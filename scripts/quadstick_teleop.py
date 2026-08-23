"""QuadStick teleop — drive the SO-101 with the QuadStick, end to end.

Wires QuadStickDevice (pygame joystick -> JogCommand on the bus)
through CartesianController (velocity caps, workspace fence, load
monitor) to the arm. The QuadStick must be in its Joystick profile
(see so101_assist/control/inputs/quadstick.py) and the arm calibrated
(scripts/calibrate_arm.py).

Jog mapping (lip switch cycles modes; a cheat-sheet prints on every
mode change). CENTER sip/puff = gripper close/open in EVERY mode —
grasping never needs a mode switch:
- SHOULDER: stick left/right -> shoulder_pan (right = arm right),
            stick up/down -> shoulder_lift (up = arm up)
- ELBOW:    stick up/down -> elbow joint (up = arm up/reach out),
            stick left/right -> shoulder_pan
- WRIST:    LEFT/RIGHT-tube sip/puff -> z velocity,
            stick left/right -> wrist roll, stick up/down -> wrist flex

The wrist camera feed is shown in an OpenCV window while driving (the
active mode name and live end-effector xyz are overlaid). It degrades
gracefully — if the camera can't open, teleop still runs; pass
--no-camera to skip it entirely.
Press q in the video window to stop (same as Ctrl-C).

Pass --detect to additionally run open-vocab object detection on the
wrist feed and overlay numbered boxes (so101_assist/perception/
detector.py, using the `perception` block of --config). Runs on its
own thread at perception.detect_hz, decoupled from the control loop.
Requires the `perception` extra and does nothing without a camera.

Safety in this loop, layered:
- deadman: if no fresh JogCommand within --stale-s (QuadStick
  unplugged, pygame hung, thread died), velocity goes to zero
- controller: velocity caps + workspace fence + load-monitor STOP
  (a load trip halts and holds; Ctrl-C then restart to resume —
  deliberately NOT auto-resumed from this script)
- driver: per-call step clamp; Ctrl-C releases torque in `finally`

    python scripts/quadstick_teleop.py --port COM3
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import yaml

from so101_assist.arm.controller import CartesianController
from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.kinematics import JOINT_LIMITS_RAD
from so101_assist.arm.safety import LoadMonitor, WorkspaceFence
from so101_assist.bus import TOPIC_DETECTIONS, TOPIC_JOG, Bus
from so101_assist.control.inputs.quadstick import QuadStickDevice
from so101_assist.control.tuning import DEFAULT_TUNING_PATH, write_status
from so101_assist.control.tuning import load as load_tuning
from so101_assist.control.tuning import resolve as resolve_tuning
from so101_assist.messages import CartesianVelocity, Detection, JogCommand, JogMode
from so101_assist.perception.camera import CameraNode
from so101_assist.perception.detector import DetectorNode
from so101_assist.ui.notes import arm_notes
from so101_assist.ui.overlay import Note, draw_detections, draw_hud

LOOP_HZ = 25.0
CAMERA_WINDOW = "so101-assist wrist"
# Fraction of the load-guard threshold at which the HUD starts warning,
# so the operator sees a servo straining BEFORE the guard stops the arm.
# Used only when arm.load_warn_threshold isn't set explicitly.
LOAD_WARN_FRACTION = 0.75
TUNING_POLL_S = 0.4
STATUS_WRITE_S = 0.2   # publish EE position ~5 Hz for the tuning GUI readout


def resolve_load_warn(arm_cfg: dict) -> float | None:
    """Load level at which the HUD warns, normalized 0..1.

    Explicit `arm.load_warn_threshold` wins. Otherwise derive it from
    the guard's stop threshold so the warning always leads the trip.
    Returns None only when neither is available — i.e. the guard is
    disabled AND no warn level was set, so there's no meaningful "high"
    to compare against.
    """
    explicit = arm_cfg.get("load_warn_threshold")
    if explicit is not None:
        return float(explicit)
    stop = arm_cfg.get("load_stop_threshold")
    return float(stop) * LOAD_WARN_FRACTION if stop is not None else None


def build_notes(controller: CartesianController, load_warn_threshold: float | None) -> list[Note]:
    """HUD notes for the current controller state (no hardware reads —
    everything here is what the last control tick already measured)."""
    return arm_notes(
        joint_load=controller.last_joint_load,
        joint_names=JOINT_NAMES,
        load_warn_threshold=load_warn_threshold,
        stopped=controller.stopped,
        fence_blocks=controller.last_fence_blocks,
        limit_clips=controller.last_limit_clips,
    )


def apply_tuning(controller: CartesianController, driver: SO101Driver, resolved: dict) -> None:
    """Push resolved tuning values onto the LIVE controller/driver.
    Every setting here is a plain mutable attribute the control loop
    reads each tick, so changes take effect on the next tick."""
    controller.max_linear_mps = resolved["max_linear_mps"]
    controller.max_wrist_radps = resolved["max_wrist_radps"]
    controller.max_elbow_radps = resolved["max_elbow_radps"]
    controller.max_shoulder_radps = resolved["max_shoulder_radps"]
    controller.setpoint_leash_rad = resolved["setpoint_leash_rad"]
    driver.max_joint_step_rad = math.radians(resolved["max_joint_step_deg"])
    controller.joint_limits = {
        j: (math.radians(lo), math.radians(hi)) for j, (lo, hi) in resolved["joint_limits_deg"].items()
    }
    fence = controller.fence
    if fence is not None:
        wf = resolved["workspace_fence"]
        fence.lo = np.array([wf["x"][0], wf["y"][0], wf["z"][0]])
        fence.hi = np.array([wf["x"][1], wf["y"][1], wf["z"][1]])
    monitor = controller.load_monitor
    if monitor is not None:
        threshold = resolved["load_stop_threshold"]
        re_enabling = monitor.threshold is None and threshold is not None
        monitor.threshold = threshold
        monitor.ticks = int(resolved["load_stop_ticks"])
        if re_enabling:
            monitor.reset()   # don't inherit a stale trip when turning the guard back on
    # Servo torque caps (serial writes — requires driver.connect() first).
    if driver.bus.is_connected:
        driver.set_torque_limit(resolved["arm_torque_limit_pct"] / 100.0, motors=JOINT_NAMES)
        driver.set_torque_limit(resolved["gripper_torque_limit_pct"] / 100.0, motors=["gripper"])


def jog_to_velocity(
    jog: JogCommand, max_linear: float, max_wrist: float, max_elbow: float, max_shoulder: float
) -> CartesianVelocity:
    """Map a normalized JogCommand to a CartesianVelocity per JogMode.

    Axes semantics per mode are set by QuadStickDevice._read_jog (see
    its module docstring); here they just get scaled to velocity caps.

    gripper_delta (center sip/puff) maps to the gripper channel in
    EVERY mode — grasping never needs a mode switch.

    Up/down sign conventions: stick-up reads negative, and positive
    elbow/lift rates move the EE down, so those values pass through
    un-negated (stick up = arm up). Pan is negated so stick-right
    moves the arm visually right (positive pan rate moves the EE
    toward -y).
    """
    linear = np.zeros(3)
    wrist = np.zeros(2)
    shoulder = np.zeros(2)
    elbow = 0.0
    if jog.mode is JogMode.SHOULDER:
        shoulder[0] = -jog.axes[0] * max_shoulder
        shoulder[1] = jog.axes[1] * max_shoulder
    elif jog.mode is JogMode.ELBOW:
        shoulder[0] = -jog.axes[0] * max_shoulder
        elbow = jog.axes[1] * max_elbow
    elif jog.mode is JogMode.WRIST:
        linear[2] = jog.axes[0] * max_linear
        wrist[0] = jog.axes[2] * max_wrist
        wrist[1] = jog.axes[1] * max_wrist
    return CartesianVelocity(
        linear=linear, wrist=wrist, shoulder=shoulder, elbow=elbow, gripper=jog.gripper_delta
    )


def mode_help(
    mode: JogMode,
    joint_limits: dict[str, tuple[float, float]],
    max_linear: float,
    max_wrist: float,
    max_elbow: float,
    max_shoulder: float,
) -> str:
    """Console cheat-sheet for one mode: controls, limits, caps —
    printed on every mode change so the operator never wonders what a
    stick deflection will do (the console stands in for the UI's
    audible/visual mode announcements until ui/ exists)."""

    def deg(name: str) -> str:
        lo, hi = joint_limits[name]
        return f"{math.degrees(lo):+.0f}..{math.degrees(hi):+.0f} deg"

    gripper_line = "  center puff/sip  -> gripper open/close (works in every mode)"

    if mode is JogMode.SHOULDER:
        return (
            f"  stick left/right -> shoulder pan (right = right) {deg('shoulder_pan')} @ {max_shoulder} rad/s\n"
            f"  stick up/down    -> shoulder lift (up = arm up)  {deg('shoulder_lift')} @ {max_shoulder} rad/s\n"
            f"{gripper_line}\n"
            "  fence: BYPASSED - can reach table/box edges, load monitor guards contact"
        )
    if mode is JogMode.ELBOW:
        return (
            f"  stick up/down    -> elbow (up = arm up)          {deg('elbow')} @ {max_elbow} rad/s\n"
            f"  stick left/right -> shoulder pan (right = right) {deg('shoulder_pan')} @ {max_shoulder} rad/s\n"
            f"{gripper_line}\n"
            "  fence: ACTIVE for elbow, BYPASSED for pan"
        )
    if mode is JogMode.WRIST:
        return (
            f"  side puff / sip  -> up / down (z)                {max_linear * 100:.0f} cm/s, fence box\n"
            f"  stick left/right -> wrist roll                   {deg('wrist_roll')} @ {max_wrist} rad/s\n"
            f"  stick up/down    -> wrist flex                   {deg('wrist_flex')} @ {max_wrist} rad/s\n"
            f"{gripper_line}\n"
            "  fence: ACTIVE for z (wrist unfenced)"
        )
    return ""


def run(
    port: str,
    config_path: Path,
    stale_s: float,
    debug: bool = False,
    camera: str | None = "wrist",
    detect: bool = False,
) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    load_warn_threshold = resolve_load_warn(cfg["arm"])
    max_linear = cfg["arm"]["max_linear_mps"]
    max_wrist = cfg["arm"]["max_wrist_radps"]
    max_elbow = cfg["arm"].get("max_elbow_radps", 0.5)
    max_shoulder = cfg["arm"].get("max_shoulder_radps", 0.4)

    joint_limits = dict(JOINT_LIMITS_RAD)
    for name, (lo_deg, hi_deg) in cfg["arm"].get("joint_limits_deg", {}).items():
        joint_limits[name] = (math.radians(lo_deg), math.radians(hi_deg))

    bus = Bus()
    sub = bus.subscribe(TOPIC_JOG, maxsize=4)
    quadstick = QuadStickDevice.from_config(bus, cfg["operator"])

    cam_node = None
    cam_sub = None
    if camera is not None:
        if camera in cfg.get("cameras", {}):
            cam_node = CameraNode.from_config(bus, camera, cfg["cameras"][camera])
            cam_sub = bus.subscribe(cam_node.topic, maxsize=1)
        else:
            print(f"[camera] '{camera}' not in config cameras block — running without video.")

    detector_node = None
    det_sub = None
    if detect:
        if cam_node is not None:
            detector_node = DetectorNode.from_config(bus, cam_node.topic, cfg["perception"], camera=camera)
            det_sub = bus.subscribe(TOPIC_DETECTIONS, maxsize=1)
        else:
            print("[detect] --detect requires a camera — running without object detection.")

    driver = SO101Driver(
        port=port,
        max_joint_step_rad=math.radians(cfg["arm"].get("max_joint_step_deg", 15)),
    )
    controller = CartesianController(
        driver,
        fence=WorkspaceFence.from_config(cfg["workspace_fence"]),
        load_monitor=LoadMonitor.from_config(cfg["arm"]),
        max_linear_mps=max_linear,
        max_wrist_radps=max_wrist,
        max_elbow_radps=max_elbow,
        max_shoulder_radps=max_shoulder,
        joint_limits=joint_limits,
        setpoint_leash_rad=cfg["arm"].get("setpoint_leash_rad", 0.2),
    )

    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit("Not calibrated — run scripts/calibrate_arm.py first.")

    # Apply settings once now (config baseline + any live-tuning
    # override), AFTER connect so servo torque writes work, then watch
    # the override file for on-the-fly changes from tuning_gui.py.
    tuning_path = DEFAULT_TUNING_PATH
    apply_tuning(controller, driver, resolve_tuning(cfg, load_tuning(tuning_path)))
    tuning_mtime = tuning_path.stat().st_mtime if tuning_path.exists() else 0.0
    if tuning_path.exists():
        print(f"[tuning] applied override from {tuning_path}")

    quadstick.start()   # daemon thread publishing JogCommand at 60 Hz
    if cam_node is not None:
        cam_node.start()   # daemon thread publishing frames; window appears once frames arrive
        print(f"Wrist camera '{camera}' starting — video window opens shortly. Press q to stop.")
    if detector_node is not None:
        detector_node.start()
        print(f"[detect] object detection running on '{camera}' at {detector_node.detect_hz} Hz.")
    print("QuadStick connected. Lip switch cycles mode; Ctrl-C stops and releases torque.")
    input("Workspace clear? Press ENTER to enable torque and start teleop...")
    driver.enable_torque()

    cv2 = None
    if cam_sub is not None:
        import cv2  # local: only needed when a camera window is shown

    zero = CartesianVelocity(linear=np.zeros(3), wrist=np.zeros(2))
    last_jog: JogCommand | None = None
    last_mode = None
    last_detections: list[Detection] | None = None
    period = 1.0 / LOOP_HZ
    last_debug = 0.0
    last_tuning_poll = 0.0
    last_status_write = 0.0
    shown_fence_blocks: list[str] = []
    shown_limit_clips: list[str] = []
    try:
        while True:
            tick_start = time.monotonic()

            if tick_start - last_tuning_poll > TUNING_POLL_S:
                last_tuning_poll = tick_start
                mtime = tuning_path.stat().st_mtime if tuning_path.exists() else 0.0
                if mtime != tuning_mtime:
                    tuning_mtime = mtime
                    apply_tuning(controller, driver, resolve_tuning(cfg, load_tuning(tuning_path)))
                    print("[tuning] settings reloaded")

            if cv2 is not None:
                frame = cam_sub.latest()
                if frame is not None:
                    image = frame.image.copy()
                    label = last_mode.name if last_mode is not None else "..."
                    lines = [f"MODE: {label}"]
                    if controller.last_ee_xyz is not None:
                        x, y, z = controller.last_ee_xyz
                        lines.append(f"xyz: {x:+.3f} {y:+.3f} {z:+.3f}")
                    if det_sub is not None:
                        # detector runs well below the 25 Hz video loop
                        # (see perception.detect_hz) — hold the last
                        # result between publishes rather than blanking
                        # the boxes out on every tick that has no fresh
                        # detections, or they visibly flash.
                        detections = det_sub.latest()
                        if detections is not None:
                            last_detections = detections
                        if last_detections:
                            image = draw_detections(image, last_detections)
                    # HUD last so a detection box can never cover it.
                    image = draw_hud(image, lines, build_notes(controller, load_warn_threshold))
                    cv2.imshow(CAMERA_WINDOW, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nstopping (q)...")
                    break

            jog = sub.latest()
            if jog is not None:
                last_jog = jog
            fresh = last_jog is not None and (time.monotonic() - last_jog.stamp) < stale_s
            # Read caps from the controller so live tuning changes flow
            # through to velocity scaling as well as the hard caps.
            cmd = (
                jog_to_velocity(
                    last_jog,
                    controller.max_linear_mps,
                    controller.max_wrist_radps,
                    controller.max_elbow_radps,
                    controller.max_shoulder_radps,
                )
                if fresh
                else zero
            )

            if last_jog is not None and last_jog.mode is not last_mode:
                last_mode = last_jog.mode
                print(f"\n== MODE: {last_mode.name} ==")
                print(mode_help(
                    last_mode, controller.joint_limits, controller.max_linear_mps,
                    controller.max_wrist_radps, controller.max_elbow_radps, controller.max_shoulder_radps,
                ))

            ticked = controller.tick(cmd, dt=period)

            # Publish EE position for the tuning GUI's live readout
            # (throttled; last_ee_xyz updates every tick, even in a hold).
            if (
                controller.last_ee_xyz is not None
                and time.monotonic() - last_status_write > STATUS_WRITE_S
            ):
                last_status_write = time.monotonic()
                write_status(controller.last_ee_xyz)

            if controller.last_fence_blocks != shown_fence_blocks:
                shown_fence_blocks = list(controller.last_fence_blocks)
                if shown_fence_blocks:
                    print(f"\n[fence] workspace boundary — blocking: {', '.join(shown_fence_blocks)}")
            if controller.last_limit_clips != shown_limit_clips:
                shown_limit_clips = list(controller.last_limit_clips)
                if shown_limit_clips:
                    print(f"\n[limit] joint at software limit: {', '.join(shown_limit_clips)}")

            if debug and time.monotonic() - last_debug > 0.3:
                last_debug = time.monotonic()
                joint_pos, joint_load, gripper_pos = driver.read_state()
                axes = last_jog.axes.round(2) if last_jog is not None else None
                fence_note = (
                    f" FENCE_BLOCKED={controller.last_fence_blocks}"
                    if controller.last_fence_blocks
                    else ""
                )
                print(
                    f"[dbg] axes={axes} grip_d={getattr(last_jog, 'gripper_delta', 0):+.1f} "
                    f"fresh={fresh} | cmd lin={cmd.linear.round(3)} wrist={cmd.wrist.round(2)} "
                    f"elbow={cmd.elbow:+.2f} grip={cmd.gripper:+.2f} | "
                    f"q={joint_pos.round(3)} load_max={joint_load.max():.2f} "
                    f"grip_pos={gripper_pos:.2f} ticked={ticked}{fence_note}"
                )

            if not ticked:
                print("\nSTOPPED (load monitor tripped?) — holding position. Ctrl-C to exit.")
                # keep looping so the hold is maintained and Ctrl-C works;
                # deliberately no auto-resume
                time.sleep(period)
                continue

            elapsed = time.monotonic() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        quadstick.stop()
        if detector_node is not None:
            detector_node.stop()
        if cam_node is not None:
            cam_node.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
        driver.torque_off()
        print("torque released.")
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--stale-s", type=float, default=0.5,
        help="deadman: zero velocity if no fresh jog input within this many seconds",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print jog input, mapped command, and joint state ~3x/sec",
    )
    parser.add_argument(
        "--camera", default="wrist",
        help="camera name from config to show while driving (default: wrist)",
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="run without the camera window",
    )
    parser.add_argument(
        "--detect", action="store_true",
        help="run open-vocab object detection on the camera feed and overlay boxes",
    )
    args = parser.parse_args()
    camera = None if args.no_camera else args.camera
    run(args.port, args.config, args.stale_s, args.debug, camera, args.detect)


if __name__ == "__main__":
    main()
