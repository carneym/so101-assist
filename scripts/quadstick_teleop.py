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
- WRIST:    LEFT-tube sip/puff -> z velocity,
            stick left/right -> wrist roll, stick up/down -> wrist flex

Named poses (taught with scripts/teach_pose.py): RIGHT puff opens the
pose menu, stick up/down chooses, lip switch goes, RIGHT sip exits or
aborts a move in progress. Jogging is suspended while the menu is open
and while a move runs.

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
import contextlib
import math
import time
from pathlib import Path

import numpy as np
import yaml

from so101_assist.arm.bus_watchdog import TRANSIENT_BUS_ERRORS, BusWatchdog
from so101_assist.arm.controller import CartesianController
from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.kinematics import JOINT_LIMITS_RAD
from so101_assist.arm.poses import (
    DEFAULT_MATCH_TOL_RAD,
    DEFAULT_POSES_PATH,
    Pose,
    load_poses,
    match_pose,
)
from so101_assist.arm.safety import LoadMonitor, WorkspaceFence
from so101_assist.arm.trajectory import (
    TrajectoryPlayer,
    nearest_waypoint_index,
    speed_scale_for_cap,
)
from so101_assist.bus import TOPIC_DETECTIONS, TOPIC_JOG, TOPIC_POSE, Bus
from so101_assist.control.inputs.quadstick import QuadStickDevice
from so101_assist.control.pose_menu import PoseMenu
from so101_assist.control.tuning import DEFAULT_TUNING_PATH, write_status
from so101_assist.control.tuning import load as load_tuning
from so101_assist.control.tuning import resolve as resolve_tuning
from so101_assist.messages import (
    CartesianVelocity,
    Detection,
    JogCommand,
    JogMode,
    PoseAction,
)
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
# A floor-guarded pose move slides along the z floor rather than through
# it, which means a move whose straight joint path is blocked makes no
# further progress instead of arriving. Call it stalled once the largest
# remaining joint error stops shrinking by this much over this long.
POSE_STALL_EPS_RAD = 0.005
POSE_STALL_S = 1.5
# Gestures replay at the tempo they were demonstrated, capped: fast
# motion near a person is a different safety case from a slow move to a
# pose above a table. The whole motion is scaled by one factor so its
# rhythm survives (see trajectory.py).
GESTURE_MAX_JOINT_RADPS = 1.0
GESTURE_REPEATS = 3          # cycles for a cyclic gesture (one taught wave -> a wave)
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


def make_player(pose: Pose, current_joints) -> TrajectoryPlayer | None:
    """Player for a recorded path, or None to move straight to the end.

    A GESTURE always starts at its first waypoint — a wave that begins
    halfway through isn't a wave. A POSE with a path joins at the
    nearest waypoint instead, so the arm doesn't first travel back to
    the start of a route it's already partway along.
    """
    if not pose.has_path:
        return None
    waypoints = pose.path
    if not pose.is_gesture and current_joints is not None:
        waypoints = waypoints[nearest_waypoint_index(waypoints, current_joints):] or waypoints
    speed = speed_scale_for_cap(pose.path, GESTURE_MAX_JOINT_RADPS)
    return TrajectoryPlayer(
        waypoints,
        speed=speed,
        repeats=GESTURE_REPEATS if (pose.is_gesture and pose.loop) else 1,
    )


def print_pose_menu(menu: PoseMenu) -> None:
    """Mirror the on-screen menu to the console — the operator may be
    looking at either, and a helper watching the terminal should see
    the same thing the video shows."""
    print("\n== POSE MENU ==  stick up/down = choose, lip switch = go, right sip = exit")
    if not menu.names:
        print("  (no poses taught — run scripts/teach_pose.py)")
        return
    for i, name in enumerate(menu.names):
        pose = menu.poses[name]
        tag = "  (gesture)" if pose.is_gesture else ("  (path)" if pose.has_path else "")
        print(f"  {'>' if i == menu.selected else ' '} {name}{tag}")


def build_notes(
    controller: CartesianController,
    load_warn_threshold: float | None,
    poses: dict[str, Pose] | None = None,
    menu: PoseMenu | None = None,
    bus_error: bool = False,
    match_tol_rad: float = DEFAULT_MATCH_TOL_RAD,
) -> list[Note]:
    """HUD notes for the current controller state (no hardware reads —
    everything here is what the last control tick already measured)."""
    pose = match_pose(controller.last_joint_pos, poses or {}, match_tol_rad) if poses else None
    return arm_notes(
        joint_load=controller.last_joint_load,
        joint_names=JOINT_NAMES,
        load_warn_threshold=load_warn_threshold,
        stopped=controller.stopped,
        bus_error=bus_error,
        fence_blocks=controller.last_fence_blocks,
        limit_clips=controller.last_limit_clips,
        pose=pose,
        pose_menu=menu.menu_lines() if menu is not None else None,
        pose_selected=menu.selected if menu is not None else 0,
        pose_moving=menu.moving.name if menu is not None and menu.moving else None,
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
    # Pose-move speed, converted to a per-tick step. Deliberately slower
    # than jogging: the operator isn't steering this one, so it has to be
    # slow enough to watch and abort.
    pose_step_rad = cfg["arm"].get("pose_speed_radps", 0.4) / LOOP_HZ
    pose_gripper_step = cfg["arm"].get("pose_gripper_speed", 0.5) / LOOP_HZ
    gesture_step_rad = cfg["arm"].get("gesture_speed_radps", GESTURE_MAX_JOINT_RADPS) / LOOP_HZ
    # Pose moves are guarded against the fence's z floor only: joint
    # interpolation sags through the table even between two safe poses.
    pose_z_floor = cfg.get("workspace_fence", {}).get("z", [None])[0]
    poses = load_poses(DEFAULT_POSES_PATH)
    if poses:
        print(f"[poses] {len(poses)} taught: {', '.join(poses)}")
    else:
        print("[poses] none taught — run scripts/teach_pose.py to add them")
    max_linear = cfg["arm"]["max_linear_mps"]
    max_wrist = cfg["arm"]["max_wrist_radps"]
    max_elbow = cfg["arm"].get("max_elbow_radps", 0.5)
    max_shoulder = cfg["arm"].get("max_shoulder_radps", 0.4)

    joint_limits = dict(JOINT_LIMITS_RAD)
    for name, (lo_deg, hi_deg) in cfg["arm"].get("joint_limits_deg", {}).items():
        joint_limits[name] = (math.radians(lo_deg), math.radians(hi_deg))

    bus = Bus()
    sub = bus.subscribe(TOPIC_JOG, maxsize=4)
    # Pose events are discrete actions, drained (not collapsed) every
    # tick — see Subscription.drain. Sized to hold a tick's worth.
    pose_sub = bus.subscribe(TOPIC_POSE, maxsize=32)
    menu = PoseMenu(poses)
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
    # Survive a USB/servo-bus hiccup instead of dying on it: hold
    # position, reopen the port, and only shut down if it stays dead.
    watchdog = BusWatchdog(driver)
    pose_best_err = float("inf")
    pose_best_at = 0.0
    player: TrajectoryPlayer | None = None      # set while following a recorded path
    try:
        # INSIDE the try: enable_torque writes two registers per motor
        # and can fail partway through, leaving earlier motors powered.
        # Outside, that partial enable escaped the finally below and the
        # script exited with the arm still energized.
        driver.enable_torque()

        while True:
            tick_start = time.monotonic()

            for event in pose_sub.drain():
                was_open, was_moving = menu.open, menu.moving
                started = menu.handle(event)
                if menu.open and (not was_open or event.action is PoseAction.CYCLE):
                    print_pose_menu(menu)     # opened, or the selection moved
                if started is not None:
                    pose_best_err, pose_best_at = float("inf"), tick_start
                    player = make_player(started, controller.last_joint_pos)
                    how = "following recorded path" if player else "moving"
                    if started.is_gesture:
                        how = "performing"
                    print(f"\n[pose] {how} {started.name} — right sip aborts.")
                elif event.action is PoseAction.CANCEL:
                    player = None
                elif event.action is PoseAction.CANCEL:
                    if was_moving is not None:
                        print(f"[pose] move to {was_moving.name} aborted — arm holds position.")
                    elif was_open:
                        print("[pose] menu closed")

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
                    image = draw_hud(
                        image, lines,
                        build_notes(
                            controller, load_warn_threshold, poses, menu,
                            bus_error=watchdog.consecutive > 0,
                        ),
                    )
                    cv2.imshow(CAMERA_WINDOW, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nstopping (q)...")
                    break

            # A pose move owns the arm while it runs: jog is drained and
            # discarded so a stick nudge can't fight the move, and the
            # only ways out are arrival, right sip (CANCEL), a load trip,
            # or Ctrl-C.
            if menu.moving is not None:
                sub.latest()   # discard jog accumulated during the move
                try:
                    if player is not None:
                        # Follow the recorded path: command the sample at
                        # the current playback time, and only advance that
                        # time while the arm is keeping up (the player's
                        # own leash) so a lagging joint can't let the
                        # target run away down the path.
                        wp_joints, wp_gripper = player.target()
                        controller.step_to_joints(
                            wp_joints, wp_gripper,
                            max_step_rad=pose_step_rad if not menu.moving.is_gesture
                            else gesture_step_rad,
                            max_gripper_step=pose_gripper_step,
                            z_floor=pose_z_floor,
                        )
                        arrived = player.advance(period, controller.last_joint_pos)
                    else:
                        arrived = controller.step_to_joints(
                            menu.moving.joints_rad,
                            menu.moving.gripper,
                            max_step_rad=pose_step_rad,
                            max_gripper_step=pose_gripper_step,
                            z_floor=pose_z_floor,
                        )
                    watchdog.record_success()
                except RuntimeError:
                    print(f"\n[pose] STOPPED during move to {menu.moving.name} — holding.")
                    menu.moving = None
                    player = None
                    arrived = False
                except TRANSIENT_BUS_ERRORS as exc:
                    if not watchdog.record_failure(exc, time.monotonic()):
                        break
                    arrived = False
                if arrived:
                    verb = "performed" if menu.moving.is_gesture else "arrived at"
                    print(f"[pose] {verb} {menu.moving.name}.")
                    menu.moving = None
                    player = None
                    last_jog = None    # don't apply a stale pre-move stick reading
                    pose_best_err = float("inf")
                elif (
                    menu.moving is not None
                    and controller.last_joint_pos is not None
                    and (player is None or not player.waiting)
                ):
                    err = float(np.max(np.abs(menu.moving.joints_rad - controller.last_joint_pos)))
                    if err < pose_best_err - POSE_STALL_EPS_RAD:
                        pose_best_err, pose_best_at = err, tick_start
                    elif tick_start - pose_best_at > POSE_STALL_S:
                        print(
                            f"\n[pose] move to {menu.moving.name} STALLED at the z floor "
                            f"({err:.2f} rad short) — the straight path there goes under the "
                            "table, and sliding along the floor can't get around it.\n"
                            "       Jog clear of the table and try again."
                        )
                        menu.moving = None
                        player = None
                        pose_best_err = float("inf")
                elapsed = time.monotonic() - tick_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue

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

            try:
                ticked = controller.tick(cmd, dt=period)
            except TRANSIENT_BUS_ERRORS as exc:
                # The arm holds by itself while the bus is down: torque
                # is a servo-side register, so commanding nothing IS the
                # safe action here.
                if not watchdog.record_failure(exc, time.monotonic()):
                    break
                elapsed = time.monotonic() - tick_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue
            watchdog.record_success()

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
        try:
            driver.torque_off()
            print("torque released.")
        except TRANSIENT_BUS_ERRORS as exc:
            # Raising here would replace the real error with this one and
            # skip the disconnect. The operator needs the plain-language
            # version anyway: software has run out of options.
            print(f"\n*** COULD NOT RELEASE TORQUE: {exc}")
            print("*** THE ARM MAY STILL BE POWERED AND HOLDING.")
            print("*** Cut power at the arm's supply — the serial link is gone,")
            print("*** so no software command can reach the servos.")
        # Closing a port that already vanished is not worth reporting.
        with contextlib.suppress(*TRANSIENT_BUS_ERRORS):
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
