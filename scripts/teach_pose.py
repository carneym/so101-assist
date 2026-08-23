"""Teach named arm poses by demonstration.

Torque stays OFF throughout — the arm is limp and you position it by
hand, exactly like calibrate_arm.py. Nothing here commands motion.

    python scripts/teach_pose.py --port /dev/ttyACM0                  # the standard set, end points only
    python scripts/teach_pose.py --port /dev/ttyACM0 --name STOW      # teach/replace one point
    python scripts/teach_pose.py --port /dev/ttyACM0 --record --name STOW   # record the whole PATH there
    python scripts/teach_pose.py --port /dev/ttyACM0 --gesture --name WAVE  # record a gesture
    python scripts/teach_pose.py --list
    python scripts/teach_pose.py --delete STOW

A recorded PATH is followed on playback instead of interpolating
straight to the end point, so the arm goes the way you showed it —
around whatever you went around. A GESTURE is a motion whose point is
the motion (wave, beckon); it ends where it began, replays at the tempo
you demonstrated, and repeats if you recorded a cycle.

Poses are saved to config/poses.json in degrees. Re-teach after any
recalibration: calibration shifts the zero pose, so previously taught
angles then point somewhere else.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from so101_assist.arm.bus_watchdog import TRANSIENT_BUS_ERRORS
from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.poses import (
    DEFAULT_POSES_PATH,
    KIND_GESTURE,
    KIND_POSE,
    Pose,
    load_poses,
    save_poses,
)
from so101_assist.arm.trajectory import (
    Waypoint,
    duration,
    is_loop,
    peak_joint_rate,
    simplify,
)

RECORD_HZ = 20.0

# The standard set offered when no --name is given. Order is the order
# the operator scrolls them in pose mode, so the safest, most-used pose
# (HOME) comes first.
STANDARD_POSES = [
    ("HOME", "tucked in close, gripper closed — the safe resting pose"),
    ("RAISED", "lifted clear of the table, ready to move"),
    ("EXTENDED", "reached out at full extension"),
]


def capture(driver: SO101Driver, name: str, description: str) -> Pose:
    joint_pos, _, gripper_pos = driver.read_state()
    print(f"    captured {name}: " + ", ".join(
        f"{j}={np.degrees(a):+.1f}deg" for j, a in zip(JOINT_NAMES, joint_pos)
    ) + f", gripper={gripper_pos:.2f}")
    return Pose(name=name, joints_rad=joint_pos.copy(), gripper=float(gripper_pos), description=description)


def record(driver: SO101Driver, name: str, description: str, kind: str) -> Pose:
    """Sample the arm while the operator moves it by hand.

    Torque stays off throughout — this only reads. Sampling runs until
    ENTER; a dropped sample from a bus hiccup is skipped rather than
    ending the take, because losing a 10-second demonstration to one
    noisy packet is not acceptable.
    """
    print("    Press ENTER to START recording, then move the arm through the motion.")
    input()
    stop = threading.Event()
    threading.Thread(
        target=lambda: (input(), stop.set()), daemon=True
    ).start()
    print("    RECORDING — press ENTER again to stop.")

    samples: list[Waypoint] = []
    dropped = 0
    t0 = time.monotonic()
    period = 1.0 / RECORD_HZ
    while not stop.is_set():
        tick = time.monotonic()
        try:
            joints, _, gripper = driver.read_state()
            samples.append(Waypoint(t=tick - t0, joints_rad=joints.copy(), gripper=float(gripper)))
        except TRANSIENT_BUS_ERRORS:
            dropped += 1
        elapsed = time.monotonic() - tick
        if elapsed < period:
            time.sleep(period - elapsed)

    if len(samples) < 2:
        raise SystemExit("Recording too short — nothing captured.")

    raw = len(samples)
    samples = simplify(samples)
    # Timestamps start at zero so playback math never depends on when
    # the take happened.
    t_start = samples[0].t
    samples = [Waypoint(w.t - t_start, w.joints_rad, w.gripper) for w in samples]
    loop = is_loop(samples)

    print(f"    captured {duration(samples):.1f}s: {raw} samples -> {len(samples)} waypoints"
          + (f", {dropped} dropped to bus errors" if dropped else ""))
    print(f"    peak joint rate {peak_joint_rate(samples):.2f} rad/s"
          + ("  (cyclic — can repeat)" if loop else ""))
    return Pose(
        name=name,
        joints_rad=samples[-1].joints_rad.copy(),
        gripper=samples[-1].gripper,
        description=description,
        kind=kind,
        path=samples,
        loop=loop,
    )


def show(poses: dict[str, Pose]) -> None:
    if not poses:
        print("No poses taught yet. Run without --list to teach the standard set.")
        return
    print(f"{len(poses)} taught in {DEFAULT_POSES_PATH}:")
    for pose in poses.values():
        if pose.has_path:
            shape = f"path {len(pose.path):>3} wpts {duration(pose.path):4.1f}s" + (
                " loop" if pose.loop else "     ")
        else:
            shape = "single point           "
        print(f"  {pose.name:<10} {pose.kind:<8} {shape}  {pose.description}")


def run(port: str, path: Path, names: list[tuple[str, str]], kind: str, as_path: bool) -> None:
    poses = load_poses(path)
    driver = SO101Driver(port=port)
    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit("Not calibrated — run scripts/calibrate_arm.py first.")

    # Explicit: this script never enables torque, and connect() doesn't
    # either. The arm is backdrivable the whole time.
    print("\nThe arm is LIMP — position it by hand. Nothing will move on its own.\n")
    try:
        for name, description in names:
            if name in poses:
                print(f"[{name}] already taught — teaching again REPLACES it.")
            print(f"[{name}] {description}")
            if as_path:
                answer = input("    ENTER to set up this recording (s = skip): ")
                if answer.strip().lower() == "s":
                    print("    skipped")
                    continue
                poses[name] = record(driver, name, description, kind)
            else:
                answer = input(f"    Move the arm into {name}, then ENTER to capture (s = skip): ")
                if answer.strip().lower() == "s":
                    print("    skipped")
                    continue
                poses[name] = capture(driver, name, description)
            save_poses(poses, path)      # save as we go, so a Ctrl-C keeps earlier poses
            print(f"    saved to {path}")
    except KeyboardInterrupt:
        print("\ninterrupted — poses captured so far are saved.")
    finally:
        driver.disconnect()

    print()
    show(load_poses(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES_PATH)
    parser.add_argument("--name", help="teach just this pose (creates or replaces it)")
    parser.add_argument("--description", default="", help="description for --name")
    parser.add_argument("--list", action="store_true", help="list taught poses and exit")
    parser.add_argument("--delete", metavar="NAME", help="remove a taught pose and exit")
    parser.add_argument(
        "--record", action="store_true",
        help="record the whole PATH (move the arm through the motion), not just the end point",
    )
    parser.add_argument(
        "--gesture", action="store_true",
        help="record a gesture (a motion, e.g. WAVE) rather than a place to be. Implies --record.",
    )
    args = parser.parse_args()

    if args.list:
        show(load_poses(args.poses))
        return

    if args.delete:
        poses = load_poses(args.poses)
        if args.delete not in poses:
            raise SystemExit(f"No pose named '{args.delete}'. Taught: {', '.join(poses) or '(none)'}")
        del poses[args.delete]
        save_poses(poses, args.poses)
        print(f"deleted {args.delete}")
        return

    if not args.port:
        raise SystemExit("--port is required to teach (use --list/--delete without it).")

    if args.gesture and not args.name:
        raise SystemExit("--gesture needs --name, e.g. --gesture --name WAVE")
    kind = KIND_GESTURE if args.gesture else KIND_POSE
    names = [(args.name, args.description)] if args.name else STANDARD_POSES
    run(args.port, args.poses, names, kind, as_path=args.record or args.gesture)


if __name__ == "__main__":
    main()
