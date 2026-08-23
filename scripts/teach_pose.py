"""Teach named arm poses by demonstration.

Torque stays OFF throughout — the arm is limp and you position it by
hand, exactly like calibrate_arm.py. Nothing here commands motion.

    python scripts/teach_pose.py --port /dev/ttyACM0                 # teach the standard set
    python scripts/teach_pose.py --port /dev/ttyACM0 --name STOW     # teach/replace one
    python scripts/teach_pose.py --list                              # show what's taught
    python scripts/teach_pose.py --delete STOW

Poses are saved to config/poses.json in degrees. Re-teach after any
recalibration: calibration shifts the zero pose, so previously taught
angles then point somewhere else.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.poses import DEFAULT_POSES_PATH, Pose, load_poses, save_poses

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


def show(poses: dict[str, Pose]) -> None:
    if not poses:
        print("No poses taught yet. Run without --list to teach the standard set.")
        return
    print(f"{len(poses)} pose(s) in {DEFAULT_POSES_PATH}:")
    for pose in poses.values():
        angles = ", ".join(f"{np.degrees(a):+.0f}" for a in pose.joints_rad)
        print(f"  {pose.name:<10} [{angles}] gripper={pose.gripper:.2f}  {pose.description}")


def run(port: str, path: Path, names: list[tuple[str, str]]) -> None:
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

    names = [(args.name, args.description)] if args.name else STANDARD_POSES
    run(args.port, args.poses, names)


if __name__ == "__main__":
    main()
