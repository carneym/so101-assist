"""Single-write diagnostic — isolates whether write_joint_targets has
any physical effect on ONE joint, without motion_test.py's repeated
ratcheting move_to loop or convergence/timeout logic in the way.

Reads current position, issues exactly ONE write_joint_targets call
with a delta small enough to land in a single step (no clipping
across multiple calls needed), waits `--settle` seconds, reads again,
and prints the before/after/delta. Torque is released afterward
regardless of what happened.

    python scripts/diag_single_write.py --port COM3 --joint wrist_roll --delta-deg 5
    python scripts/diag_single_write.py --port COM3 --gripper --delta 0.1
"""
from __future__ import annotations

import argparse
import math
import time

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver


def run(port: str, joint: str | None, gripper: bool, delta_deg: float, delta_gripper: float, settle_s: float) -> None:
    driver = SO101Driver(port=port)
    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit("Not calibrated — run scripts/calibrate_arm.py first.")

    before_pos, _, before_gripper = driver.read_state()
    print("before:   ", dict(zip(JOINT_NAMES, before_pos.round(4))), "gripper=", round(before_gripper, 4))

    target_pos = before_pos.copy()
    target_gripper = before_gripper
    if joint:
        target_pos[JOINT_NAMES.index(joint)] += math.radians(delta_deg)
    if gripper:
        target_gripper = before_gripper + delta_gripper

    print("commanding:", dict(zip(JOINT_NAMES, target_pos.round(4))), "gripper=", round(target_gripper, 4))
    input("Workspace clear? Press ENTER for ONE write, Ctrl-C to abort...")

    driver.enable_torque()
    try:
        driver.write_joint_targets(target_pos, target_gripper)
        time.sleep(settle_s)
        after_pos, _, after_gripper = driver.read_state()
        print("after:    ", dict(zip(JOINT_NAMES, after_pos.round(4))), "gripper=", round(after_gripper, 4))
        print(
            "delta:    ",
            dict(zip(JOINT_NAMES, (after_pos - before_pos).round(4))),
            "gripper=",
            round(after_gripper - before_gripper, 4),
        )
    finally:
        driver.torque_off()
        print("torque released.")
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyACM0")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--joint", choices=JOINT_NAMES, help="joint to move")
    target.add_argument("--gripper", action="store_true", help="move the gripper instead")
    parser.add_argument("--delta-deg", type=float, default=5.0, help="joint delta, degrees (<= default step cap)")
    parser.add_argument("--delta", type=float, default=0.1, help="gripper delta, 0..1 fraction (<= default step cap)")
    parser.add_argument("--settle", type=float, default=1.5, help="seconds to wait before re-reading")
    args = parser.parse_args()
    run(args.port, args.joint, args.gripper, args.delta_deg, args.delta, args.settle)


if __name__ == "__main__":
    main()
