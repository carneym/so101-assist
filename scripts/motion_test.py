"""First-motion smoke test — SO-101 arm, one joint (or the gripper),
small and slow.

This is the first script in this project that can move the arm under
power. Clear the workspace and be ready to Ctrl-C before confirming.

Moves the chosen joint by a small commanded delta and back to where
it started, then releases torque. Every write goes through
SO101Driver.write_joint_targets, so each individual call can only
move max_joint_step_rad / max_gripper_step (see arm/driver.py) —
this script reaches the requested delta by calling it repeatedly in
small steps, not with one big jump. Requires a calibration file
(config/calibration/arm.json — see calibrate_arm.py): calibration is
also what gives the servos real firmware-enforced position limits, a
second, independent backstop.

    python scripts/motion_test.py --port COM3 --joint wrist_roll --delta-deg 15
    python scripts/motion_test.py --port COM3 --gripper --delta 0.3
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver

STEP_HZ = 10.0
# Looser than the servo's raw encoder resolution (~0.0015 rad/tick) but
# comfortably above its observed positioning deadband — a residual error
# smaller than this isn't worth waiting on, the servo may just never
# bother closing it.
TOL_RAD = 0.02
TOL_GRIPPER = 0.02
TIMEOUT_S = 10.0

# If position hasn't moved more than this in STALL_ITERS consecutive
# iterations, stop waiting instead of burning the full timeout — a real
# stall (servo considers itself "close enough" short of TOL_RAD) looks
# identical to a stuck loop otherwise.
STALL_EPS = 0.002
STALL_ITERS = 10   # ~1s at STEP_HZ


def move_to(driver: SO101Driver, target_pos: np.ndarray, target_gripper: float, verbose: bool = False) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    prev_pos, prev_gripper = None, None
    stall_count = 0
    i = 0
    while time.monotonic() < deadline:
        joint_pos, _, gripper_pos = driver.read_state()
        pos_err = np.abs(target_pos - joint_pos)
        gripper_err = abs(target_gripper - gripper_pos)
        pos_done = np.all(pos_err < TOL_RAD)
        gripper_done = gripper_err < TOL_GRIPPER

        if prev_pos is not None and np.all(np.abs(joint_pos - prev_pos) < STALL_EPS) and (
            abs(gripper_pos - prev_gripper) < STALL_EPS
        ):
            stall_count += 1
        else:
            stall_count = 0
        prev_pos, prev_gripper = joint_pos, gripper_pos

        if verbose:
            print(
                f"  [{i:03d}] pos={joint_pos.round(4)} err={pos_err.round(4)} "
                f"gripper={gripper_pos:.4f} gripper_err={gripper_err:.4f} "
                f"pos_done={pos_done} gripper_done={gripper_done} stall={stall_count}"
            )
        if pos_done and gripper_done:
            if verbose:
                print(f"  converged after {i} iterations")
            return
        if stall_count >= STALL_ITERS:
            print(
                f"  (stalled short of tolerance — treating as converged; "
                f"residual pos_err={pos_err.round(4)} gripper_err={gripper_err:.4f})"
            )
            return
        driver.write_joint_targets(target_pos, target_gripper)
        time.sleep(1.0 / STEP_HZ)
        i += 1
    print("  (timed out approaching target — stopped where it got to)")


def run(
    port: str, joint: str | None, gripper: bool, delta_deg: float, delta_gripper: float, verbose: bool = False
) -> None:
    driver = SO101Driver(port=port)
    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit(
            "Arm is not calibrated (config/calibration/arm.json missing). "
            "Run scripts/calibrate_arm.py first — refusing to move an uncalibrated arm."
        )

    start_pos, _, start_gripper = driver.read_state()
    print("start:", dict(zip(JOINT_NAMES, start_pos.round(3))), "gripper=", round(start_gripper, 3))

    target_pos = start_pos.copy()
    target_gripper = start_gripper
    if joint:
        target_pos[JOINT_NAMES.index(joint)] += math.radians(delta_deg)
    if gripper:
        target_gripper = start_gripper + delta_gripper

    input("Workspace clear? Press ENTER to enable torque and move, Ctrl-C to abort...")

    try:
        # Inside the try: a partial enable_torque (it can fail
        # midway, leaving earlier motors powered) must still hit
        # the torque_off below.
        driver.enable_torque()
        print("moving out...")
        move_to(driver, target_pos, target_gripper, verbose)
        time.sleep(0.5)
        print("moving back to start...")
        move_to(driver, start_pos, start_gripper, verbose)
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
    parser.add_argument("--delta-deg", type=float, default=15.0, help="joint delta, degrees")
    parser.add_argument("--delta", type=float, default=0.3, help="gripper delta, 0..1 fraction")
    parser.add_argument("--verbose", action="store_true", help="print per-iteration progress")
    args = parser.parse_args()
    run(args.port, args.joint, args.gripper, args.delta_deg, args.delta, args.verbose)


if __name__ == "__main__":
    main()
