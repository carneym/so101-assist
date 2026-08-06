"""IK-commanded Cartesian move test — first commanded Cartesian motion.

Reads the current end-effector position via FK, offsets it by a small
Cartesian delta (default 3 cm, one axis at a time), solves
ik_position() seeded from the CURRENT joint config (so the solver
returns the nearby solution, not a wildly different arm posture for
the same point), previews the joint-space delta, and only moves after
explicit confirmation. Motion goes through the same capped
write_joint_targets ratcheting loop validated by motion_test.py, then
returns to the start pose and releases torque.

Refuses to run if the IK solution moves any joint more than
--max-joint-jump-deg from the current config — a solution that far
away means the solver picked a different posture branch and the arm
would swing through space to reach the "nearby" target.

    python scripts/ik_move_test.py --port COM3 --axis z --delta-cm 3
    python scripts/ik_move_test.py --port COM3 --axis x --delta-cm -3
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.kinematics import fk, ik_position

STEP_HZ = 10.0
TOL_RAD = 0.02
TIMEOUT_S = 10.0
STALL_EPS = 0.002
STALL_ITERS = 10

AXES = {"x": 0, "y": 1, "z": 2}


def move_to(driver: SO101Driver, target_pos: np.ndarray, target_gripper: float) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    prev = None
    stall = 0
    while time.monotonic() < deadline:
        joint_pos, _, gripper_pos = driver.read_state()
        if np.all(np.abs(target_pos - joint_pos) < TOL_RAD):
            return
        if prev is not None and np.all(np.abs(joint_pos - prev) < STALL_EPS):
            stall += 1
            if stall >= STALL_ITERS:
                print(f"  (stalled; residual err={np.abs(target_pos - joint_pos).round(4)})")
                return
        else:
            stall = 0
        prev = joint_pos
        driver.write_joint_targets(target_pos, gripper_pos)
        time.sleep(1.0 / STEP_HZ)
    print("  (timed out approaching target)")


def run(port: str, axis: str, delta_cm: float, max_joint_jump_deg: float) -> None:
    driver = SO101Driver(port=port)
    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit("Not calibrated — run scripts/calibrate_arm.py first.")

    start_q, _, start_gripper = driver.read_state()
    start_xyz = fk(start_q)[:3, 3]
    target_xyz = start_xyz.copy()
    target_xyz[AXES[axis]] += delta_cm / 100.0

    print(f"current ee: {start_xyz.round(4)}")
    print(f"target  ee: {target_xyz.round(4)}  ({axis:s} {delta_cm:+.1f} cm)")

    target_q = ik_position(target_xyz, seed=start_q)
    if target_q is None:
        driver.disconnect()
        raise SystemExit("IK found no solution for that target — try a smaller/different delta.")

    jump_deg = np.degrees(np.abs(target_q - start_q))
    print("joint deltas (deg):", dict(zip(JOINT_NAMES, jump_deg.round(1))))
    if np.any(jump_deg > max_joint_jump_deg):
        driver.disconnect()
        raise SystemExit(
            f"IK solution moves a joint {jump_deg.max():.0f}deg (> {max_joint_jump_deg:.0f} limit) — "
            "different posture branch, refusing. Try a smaller delta."
        )

    achieved = fk(target_q)[:3, 3]
    print(f"IK residual: {np.linalg.norm(achieved - target_xyz) * 1000:.2f} mm")
    input("Workspace clear? Press ENTER to enable torque and move, Ctrl-C to abort...")

    driver.enable_torque()
    try:
        print("moving to target...")
        move_to(driver, target_q, start_gripper)
        end_xyz = fk(driver.read_state()[0])[:3, 3]
        err_mm = np.linalg.norm(end_xyz - target_xyz) * 1000
        print(f"reached ee: {end_xyz.round(4)}  (cartesian error {err_mm:.1f} mm)")
        time.sleep(0.5)
        print("returning to start...")
        move_to(driver, start_q, start_gripper)
    finally:
        driver.torque_off()
        print("torque released.")
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--axis", choices=AXES, default="z", help="base-frame axis to move along")
    parser.add_argument("--delta-cm", type=float, default=3.0, help="signed offset in cm")
    parser.add_argument(
        "--max-joint-jump-deg", type=float, default=30.0,
        help="refuse IK solutions that move any joint more than this",
    )
    args = parser.parse_args()
    run(args.port, args.axis, args.delta_cm, args.max_joint_jump_deg)


if __name__ == "__main__":
    main()
