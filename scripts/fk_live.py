"""Live FK validation against the physical arm — read-only, no motion.

Streams joint angles and the FK-computed end-effector position while
you move the arm by hand (torque stays off throughout). Use it to
verify the kinematics chain against reality before trusting IK for
commanded Cartesian motion:

- park the arm in its calibration middle pose: all joints should read
  ~0 rad and the printed xyz should match fk(zeros) (~[0.39, 0, 0.23])
- move the gripper straight up: z should increase, x/y roughly hold
- swing the base left/right: x/y should trace an arc, z hold
- measure gripper height above the base plate with a ruler at a couple
  of poses: should agree with z to within ~1-2 cm

If a direction is MIRRORED (e.g. physically moving up decreases z),
that joint's sign convention differs between our calibration and the
URDF — note which joint and we fix it in one place (driver or
kinematics), not by fudging downstream code.

    python scripts/fk_live.py --port COM3 [--hz 5]
"""
from __future__ import annotations

import argparse
import time

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver
from so101_assist.arm.kinematics import fk


def stream(port: str, hz: float) -> None:
    driver = SO101Driver(port=port)
    driver.connect()
    if not driver.is_calibrated:
        driver.disconnect()
        raise SystemExit("Not calibrated — run scripts/calibrate_arm.py first.")
    print("Connected. Torque is OFF — move the arm by hand. Ctrl-C to stop.")
    try:
        while True:
            joint_pos, _, gripper_pos = driver.read_state()
            ee_xyz = fk(joint_pos)[:3, 3]
            joints = " ".join(f"{n}={p:+.2f}" for n, p in zip(JOINT_NAMES, joint_pos))
            print(
                f"{joints}  ee_xyz=[{ee_xyz[0]:+.3f} {ee_xyz[1]:+.3f} {ee_xyz[2]:+.3f}]  "
                f"gripper={gripper_pos:.2f}"
            )
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        driver.disconnect()
        print("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--hz", type=float, default=5.0, help="print rate")
    args = parser.parse_args()
    stream(args.port, args.hz)


if __name__ == "__main__":
    main()
