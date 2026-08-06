"""Guided SO-101 arm calibration.

Interactive, torque-disabled (the arm is backdrivable throughout —
you move it by hand, nothing here commands motion): homes each joint
to the middle of its range, then has you sweep every joint except
wrist_roll through its full range of motion while it records the
min/max encoder ticks. Saves the result as a plain JSON dict of
MotorCalibration fields to config/calibration/arm.json, which
so101_assist/arm/driver.py's SO101Driver reads on construction.

Torque is left DISABLED when this finishes — re-enabling it (so the
arm holds position under power) is deliberately not this script's
job; that belongs with the safety layer (workspace fence, load
monitoring, STOP handling) once motion is implemented.

Run scripts/arm_test.py --list-only first if you don't know the port.

    python scripts/calibrate_arm.py --port COM5 [--output config/calibration/arm.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode

from so101_assist.arm.driver import DEFAULT_CALIBRATION_PATH, SO101Driver

FULL_TURN_MOTOR = "wrist_roll"


def calibrate(port: str, output_path: Path) -> None:
    # Any existing calibration at output_path is reset by set_half_turn_homings()
    # below regardless, so loading it here (if present) is harmless.
    driver = SO101Driver(port=port, calibration_path=output_path)
    driver.connect()
    bus = driver.bus
    try:
        print("Disabling torque — the arm is safe to move by hand now.")
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move the arm to the middle of its range of motion, then press ENTER...")
        homing_offsets = bus.set_half_turn_homings()

        other_motors = [m for m in bus.motors if m != FULL_TURN_MOTOR]
        print(
            f"Now move every joint except '{FULL_TURN_MOTOR}' through its full range of "
            "motion. Recording live — press ENTER when done."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion(other_motors)
        range_mins[FULL_TURN_MOTOR] = 0
        range_maxes[FULL_TURN_MOTOR] = 4095   # continuous-rotation joint, full encoder range

        calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, m in bus.motors.items()
        }
        bus.write_calibration(calibration)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump({name: asdict(cal) for name, cal in calibration.items()}, f, indent=2)
        print(f"Calibration saved to {output_path}")
        print("Torque is still disabled — the arm will stay compliant until motion is implemented.")
    finally:
        driver.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    args = parser.parse_args()
    calibrate(args.port, args.output)


if __name__ == "__main__":
    main()
