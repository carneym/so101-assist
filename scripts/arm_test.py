"""Read-only SO-101 arm connectivity smoke test.

Lists serial ports (use this to find the arm's port — on Windows
that's a COMn port, on Linux/Mac typically /dev/ttyACM0 or
/dev/tty.usbmodem*), then connects and streams joint_pos/joint_load/
gripper_pos. Nothing here writes to the servos or can move the arm —
see so101_assist/arm/driver.py for what's implemented vs. still
stubbed (write_joint_targets, torque_off).

    python scripts/arm_test.py --port COM5 [--hz 5]
    python scripts/arm_test.py --list-only
"""
from __future__ import annotations

import argparse
import time

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver


def list_ports() -> None:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected.")
        return
    for p in ports:
        print(f"{p.device}  {p.description}")


def stream(port: str, hz: float, calibration_path: str | None) -> None:
    kwargs = {"calibration_path": calibration_path} if calibration_path else {}
    driver = SO101Driver(port=port, **kwargs)
    print(f"Connecting to {port}...")
    driver.connect()
    print(f"Connected. calibrated={driver.is_calibrated}  Ctrl-C to stop.")
    try:
        while True:
            joint_pos, joint_load, gripper_pos = driver.read_state()
            joints = ", ".join(
                f"{name}={pos:+.2f}rad load={load:.2f}"
                for name, pos, load in zip(JOINT_NAMES, joint_pos, joint_load)
            )
            print(f"{joints}  gripper={gripper_pos:.2f}")
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        driver.disconnect()
        print("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--hz", type=float, default=5.0, help="print rate")
    parser.add_argument("--calibration", help="path to a calibration JSON (default: config/calibration/arm.json)")
    parser.add_argument("--list-only", action="store_true", help="list serial ports and exit")
    args = parser.parse_args()

    list_ports()
    if args.list_only:
        return
    if not args.port:
        parser.error("--port is required unless --list-only")
    stream(args.port, args.hz, args.calibration)


if __name__ == "__main__":
    main()
