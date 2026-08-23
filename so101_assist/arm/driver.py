"""Low-level SO-101 servo interface.

Wraps the Feetech STS3215 bus (via LeRobot's feetech driver) behind a
minimal API. This is the ONLY file that talks to hardware; everything
else consumes ArmState and emits CartesianVelocity.

Responsibilities:
- open/close serial bus, ping servos, load calibration offsets
- read joint positions + load at CONTROL_HZ, publish ArmState
- accept joint velocity/position setpoints from controller.py
- enforce firmware-level torque and velocity limits at init

Current status: connect/disconnect/read_state/write_joint_targets/
enable_torque/torque_off are all implemented. write_joint_targets is
deliberately conservative: it requires a calibration file (which is
also what gives the servos real firmware-enforced position limits —
see calibrate_arm.py) and clips every call's movement to a small
per-joint step, independent of and in addition to whatever caps
controller.py will eventually apply. Nothing auto-enables torque;
callers must call enable_torque() explicitly before anything can
move — see scripts/motion_test.py for the smallest possible example.

Uses lerobot.motors.feetech.FeetechMotorsBus directly rather than
lerobot's higher-level Robot/SOFollower wrapper — that wrapper pulls
in lerobot's own camera/dataset feature system and an interactive
`input()`-driven calibration flow, neither of which fit this
project's own camera/config architecture. Calibration is instead
loaded from a plain JSON dict of MotorCalibration fields at
`config/calibration/arm.json` if present (produce one with lerobot's
own calibration tooling, e.g. `lerobot-calibrate`, and dump
`bus.read_calibration()` — see the `MotorCalibration` fields).

Without a calibration file, joint_pos/gripper_pos are reported in an
UNCALIBRATED fallback (raw encoder ticks scaled by motor resolution,
arbitrary zero point) — good enough to confirm the servos respond,
NOT good enough to trust for kinematics or the workspace fence.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)

CONTROL_HZ = 50
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]

# Feetech servo IDs, factory/LeRobot-standard SO-101 wiring order.
_MOTOR_IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
_STS3215_RESOLUTION = 4096   # encoder ticks per revolution

DEFAULT_CALIBRATION_PATH = Path("config/calibration/arm.json")

# sync_read defaults to num_retry=0 — a single dropped status packet
# (USB/serial noise, briefly starved by other threads e.g. --detect's
# inference) would otherwise kill read_state and crash the whole
# control loop. Matches the retry count torque_off already uses.
READ_NUM_RETRIES = 3

# Per-call safety backstop for write_joint_targets, independent of
# controller.py's own velocity caps: bounds how far a single call can
# move a joint/gripper, so a bad target from upstream can only move
# the arm a small, bounded amount, never jump anywhere arbitrary.
#
# NOTE: this is also the tightest bound on how far the commanded
# position may lead the MEASURED position, which caps the torque a
# position-mode servo applies. If the arm can't lift its own weight,
# this (together with controller.setpoint_leash_rad) is what to raise
# — the effective lead is min(leash, this). Raising it lets a lagging
# servo push harder to catch up, at the cost of a larger single-tick
# jump if a target is bad or a joint is blocked.
DEFAULT_MAX_JOINT_STEP_RAD = math.radians(15)
DEFAULT_MAX_GRIPPER_STEP = 0.1


def _build_motors() -> dict[str, Motor]:
    return {
        name: Motor(
            motor_id,
            "sts3215",
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for name, motor_id in _MOTOR_IDS.items()
    }


class SO101Driver:
    def __init__(
        self,
        port: str,
        baud: int = 1_000_000,
        calibration_path: Path | str = DEFAULT_CALIBRATION_PATH,
        max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
        max_gripper_step: float = DEFAULT_MAX_GRIPPER_STEP,
    ) -> None:
        self.port = port
        self.baud = baud
        self.calibration_path = Path(calibration_path)
        self.max_joint_step_rad = max_joint_step_rad
        self.max_gripper_step = max_gripper_step
        self._torque_limit: dict[str, int] = {}   # last value written per motor (cache)
        calibration = self._load_calibration()
        self.bus = FeetechMotorsBus(port=port, motors=_build_motors(), calibration=calibration)

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        if not self.calibration_path.exists():
            logger.warning(
                "No arm calibration at %s — joint_pos/gripper_pos will be UNCALIBRATED "
                "(arbitrary zero, raw tick scale). Do not trust this for kinematics or the "
                "workspace fence until a calibration file is in place.",
                self.calibration_path,
            )
            return {}
        with self.calibration_path.open() as f:
            raw = json.load(f)
        return {name: MotorCalibration(**fields) for name, fields in raw.items()}

    @property
    def is_calibrated(self) -> bool:
        return bool(self.bus.calibration)

    def connect(self) -> None:
        """Open the serial port and ping every motor (handshake=True).

        Read-only: pings and register reads, no writes. Raises
        ConnectionError if the port can't be opened or a motor doesn't
        respond — this IS the hardware smoke test.
        """
        self.bus.connect()

    def disconnect(self) -> None:
        self.bus.disconnect()

    def read_state(self):
        """Return (joint_pos, joint_load, gripper_pos).

        joint_pos: radians, JOINT_NAMES order.
        joint_load: normalized servo load magnitude in [0, 1] (matches
            config's load_stop_threshold), JOINT_NAMES order.
        gripper_pos: 0 closed .. 1 open if calibrated; UNCALIBRATED
            raw-tick fraction otherwise (see class docstring).
        """
        positions = self.bus.sync_read(
            "Present_Position", normalize=self.is_calibrated, num_retry=READ_NUM_RETRIES
        )
        loads = self.bus.sync_read("Present_Load", normalize=False, num_retry=READ_NUM_RETRIES)

        joint_pos = np.array([self._position_to_radians(name, positions[name]) for name in JOINT_NAMES])
        joint_load = np.array([abs(loads[name]) / 1000.0 for name in JOINT_NAMES])
        gripper_pos = self._gripper_to_unit(positions["gripper"])
        return joint_pos, joint_load, gripper_pos

    def _position_to_radians(self, name: str, value: float) -> float:
        if self.is_calibrated:
            return math.radians(value)   # DEGREES norm_mode
        return value * (2 * math.pi / _STS3215_RESOLUTION)   # raw ticks, uncalibrated

    def _gripper_to_unit(self, value: float) -> float:
        if self.is_calibrated:
            return value / 100.0   # RANGE_0_100 norm_mode
        return value / _STS3215_RESOLUTION   # raw ticks, uncalibrated

    def enable_torque(self) -> None:
        """Power the servos so they hold and move to commanded positions.

        Safe on its own: a servo holds its CURRENT position when torque
        is enabled, it doesn't jump anywhere. Nothing moves until
        write_joint_targets() is also called.

        Retries like read_state and torque_off do: this writes two
        registers per motor across twelve round trips, and a single
        dropped status packet (USB/serial noise) would otherwise abort
        the whole call PARTWAY THROUGH — leaving some motors powered and
        the rest limp, which reads to the operator as a dead arm while
        it is in fact energized. Callers must still release torque if
        this raises; a partial enable is not a no-op.
        """
        self.bus.enable_torque(num_retry=READ_NUM_RETRIES)

    def set_torque_limit(self, fraction: float, motors: list[str] | None = None) -> None:
        """Set the runtime torque cap (0..1 of rated) on the given motors
        (default all). Writes the SRAM Torque_Limit register, not the
        EEPROM Max_Torque_Limit: it takes effect immediately, resets to
        the servo's stored ceiling on power-cycle (so it never
        permanently reconfigures the servo), and doesn't wear EEPROM —
        which matters because the tuning GUI writes this on every change.

        Higher = more lifting/holding force but more heat and stall
        burnout risk on these small servos; keep the gripper limited so
        a stalled grasp can't cook it. Cached per motor so repeated
        no-op applies don't spam the bus. Requires connect() first.
        """
        value = int(np.clip(fraction, 0.0, 1.0) * 1000)
        for motor in (motors if motors is not None else list(self.bus.motors)):
            if self._torque_limit.get(motor) != value:
                self.bus.write("Torque_Limit", motor, value, normalize=False)
                self._torque_limit[motor] = value

    def torque_off(self) -> None:
        """Immediate compliance — used by HALTED state and e-stop.

        Attempts every motor independently and retries each a few
        times, so one motor's comm failure never blocks releasing the
        rest. Raises listing any motor that could not be confirmed
        released — callers on an e-stop path must treat that as "the
        arm may still be powered," never as a silent success.
        """
        failed = []
        for motor in self.bus.motors:
            try:
                self.bus.disable_torque(motor, num_retry=3)
            except ConnectionError:
                failed.append(motor)
        if failed:
            raise ConnectionError(f"Failed to release torque on: {failed}. Arm may still be powered.")

    def write_joint_targets(self, pos: np.ndarray, gripper: float) -> None:
        """Command joint positions (radians, JOINT_NAMES order) and
        gripper (0 closed .. 1 open).

        Requires calibration — refuses otherwise. calibrate_arm.py is
        what writes the firmware Min/Max_Position_Limit registers that
        give this a hard, servo-enforced range backstop; without it
        there's only the per-call step clamp below, which alone isn't
        enough to trust.

        Safety backstop, independent of and in addition to
        controller.py's own velocity caps: clips every joint's and the
        gripper's movement in this call to max_joint_step_rad /
        max_gripper_step relative to the CURRENT measured position
        (re-read every call, never a stale cache) — a bad target from
        upstream can only move the arm a small, bounded amount, never
        jump anywhere arbitrary.

        Does nothing to a joint whose servo has torque disabled — call
        enable_torque() first.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Arm is not calibrated — refusing to write joint targets. Run "
                "scripts/calibrate_arm.py first (it also sets the firmware position "
                "limits this method relies on as a safety backstop)."
            )

        current_pos, _, current_gripper = self.read_state()
        clipped_pos = current_pos + np.clip(
            np.asarray(pos) - current_pos, -self.max_joint_step_rad, self.max_joint_step_rad
        )
        clipped_gripper = current_gripper + np.clip(
            gripper - current_gripper, -self.max_gripper_step, self.max_gripper_step
        )
        clipped_gripper = float(np.clip(clipped_gripper, 0.0, 1.0))

        goal = {name: math.degrees(angle) for name, angle in zip(JOINT_NAMES, clipped_pos)}
        goal["gripper"] = clipped_gripper * 100.0
        self.bus.sync_write("Goal_Position", goal, normalize=True)
