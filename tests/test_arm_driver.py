"""SO101Driver: calibration loading and position/load unit conversion.
Read-only — never calls bus.connect()/sync_read() against real
hardware; sync_read is faked so these run without a servo attached."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from lerobot.motors import MotorCalibration

from so101_assist.arm.driver import JOINT_NAMES, SO101Driver


@pytest.fixture
def driver(tmp_path) -> SO101Driver:
    return SO101Driver(port="COM_FAKE", calibration_path=tmp_path / "missing.json")


@pytest.fixture
def calibrated_driver(tmp_path) -> SO101Driver:
    cal = {
        name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(
            ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "gripper"], start=1
        )
    }
    cal_path = tmp_path / "arm.json"
    cal_path.write_text(json.dumps(cal))
    return SO101Driver(port="COM_FAKE", calibration_path=cal_path)


def test_missing_calibration_file_leaves_driver_uncalibrated(driver: SO101Driver):
    assert driver.is_calibrated is False
    assert driver.bus.calibration == {}


def test_calibration_file_is_loaded(tmp_path):
    cal_path = tmp_path / "arm.json"
    cal_path.write_text(
        json.dumps(
            {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            }
        )
    )
    driver = SO101Driver(port="COM_FAKE", calibration_path=cal_path)
    assert driver.is_calibrated is True
    assert isinstance(driver.bus.calibration["shoulder_pan"], MotorCalibration)
    assert driver.bus.calibration["shoulder_pan"].range_max == 4095


def test_uncalibrated_position_uses_raw_tick_scale(driver: SO101Driver):
    # full turn (4096 ticks) -> 2*pi radians
    assert driver._position_to_radians("shoulder_pan", 4096) == pytest.approx(2 * math.pi)
    assert driver._position_to_radians("shoulder_pan", 0) == 0.0


def test_calibrated_position_uses_degrees(driver: SO101Driver):
    driver.bus.calibration = {"shoulder_pan": object()}   # any truthy dict marks calibrated
    assert driver._position_to_radians("shoulder_pan", 180) == pytest.approx(math.pi)
    assert driver._position_to_radians("shoulder_pan", 90) == pytest.approx(math.pi / 2)


def test_uncalibrated_gripper_uses_raw_tick_fraction(driver: SO101Driver):
    assert driver._gripper_to_unit(4096) == pytest.approx(1.0)
    assert driver._gripper_to_unit(0) == 0.0


def test_calibrated_gripper_uses_percent(driver: SO101Driver):
    driver.bus.calibration = {"gripper": object()}
    assert driver._gripper_to_unit(50) == pytest.approx(0.5)
    assert driver._gripper_to_unit(100) == pytest.approx(1.0)


def test_read_state_shapes_and_orders_by_joint_names(driver: SO101Driver, monkeypatch):
    positions = {name: i * 100 for i, name in enumerate(JOINT_NAMES, start=1)}
    positions["gripper"] = 2048
    loads = {name: -300 for name in JOINT_NAMES}   # negative = sign-decoded direction

    def fake_sync_read(data_name, normalize=True, num_retry=0):
        return positions if data_name == "Present_Position" else loads

    monkeypatch.setattr(driver.bus, "sync_read", fake_sync_read)

    joint_pos, joint_load, gripper_pos = driver.read_state()

    assert joint_pos.shape == (5,)
    assert joint_load.shape == (5,)
    assert np.all(joint_load == pytest.approx(0.3))   # abs(-300) / 1000
    assert gripper_pos == pytest.approx(2048 / 4096)
    # order matches JOINT_NAMES, not dict insertion order
    assert joint_pos[0] < joint_pos[1] < joint_pos[2] < joint_pos[3] < joint_pos[4]


def test_write_joint_targets_requires_calibration(driver: SO101Driver):
    with pytest.raises(RuntimeError, match="not calibrated"):
        driver.write_joint_targets(np.zeros(5), 0.5)


def test_write_joint_targets_clips_large_step_per_call(calibrated_driver: SO101Driver, monkeypatch):
    monkeypatch.setattr(calibrated_driver, "read_state", lambda: (np.zeros(5), np.zeros(5), 0.0))
    sent = {}
    monkeypatch.setattr(calibrated_driver.bus, "sync_write", lambda name, values, **kw: sent.update(values))

    huge_target = np.full(5, math.radians(90))   # way past the default 5deg per-call cap
    calibrated_driver.write_joint_targets(huge_target, 0.0)

    expected_deg = math.degrees(calibrated_driver.max_joint_step_rad)
    for name in JOINT_NAMES:
        assert sent[name] == pytest.approx(expected_deg)


def test_write_joint_targets_clips_gripper_step_per_call(calibrated_driver: SO101Driver, monkeypatch):
    monkeypatch.setattr(calibrated_driver, "read_state", lambda: (np.zeros(5), np.zeros(5), 0.5))
    sent = {}
    monkeypatch.setattr(calibrated_driver.bus, "sync_write", lambda name, values, **kw: sent.update(values))

    calibrated_driver.write_joint_targets(np.zeros(5), 1.0)   # full open, requested from 0.5

    expected = (0.5 + calibrated_driver.max_gripper_step) * 100.0
    assert sent["gripper"] == pytest.approx(expected)


def test_write_joint_targets_clamps_gripper_to_unit_range(calibrated_driver: SO101Driver, monkeypatch):
    monkeypatch.setattr(calibrated_driver, "read_state", lambda: (np.zeros(5), np.zeros(5), 0.0))
    sent = {}
    monkeypatch.setattr(calibrated_driver.bus, "sync_write", lambda name, values, **kw: sent.update(values))

    calibrated_driver.write_joint_targets(np.zeros(5), -5.0)   # nonsensical negative target

    assert sent["gripper"] == pytest.approx(0.0)


def test_enable_torque_delegates_to_bus(driver: SO101Driver, monkeypatch):
    called = []
    monkeypatch.setattr(driver.bus, "enable_torque", lambda: called.append(True))
    driver.enable_torque()
    assert called == [True]


def test_set_torque_limit_writes_scaled_value_to_selected_motors(driver: SO101Driver, monkeypatch):
    writes = []
    monkeypatch.setattr(driver.bus, "write", lambda name, motor, value, **kw: writes.append((name, motor, value)))

    driver.set_torque_limit(0.5, motors=["shoulder_lift", "elbow"])

    assert writes == [("Torque_Limit", "shoulder_lift", 500), ("Torque_Limit", "elbow", 500)]


def test_set_torque_limit_clamps_fraction_to_unit_range(driver: SO101Driver, monkeypatch):
    writes = []
    monkeypatch.setattr(driver.bus, "write", lambda name, motor, value, **kw: writes.append(value))
    driver.set_torque_limit(5.0, motors=["gripper"])   # absurd
    assert writes == [1000]


def test_set_torque_limit_caches_and_skips_unchanged(driver: SO101Driver, monkeypatch):
    writes = []
    monkeypatch.setattr(driver.bus, "write", lambda name, motor, value, **kw: writes.append(value))
    driver.set_torque_limit(0.8, motors=["elbow"])
    driver.set_torque_limit(0.8, motors=["elbow"])   # no-op, cached
    driver.set_torque_limit(0.6, motors=["elbow"])   # changed
    assert writes == [800, 600]


def test_torque_off_attempts_every_motor_even_if_one_fails(driver: SO101Driver, monkeypatch):
    attempted = []

    def fake_disable_torque(motor, num_retry=0):
        attempted.append(motor)
        if motor == "elbow":
            raise ConnectionError("simulated comm failure")

    monkeypatch.setattr(driver.bus, "disable_torque", fake_disable_torque)

    with pytest.raises(ConnectionError, match="elbow"):
        driver.torque_off()

    assert set(attempted) == set(driver.bus.motors)   # every motor attempted, not just up to the failure
