"""Live-tuning override: baseline extraction, override merge, and
atomic round-trip. The GUI and teleop both depend on resolve() giving
the same answer, so it's worth pinning."""
from __future__ import annotations

import math

from so101_assist.arm.kinematics import JOINT_LIMITS_RAD
from so101_assist.control import tuning as T

CFG = {
    "arm": {
        "max_linear_mps": 0.06,
        "max_wrist_radps": 0.5,
        "max_elbow_radps": 0.5,
        "max_shoulder_radps": 0.4,
        "setpoint_leash_rad": 0.3,
        "max_joint_step_deg": 15,
        "load_stop_threshold": 0.9,
        "load_stop_ticks": 10,
        "joint_limits_deg": {"shoulder_lift": [-150, 150], "elbow": [-135, 135]},
    },
    "workspace_fence": {"x": [-0.15, 0.50], "y": [-0.45, 0.45], "z": [0.08, 0.65]},
}


def test_baseline_reads_scalars_from_config():
    base = T.baseline(CFG)
    assert base["setpoint_leash_rad"] == 0.3
    assert base["load_stop_threshold"] == 0.9
    assert base["max_joint_step_deg"] == 15


def test_baseline_torque_limits_default_when_absent():
    # CFG has no torque keys -> baseline supplies the documented defaults
    base = T.baseline(CFG)
    assert base["arm_torque_limit_pct"] == 100
    assert base["gripper_torque_limit_pct"] == 50


def test_resolve_overlays_torque_limit():
    resolved = T.resolve(CFG, {"arm_torque_limit_pct": 70})
    assert resolved["arm_torque_limit_pct"] == 70
    assert resolved["gripper_torque_limit_pct"] == 50   # untouched -> baseline default


def test_baseline_joint_limits_overlay_urdf_with_config():
    base = T.baseline(CFG)
    # overridden joints come from config
    assert base["joint_limits_deg"]["shoulder_lift"] == [-150, 150]
    assert base["joint_limits_deg"]["elbow"] == [-135, 135]
    # non-overridden joints come from the URDF model
    urdf_pan = JOINT_LIMITS_RAD["shoulder_pan"]
    assert base["joint_limits_deg"]["shoulder_pan"] == [
        round(math.degrees(urdf_pan[0])), round(math.degrees(urdf_pan[1]))
    ]


def test_resolve_overlays_override_scalars():
    resolved = T.resolve(CFG, {"max_shoulder_radps": 0.8, "setpoint_leash_rad": 0.45})
    assert resolved["max_shoulder_radps"] == 0.8
    assert resolved["setpoint_leash_rad"] == 0.45
    assert resolved["max_linear_mps"] == 0.06   # untouched -> baseline


def test_resolve_overlays_override_joint_limits():
    resolved = T.resolve(CFG, {"joint_limits_deg": {"elbow": [-160, 160]}})
    assert resolved["joint_limits_deg"]["elbow"] == [-160, 160]
    assert resolved["joint_limits_deg"]["shoulder_lift"] == [-150, 150]   # baseline kept


def test_resolve_can_disable_load_guard():
    resolved = T.resolve(CFG, {"load_stop_threshold": None})
    assert resolved["load_stop_threshold"] is None


def test_baseline_reads_workspace_fence():
    base = T.baseline(CFG)
    assert base["workspace_fence"]["z"] == [0.08, 0.65]
    assert base["workspace_fence"]["x"] == [-0.15, 0.50]


def test_resolve_overlays_fence_per_axis():
    resolved = T.resolve(CFG, {"workspace_fence": {"z": [0.0, 0.7]}})
    assert resolved["workspace_fence"]["z"] == [0.0, 0.7]
    assert resolved["workspace_fence"]["x"] == [-0.15, 0.50]   # untouched axis kept


def test_load_missing_file_returns_empty(tmp_path):
    assert T.load(tmp_path / "nope.json") == {}


def test_load_malformed_file_returns_empty(tmp_path):
    bad = tmp_path / "tuning.json"
    bad.write_text("{ not valid json")
    assert T.load(bad) == {}


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "tuning.json"
    data = {"max_elbow_radps": 0.7, "joint_limits_deg": {"elbow": [-140, 140]}}
    T.save(data, path)
    assert T.load(path) == data


def test_save_is_atomic_leaves_no_tmp(tmp_path):
    path = tmp_path / "tuning.json"
    T.save({"max_linear_mps": 0.05}, path)
    assert path.exists()
    assert not (tmp_path / "tuning.json.tmp").exists()
