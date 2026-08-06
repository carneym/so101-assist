"""Live-tunable teleop settings shared by the tuning GUI and teleop.

The GUI (scripts/tuning_gui.py) writes these values to a JSON override
file; teleop (scripts/quadstick_teleop.py) reads it at startup AND
re-reads it whenever it changes, applying the new values to the live
controller/driver — so speeds, limits, leash, and the load guard can
be fine-tuned on the fly without restarting.

Layering: config/default.yaml holds the documented baseline (with
comments); the JSON override holds only what the operator has tuned
away from it. Resolving merges the override over the baseline. The
override file is machine-specific tuning, not committed.

A second, reverse channel runs teleop -> GUI: teleop can only hold the
serial port from one process, so the GUI (a separate process) cannot
read the arm directly. Instead teleop publishes the live end-effector
position to a small status file (write_status) that the GUI polls
(read_status), giving a live readout to tune the fence/limits against.
Both files are machine-specific runtime state, not committed.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from ..arm.kinematics import JOINT_LIMITS_RAD

DEFAULT_TUNING_PATH = Path("config/tuning.json")
DEFAULT_STATUS_PATH = Path("config/arm_status.json")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]

# key -> (label, min, max, step) for GUI sliders. load_stop_threshold
# has a separate enable checkbox (None disables the monitor).
SCALARS: dict[str, tuple[str, float, float, float]] = {
    "max_linear_mps": ("Linear speed (m/s)", 0.01, 0.20, 0.005),
    "max_wrist_radps": ("Wrist speed (rad/s)", 0.1, 2.0, 0.05),
    "max_elbow_radps": ("Elbow speed (rad/s)", 0.1, 2.0, 0.05),
    "max_shoulder_radps": ("Shoulder speed (rad/s)", 0.1, 2.0, 0.05),
    "setpoint_leash_rad": ("Setpoint leash (rad)", 0.05, 0.60, 0.01),
    "max_joint_step_deg": ("Driver step clamp (deg)", 2, 30, 1),
    "arm_torque_limit_pct": ("Arm torque limit (%)", 10, 100, 5),
    "gripper_torque_limit_pct": ("Gripper torque limit (%)", 10, 100, 5),
    "load_stop_threshold": ("Load stop threshold", 0.1, 1.5, 0.05),
    "load_stop_ticks": ("Load stop ticks", 1, 30, 1),
}

# Per-joint operating-range slider bounds (degrees), generous enough to
# cover the physical arm past the URDF model.
JOINT_LIMIT_SLIDER = (-180, 180, 1)

# Workspace-fence axes and slider bounds (meters, robot base frame).
# Range is wider than the arm's reach so no configured bound is ever
# clipped by the slider.
FENCE_AXES = ["x", "y", "z"]
FENCE_SLIDER = (-0.6, 0.9, 0.01)


def baseline(cfg: dict) -> dict:
    """Resolved settings from config/default.yaml alone (no override).

    joint_limits_deg covers all arm joints: URDF-modeled defaults
    overlaid with any config `arm.joint_limits_deg` entries.
    """
    arm = cfg.get("arm", {})
    scalars = {
        "max_linear_mps": arm.get("max_linear_mps", 0.06),
        "max_wrist_radps": arm.get("max_wrist_radps", 0.5),
        "max_elbow_radps": arm.get("max_elbow_radps", 0.5),
        "max_shoulder_radps": arm.get("max_shoulder_radps", 0.4),
        "setpoint_leash_rad": arm.get("setpoint_leash_rad", 0.2),
        "max_joint_step_deg": arm.get("max_joint_step_deg", 15),
        "arm_torque_limit_pct": arm.get("arm_torque_limit_pct", 100),
        "gripper_torque_limit_pct": arm.get("gripper_torque_limit_pct", 50),
        "load_stop_threshold": arm.get("load_stop_threshold"),   # may be None
        "load_stop_ticks": arm.get("load_stop_ticks", 5),
    }
    limits = {j: [round(math.degrees(lo)), round(math.degrees(hi))] for j, (lo, hi) in JOINT_LIMITS_RAD.items()}
    for j, pair in arm.get("joint_limits_deg", {}).items():
        limits[j] = [pair[0], pair[1]]
    fence_cfg = cfg.get("workspace_fence", {})
    fence = {ax: [float(fence_cfg.get(ax, [0.0, 0.0])[0]), float(fence_cfg.get(ax, [0.0, 0.0])[1])] for ax in FENCE_AXES}
    return {**scalars, "joint_limits_deg": limits, "workspace_fence": fence}


def resolve(cfg: dict, override: dict) -> dict:
    """Baseline with the override layered on top."""
    resolved = baseline(cfg)
    for key in SCALARS:
        if key in override:
            resolved[key] = override[key]
    for j, pair in override.get("joint_limits_deg", {}).items():
        if j in resolved["joint_limits_deg"]:
            resolved["joint_limits_deg"][j] = [pair[0], pair[1]]
    for ax, pair in override.get("workspace_fence", {}).items():
        if ax in resolved["workspace_fence"]:
            resolved["workspace_fence"][ax] = [pair[0], pair[1]]
    return resolved


def load(path: Path | str = DEFAULT_TUNING_PATH) -> dict:
    """Read the override file; {} if absent or malformed (never raises,
    so a mid-write read during hot-reload just skips this cycle)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save(values: dict, path: Path | str = DEFAULT_TUNING_PATH) -> None:
    """Atomically write the override file (temp + os.replace) so a
    reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(values, indent=2))
    os.replace(tmp, path)


def write_status(ee_xyz, path: Path | str = DEFAULT_STATUS_PATH) -> None:
    """Publish the live end-effector position for the GUI to read.

    Stamped with wall-clock time so the reader (a different process) can
    tell a fresh reading from a stale one left by an exited teleop.
    Atomic write, same as save()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ee_xyz": [float(v) for v in ee_xyz], "stamp": time.time()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_status(path: Path | str = DEFAULT_STATUS_PATH) -> dict:
    """Read the status file; {} if absent or malformed (never raises, so
    a mid-write read just skips this cycle). Callers should treat a
    stamp older than a second or two as stale (teleop not running)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
