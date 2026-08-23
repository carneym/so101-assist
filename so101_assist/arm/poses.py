"""Named arm poses — stored, matched, and interpolated toward.

Poses are TAUGHT, not hard-coded: you move the limp arm by hand and
scripts/teach_pose.py records the measured joint angles (same workflow
as calibrate_arm.py). Hard-coded angles would be wrong on any other
unit, and wrong on THIS unit after a recalibration — every calibration
shifts the zero pose, so a literal angle means something different
afterwards. A taught pose is re-taught in seconds; a hard-coded table
silently sends the arm somewhere unexpected.

Stored in degrees at `config/poses.json` — human-readable and
hand-editable, machine-specific, gitignored like the calibration it
depends on.

Matching is JOINT-ONLY: the gripper is stored (a pose move closes or
opens it) but ignored when deciding "is the arm at this pose", so
holding an object never hides the HOME readout.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .driver import JOINT_NAMES
from .kinematics import fk, jacobian

DEFAULT_POSES_PATH = Path("config/poses.json")

# How close every joint must be for the arm to read as "at" a pose.
# Generous by design: this drives a status readout, not a control
# decision, and a taught pose is only ever as repeatable as the hand
# that taught it.
DEFAULT_MATCH_TOL_RAD = math.radians(10)


@dataclass(frozen=True)
class Pose:
    """One named arm configuration, in JOINT_NAMES order."""
    name: str
    joints_rad: np.ndarray
    gripper: float = 0.0          # 0 closed .. 1 open
    description: str = ""

    def max_error(self, joint_pos: np.ndarray) -> float:
        """Largest per-joint deviation from this pose, radians."""
        return float(np.max(np.abs(np.asarray(joint_pos) - self.joints_rad)))


def load_poses(path: Path | str = DEFAULT_POSES_PATH) -> dict[str, Pose]:
    """Read taught poses. Missing file -> {} (nothing taught yet)."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())

    stored_names = raw.get("joint_names")
    if stored_names is not None and stored_names != JOINT_NAMES:
        raise ValueError(
            f"{path} was taught with joint order {stored_names}, but this build uses "
            f"{JOINT_NAMES}. Re-teach the poses rather than trusting a reordered mapping."
        )

    poses: dict[str, Pose] = {}
    for name, fields in raw.get("poses", {}).items():
        joints_deg = fields["joints_deg"]
        if len(joints_deg) != len(JOINT_NAMES):
            raise ValueError(
                f"pose '{name}' in {path} has {len(joints_deg)} joints, expected {len(JOINT_NAMES)}"
            )
        poses[name] = Pose(
            name=name,
            joints_rad=np.radians(np.asarray(joints_deg, dtype=float)),
            gripper=float(fields.get("gripper", 0.0)),
            description=fields.get("description", ""),
        )
    return poses


def save_poses(poses: dict[str, Pose], path: Path | str = DEFAULT_POSES_PATH) -> None:
    """Write poses back, preserving insertion order (which is the order
    they're offered to the operator)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_names": JOINT_NAMES,
        "poses": {
            pose.name: {
                "joints_deg": [round(float(a), 2) for a in np.degrees(pose.joints_rad)],
                "gripper": round(pose.gripper, 3),
                "description": pose.description,
            }
            for pose in poses.values()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def match_pose(
    joint_pos: np.ndarray,
    poses: dict[str, Pose],
    tol_rad: float = DEFAULT_MATCH_TOL_RAD,
) -> str | None:
    """Name of the pose the arm is currently in, or None.

    Every joint must be within `tol_rad`. If several poses qualify (they
    were taught close together), the closest one wins so the readout is
    never ambiguous.
    """
    if joint_pos is None or not poses:
        return None
    best: tuple[float, str] | None = None
    for name, pose in poses.items():
        error = pose.max_error(joint_pos)
        if error <= tol_rad and (best is None or error < best[0]):
            best = (error, name)
    return best[1] if best else None


def step_toward(
    current_rad: np.ndarray,
    target_rad: np.ndarray,
    max_step_rad: float,
) -> tuple[np.ndarray, bool]:
    """One interpolation step from `current` toward `target`.

    Returns (next_target, arrived). Every joint moves at most
    `max_step_rad`, so a pose move is a sequence of small bounded
    commands rather than one large jump — the arm eases into the pose
    and the driver's own per-call clamp is never the thing saving it.
    Arrival is reported when the remaining error is within one step.
    """
    current = np.asarray(current_rad, dtype=float)
    target = np.asarray(target_rad, dtype=float)
    delta = target - current
    if np.max(np.abs(delta)) <= max_step_rad:
        return target.copy(), True
    return current + np.clip(delta, -max_step_rad, max_step_rad), False


# --------------------------------------------------------------- path safety

# How finely a candidate path is sampled when checking it against the
# floor. The dip between two poses is smooth, so this only has to be
# fine enough not to step over the lowest point; 60 samples across a
# move of a radian or so is ~1 degree per sample.
PATH_SAMPLES = 60


def ee_height(joints_rad: np.ndarray) -> float:
    """End-effector z in the base frame, meters."""
    return float(fk(np.asarray(joints_rad))[2, 3])


def lowest_on_path(start_rad: np.ndarray, end_rad: np.ndarray, samples: int = PATH_SAMPLES) -> float:
    """Lowest end-effector z along the straight JOINT-space path.

    This is the whole problem with interpolating in joint space: both
    endpoints can be well clear of the table while the path between them
    swings the gripper straight through it. Nothing about the endpoints
    tells you that — you have to look at the path.
    """
    start = np.asarray(start_rad, dtype=float)
    end = np.asarray(end_rad, dtype=float)
    return min(ee_height(start + (end - start) * t) for t in np.linspace(0.0, 1.0, samples))


def path_clears_floor(
    start_rad: np.ndarray,
    end_rad: np.ndarray,
    z_floor: float,
    samples: int = PATH_SAMPLES,
) -> bool:
    return lowest_on_path(start_rad, end_rad, samples) >= z_floor


def floor_guarded_step(
    current_rad: np.ndarray,
    next_rad: np.ndarray,
    z_floor: float,
    corrections: int = 3,
) -> np.ndarray:
    """Keep one interpolation step from putting the gripper under the table.

    Joint-space interpolation does not preserve Cartesian constraints:
    measured on real taught poses, a path between two configurations
    BOTH clear of the table sags through it around the middle (worst
    seen: endpoints at +0.07 and +0.03, sagging to -0.04). Neither
    endpoint reveals it, so the only reliable place to catch it is at
    every step.

    Applies the same rule WorkspaceFence uses for Cartesian velocity,
    expressed in joint space through the Jacobian's z row: strip the
    step's downward component so lateral progress survives, then push
    out any residual breach. The arm slides along the floor instead of
    through it.

    This is a local constraint, not a planner — it cannot route around
    an obstruction, so a move whose straight path is blocked will slide
    until it stalls. Callers must detect that stall (see
    scripts/quadstick_teleop.py) rather than assume arrival.
    """
    q = np.asarray(current_rad, dtype=float)
    step = np.asarray(next_rad, dtype=float) - q

    Jz = jacobian(q)[2, :]
    denom = float(Jz @ Jz)
    if denom > 1e-9 and ee_height(q + step) < z_floor:
        along = float(Jz @ step)
        if along < 0:
            step = step - Jz * (along / denom)

    guarded = q + step
    for _ in range(corrections):
        z = ee_height(guarded)
        if z >= z_floor:
            break
        Jz2 = jacobian(guarded)[2, :]
        denom2 = float(Jz2 @ Jz2)
        if denom2 < 1e-9:
            break      # singular in z: nothing to push against
        guarded = guarded + Jz2 * ((z_floor - z) / denom2)
    return guarded
