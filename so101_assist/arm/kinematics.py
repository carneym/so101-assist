"""FK / IK for the SO-101 5DOF chain.

The arm has no wrist yaw, so full 6DOF poses are generally unreachable.
IK therefore solves for a RELAXED target: position (3) + approach
direction constrained to the arm's reachable orientation manifold.
grasp.py is responsible for choosing grasp poses that respect this.

Geometry (JOINT_FRAMES, JOINT_LIMITS_RAD) is sourced exactly from the
official URDF — TheRobotStudio/SO-ARM100, Simulation/SO101/
so101_new_calib.urdf, fetched 2026-07-25 — rather than manual
measurement: every SO-101 joint's <axis> is local +Z, so FK/the
Jacobian are a straightforward per-joint (translate + rotate-by-rpy,
then rotate-by-theta-about-Z) chain, no DH-parameter guesswork needed.

ik_position is a numeric damped-least-squares solve using jacobian()
below — no ikpy/placo dependency needed given the Jacobian is exact.
It only targets POSITION: the 5DOF/no-wrist-yaw constraint means
orientation generally can't be fully specified anyway, so `approach`
is accepted for interface stability but not yet used to bias which of
several reachable joint solutions comes back — that needs a concrete
consumer (grasp.py) to define what "top" vs "front" should actually
constrain, which doesn't exist yet.
"""
from __future__ import annotations

import math

import numpy as np

from .driver import JOINT_NAMES


class JointFrame:
    """Fixed offset (URDF <joint><origin>) from the parent joint's
    frame to this joint's own frame, whose rotation axis is always
    local +Z (every SO-101 joint's URDF <axis> is "0 0 1")."""

    __slots__ = ("rpy", "xyz")

    def __init__(self, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> None:
        self.xyz = xyz
        self.rpy = rpy


# parent joint's frame -> this joint's frame. Order matches JOINT_NAMES.
JOINT_FRAMES: dict[str, JointFrame] = {
    "shoulder_pan": JointFrame((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    "shoulder_lift": JointFrame((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    "elbow": JointFrame((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    "wrist_flex": JointFrame((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    "wrist_roll": JointFrame((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
}
# wrist_roll's child link (gripper_link) -> the URDF's declared
# end-effector reference frame (gripper_frame_link); fixed, no joint.
GRIPPER_FRAME = JointFrame((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

# radians, from the URDF's <joint><limit lower=".." upper=".."/>.
# These are the MODELED limits, used for IK solving/sampling and as
# the controller's default operating range. The range the operator
# may actually drive is policy, not model — override it per-joint via
# `arm.joint_limits_deg` in config/default.yaml (the physical arm
# safely exceeds the URDF model on some joints, and calibration
# changes shift where the model's zero sits, so measured overrides
# belong in per-setup config, not here).
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _frame_transform(frame: JointFrame) -> np.ndarray:
    """4x4 fixed offset transform for a JointFrame (no joint rotation)."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(frame.rpy)
    T[:3, 3] = frame.xyz
    return T


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.eye(4)
    Rz[:2, :2] = [[c, -s], [s, c]]
    return Rz


def fk(joint_pos: np.ndarray) -> np.ndarray:
    """Joint angles (5,) -> end-effector pose (4,4) in base frame."""
    T = np.eye(4)
    for name, theta in zip(JOINT_NAMES, joint_pos):
        T = T @ _frame_transform(JOINT_FRAMES[name]) @ _rotz(theta)
    return T @ _frame_transform(GRIPPER_FRAME)


def jacobian(joint_pos: np.ndarray) -> np.ndarray:
    """(6,5) geometric Jacobian, for Cartesian velocity control.

    Rows 0:3 are linear velocity, 3:6 angular velocity, both in the
    base frame; columns are JOINT_NAMES order.
    """
    T = np.eye(4)
    axis_origins = []   # (axis, origin), both in base frame, one per joint
    for name, theta in zip(JOINT_NAMES, joint_pos):
        T = T @ _frame_transform(JOINT_FRAMES[name])
        axis_origins.append((T[:3, :3] @ np.array([0.0, 0.0, 1.0]), T[:3, 3].copy()))
        T = T @ _rotz(theta)

    p_end = (T @ _frame_transform(GRIPPER_FRAME))[:3, 3]

    J = np.zeros((6, len(JOINT_NAMES)))
    for i, (axis, origin) in enumerate(axis_origins):
        J[:3, i] = np.cross(axis, p_end - origin)
        J[3:, i] = axis
    return J


def _clip_to_limits(joint_pos: np.ndarray) -> np.ndarray:
    return np.array(
        [np.clip(theta, *JOINT_LIMITS_RAD[name]) for name, theta in zip(JOINT_NAMES, joint_pos)]
    )


def _ik_attempt(
    target_xyz: np.ndarray, q0: np.ndarray, max_iters: int, tol_m: float, damping: float
) -> np.ndarray | None:
    q = _clip_to_limits(q0)
    for _ in range(max_iters):
        err = target_xyz - fk(q)[:3, 3]
        if np.linalg.norm(err) < tol_m:
            return q

        Jv = jacobian(q)[:3]
        dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + damping * np.eye(3), err)
        q = _clip_to_limits(q + dq)

    return q if np.linalg.norm(target_xyz - fk(q)[:3, 3]) < tol_m else None


def ik_position(
    target_xyz: np.ndarray,
    approach: str = "top",
    *,
    seed: np.ndarray | None = None,
    max_iters: int = 200,
    tol_m: float = 1e-4,
    damping: float = 1e-3,
    num_restarts: int = 15,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Solve for joints reaching target_xyz with a 'top' or 'front'
    approach. Returns None if unreachable — the planner must handle it.

    Numeric damped-least-squares position IK: iteratively moves along
    jacobian()'s linear rows to reduce position error, clipping to
    JOINT_LIMITS_RAD every step so the search never leaves the
    feasible region. `approach` is currently unused (see module
    docstring).

    Gradient IK from a single start can land in a local minimum well
    short of the target (joint-limit clipping makes the feasible
    region non-smooth) even when the target is genuinely reachable —
    so on failure this retries from `num_restarts` random seeds within
    JOINT_LIMITS_RAD before giving up. `seed` is only used for the
    first attempt.
    """
    rng = rng or np.random.default_rng()
    q0 = np.zeros(len(JOINT_NAMES)) if seed is None else np.array(seed, dtype=float)

    for attempt in range(1 + num_restarts):
        result = _ik_attempt(target_xyz, q0, max_iters, tol_m, damping)
        if result is not None:
            return result
        q0 = np.array([rng.uniform(*JOINT_LIMITS_RAD[name]) for name in JOINT_NAMES])

    return None
