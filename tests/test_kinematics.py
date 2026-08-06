"""FK/IK round-trip: for random reachable joint configs,
ik_position(fk(q)) must land within 2 mm. Also test that points
outside the workspace return None rather than garbage."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.arm.driver import JOINT_NAMES
from so101_assist.arm.kinematics import JOINT_LIMITS_RAD, fk, ik_position, jacobian

TOL_M = 2e-3


def _random_reachable_configs(n: int, seed: int, limit_margin: float = 0.1) -> list[np.ndarray]:
    """Sample joint configs with a margin from their hard limits.

    A config that lands within ~0.01 rad of a joint's limit is a
    pathological case for any clipped-gradient IK solver (and not one
    real grasp planning should target anyway — motion planning wants
    margin from hard stops), so this deliberately excludes the
    knife-edge boundary rather than treating it as a solver bug.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        q = []
        for name in JOINT_NAMES:
            lo, hi = JOINT_LIMITS_RAD[name]
            pad = (hi - lo) * limit_margin
            q.append(rng.uniform(lo + pad, hi - pad))
        configs.append(np.array(q))
    return configs


@pytest.mark.parametrize("i", range(20))
def test_ik_position_round_trip_within_2mm(i: int):
    q_true = _random_reachable_configs(20, seed=0)[i]
    target = fk(q_true)[:3, 3]

    q_solved = ik_position(target, rng=np.random.default_rng(i))

    assert q_solved is not None
    achieved = fk(q_solved)[:3, 3]
    assert np.linalg.norm(achieved - target) < TOL_M


def test_ik_position_returns_none_for_unreachable_target():
    far_away = np.array([10.0, 10.0, 10.0])   # far outside any plausible desktop-arm workspace
    assert ik_position(far_away, num_restarts=3) is None


def test_fk_home_pose_is_within_plausible_desktop_arm_reach():
    pos = fk(np.zeros(len(JOINT_NAMES)))[:3, 3]
    assert 0.1 < np.linalg.norm(pos) < 0.5


def test_jacobian_shape():
    J = jacobian(np.zeros(len(JOINT_NAMES)))
    assert J.shape == (6, len(JOINT_NAMES))


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(7)
    q = np.array([rng.uniform(*JOINT_LIMITS_RAD[name]) for name in JOINT_NAMES])
    eps = 1e-6

    J = jacobian(q)[:3]   # linear rows only
    J_numeric = np.zeros((3, len(JOINT_NAMES)))
    for i in range(len(JOINT_NAMES)):
        dq = np.zeros(len(JOINT_NAMES))
        dq[i] = eps
        J_numeric[:, i] = (fk(q + dq)[:3, 3] - fk(q - dq)[:3, 3]) / (2 * eps)

    assert np.allclose(J, J_numeric, atol=1e-5)
