"""Named poses: round-trip storage, "am I at this pose" matching, and
the bounded interpolation a pose move is built from."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from so101_assist.arm.driver import JOINT_NAMES
from so101_assist.arm.poses import (
    Pose,
    ee_height,
    floor_guarded_step,
    load_poses,
    lowest_on_path,
    match_pose,
    path_clears_floor,
    save_poses,
    step_toward,
)

N = len(JOINT_NAMES)


def make_pose(name="HOME", value=0.0, gripper=0.0):
    return Pose(name=name, joints_rad=np.full(N, value), gripper=gripper)


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "poses.json"
    poses = {
        "HOME": Pose("HOME", np.radians(np.arange(N, dtype=float)), 0.0, "tucked"),
        "RAISED": Pose("RAISED", np.radians(np.full(N, 45.0)), 1.0, "up"),
    }

    save_poses(poses, path)
    loaded = load_poses(path)

    assert list(loaded) == ["HOME", "RAISED"]        # order preserved
    assert loaded["HOME"].description == "tucked"
    assert loaded["RAISED"].gripper == 1.0
    np.testing.assert_allclose(loaded["HOME"].joints_rad, poses["HOME"].joints_rad, atol=1e-4)


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert load_poses(tmp_path / "nope.json") == {}


def test_stored_in_degrees_for_hand_editing(tmp_path):
    path = tmp_path / "poses.json"
    save_poses({"X": Pose("X", np.radians(np.full(N, 90.0)))}, path)

    raw = json.loads(path.read_text())

    assert raw["poses"]["X"]["joints_deg"] == [90.0] * N
    assert raw["joint_names"] == JOINT_NAMES


def test_reordered_joint_names_is_rejected(tmp_path):
    """A pose file taught under a different joint order would map angles
    onto the wrong joints — refuse it rather than move the arm wrongly."""
    path = tmp_path / "poses.json"
    path.write_text(json.dumps({
        "joint_names": list(reversed(JOINT_NAMES)),
        "poses": {"X": {"joints_deg": [0.0] * N}},
    }))

    with pytest.raises(ValueError, match="Re-teach"):
        load_poses(path)


def test_wrong_joint_count_is_rejected(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text(json.dumps({"poses": {"X": {"joints_deg": [0.0, 1.0]}}}))

    with pytest.raises(ValueError, match="expected"):
        load_poses(path)


def test_match_pose_within_tolerance():
    poses = {"HOME": make_pose("HOME", 0.0)}
    near = np.full(N, math.radians(5))

    assert match_pose(near, poses, tol_rad=math.radians(10)) == "HOME"


def test_match_pose_rejects_one_joint_out_of_tolerance():
    poses = {"HOME": make_pose("HOME", 0.0)}
    off = np.zeros(N)
    off[2] = math.radians(30)

    assert match_pose(off, poses, tol_rad=math.radians(10)) is None


def test_match_pose_picks_the_closest_when_two_overlap():
    poses = {
        "A": make_pose("A", math.radians(0)),
        "B": make_pose("B", math.radians(6)),
    }
    at = np.full(N, math.radians(5))

    assert match_pose(at, poses, tol_rad=math.radians(10)) == "B"


def test_match_pose_handles_no_poses_and_no_reading():
    assert match_pose(np.zeros(N), {}, tol_rad=1.0) is None
    assert match_pose(None, {"HOME": make_pose()}, tol_rad=1.0) is None


def test_gripper_is_ignored_when_matching():
    """Holding an object must not hide the pose readout."""
    poses = {"HOME": make_pose("HOME", 0.0, gripper=0.0)}

    assert match_pose(np.zeros(N), poses, tol_rad=0.1) == "HOME"


def test_step_toward_clamps_each_joint():
    current = np.zeros(N)
    target = np.full(N, 1.0)

    nxt, arrived = step_toward(current, target, max_step_rad=0.1)

    assert not arrived
    np.testing.assert_allclose(nxt, np.full(N, 0.1))


def test_step_toward_reports_arrival_and_lands_exactly():
    current = np.zeros(N)
    target = np.full(N, 0.05)

    nxt, arrived = step_toward(current, target, max_step_rad=0.1)

    assert arrived
    np.testing.assert_allclose(nxt, target)


def test_step_toward_converges():
    current = np.zeros(N)
    target = np.array([1.0, -0.5, 0.25, -1.2, 0.8])[:N]

    for _ in range(1000):
        current, arrived = step_toward(current, target, max_step_rad=0.05)
        if arrived:
            break

    assert arrived
    np.testing.assert_allclose(current, target)


# --------------------------------------------------------------- path safety

# Measured, not invented: a jogged start (ee z = +0.068) moving to this
# arm's real taught HOME (+0.028). Both clear of the table, yet the
# straight joint path sags to -0.030 — 3 cm under it, around t=0.56.
# Nothing about the endpoints reveals it; this is the field bug.
SAG_START = np.array([-1.1647, 0.9455, 1.4436, 0.7315, 0.0])
SAG_END = np.array([-0.1979, -1.7254, 1.5574, 1.0434, -0.3522])


def test_lowest_on_path_finds_a_sag_between_clear_endpoints():
    assert ee_height(SAG_START) > 0.0 and ee_height(SAG_END) > 0.0

    lowest = lowest_on_path(SAG_START, SAG_END)

    assert lowest < 0.0                      # under the table
    assert not path_clears_floor(SAG_START, SAG_END, z_floor=0.0)


def test_path_clears_floor_accepts_a_short_safe_move():
    a = np.zeros(N)
    assert path_clears_floor(a, a + 0.01, z_floor=-1.0)


def test_guard_leaves_a_safe_step_untouched():
    q = np.zeros(N)
    desired = q + 0.01

    guarded = floor_guarded_step(q, desired, z_floor=-1.0)

    np.testing.assert_allclose(guarded, desired)


def test_guard_lifts_a_step_that_would_breach_the_floor():
    q = np.zeros(N)
    floor = ee_height(q)                    # standing exactly on the floor
    down = q.copy()
    down[1] += 0.3                          # shoulder_lift down -> EE drops
    assert ee_height(down) < floor          # unguarded, this breaches

    guarded = floor_guarded_step(q, down, z_floor=floor)

    assert ee_height(guarded) >= floor - 1e-6


def test_guard_keeps_every_step_of_a_sagging_move_above_the_floor():
    """The end-to-end property: replay the whole move under the guard
    and prove the gripper never goes under the table."""
    floor = 0.0
    q = SAG_START.copy()
    step = 0.4 / 25                          # config default speed, one tick
    lowest = ee_height(q)

    for _ in range(2000):
        delta = SAG_END - q
        nxt = SAG_END.copy() if np.max(np.abs(delta)) <= step else q + np.clip(delta, -step, step)
        q = floor_guarded_step(q, nxt, floor)
        lowest = min(lowest, ee_height(q))
        if np.max(np.abs(SAG_END - q)) <= 1e-3:
            break

    assert lowest >= floor - 1e-6            # never under the table
    assert lowest < ee_height(SAG_START) + 1.0   # it did actually move


def test_guard_is_a_local_constraint_not_a_planner():
    """Documented limitation: sliding along the floor cannot route
    around a blockage, so some moves stall short instead of arriving.
    The teleop loop detects that; the guard must not pretend otherwise."""
    floor = 0.0
    q = SAG_START.copy()
    step = 0.4 / 25
    for _ in range(2000):
        delta = SAG_END - q
        nxt = SAG_END.copy() if np.max(np.abs(delta)) <= step else q + np.clip(delta, -step, step)
        nq = floor_guarded_step(q, nxt, floor)
        if np.max(np.abs(nq - q)) < 1e-9:
            break
        q = nq
    assert ee_height(q) >= floor - 1e-6
