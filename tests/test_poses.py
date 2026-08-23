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
    load_poses,
    match_pose,
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
