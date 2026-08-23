"""Recorded paths and gestures: simplification, loop detection, the
speed cap, time sampling, and the follow-leash that keeps a taught path
from degenerating into the straight line it existed to avoid."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.arm.trajectory import (
    TrajectoryPlayer,
    Waypoint,
    duration,
    is_loop,
    nearest_waypoint_index,
    peak_joint_rate,
    sample_at,
    simplify,
    speed_scale_for_cap,
)

N = 5


def wp(t, value, gripper=0.0):
    return Waypoint(t=t, joints_rad=np.full(N, value), gripper=gripper)


def ramp(n=10, span=1.0, dt=0.1):
    return [wp(i * dt, span * i / (n - 1)) for i in range(n)]


# ------------------------------------------------------------ simplify

def test_simplify_drops_samples_that_add_no_shape():
    held = [wp(i * 0.05, 0.0) for i in range(50)]      # arm held still

    kept = simplify(held)

    assert len(kept) == 2                              # just the endpoints


def test_simplify_always_keeps_the_endpoints_exactly():
    path = ramp(20)

    kept = simplify(path)

    assert kept[0] is path[0]
    assert kept[-1] is path[-1]


def test_simplify_preserves_the_shape_of_real_motion():
    path = ramp(40, span=2.0)

    kept = simplify(path, min_step_rad=0.035)

    assert 2 < len(kept) <= len(path)
    # every dropped sample is still within tolerance of the kept line
    assert np.isclose(kept[-1].joints_rad[0], 2.0)


def test_simplify_handles_degenerate_input():
    assert simplify([]) == []
    assert len(simplify([wp(0, 0.0)])) == 1


# ---------------------------------------------------------- loop / timing

def test_a_motion_returning_to_its_start_is_cyclic():
    wave = [wp(0.0, 0.0), wp(0.2, 0.5), wp(0.4, -0.5), wp(0.6, 0.02)]

    assert is_loop(wave)


def test_a_motion_ending_elsewhere_is_not_cyclic():
    assert not is_loop(ramp(10, span=1.5))


def test_two_samples_are_never_a_loop():
    assert not is_loop([wp(0.0, 0.0), wp(0.1, 0.0)])


def test_duration_is_measured_end_to_end():
    assert duration(ramp(11, dt=0.1)) == pytest.approx(1.0)


# ------------------------------------------------------------ speed cap

def test_peak_joint_rate_finds_the_fastest_segment():
    path = [wp(0.0, 0.0), wp(1.0, 0.1), wp(1.1, 1.0)]   # slow then fast

    assert peak_joint_rate(path) == pytest.approx(9.0, rel=1e-3)


def test_a_slow_recording_plays_at_full_speed():
    assert speed_scale_for_cap(ramp(10, span=0.5), max_joint_radps=10.0) == 1.0


def test_a_fast_gesture_is_scaled_down_to_the_cap():
    path = [wp(0.0, 0.0), wp(0.1, 1.0)]                 # 10 rad/s

    scale = speed_scale_for_cap(path, max_joint_radps=1.0)

    assert scale == pytest.approx(0.1)


def test_scaling_is_uniform_so_rhythm_survives():
    """One factor for the whole motion, not per-segment clamping —
    clamping would flatten exactly the fast parts that carry meaning."""
    slow_then_fast = [wp(0.0, 0.0), wp(1.0, 0.2), wp(1.1, 1.2)]

    scale = speed_scale_for_cap(slow_then_fast, max_joint_radps=5.0)

    # the fast segment sets the factor; the slow one keeps its relative
    # proportion because the same factor applies to both
    assert scale == pytest.approx(0.5, rel=1e-3)


# --------------------------------------------------------------- sampling

def test_sample_at_interpolates_between_waypoints():
    path = [wp(0.0, 0.0), wp(1.0, 1.0)]

    joints, _ = sample_at(path, 0.25)

    np.testing.assert_allclose(joints, np.full(N, 0.25))


def test_sample_at_clamps_outside_the_recording():
    path = [wp(0.0, 0.0), wp(1.0, 1.0)]

    np.testing.assert_allclose(sample_at(path, -5)[0], np.zeros(N))
    np.testing.assert_allclose(sample_at(path, 99)[0], np.ones(N))


def test_sample_at_interpolates_the_gripper_too():
    path = [wp(0.0, 0.0, gripper=0.0), wp(1.0, 0.0, gripper=1.0)]

    assert sample_at(path, 0.5)[1] == pytest.approx(0.5)


def test_sample_at_rejects_an_empty_recording():
    with pytest.raises(ValueError):
        sample_at([], 0.0)


def test_nearest_waypoint_finds_where_to_join_a_path():
    path = ramp(11, span=1.0)

    assert nearest_waypoint_index(path, np.full(N, 0.5)) == 5
    assert nearest_waypoint_index(path, np.full(N, -3.0)) == 0


# ---------------------------------------------------------------- player

def test_player_walks_the_path_and_reports_completion():
    player = TrajectoryPlayer(ramp(11, span=1.0))

    done = False
    for _ in range(200):
        joints, _ = player.target()
        done = player.advance(0.04, joints)      # arm tracks perfectly
        if done:
            break

    assert done


def test_player_holds_time_while_the_arm_lags_behind():
    """The leash: if the target ran ahead of a lagging joint, the arm
    would cut the corner back to it — turning a taught path into the
    straight line it existed to avoid."""
    player = TrajectoryPlayer(ramp(11, span=2.0), follow_tolerance_rad=0.1)
    player.advance(0.04, np.zeros(N))
    t_before = player.t

    for _ in range(20):
        player.advance(0.04, np.full(N, -5.0))   # arm stuck far away

    assert player.waiting
    assert player.t == pytest.approx(t_before)


def test_player_resumes_once_the_arm_catches_up():
    player = TrajectoryPlayer(ramp(11, span=2.0), follow_tolerance_rad=0.1)
    player.advance(0.04, np.full(N, -5.0))
    assert player.waiting

    joints, _ = player.target()
    player.advance(0.04, joints)

    assert not player.waiting


def test_a_cyclic_gesture_repeats_the_requested_number_of_times():
    player = TrajectoryPlayer(ramp(11, span=1.0), repeats=3)

    completions = 0
    for _ in range(2000):
        joints, _ = player.target()
        if player.advance(0.04, joints):
            completions += 1
            break

    assert completions == 1
    assert player.cycle == 3          # went round three times before finishing


def test_speed_scales_playback_time():
    fast = TrajectoryPlayer(ramp(11, span=1.0), speed=1.0)
    slow = TrajectoryPlayer(ramp(11, span=1.0), speed=0.5)

    for p in (fast, slow):
        p.advance(0.1, p.target()[0])

    assert slow.t == pytest.approx(fast.t / 2)


def test_player_rejects_an_empty_recording():
    with pytest.raises(ValueError):
        TrajectoryPlayer([])
