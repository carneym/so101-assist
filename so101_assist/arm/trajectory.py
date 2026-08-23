"""Recorded motion paths — taught by moving the limp arm by hand.

A Pose says WHERE to end up. A trajectory says HOW to get there, and
that difference buys two things a single target cannot:

- paths that go AROUND things. The floor guard in poses.py only knows
  about a horizontal plane; it cannot route around a wheelchair, a box,
  or the operator's own body. A path a human physically traced avoids
  whatever that human avoided.
- GESTURES. A wave, a beckon, a nod — motions whose point is the motion.
  For an operator who cannot wave, that is a communication channel, not
  a robotics feature.

Timestamps are recorded even though pose playback ignores them. For a
pose the recorded tempo is irrelevant (replay at a safe fixed rate); for
a gesture the tempo IS the content — a wave replayed at pose speed is
just the arm leaning about. Timing costs one float per sample and cannot
be recovered later, so it is always captured.

Speed is limited by scaling the whole trajectory UNIFORMLY rather than
clamping individual segments: uniform scaling preserves the rhythm and
shape that make a gesture legible, where per-segment clamping would
flatten exactly the fast parts that carry the meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# Samples closer together than this add no shape, only file size and
# hand tremor. ~2 degrees.
DEFAULT_SIMPLIFY_RAD = 0.035

# A path whose end returns to its start is cyclic and can be repeated
# (one recorded wave -> N waves). Tolerance is generous: a hand-traced
# loop never closes exactly.
LOOP_TOLERANCE_RAD = 0.15


@dataclass(frozen=True)
class Waypoint:
    """One sample of a recorded motion."""
    t: float                     # seconds from the start of the recording
    joints_rad: np.ndarray
    gripper: float = 0.0


def simplify(waypoints: list[Waypoint], min_step_rad: float = DEFAULT_SIMPLIFY_RAD) -> list[Waypoint]:
    """Drop samples that don't change the shape.

    Hand-recorded motion is full of pauses and tremor: a 20 Hz recording
    of someone holding still is hundreds of identical samples. Keeps the
    first and last always, so the endpoints — the parts that have to be
    exact — survive untouched.
    """
    if len(waypoints) <= 2:
        return list(waypoints)
    kept = [waypoints[0]]
    for wp in waypoints[1:-1]:
        if np.max(np.abs(wp.joints_rad - kept[-1].joints_rad)) >= min_step_rad:
            kept.append(wp)
    kept.append(waypoints[-1])
    return kept


def is_loop(waypoints: list[Waypoint], tolerance_rad: float = LOOP_TOLERANCE_RAD) -> bool:
    """True if the motion ends where it began, so it can be repeated."""
    if len(waypoints) < 3:
        return False
    return bool(np.max(np.abs(waypoints[-1].joints_rad - waypoints[0].joints_rad)) <= tolerance_rad)


def duration(waypoints: list[Waypoint]) -> float:
    return waypoints[-1].t - waypoints[0].t if waypoints else 0.0


def peak_joint_rate(waypoints: list[Waypoint]) -> float:
    """Fastest joint rate demanded by the recording, rad/s.

    This is what a speed cap is compared against — it answers "how fast
    would replaying this exactly actually move a joint".
    """
    fastest = 0.0
    for a, b in pairwise(waypoints):
        dt = b.t - a.t
        if dt <= 0:
            continue
        fastest = max(fastest, float(np.max(np.abs(b.joints_rad - a.joints_rad)) / dt))
    return fastest


def speed_scale_for_cap(waypoints: list[Waypoint], max_joint_radps: float) -> float:
    """Playback rate multiplier that keeps every joint under the cap.

    1.0 means the recording is already slow enough to replay as taught;
    below 1.0 slows the WHOLE motion by one factor, preserving its
    relative rhythm.
    """
    peak = peak_joint_rate(waypoints)
    if peak <= 0.0 or peak <= max_joint_radps:
        return 1.0
    return max_joint_radps / peak


def sample_at(waypoints: list[Waypoint], t: float) -> tuple[np.ndarray, float]:
    """Interpolated (joints, gripper) at time `t` along the recording."""
    if not waypoints:
        raise ValueError("empty trajectory")
    if t <= waypoints[0].t:
        return waypoints[0].joints_rad.copy(), waypoints[0].gripper
    if t >= waypoints[-1].t:
        return waypoints[-1].joints_rad.copy(), waypoints[-1].gripper
    for a, b in pairwise(waypoints):
        if a.t <= t <= b.t:
            span = b.t - a.t
            frac = 0.0 if span <= 0 else (t - a.t) / span
            return (
                a.joints_rad + (b.joints_rad - a.joints_rad) * frac,
                a.gripper + (b.gripper - a.gripper) * frac,
            )
    return waypoints[-1].joints_rad.copy(), waypoints[-1].gripper


def nearest_waypoint_index(waypoints: list[Waypoint], joints_rad: np.ndarray) -> int:
    """Index of the waypoint closest to the arm's current configuration.

    Entry point for replaying a PATH from wherever the operator happens
    to be: joining at the nearest point keeps the unguarded approach
    segment short. Gestures ignore this and always start at index 0 —
    a wave that begins halfway through isn't a wave.
    """
    errors = [float(np.max(np.abs(wp.joints_rad - joints_rad))) for wp in waypoints]
    return int(np.argmin(errors))


class TrajectoryPlayer:
    """Follows a recorded trajectory in playback time.

    Playback time advances with the clock, but ONLY while the arm is
    actually keeping up: if the measured position falls further than
    `follow_tolerance_rad` behind the commanded sample, time holds until
    it catches up. Without that leash a blocked or slow joint would let
    the target run away down the path and the arm would then cut the
    corner back to it — turning a taught path into exactly the straight
    line the path existed to avoid.
    """

    def __init__(
        self,
        waypoints: list[Waypoint],
        *,
        speed: float = 1.0,
        repeats: int = 1,
        follow_tolerance_rad: float = 0.25,
    ) -> None:
        if not waypoints:
            raise ValueError("cannot play an empty trajectory")
        self.waypoints = waypoints
        self.speed = speed
        self.repeats = max(1, repeats)
        self.follow_tolerance_rad = follow_tolerance_rad
        self.t = waypoints[0].t
        self.cycle = 0
        self.waiting = False        # True while held back by the leash

    @property
    def end_t(self) -> float:
        return self.waypoints[-1].t

    def target(self) -> tuple[np.ndarray, float]:
        return sample_at(self.waypoints, self.t)

    def advance(self, dt: float, measured_joints: np.ndarray | None) -> bool:
        """Move playback time on by `dt` (scaled). Returns True when the
        whole motion, including any repeats, has finished."""
        commanded, _ = self.target()
        if measured_joints is not None:
            behind = float(np.max(np.abs(commanded - np.asarray(measured_joints))))
            self.waiting = behind > self.follow_tolerance_rad
            if self.waiting:
                return False        # let the arm catch up before moving on

        self.t += dt * self.speed
        if self.t < self.end_t:
            return False

        self.cycle += 1
        if self.cycle >= self.repeats:
            self.t = self.end_t
            return True
        self.t = self.waypoints[0].t   # loop again
        return False
