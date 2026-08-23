"""Safety layer: workspace fence, load monitoring, stop handling.

- Workspace fence: axis-aligned box in base frame (config/default.yaml).
  Velocity components that would exit the box are zeroed, not clipped
  at the boundary (prevents sliding along the fence into singularities).
- Load monitor: servo load above threshold for N consecutive ticks
  => trip (probable collision). The controller treats a trip as STOP.
- STOP semantics: zero velocity, hold position, keep torque ON
  (torque off would drop a held object). Torque-off is a separate,
  explicit action.
"""
from __future__ import annotations

import numpy as np


class WorkspaceFence:
    """Axis-aligned box in the robot base frame, meters.

    clamp() zeroes any velocity component pointing further out of the
    box when the end effector is at/past that face. Components moving
    back INTO the box always pass through, so the arm can never get
    stuck outside the fence (e.g. after being hand-moved while limp).
    """

    def __init__(self, x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> None:
        self.lo = np.array([x[0], y[0], z[0]])
        self.hi = np.array([x[1], y[1], z[1]])

    @classmethod
    def from_config(cls, fence_cfg: dict) -> WorkspaceFence:
        """Build from the `workspace_fence` block of config/default.yaml."""
        return cls(tuple(fence_cfg["x"]), tuple(fence_cfg["y"]), tuple(fence_cfg["z"]))

    def clamp(self, ee_xyz: np.ndarray, linear_vel: np.ndarray) -> np.ndarray:
        v = np.array(linear_vel, dtype=float)
        for i in range(3):
            # A degenerate axis (lo >= hi) is treated as UNBOUNDED, not
            # as a zero-width wall — a misconfigured fence that collapsed
            # to a point must never silently block all motion.
            if self.lo[i] >= self.hi[i]:
                continue
            if ee_xyz[i] <= self.lo[i] and v[i] < 0:
                v[i] = 0.0
            if ee_xyz[i] >= self.hi[i] and v[i] > 0:
                v[i] = 0.0
        return v

    def contains(self, ee_xyz: np.ndarray) -> bool:
        return bool(np.all(ee_xyz >= self.lo) and np.all(ee_xyz <= self.hi))


def joints_over(joint_load: np.ndarray, threshold: float | None, names: list[str]) -> list[str]:
    """Names of joints whose normalized load exceeds `threshold`.

    Pure and read-only — this is the "is anything pushing hard right
    now" question, separate from LoadMonitor's "has it pushed hard long
    enough to be a collision" one. Used to warn the operator BEFORE the
    guard trips, and to give feedback at all when the guard is disabled
    (`load_stop_threshold: null`), which is common during bring-up.

    `threshold` of None means "no warning level configured" -> never.
    """
    if threshold is None:
        return []
    load = np.asarray(joint_load)
    return [name for name, value in zip(names, load) if value > threshold]


class LoadMonitor:
    """Trips when any joint's normalized load exceeds `threshold` for
    `ticks` CONSECUTIVE feeds — a sustained overload reads as a
    probable collision or jam, a single spike as noise.

    Once tripped it stays tripped until reset() — a collision must not
    "clear itself" just because the load dropped after the arm stopped
    pushing.
    """

    def __init__(self, threshold: float | None, ticks: int) -> None:
        # threshold None -> monitor disabled (feed never trips). Use for
        # bring-up / when the arm can't lift its own weight under the
        # guard; you lose collision-stop protection, so only off
        # deliberately.
        self.threshold = threshold
        self.ticks = ticks
        self._over_count = 0
        self.tripped = False

    @classmethod
    def from_config(cls, arm_cfg: dict) -> LoadMonitor:
        """Build from the `arm` block of config/default.yaml.
        `load_stop_threshold: null` (or absent) disables the monitor."""
        return cls(arm_cfg.get("load_stop_threshold"), arm_cfg.get("load_stop_ticks", 5))

    def feed(self, joint_load: np.ndarray) -> bool:
        """Feed one tick's load reading; returns current tripped state."""
        if self.threshold is None:
            return False
        if self.tripped:
            return True
        if np.any(np.asarray(joint_load) > self.threshold):
            self._over_count += 1
            if self._over_count >= self.ticks:
                self.tripped = True
        else:
            self._over_count = 0
        return self.tripped

    def reset(self) -> None:
        self._over_count = 0
        self.tripped = False
