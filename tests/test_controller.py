"""Safety layer + Cartesian controller: fence zeroing semantics, load
monitor trip/latch behavior, velocity caps, STOP semantics, and that
commanded Cartesian velocities move the FK position the right way.
Driven against a fake driver — no hardware."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.arm.controller import CartesianController
from so101_assist.arm.kinematics import fk
from so101_assist.arm.safety import LoadMonitor, WorkspaceFence
from so101_assist.messages import CartesianVelocity


class FakeDriver:
    """Holds a joint state; write_joint_targets updates it directly
    (perfect tracking), so controller ticks integrate like hardware."""

    def __init__(self, joint_pos: np.ndarray | None = None):
        self.joint_pos = joint_pos if joint_pos is not None else np.zeros(5)
        self.joint_load = np.zeros(5)
        self.gripper_pos = 0.5
        self.writes: list[tuple[np.ndarray, float]] = []

    def read_state(self):
        return self.joint_pos.copy(), self.joint_load.copy(), self.gripper_pos

    def write_joint_targets(self, pos: np.ndarray, gripper: float) -> None:
        self.writes.append((pos.copy(), gripper))
        self.joint_pos = pos.copy()
        self.gripper_pos = gripper


def _vel(
    linear=(0.0, 0.0, 0.0), wrist=(0.0, 0.0), shoulder=(0.0, 0.0), elbow=0.0, gripper=0.0
) -> CartesianVelocity:
    return CartesianVelocity(
        linear=np.array(linear),
        wrist=np.array(wrist),
        shoulder=np.array(shoulder),
        elbow=elbow,
        gripper=gripper,
    )


# ---------------------------------------------------------------- fence

def test_fence_zeroes_outward_component_at_boundary():
    fence = WorkspaceFence(x=(0.1, 0.3), y=(-0.2, 0.2), z=(0.0, 0.3))
    v = fence.clamp(np.array([0.3, 0.0, 0.1]), np.array([0.05, 0.02, 0.0]))
    assert v[0] == 0.0          # +x at the +x face: zeroed, not clipped
    assert v[1] == 0.02         # other components untouched
    assert v[2] == 0.0


def test_fence_allows_moving_back_inside():
    fence = WorkspaceFence(x=(0.1, 0.3), y=(-0.2, 0.2), z=(0.0, 0.3))
    # EE outside the +x face, commanding -x back toward the box
    v = fence.clamp(np.array([0.35, 0.0, 0.1]), np.array([-0.05, 0.0, 0.0]))
    assert v[0] == -0.05


def test_fence_passes_velocity_through_in_interior():
    fence = WorkspaceFence(x=(0.1, 0.3), y=(-0.2, 0.2), z=(0.0, 0.3))
    v = fence.clamp(np.array([0.2, 0.0, 0.15]), np.array([0.05, -0.03, 0.02]))
    assert np.array_equal(v, [0.05, -0.03, 0.02])


def test_degenerate_fence_axis_is_unbounded_not_a_wall():
    """A collapsed/inverted axis (lo >= hi) must NOT block all motion —
    a misconfigured fence collapsed to a point once killed live jogging."""
    fence = WorkspaceFence(x=(0.9, 0.9), y=(0.9, 0.9), z=(0.9, 0.9))
    v = fence.clamp(np.array([0.2, 0.0, 0.1]), np.array([0.05, -0.03, 0.02]))
    assert np.array_equal(v, [0.05, -0.03, 0.02])   # nothing zeroed


# --------------------------------------------------------- load monitor

def test_load_monitor_needs_consecutive_ticks_to_trip():
    monitor = LoadMonitor(threshold=0.5, ticks=3)
    high, low = np.array([0.6, 0, 0, 0, 0]), np.zeros(5)
    assert monitor.feed(high) is False
    assert monitor.feed(high) is False
    assert monitor.feed(low) is False     # streak broken
    assert monitor.feed(high) is False
    assert monitor.feed(high) is False
    assert monitor.feed(high) is True     # 3 consecutive


def test_load_monitor_latches_until_reset():
    monitor = LoadMonitor(threshold=0.5, ticks=1)
    monitor.feed(np.array([0.9, 0, 0, 0, 0]))
    assert monitor.tripped
    assert monitor.feed(np.zeros(5)) is True   # load gone, still tripped
    monitor.reset()
    assert monitor.feed(np.zeros(5)) is False


def test_load_monitor_disabled_never_trips():
    monitor = LoadMonitor(threshold=None, ticks=1)
    for _ in range(20):
        assert monitor.feed(np.array([9.9, 0, 0, 0, 0])) is False


def test_load_monitor_from_config_null_threshold_disables():
    from so101_assist.arm.safety import LoadMonitor as LM

    monitor = LM.from_config({"load_stop_threshold": None, "load_stop_ticks": 5})
    assert monitor.feed(np.full(5, 9.9)) is False


# ----------------------------------------------------------- controller

def test_positive_z_velocity_raises_ee():
    driver = FakeDriver()
    controller = CartesianController(driver)
    z_before = fk(driver.joint_pos)[:3, 3][2]

    for _ in range(10):
        assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02)

    z_after = fk(driver.joint_pos)[:3, 3][2]
    assert z_after > z_before + 1e-3


def test_last_ee_xyz_tracks_fk_including_while_stopped():
    """The controller exposes the live EE position for the tuning-GUI
    readout, and must keep it current even during a STOP hold (the
    readout stays live so the operator can still tune the fence)."""
    driver = FakeDriver()
    controller = CartesianController(driver)
    assert controller.last_ee_xyz is None

    before = driver.joint_pos.copy()
    controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02)
    # reflects the position measured at the START of the tick
    assert np.allclose(controller.last_ee_xyz, fk(before)[:3, 3])

    controller.stop()
    driver.joint_pos = np.array([0.2, 0.1, 0.0, 0.0, 0.0])   # arm hand-moved while halted
    assert controller.tick(_vel(), dt=0.02) is False          # still stopped
    assert np.allclose(controller.last_ee_xyz, fk(driver.joint_pos)[:3, 3])


def test_linear_velocity_is_capped_by_norm():
    driver = FakeDriver()
    controller = CartesianController(driver, max_linear_mps=0.06)
    dt = 0.02

    controller.tick(_vel(linear=(10.0, 0.0, 0.0)), dt=dt)   # absurd command

    p_moved = fk(driver.joint_pos)[:3, 3] - fk(np.zeros(5))[:3, 3]
    # one tick at the cap can move at most max_linear * dt (plus damped-solve slack)
    assert np.linalg.norm(p_moved) < 0.06 * dt * 1.5


def test_wrist_rates_pass_through_to_wrist_joints():
    driver = FakeDriver()
    controller = CartesianController(driver)
    controller.tick(_vel(wrist=(0.4, -0.4)), dt=0.1)

    assert driver.joint_pos[3] == pytest.approx(0.04)    # wrist_flex
    assert driver.joint_pos[4] == pytest.approx(-0.04)   # wrist_roll
    assert np.allclose(driver.joint_pos[:3], 0.0)        # arm untouched


def test_stop_halts_motion_until_resume():
    driver = FakeDriver()
    controller = CartesianController(driver)
    controller.stop()

    assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02) is False
    assert driver.writes == []

    controller.resume()
    assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02) is True
    assert len(driver.writes) == 1


def test_load_trip_stops_motion_and_resume_clears_it():
    driver = FakeDriver()
    monitor = LoadMonitor(threshold=0.5, ticks=1)
    controller = CartesianController(driver, load_monitor=monitor)

    driver.joint_load = np.array([0.9, 0, 0, 0, 0])
    assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02) is False
    assert driver.writes == []

    driver.joint_load = np.zeros(5)
    assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02) is False   # latched

    controller.resume()
    assert controller.tick(_vel(linear=(0.0, 0.0, 0.05)), dt=0.02) is True


def test_fence_blocks_outward_motion_in_controller():
    driver = FakeDriver()
    start_xyz = fk(driver.joint_pos)[:3, 3]
    # a fence whose +x face is exactly at the current EE x
    fence = WorkspaceFence(
        x=(start_xyz[0] - 0.2, start_xyz[0]), y=(-0.3, 0.3), z=(-0.3, 0.5)
    )
    controller = CartesianController(driver, fence=fence)

    controller.tick(_vel(linear=(0.05, 0.0, 0.0)), dt=0.02)   # push outward in +x

    moved = fk(driver.joint_pos)[:3, 3] - start_xyz
    assert abs(moved[0]) < 1e-6   # x motion zeroed by the fence


def test_elbow_rate_passes_through_to_elbow_joint():
    driver = FakeDriver()
    controller = CartesianController(driver)
    controller.tick(_vel(elbow=0.4), dt=0.1)

    assert driver.joint_pos[2] == pytest.approx(0.04)   # elbow
    assert np.allclose(np.delete(driver.joint_pos, 2), 0.0)


def test_elbow_rate_is_capped():
    driver = FakeDriver()
    controller = CartesianController(driver, max_elbow_radps=0.5)
    controller.tick(_vel(elbow=10.0), dt=0.1)

    assert driver.joint_pos[2] == pytest.approx(0.05)   # 0.5 rad/s * 0.1 s


def test_fence_zeroes_elbow_that_would_exit_the_box():
    driver = FakeDriver()
    start_xyz = fk(driver.joint_pos)[:3, 3]
    # At zero pose, positive elbow rate moves the EE in -z; put the
    # fence floor exactly at the current EE height so that motion is
    # outward at the -z face.
    fence = WorkspaceFence(x=(-0.5, 0.5), y=(-0.5, 0.5), z=(start_xyz[2], start_xyz[2] + 0.3))
    controller = CartesianController(driver, fence=fence)

    controller.tick(_vel(elbow=0.4), dt=0.1)
    assert driver.joint_pos[2] == pytest.approx(0.0)    # elbow motion blocked

    controller.tick(_vel(elbow=-0.4), dt=0.1)           # EE up = back into the box
    assert driver.joint_pos[2] < 0.0                    # allowed


def test_shoulder_rates_pass_through_to_shoulder_joints():
    driver = FakeDriver()
    controller = CartesianController(driver)
    controller.tick(_vel(shoulder=(0.3, -0.3)), dt=0.1)

    assert driver.joint_pos[0] == pytest.approx(0.03)    # shoulder_pan
    assert driver.joint_pos[1] == pytest.approx(-0.03)   # shoulder_lift
    assert np.allclose(driver.joint_pos[2:], 0.0)


def test_shoulder_rate_is_capped():
    driver = FakeDriver()
    controller = CartesianController(driver, max_shoulder_radps=0.4)
    controller.tick(_vel(shoulder=(10.0, -10.0)), dt=0.1)

    assert driver.joint_pos[0] == pytest.approx(0.04)
    assert driver.joint_pos[1] == pytest.approx(-0.04)


def test_shoulder_channels_are_not_fence_checked():
    """Deliberate operator decision (2026-07-26): SHOULDER mode is the
    direct-drive fallback and the any-outward-component fence rule
    blocked it far too early — shoulder pan/lift bypass the fence.
    Guarding falls to the load monitor, step clamps, and joint limits."""
    driver = FakeDriver()
    start_xyz = fk(driver.joint_pos)[:3, 3]
    # A fence whose floor is AT the current EE height: lift motion
    # heading below it must still be allowed through.
    fence = WorkspaceFence(x=(-0.5, 0.5), y=(-0.5, 0.5), z=(start_xyz[2], start_xyz[2] + 0.3))
    controller = CartesianController(driver, fence=fence)

    controller.tick(_vel(shoulder=(0.0, 0.3)), dt=0.1)
    assert driver.joint_pos[1] == pytest.approx(0.03)    # NOT blocked
    assert controller.last_fence_blocks == []


def test_custom_joint_limits_override_urdf_defaults():
    """Operating limits are per-setup policy: a config override must
    widen the range the controller allows past the URDF default."""
    from so101_assist.arm.kinematics import JOINT_LIMITS_RAD

    driver = FakeDriver(joint_pos=np.array([0.0, 1.7, 0.0, 0.0, 0.0]))   # near URDF lift limit
    wide = dict(JOINT_LIMITS_RAD)
    wide["shoulder_lift"] = (-2.6, 2.6)
    controller = CartesianController(driver, joint_limits=wide)

    for _ in range(30):
        controller.tick(_vel(shoulder=(0.0, 0.4)), dt=0.1)

    assert driver.joint_pos[1] > 1.75           # drove past the URDF limit
    assert "shoulder_lift" not in controller.last_limit_clips or driver.joint_pos[1] > 2.5


def test_default_joint_limits_still_clip():
    driver = FakeDriver(joint_pos=np.array([0.0, 1.7, 0.0, 0.0, 0.0]))
    controller = CartesianController(driver)

    for _ in range(10):
        controller.tick(_vel(shoulder=(0.0, 0.4)), dt=0.1)

    assert driver.joint_pos[1] == pytest.approx(1.74533)   # pinned at URDF limit
    assert "shoulder_lift" in controller.last_limit_clips


def test_gripper_velocity_integrates_and_clamps():
    driver = FakeDriver()
    controller = CartesianController(driver)

    controller.tick(_vel(gripper=1.0), dt=0.05)
    assert driver.gripper_pos == pytest.approx(0.55)

    for _ in range(20):
        controller.tick(_vel(gripper=1.0), dt=0.5)
    assert driver.gripper_pos == 1.0   # clamped at fully open


class StuckDriver(FakeDriver):
    """Ignores writes — simulates a servo pinned by stiction (or a
    physically blocked joint): commands are accepted, position never
    changes."""

    def write_joint_targets(self, pos: np.ndarray, gripper: float) -> None:
        self.writes.append((pos.copy(), gripper))
        # position deliberately NOT updated


def test_setpoint_accumulates_against_stiction_up_to_leash():
    """Regression: hardware wrist_flex froze under jog because each
    tick re-anchored the target to the measured position, so the
    commanded step (0.02 rad) never exceeded the servo's stiction
    deadband (~0.03 rad). The persistent setpoint must keep growing
    the commanded error across ticks — but never beyond the leash."""
    leash = 0.08
    driver = StuckDriver()
    controller = CartesianController(driver, setpoint_leash_rad=leash)

    for _ in range(20):
        controller.tick(_vel(wrist=(0.5, 0.0)), dt=0.04)   # 0.02 rad/tick commanded

    commanded_flex = [w[0][3] for w in driver.writes]
    assert commanded_flex[1] > commanded_flex[0]           # error accumulates across ticks
    # 20 ticks * 0.02 = 0.4 rad requested, but the leash bounds it
    assert commanded_flex[-1] == pytest.approx(leash)


def test_resume_reseeds_setpoint_from_measured_position():
    """After STOP/resume the setpoint must re-seed from the measured
    position — never command a jump back to a stale setpoint."""
    driver = FakeDriver()
    controller = CartesianController(driver)
    for _ in range(5):
        controller.tick(_vel(wrist=(0.5, 0.0)), dt=0.04)

    controller.stop()
    driver.joint_pos = np.array([0.0, 0.0, 0.0, 0.5, 0.0])   # arm hand-moved while halted
    controller.resume()
    controller.tick(_vel(), dt=0.04)   # zero velocity

    assert driver.writes[-1][0][3] == pytest.approx(0.5)   # holds where it IS, no jump
