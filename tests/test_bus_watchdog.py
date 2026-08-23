"""BusWatchdog: how long the control loop tolerates a dead servo bus.

Holding position through a hiccup is safe (a torqued servo holds with
no host involvement); holding forever is not, because the operator has
no control while the bus is down.
"""
from __future__ import annotations

import pytest
from serial import SerialException

from so101_assist.arm.bus_watchdog import TRANSIENT_BUS_ERRORS, BusWatchdog


class FakeDriver:
    def __init__(self, reconnect_ok: bool = True):
        self.reconnect_ok = reconnect_ok
        self.connects = 0
        self.disconnects = 0
        self.torque_enables = 0

    def connect(self):
        self.connects += 1
        if not self.reconnect_ok:
            raise ConnectionError("still gone")

    def disconnect(self):
        self.disconnects += 1

    def enable_torque(self):
        self.torque_enables += 1


@pytest.fixture
def driver() -> FakeDriver:
    return FakeDriver()


def watchdog(driver, **kw) -> BusWatchdog:
    kw.setdefault("log", lambda *_: None)
    return BusWatchdog(driver, **kw)


def test_both_error_families_are_recognized():
    """LeRobot raises ConnectionError, pyserial raises SerialException —
    a policy that catches only one still dies on the other."""
    assert isinstance(ConnectionError("x"), TRANSIENT_BUS_ERRORS)
    assert isinstance(SerialException("x"), TRANSIENT_BUS_ERRORS)


def test_a_brief_glitch_keeps_the_loop_running(driver):
    wd = watchdog(driver)

    assert wd.record_failure(ConnectionError(), now=0.0) is True
    assert wd.record_failure(ConnectionError(), now=0.04) is True


def test_recovery_resets_the_outage(driver):
    wd = watchdog(driver, give_up_after_s=3.0)
    wd.record_failure(ConnectionError(), now=0.0)
    wd.record_failure(ConnectionError(), now=1.0)

    wd.record_success()

    assert wd.consecutive == 0
    # A later failure starts a fresh 3s budget rather than inheriting
    # the old one — otherwise sporadic glitches would eventually add up
    # to a shutdown despite the arm working fine in between.
    assert wd.record_failure(ConnectionError(), now=2.9) is True
    assert wd.record_failure(ConnectionError(), now=5.5) is True


def test_gives_up_once_the_outage_exceeds_the_budget(driver):
    wd = watchdog(driver, give_up_after_s=3.0)

    assert wd.record_failure(ConnectionError(), now=0.0) is True
    assert wd.record_failure(ConnectionError(), now=2.9) is True
    assert wd.record_failure(ConnectionError(), now=3.0) is False


def test_reopens_the_port_after_repeated_failures(driver):
    """A re-enumerated device leaves a stale fd — retrying reads alone
    would never recover."""
    wd = watchdog(driver, reconnect_after=3)

    for tick, now in enumerate([0.0, 0.04, 0.08]):
        wd.record_failure(ConnectionError(), now=now)

    assert driver.disconnects == 1
    assert driver.connects == 1


def test_does_not_reopen_on_the_very_first_failure(driver):
    wd = watchdog(driver, reconnect_after=3)

    wd.record_failure(ConnectionError(), now=0.0)

    assert driver.connects == 0      # a single dropped packet isn't a dead port


def test_a_failed_reconnect_is_not_fatal_on_its_own(driver):
    """The port may take a moment to come back; keep holding until the
    time budget decides, not the first failed reopen."""
    driver.reconnect_ok = False
    wd = watchdog(driver, reconnect_after=1, give_up_after_s=3.0)

    assert wd.record_failure(ConnectionError(), now=0.0) is True
    assert driver.connects == 1


def test_reconnect_never_re_enables_torque(driver):
    """Torque lives on the servos' own rail and survives the host losing
    the port. Re-enabling it automatically after an error would be a
    motion-adjacent write nobody asked for."""
    wd = watchdog(driver, reconnect_after=1)

    wd.record_failure(ConnectionError(), now=0.0)

    assert driver.torque_enables == 0


def test_success_before_any_failure_is_harmless(driver):
    wd = watchdog(driver)

    wd.record_success()

    assert wd.consecutive == 0
    assert wd.first_failure_at is None


# ------------------------------------------- integration with the real loop

class FlakyDriver:
    """Real-ish driver whose bus dies for a window of ticks, reproducing
    the field failure: 'device reports readiness to read but returned no
    data (device disconnected...)' raised from inside read_state."""

    def __init__(self, fail_from: int, fail_until: int):
        import numpy as np
        self.np = np
        self.fail_from = fail_from
        self.fail_until = fail_until
        self.tick = 0
        self.joint_pos = np.zeros(5)
        self.joint_load = np.zeros(5)
        self.gripper_pos = 0.5
        self.connects = 0

    def read_state(self):
        self.tick += 1
        if self.fail_from <= self.tick < self.fail_until:
            raise SerialException(
                "device reports readiness to read but returned no data "
                "(device disconnected or multiple access on port?)"
            )
        return self.joint_pos.copy(), self.joint_load.copy(), self.gripper_pos

    def write_joint_targets(self, pos, gripper):
        self.joint_pos = pos.copy()
        self.gripper_pos = gripper

    def connect(self):
        self.connects += 1

    def disconnect(self):
        pass


def run_loop(driver, ticks: int, watchdog: BusWatchdog):
    """Miniature of the teleop control loop: tick, hold on bus error,
    stop when the watchdog gives up. Returns (completed, held)."""
    import numpy as np

    from so101_assist.arm.controller import CartesianController
    from so101_assist.messages import CartesianVelocity

    controller = CartesianController(driver)
    zero = CartesianVelocity(linear=np.zeros(3), wrist=np.zeros(2))
    held = 0
    for i in range(ticks):
        try:
            controller.tick(zero, dt=0.04)
        except TRANSIENT_BUS_ERRORS as exc:
            held += 1
            if not watchdog.record_failure(exc, now=i * 0.04):
                return False, held
            continue
        watchdog.record_success()
    return True, held


def test_loop_survives_a_transient_outage_and_resumes():
    """The field failure: the port drops for ~10 ticks (0.4 s) and comes
    back. The loop must hold through it and keep driving afterwards."""
    driver = FlakyDriver(fail_from=5, fail_until=15)
    wd = watchdog(driver, give_up_after_s=3.0)

    completed, held = run_loop(driver, 40, wd)

    assert completed          # never gave up
    assert held == 10         # held exactly through the outage
    assert wd.consecutive == 0    # and recovered


def test_loop_shuts_down_when_the_bus_never_comes_back():
    """A permanently dead port must NOT hold forever — the operator has
    no control while it's down, so the loop exits and releases torque."""
    driver = FlakyDriver(fail_from=5, fail_until=10_000)
    wd = watchdog(driver, give_up_after_s=1.0)

    completed, held = run_loop(driver, 400, wd)

    assert not completed
    assert held < 400         # gave up rather than spinning to the end


def test_a_dead_port_is_reopened_while_holding():
    driver = FlakyDriver(fail_from=1, fail_until=10_000)
    wd = watchdog(driver, reconnect_after=3, give_up_after_s=1.0)

    run_loop(driver, 400, wd)

    assert driver.connects > 0
