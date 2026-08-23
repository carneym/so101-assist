"""Ride out transient servo-bus failures instead of dying on them.

A USB hiccup — another device on the bus re-enumerating, a noisy
packet, a momentary brownout — surfaces here as a SerialException or
ConnectionError from deep inside the servo SDK. Before this, any one of
them killed the teleop loop outright, which is the worst response
available: the arm is left under torque holding position while the
process unwinds, and the operator sees a wall of traceback instead of a
working arm.

The right behavior is the one the arm already has for a load trip: HOLD.
A servo under torque keeps its last commanded position with no host
involvement, so a bus outage is survivable by simply not commanding
anything until the bus returns. This class decides how long to wait and
when to give up.

Deliberately does NOT re-enable torque after a reconnect: torque lives
in a servo register, on the servos' own power rail, so it survives the
host losing the port. Re-enabling would be a motion-adjacent write
issued automatically after an error — exactly the kind of thing that
should stay an explicit operator action.
"""
from __future__ import annotations

import contextlib

from serial import SerialException

# What "the bus glitched" looks like. ConnectionError is what LeRobot
# raises for a failed read/write; SerialException is pyserial's when the
# port itself goes away. Both are OSError subclasses, but catching
# OSError would also swallow unrelated bugs.
TRANSIENT_BUS_ERRORS = (ConnectionError, SerialException)

DEFAULT_RECONNECT_AFTER = 3      # consecutive bad ticks before reopening the port
DEFAULT_GIVE_UP_AFTER_S = 3.0    # total outage the loop will tolerate


class BusWatchdog:
    """Tracks consecutive bus failures and says whether to keep going.

    The caller holds position while this returns True, and shuts down
    (releasing torque) the first time it returns False.
    """

    def __init__(
        self,
        driver,
        *,
        reconnect_after: int = DEFAULT_RECONNECT_AFTER,
        give_up_after_s: float = DEFAULT_GIVE_UP_AFTER_S,
        log=print,
    ) -> None:
        self.driver = driver
        self.reconnect_after = reconnect_after
        self.give_up_after_s = give_up_after_s
        self.log = log
        self.consecutive = 0
        self.first_failure_at: float | None = None
        self.reconnects = 0

    def record_success(self) -> None:
        """Call after any tick that talked to the arm without error."""
        if self.consecutive:
            self.log(f"[bus] recovered after {self.consecutive} failed tick(s)")
        self.consecutive = 0
        self.first_failure_at = None

    def record_failure(self, exc: Exception, now: float) -> bool:
        """Record one failed tick. True = hold and retry, False = give up.

        `now` is passed in rather than read from the clock so the retry
        policy is testable without sleeping.
        """
        self.consecutive += 1
        if self.first_failure_at is None:
            self.first_failure_at = now
            self.log(f"[bus] servo bus error — holding position, retrying. ({exc})")

        if now - self.first_failure_at >= self.give_up_after_s:
            self.log(
                f"[bus] no response for {self.give_up_after_s:.0f}s — stopping and "
                "releasing torque. Check the USB cable and whether another device "
                "on the bus is re-enumerating."
            )
            return False

        if self.consecutive % self.reconnect_after == 0:
            self._reopen()
        return True

    def _reopen(self) -> None:
        """Reopen the port. A re-enumerated device leaves a stale file
        descriptor that will never recover on its own, so retrying reads
        alone isn't enough."""
        # Already gone is the normal case here; nothing to salvage.
        with contextlib.suppress(*TRANSIENT_BUS_ERRORS):
            self.driver.disconnect()
        try:
            self.driver.connect()
        except TRANSIENT_BUS_ERRORS as exc:
            self.log(f"[bus] reconnect failed: {exc}")
            return
        self.reconnects += 1
        self.log("[bus] port reopened")
