"""Shared plumbing for pygame-joystick-backed input devices.

gamepad.py and quadstick.py both read a pygame joystick, apply a
deadzone + expo curve to raw axes so JogCommand semantics stay
device-independent, and publish JogCommand at a fixed rate. This
module holds exactly that shared plumbing; device-specific axis and
button meaning stays in each driver.

Uses pygame rather than evdev so input drivers work cross-platform
(evdev is Linux-only). pygame is imported lazily inside connect() so
importing this module — and unit-testing driver logic against a fake
joystick — never requires pygame or a physical device to be present.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Protocol

from ...bus import TOPIC_JOG, Bus
from ...messages import JogCommand


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero out |value| < deadzone; rescale the remainder to fill [-1, 1]."""
    if abs(value) < deadzone:
        return 0.0
    sign = math.copysign(1.0, value)
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def apply_expo(value: float, expo: float) -> float:
    """Expo curve: gentler near center, full authority at the ends.

    expo == 1.0 is linear; > 1.0 softens small deflections.
    """
    return math.copysign(abs(value) ** expo, value)


def shape_axis(value: float, deadzone: float, expo: float) -> float:
    return apply_expo(apply_deadzone(value, deadzone), expo)


class JoystickLike(Protocol):
    """Minimal surface this module needs from a pygame.joystick.Joystick.

    Lets tests drive driver logic with a plain fake, without pygame or
    a physical device.
    """

    def get_axis(self, index: int) -> float: ...
    def get_button(self, index: int) -> bool: ...


class PygameJoystickDevice:
    """Base for input drivers backed by a pygame joystick.

    Subclasses implement `_read_jog(joystick) -> JogCommand`, called
    once per poll. `run()` blocks and publishes at `poll_hz`; `start()`
    runs it on a daemon thread instead.
    """

    def __init__(self, bus: Bus, joystick_index: int = 0, poll_hz: float = 60.0) -> None:
        self.bus = bus
        self.joystick_index = joystick_index
        self.poll_hz = poll_hz
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._joystick: JoystickLike | None = None

    def connect(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self.joystick_index:
            raise RuntimeError(f"no joystick at index {self.joystick_index}")
        joystick = pygame.joystick.Joystick(self.joystick_index)
        joystick.init()
        self._joystick = joystick

    def start(self) -> None:
        """Connect and run the poll loop on a daemon thread."""
        self.connect()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def run(self) -> None:
        """Blocking poll loop: read the device, publish JogCommand, repeat."""
        import pygame

        if self._joystick is None:
            self.connect()
        period = 1.0 / self.poll_hz
        while not self._stop_event.is_set():
            pygame.event.pump()
            self.bus.publish(TOPIC_JOG, self._read_jog(self._joystick))
            time.sleep(period)

    def _read_jog(self, joystick: JoystickLike) -> JogCommand:
        raise NotImplementedError
