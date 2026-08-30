"""Keyboard jog — the fallback when no QuadStick is plugged in.

Publishes the same JogCommand and PoseEvent messages the QuadStick
driver does, so modes, the gripper channel, the pose menu and every
safety layer behave identically. Only the input surface differs.

Reads raw keys from the terminal (termios/cbreak) rather than pygame or
the OpenCV window, so it works headless over SSH and with --no-camera.
The terminal running teleop must stay focused.

THE KEY-RELEASE PROBLEM, and why jogging feels different here: a
terminal reports key PRESSES, never releases. There is no way to ask
"is A still down". Holding a key produces the OS auto-repeat stream —
one character, a ~0.5 s gap, then a fast repeat — so this driver treats
a key as active until `key_hold_s` has passed without another repeat.

Two consequences, both deliberate and both worth knowing before driving:
- holding a key stutters for the first half second, until auto-repeat
  kicks in. Tapping repeatedly is smoother for small corrections.
- releasing a key leaves up to `key_hold_s` of coast. At the default
  0.25 s and the 0.4 rad/s shoulder cap that is about 6 degrees. Keep it
  small; teleop's deadman (--stale-s) is the backstop, not this.

On X11 you can make holds much smoother with a shorter repeat delay:
    xset r rate 200 30

This is a bring-up and fallback input. The QuadStick remains the
interface the system is designed around — see quadstick.py.
"""
from __future__ import annotations

import contextlib
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from ...bus import TOPIC_JOG, TOPIC_POSE, Bus
from ...messages import JogCommand, JogMode, PoseAction, PoseEvent

# How long a key counts as held after its most recent repeat. See the
# key-release discussion above: too short stutters, too long coasts.
DEFAULT_KEY_HOLD_S = 0.25

KEY_LEFT = "a"
KEY_RIGHT = "d"
KEY_UP = "w"
KEY_DOWN = "s"
KEY_SIDE_UP = "r"          # z up in WRIST mode (the QuadStick's left-tube puff)
KEY_SIDE_DOWN = "f"        # z down (left-tube sip)
KEY_GRIP_OPEN = "o"        # centre puff
KEY_GRIP_CLOSE = "c"       # centre sip
KEY_MODE_NEXT = "m"        # lip switch
KEY_POSE_MENU = "p"        # right puff
KEY_POSE_CANCEL = "x"      # right sip — leave the menu / abort a move
KEY_SELECT = "\r"          # lip switch, while the menu is open
KEY_QUIT = "q"

_MODE_ORDER = (JogMode.SHOULDER, JogMode.ELBOW, JogMode.WRIST)


class KeyboardDevice:
    """Terminal-key jog, publishing the QuadStick's message set.

    `run()` blocks; `start()` runs it on a daemon thread. stop() always
    restores the terminal — a raw-mode terminal left behind by a crash
    is unusable, so the restore is in a finally and is idempotent.
    """

    def __init__(
        self,
        bus: Bus,
        *,
        poll_hz: float = 60.0,
        key_hold_s: float = DEFAULT_KEY_HOLD_S,
        stream=None,
    ) -> None:
        self.bus = bus
        self.poll_hz = poll_hz
        self.key_hold_s = key_hold_s
        self._stream = stream or sys.stdin
        self.mode = JogMode.SHOULDER
        self.pose_menu_open = False
        self.quit_requested = False
        self._held: dict[str, float] = {}      # key -> time it was last seen
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_term: list | None = None

    @classmethod
    def from_config(cls, bus: Bus, operator_cfg: dict) -> KeyboardDevice:
        kb_cfg = operator_cfg.get("keyboard", {})
        return cls(bus, key_hold_s=kb_cfg.get("key_hold_s", DEFAULT_KEY_HOLD_S))

    # ------------------------------------------------------------ terminal

    def connect(self) -> None:
        if not self._stream.isatty():
            raise RuntimeError(
                "keyboard input needs a terminal (stdin is not a TTY). "
                "Run teleop directly in a terminal, not through a pipe."
            )
        self._saved_term = termios.tcgetattr(self._stream)
        tty.setcbreak(self._stream.fileno())

    def restore(self) -> None:
        """Put the terminal back. Safe to call more than once."""
        if self._saved_term is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(self._stream, termios.TCSADRAIN, self._saved_term)
            self._saved_term = None

    def start(self) -> None:
        self.connect()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True, name="keyboard-input")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.restore()

    # ---------------------------------------------------------------- loop

    def run(self) -> None:
        period = 1.0 / self.poll_hz
        try:
            while not self._stop_event.is_set():
                self._drain_keys()
                self.bus.publish(TOPIC_JOG, self._read_jog())
                time.sleep(period)
        finally:
            self.restore()

    def _drain_keys(self) -> None:
        """Consume everything typed since the last poll.

        Each key seen refreshes its hold timestamp; edge-triggered
        actions (mode, menu) fire once per keystroke rather than per
        repeat, so a leaning finger can't cycle modes continuously.
        """
        now = time.monotonic()
        while select.select([self._stream], [], [], 0)[0]:
            char = self._stream.read(1)
            if not char:
                break
            key = char.lower() if char.isalpha() else char
            was_held = self._is_held(key, now)
            self._held[key] = now
            if not was_held:
                self._on_key_down(key)

    def _is_held(self, key: str, now: float) -> bool:
        last = self._held.get(key)
        return last is not None and (now - last) <= self.key_hold_s

    def _axis(self, negative: str, positive: str) -> float:
        now = time.monotonic()
        return float(self._is_held(positive, now)) - float(self._is_held(negative, now))

    def _on_key_down(self, key: str) -> None:
        if key == KEY_QUIT:
            self.quit_requested = True
            return
        if key == KEY_POSE_CANCEL:
            # Published even when the menu is closed: this is also the
            # abort for a pose move already running.
            self.pose_menu_open = False
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.CANCEL))
            return
        if key == KEY_POSE_MENU and not self.pose_menu_open:
            self.pose_menu_open = True
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.ENTER))
            return
        if self.pose_menu_open:
            if key == KEY_SELECT or key == "\n":
                self.pose_menu_open = False
                self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.SELECT))
            elif key == KEY_UP:
                self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.CYCLE, delta=-1))
            elif key == KEY_DOWN:
                self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.CYCLE, delta=1))
            return
        if key == KEY_MODE_NEXT:
            self.mode = _MODE_ORDER[(_MODE_ORDER.index(self.mode) + 1) % len(_MODE_ORDER)]

    def _read_jog(self) -> JogCommand:
        if self.pose_menu_open:
            # Browsing is not driving — same rule as the QuadStick.
            return JogCommand(axes=np.zeros(3), mode=self.mode, gripper_delta=0.0)

        x = self._axis(KEY_LEFT, KEY_RIGHT)
        y = self._axis(KEY_DOWN, KEY_UP)
        side = self._axis(KEY_SIDE_DOWN, KEY_SIDE_UP)
        gripper = self._axis(KEY_GRIP_CLOSE, KEY_GRIP_OPEN)

        # Sign conventions match the QuadStick: stick-up reads negative
        # there, and jog_to_velocity is written against that, so "up"
        # here has to be negative too or W would drive the arm down.
        axes = np.zeros(3)
        if self.mode in (JogMode.SHOULDER, JogMode.ELBOW):
            axes[0], axes[1] = x, -y
        elif self.mode is JogMode.WRIST:
            axes[0], axes[1], axes[2] = side, x, -y
        return JogCommand(axes=axes, mode=self.mode, gripper_delta=gripper)


def help_text() -> str:
    return (
        "KEYBOARD CONTROLS\n"
        "  w / s        up / down        (mode-dependent)\n"
        "  a / d        left / right     (shoulder pan)\n"
        "  r / f        z up / down      (WRIST mode)\n"
        "  o / c        gripper open / close   (every mode)\n"
        "  m            next mode: SHOULDER -> ELBOW -> WRIST\n"
        "  p            open the pose menu\n"
        "                 w / s = choose, ENTER = go, x = exit\n"
        "  x            abort a pose move in progress\n"
        "  q            stop and release torque\n"
        "\n"
        "  Holding a key stutters until the terminal's auto-repeat starts,\n"
        "  and there is a short coast after release (see keyboard.py).\n"
        "  On X11, 'xset r rate 200 30' makes holds much smoother."
    )
