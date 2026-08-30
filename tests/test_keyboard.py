"""Keyboard fallback driver: emits the same messages the QuadStick does,
so modes, the gripper channel and the pose menu behave identically.

Driven against a fake stdin, so no terminal and no key presses are
needed. The held-key model (a terminal reports presses, never releases)
is the part most worth pinning down.
"""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.bus import TOPIC_POSE, Bus
from so101_assist.control.inputs.keyboard import KeyboardDevice
from so101_assist.messages import JogMode, PoseAction


class FakeStdin:
    """Feedable stand-in for a raw terminal."""

    def __init__(self) -> None:
        self.buffer = ""

    def isatty(self) -> bool:
        return True

    def read(self, n: int) -> str:
        chunk, self.buffer = self.buffer[:n], self.buffer[n:]
        return chunk

    def fileno(self) -> int:
        return 0

    def type(self, keys: str) -> None:
        self.buffer += keys


class RecordingBus(Bus):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, object]] = []

    def publish(self, topic: str, msg: object) -> None:
        self.published.append((topic, msg))
        super().publish(topic, msg)

    def pose_events(self) -> list:
        return [m for topic, m in self.published if topic == TOPIC_POSE]

    def clear(self) -> None:
        self.published.clear()


@pytest.fixture
def kit():
    bus = RecordingBus()
    stdin = FakeStdin()
    device = KeyboardDevice(bus, stream=stdin)
    return bus, stdin, device


def press(device, stdin, keys: str):
    """Type keys and read one jog command, as one poll would."""
    stdin.type(keys)
    device._drain_keys()
    return device._read_jog()


# ------------------------------------------------------------------- jog

def test_idle_commands_no_motion(kit):
    _, stdin, device = kit

    cmd = press(device, stdin, "")

    assert np.array_equal(cmd.axes, np.zeros(3))
    assert cmd.gripper_delta == 0.0


def test_left_and_right_drive_the_x_axis(kit):
    _, stdin, device = kit

    assert press(device, stdin, "d").axes[0] == 1.0
    device._held.clear()
    assert press(device, stdin, "a").axes[0] == -1.0


def test_up_reads_negative_to_match_the_quadstick(kit):
    """jog_to_velocity is written against the QuadStick's convention,
    where stick-up is negative. W has to match or it drives the arm
    down."""
    _, stdin, device = kit

    assert press(device, stdin, "w").axes[1] == -1.0
    device._held.clear()
    assert press(device, stdin, "s").axes[1] == 1.0


def test_opposite_keys_cancel(kit):
    _, stdin, device = kit

    cmd = press(device, stdin, "ad")

    assert cmd.axes[0] == 0.0


def test_gripper_works_in_every_mode(kit):
    _, stdin, device = kit

    for mode in (JogMode.SHOULDER, JogMode.ELBOW, JogMode.WRIST):
        device.mode = mode
        device._held.clear()
        assert press(device, stdin, "o").gripper_delta == 1.0
        device._held.clear()
        assert press(device, stdin, "c").gripper_delta == -1.0


def test_side_channel_only_reaches_the_axes_in_wrist_mode(kit):
    _, stdin, device = kit
    device.mode = JogMode.SHOULDER

    assert press(device, stdin, "r").axes[0] == 0.0

    device.mode = JogMode.WRIST
    device._held.clear()
    assert press(device, stdin, "r").axes[0] == 1.0


def test_wrist_mode_maps_side_and_stick(kit):
    _, stdin, device = kit
    device.mode = JogMode.WRIST

    cmd = press(device, stdin, "rdw")

    assert cmd.axes[0] == 1.0        # side -> z
    assert cmd.axes[1] == 1.0        # x -> wrist roll
    assert cmd.axes[2] == -1.0       # y -> wrist flex


# ------------------------------------------------------------- held keys

def test_a_key_stays_held_briefly_after_its_last_repeat(kit):
    """A terminal never reports key-up, so a press has to decay."""
    _, stdin, device = kit
    device.key_hold_s = 10.0

    press(device, stdin, "d")

    assert device._read_jog().axes[0] == 1.0     # still held on the next poll


def test_a_held_key_expires(kit):
    _, stdin, device = kit
    device.key_hold_s = 0.0

    press(device, stdin, "d")

    assert device._read_jog().axes[0] == 0.0


# ----------------------------------------------------------------- modes

def test_m_cycles_the_mode_and_wraps(kit):
    _, stdin, device = kit

    for expected in (JogMode.ELBOW, JogMode.WRIST, JogMode.SHOULDER):
        device._held.clear()
        press(device, stdin, "m")
        assert device.mode is expected


def test_mode_does_not_advance_on_auto_repeat(kit):
    """A leaning finger must not cycle modes continuously."""
    _, stdin, device = kit
    device.key_hold_s = 10.0

    press(device, stdin, "mmmm")

    assert device.mode is JogMode.ELBOW      # one press, not four


# ------------------------------------------------------------- pose menu

def test_p_opens_the_menu_and_suspends_jogging(kit):
    bus, stdin, device = kit

    press(device, stdin, "p")

    assert device.pose_menu_open
    assert [e.action for e in bus.pose_events()] == [PoseAction.ENTER]
    cmd = press(device, stdin, "dw")
    assert np.array_equal(cmd.axes, np.zeros(3))
    assert cmd.gripper_delta == 0.0


def test_menu_cycles_with_w_and_s(kit):
    bus, stdin, device = kit
    press(device, stdin, "p")
    bus.clear()

    device._held.clear()
    press(device, stdin, "s")
    device._held.clear()
    press(device, stdin, "w")

    assert [e.delta for e in bus.pose_events()] == [1, -1]


def test_enter_selects_and_closes_the_menu(kit):
    bus, stdin, device = kit
    press(device, stdin, "p")
    bus.clear()

    press(device, stdin, "\r")

    assert [e.action for e in bus.pose_events()] == [PoseAction.SELECT]
    assert not device.pose_menu_open


def test_x_leaves_the_menu(kit):
    bus, stdin, device = kit
    press(device, stdin, "p")
    bus.clear()

    press(device, stdin, "x")

    assert [e.action for e in bus.pose_events()] == [PoseAction.CANCEL]
    assert not device.pose_menu_open


def test_x_aborts_a_move_even_with_the_menu_closed(kit):
    bus, stdin, device = kit

    press(device, stdin, "x")

    assert [e.action for e in bus.pose_events()] == [PoseAction.CANCEL]


def test_jogging_resumes_after_leaving_the_menu(kit):
    _, stdin, device = kit
    press(device, stdin, "p")
    press(device, stdin, "x")
    device._held.clear()

    assert press(device, stdin, "d").axes[0] == 1.0


# ------------------------------------------------------------------ quit

def test_q_requests_a_stop(kit):
    _, stdin, device = kit

    press(device, stdin, "q")

    assert device.quit_requested


def test_non_tty_is_refused_with_a_clear_message(kit):
    _, stdin, device = kit
    stdin.isatty = lambda: False

    with pytest.raises(RuntimeError, match="TTY"):
        device.connect()
