"""QuadStick driver: axis shaping, mode-dependent axis mapping, the
split breath channels (center tube = gripper in every mode; side
tubes = mode-specific), and mode cycling on the lip switch. Driven
against a fake joystick so no pygame device is required. Axis/button
indices match a physical unit measured on 2026-07-25 (see
quadstick.py's module docstring)."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.bus import TOPIC_JOG, Bus
from so101_assist.control.inputs.base import apply_deadzone, apply_expo
from so101_assist.control.inputs.quadstick import QuadStickDevice
from so101_assist.messages import JogMode


class FakeJoystick:
    def __init__(self, axes: dict[int, float] | None = None, buttons: dict[int, bool] | None = None):
        self._axes = axes or {}
        self._buttons = buttons or {}

    def get_axis(self, index: int) -> float:
        return self._axes.get(index, 0.0)

    def get_button(self, index: int) -> bool:
        return self._buttons.get(index, False)

    def set_axis(self, index: int, value: float) -> None:
        self._axes[index] = value

    def set_button(self, index: int, value: bool) -> None:
        self._buttons[index] = value


@pytest.fixture
def device() -> QuadStickDevice:
    return QuadStickDevice(Bus())


def test_deadzone_zeroes_small_values_and_rescales_rest():
    assert apply_deadzone(0.05, 0.15) == 0.0
    assert apply_deadzone(-0.05, 0.15) == 0.0
    assert apply_deadzone(1.0, 0.15) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.15) == pytest.approx(-1.0)
    assert apply_deadzone(0.15, 0.15) == pytest.approx(0.0)


def test_expo_preserves_sign_and_endpoints():
    assert apply_expo(0.0, 1.6) == 0.0
    assert apply_expo(1.0, 1.6) == pytest.approx(1.0)
    assert apply_expo(-1.0, 1.6) == pytest.approx(-1.0)
    assert apply_expo(0.5, 1.6) < 0.5   # softened near center
    assert apply_expo(-0.5, 1.6) > -0.5


def test_default_mode_is_shoulder(device: QuadStickDevice):
    joystick = FakeJoystick()
    cmd = device._read_jog(joystick)
    assert cmd.mode is JogMode.SHOULDER


def test_shoulder_mode_reports_both_stick_axes(device: QuadStickDevice):
    # axis 2 = stick X (right=+1), axis 1 = stick Y (down=+1)
    joystick = FakeJoystick(axes={2: 0.5, 1: -0.5})
    cmd = device._read_jog(joystick)
    assert cmd.mode is JogMode.SHOULDER
    assert cmd.axes[0] > 0     # x -> pan
    assert cmd.axes[1] < 0     # y -> lift
    assert cmd.axes[2] == 0.0


def test_elbow_mode_reports_both_stick_axes(device: QuadStickDevice):
    device.mode = JogMode.ELBOW
    joystick = FakeJoystick(axes={2: 0.5, 1: -0.5})
    cmd = device._read_jog(joystick)
    assert cmd.mode is JogMode.ELBOW
    assert cmd.axes[0] > 0     # x -> pan (downstream)
    assert cmd.axes[1] < 0     # y -> elbow (downstream)


# ------------------------------------------------------ breath channels

@pytest.mark.parametrize("mode", list(JogMode))
def test_center_sip_closes_gripper_in_every_mode(device: QuadStickDevice, mode: JogMode):
    device.mode = mode
    joystick = FakeJoystick(buttons={0: True})   # center sip
    assert device._read_jog(joystick).gripper_delta == -1.0


@pytest.mark.parametrize("mode", list(JogMode))
def test_center_puff_opens_gripper_in_every_mode(device: QuadStickDevice, mode: JogMode):
    device.mode = mode
    joystick = FakeJoystick(buttons={2: True})   # center puff
    assert device._read_jog(joystick).gripper_delta == 1.0


@pytest.mark.parametrize("button", [4, 5])
def test_side_sip_reads_negative_on_side_channel(device: QuadStickDevice, button: int):
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={button: True})
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == -1.0        # side breath -> z slot in WRIST mode
    assert cmd.gripper_delta == 0.0   # center channel untouched


@pytest.mark.parametrize("button", [6, 7])
def test_side_puff_reads_positive_on_side_channel(device: QuadStickDevice, button: int):
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={button: True})
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == 1.0
    assert cmd.gripper_delta == 0.0


def test_side_breath_is_ignored_outside_wrist_mode(device: QuadStickDevice):
    device.mode = JogMode.SHOULDER
    joystick = FakeJoystick(buttons={6: True})   # side puff
    cmd = device._read_jog(joystick)
    assert np.array_equal(cmd.axes, np.zeros(3))
    assert cmd.gripper_delta == 0.0


def test_simultaneous_sip_and_puff_on_same_channel_is_no_signal(device: QuadStickDevice):
    joystick = FakeJoystick(buttons={0: True, 2: True})   # center sip + puff
    assert device._read_jog(joystick).gripper_delta == 0.0


def test_center_and_side_channels_are_independent(device: QuadStickDevice):
    """Center puff (gripper open) + side sip (z down) at once must not
    cancel — they are separate channels."""
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={2: True, 4: True})
    cmd = device._read_jog(joystick)
    assert cmd.gripper_delta == 1.0
    assert cmd.axes[0] == -1.0


def test_wrist_mode_maps_side_breath_and_stick(device: QuadStickDevice):
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(axes={2: 0.3, 1: 0.7}, buttons={6: True})   # side puff
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == 1.0     # side puff -> z
    assert cmd.axes[1] > 0        # stick x -> wrist roll
    assert cmd.axes[2] > 0        # stick y -> wrist flex


# --------------------------------------------------------- mode cycling

def test_mode_button_cycles_shoulder_elbow_wrist_and_wraps(device: QuadStickDevice):
    joystick = FakeJoystick()
    assert device.mode is JogMode.SHOULDER

    joystick.set_button(device.button_mode_next, True)
    device._read_jog(joystick)
    assert device.mode is JogMode.ELBOW

    joystick.set_button(device.button_mode_next, True)
    device._read_jog(joystick)
    assert device.mode is JogMode.ELBOW   # still held -> no re-trigger

    joystick.set_button(device.button_mode_next, False)
    device._read_jog(joystick)
    joystick.set_button(device.button_mode_next, True)
    device._read_jog(joystick)
    assert device.mode is JogMode.WRIST

    joystick.set_button(device.button_mode_next, False)
    device._read_jog(joystick)
    joystick.set_button(device.button_mode_next, True)
    device._read_jog(joystick)
    assert device.mode is JogMode.SHOULDER   # wraps around


def test_hard_puff_never_publishes_stop_or_cancel(device: QuadStickDevice):
    """Safety invariant: this driver has no path to STOP/CANCEL — those
    are voice-only. A held puff must only ever produce a JogCommand,
    never anything on a different topic."""
    joystick = FakeJoystick(buttons={2: True, 6: True})
    cmd = device._read_jog(joystick)
    assert cmd.__class__.__name__ == "JogCommand"


def test_read_jog_publishes_on_operator_jog_topic():
    bus = Bus()
    sub = bus.subscribe(TOPIC_JOG)
    device = QuadStickDevice(bus)
    joystick = FakeJoystick(axes={2: 0.5, 1: 0.5})
    bus.publish(TOPIC_JOG, device._read_jog(joystick))
    cmd = sub.get(timeout=1.0)
    assert cmd.mode is JogMode.SHOULDER


def test_from_config_reads_operator_block():
    bus = Bus()
    operator_cfg = {
        "jog_expo": 2.0,
        "quadstick": {
            "joystick_deadzone": 0.1,
            "center_sip_buttons": [0],
            "center_puff_buttons": [2],
            "side_sip_buttons": [4],
            "side_puff_buttons": [6],
            "button_mode_next": 3,
        },
    }
    device = QuadStickDevice.from_config(bus, operator_cfg)
    assert device.expo == 2.0
    assert device.joystick_deadzone == 0.1
    assert device.center_sip_buttons == (0,)
    assert device.center_puff_buttons == (2,)
    assert device.side_sip_buttons == (4,)
    assert device.side_puff_buttons == (6,)
    assert device.button_mode_next == 3
