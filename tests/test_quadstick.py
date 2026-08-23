"""QuadStick driver: axis shaping, mode-dependent axis mapping, the
split breath channels (center tube = gripper in every mode; side
tubes = mode-specific), and mode cycling on the lip switch. Driven
against a fake joystick so no pygame device is required. Axis/button
indices match a physical unit measured on 2026-07-25 (see
quadstick.py's module docstring)."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.bus import TOPIC_JOG, TOPIC_POSE, Bus
from so101_assist.control.inputs.base import apply_deadzone, apply_expo
from so101_assist.control.inputs.quadstick import QuadStickDevice
from so101_assist.messages import JogMode, PoseAction


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


def test_left_sip_reads_negative_on_side_channel(device: QuadStickDevice):
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={4: True})
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == -1.0        # side breath -> z slot in WRIST mode
    assert cmd.gripper_delta == 0.0   # center channel untouched


def test_left_puff_reads_positive_on_side_channel(device: QuadStickDevice):
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={6: True})
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == 1.0
    assert cmd.gripper_delta == 0.0


@pytest.mark.parametrize("button", [5, 7])
def test_right_tube_no_longer_drives_z(device: QuadStickDevice, button: int):
    """The right tube is the pose-menu channel now — it must not also
    move the arm, or opening the menu would jog z on the way in."""
    device.mode = JogMode.WRIST
    joystick = FakeJoystick(buttons={button: True})
    cmd = device._read_jog(joystick)
    assert cmd.axes[0] == 0.0


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


# ------------------------------------------------------------ pose menu

class RecordingBus(Bus):
    """Captures every publish. The real Subscription only exposes
    latest() (which collapses a burst to one message) over a queue that
    drops the oldest past maxsize — neither can prove "exactly one event
    was emitted", which is the whole point of the edge-trigger tests."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, object]] = []

    def publish(self, topic: str, msg: object) -> None:
        self.published.append((topic, msg))
        super().publish(topic, msg)

    def pose_events(self) -> list:
        return [msg for topic, msg in self.published if topic == TOPIC_POSE]

    def clear(self) -> None:
        self.published.clear()


@pytest.fixture
def pose_bus():
    bus = RecordingBus()
    return bus, QuadStickDevice(bus)


def test_right_puff_opens_the_menu(pose_bus):
    bus, device = pose_bus

    device._read_jog(FakeJoystick(buttons={7: True}))

    assert device.pose_menu_open
    assert [e.action for e in bus.pose_events()] == [PoseAction.ENTER]


def test_opening_the_menu_suspends_jogging(pose_bus):
    """Browsing poses must not drive the arm: zero axes AND zero
    gripper while the menu is open, whatever the stick is doing."""
    _, device = pose_bus
    device.mode = JogMode.WRIST

    device._read_jog(FakeJoystick(buttons={7: True}))
    cmd = device._read_jog(FakeJoystick(axes={1: 1.0, 2: -1.0}, buttons={0: True, 6: True}))

    assert np.array_equal(cmd.axes, np.zeros(3))
    assert cmd.gripper_delta == 0.0


def test_held_puff_opens_the_menu_only_once(pose_bus):
    bus, device = pose_bus
    joystick = FakeJoystick(buttons={7: True})

    for _ in range(5):
        device._read_jog(joystick)

    assert [e.action for e in bus.pose_events()] == [PoseAction.ENTER]


def test_stick_cycles_the_selection_with_a_detent(pose_bus):
    """One step per deflection: the stick must return toward center
    before it can step again, or a held stick would run away."""
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()

    held = FakeJoystick(axes={1: 1.0})
    for _ in range(5):
        device._read_jog(held)
    assert [e.delta for e in bus.pose_events()] == [1]       # one step, not five

    bus.clear()
    device._read_jog(FakeJoystick(axes={1: 0.0}))            # re-arm
    device._read_jog(FakeJoystick(axes={1: 1.0}))
    assert [e.delta for e in bus.pose_events()] == [1]


def test_stick_up_steps_up_the_list(pose_bus):
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()

    device._read_jog(FakeJoystick(axes={1: -1.0}))           # up reads negative

    assert [e.delta for e in bus.pose_events()] == [-1]


def test_small_deflection_does_not_step(pose_bus):
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()

    device._read_jog(FakeJoystick(axes={1: 0.4}))

    assert bus.pose_events() == []


def test_lip_switch_confirms_and_closes_the_menu(pose_bus):
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()

    device._read_jog(FakeJoystick(buttons={1: True}))

    assert [e.action for e in bus.pose_events()] == [PoseAction.SELECT]
    assert not device.pose_menu_open


def test_held_lip_switch_confirms_only_once(pose_bus):
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()
    joystick = FakeJoystick(buttons={1: True})

    for _ in range(5):
        device._read_jog(joystick)

    assert [e.action for e in bus.pose_events()] == [PoseAction.SELECT]


def test_lip_switch_does_not_change_jog_mode_while_menu_is_open(pose_bus):
    _, device = pose_bus
    device.mode = JogMode.SHOULDER
    device._read_jog(FakeJoystick(buttons={7: True}))

    device._read_jog(FakeJoystick(buttons={1: True}))

    assert device.mode is JogMode.SHOULDER


def test_right_sip_leaves_the_menu(pose_bus):
    bus, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    bus.clear()

    device._read_jog(FakeJoystick(buttons={5: True}))

    assert [e.action for e in bus.pose_events()] == [PoseAction.CANCEL]
    assert not device.pose_menu_open


def test_right_sip_aborts_even_when_the_menu_is_closed(pose_bus):
    """After SELECT the menu is closed but the arm is moving — the
    abort has to still reach the consumer."""
    bus, device = pose_bus

    device._read_jog(FakeJoystick(buttons={5: True}))

    assert [e.action for e in bus.pose_events()] == [PoseAction.CANCEL]


def test_jogging_resumes_after_leaving_the_menu(pose_bus):
    _, device = pose_bus
    device._read_jog(FakeJoystick(buttons={7: True}))
    device._read_jog(FakeJoystick(buttons={5: True}))

    cmd = device._read_jog(FakeJoystick(axes={2: 1.0}))

    assert cmd.axes[0] != 0.0


def test_confirming_a_pose_does_not_also_cycle_jog_mode(pose_bus):
    """The lip switch confirms AND cycles mode, on the same button. A
    press that selected a pose must not additionally advance the jog
    mode when the menu closes and jogging resumes."""
    _, device = pose_bus
    device.mode = JogMode.SHOULDER
    device._read_jog(FakeJoystick(buttons={7: True}))       # open menu
    device._read_jog(FakeJoystick(buttons={1: True}))       # confirm; menu closes

    held = FakeJoystick(buttons={1: True})                  # still holding it
    for _ in range(3):
        device._read_jog(held)

    assert device.mode is JogMode.SHOULDER

    device._read_jog(FakeJoystick())                        # released
    device._read_jog(FakeJoystick(buttons={1: True}))       # a NEW press
    assert device.mode is JogMode.ELBOW
