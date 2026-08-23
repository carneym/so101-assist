"""QuadStick input mapping.

The QuadStick presents as a USB joystick (pygame backend — see
base.py) only when its QuadStick Configurator profile is set to
Joystick/Gamepad mode — NOT Mouse mode. Mouse mode routes movement
through the OS cursor instead of joystick axes (which this driver
can't read), and would fight with the operator's normal computer use
since the QuadStick doubles as their everyday input device: capturing
system mouse deltas for arm control would send jog commands from
ordinary computer use and/or block normal mouse access while this app
runs. Keep "driving the arm" (Joystick profile) and "using the
computer" (Mouse profile) as separate, deliberately-switched profiles
on the device.

Key design constraints:

- few simultaneous channels: joystick (2 axes) + sip/puff breath on
  three tubes (left/center/right) + lip switch. Everything else is
  MODE SWITCHING, announced on the console (via the `mode` field on
  every published JogCommand; the UI/voice layers own the eventual
  audible announcement).
- the CENTER tube is the gripper channel in EVERY mode (puff = open,
  sip = close) so grasping never requires a mode switch.
- the LEFT tube carries the mode-specific breath signal (z in WRIST
  mode). The RIGHT tube is the POSE MENU channel: puff opens the menu,
  sip leaves it (or aborts a move in progress).

  The two side tubes used to be interchangeable for the z channel.
  Splitting them is what buys a pose menu without stealing a control
  the operator already relies on, and it gives the menu a dedicated
  abort that is reachable while the arm is moving under its own power
  — worth more than a redundant second z tube.
- breath is NEVER a safety-critical action since it can be noisy or
  accidental; "stop" stays on voice (this driver never publishes
  anything that halts the arm).
- all thresholds/curves per-user in config (fatigue varies day to day)

Measured against a physical unit (QuadStick Configurator, Joystick
profile) on 2026-07-25:

- axis[1] = stick Y: down=+1, up=-1, centered=0
- axis[2] = stick X: right=+1, left=-1, centered=0
- axis[0], axis[3], axis[4], axis[5]: unused/unverified — not wired

- sip/puff is NOT an analog axis on this profile; each tube+direction
  is a distinct button:
  sip-left=4, sip-center=0, sip-right=5,
  puff-left=6, puff-center=2, puff-right=7.
  Center buttons (0/2) form the gripper channel; LEFT buttons (4/6)
  form the side channel; RIGHT buttons (5/7) drive the pose menu.
- lip switch = button 1: cycles jog mode normally, confirms the
  highlighted pose while the menu is open.

Axis mapping (2 continuous joystick signals + the side-breath signal
-> a 3-wide JogCommand.axes; gripper_delta always carries the
center-breath signal):

- SHOULDER: axes = [joystick_x, joystick_y, 0]
            downstream: x -> shoulder_pan rate, y -> shoulder_lift
- ELBOW:    axes = [joystick_x, joystick_y, 0]
            downstream: x -> shoulder_pan rate, y -> elbow rate
- WRIST:    axes = [side_breath, joystick_x, joystick_y]
            downstream: z velocity, wrist roll, wrist flex

This file should stay thin; per-user tuning lives in config, and the
interaction design will be reshaped by end-user testing (Phase 5).

All of AXIS_X / AXIS_Y / *_SIP_BUTTONS / *_PUFF_BUTTONS /
BUTTON_MODE_NEXT below are confirmed against the physical unit above
— re-verify with scripts/quadstick_test.py --raw if you calibrate a
different unit or QuadStick Configurator profile, and override via
config if they differ.
"""
from __future__ import annotations

import numpy as np

from ...bus import TOPIC_POSE, Bus
from ...messages import JogCommand, JogMode, PoseAction, PoseEvent
from .base import PygameJoystickDevice, shape_axis

AXIS_X = 2
AXIS_Y = 1
CENTER_SIP_BUTTONS = (0,)       # gripper close
CENTER_PUFF_BUTTONS = (2,)      # gripper open
SIDE_SIP_BUTTONS = (4,)         # LEFT tube only — z down (see note below)
SIDE_PUFF_BUTTONS = (6,)        # LEFT tube only — z up
BUTTON_MODE_NEXT = 1            # lip switch
BUTTON_POSE_ENTER = 7           # RIGHT puff — open the pose menu
BUTTON_POSE_CANCEL = 5          # RIGHT sip  — leave the menu / abort a move

# Stick deflection that counts as one menu step, and the value it must
# fall back below before another step registers. A menu needs discrete
# detents, not a rate: without the release threshold a held stick would
# scroll the list continuously and the operator would overshoot.
MENU_STEP_THRESHOLD = 0.6
MENU_RELEASE_THRESHOLD = 0.3

_MODE_ORDER = (JogMode.SHOULDER, JogMode.ELBOW, JogMode.WRIST)


class QuadStickDevice(PygameJoystickDevice):
    def __init__(
        self,
        bus: Bus,
        *,
        joystick_deadzone: float = 0.08,
        expo: float = 1.6,
        joystick_index: int = 0,
        poll_hz: float = 60.0,
        axis_x: int = AXIS_X,
        axis_y: int = AXIS_Y,
        center_sip_buttons: tuple[int, ...] = CENTER_SIP_BUTTONS,
        center_puff_buttons: tuple[int, ...] = CENTER_PUFF_BUTTONS,
        side_sip_buttons: tuple[int, ...] = SIDE_SIP_BUTTONS,
        side_puff_buttons: tuple[int, ...] = SIDE_PUFF_BUTTONS,
        button_mode_next: int = BUTTON_MODE_NEXT,
        button_pose_enter: int = BUTTON_POSE_ENTER,
        button_pose_cancel: int = BUTTON_POSE_CANCEL,
    ) -> None:
        super().__init__(bus, joystick_index=joystick_index, poll_hz=poll_hz)
        self.joystick_deadzone = joystick_deadzone
        self.expo = expo
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.center_sip_buttons = center_sip_buttons
        self.center_puff_buttons = center_puff_buttons
        self.side_sip_buttons = side_sip_buttons
        self.side_puff_buttons = side_puff_buttons
        self.button_mode_next = button_mode_next
        self.button_pose_enter = button_pose_enter
        self.button_pose_cancel = button_pose_cancel
        self.mode = JogMode.SHOULDER
        self._prev_mode_button = False
        # Pose-menu state. `pose_menu_open` gates jogging: while the
        # menu is up the stick scrolls the list instead of driving the
        # arm, so this driver publishes ZERO axes and zero gripper —
        # the operator can never move the arm by browsing poses.
        self.pose_menu_open = False
        self._prev_pose_enter = False
        self._prev_pose_cancel = False
        self._menu_detent_armed = True

    @classmethod
    def from_config(cls, bus: Bus, operator_cfg: dict) -> QuadStickDevice:
        """Build from the `operator` block of config/default.yaml."""
        qs_cfg = operator_cfg.get("quadstick", {})
        return cls(
            bus,
            joystick_deadzone=qs_cfg.get("joystick_deadzone", 0.08),
            expo=operator_cfg.get("jog_expo", 1.6),
            axis_x=qs_cfg.get("axis_x", AXIS_X),
            axis_y=qs_cfg.get("axis_y", AXIS_Y),
            center_sip_buttons=tuple(qs_cfg.get("center_sip_buttons", CENTER_SIP_BUTTONS)),
            center_puff_buttons=tuple(qs_cfg.get("center_puff_buttons", CENTER_PUFF_BUTTONS)),
            side_sip_buttons=tuple(qs_cfg.get("side_sip_buttons", SIDE_SIP_BUTTONS)),
            side_puff_buttons=tuple(qs_cfg.get("side_puff_buttons", SIDE_PUFF_BUTTONS)),
            button_mode_next=qs_cfg.get("button_mode_next", BUTTON_MODE_NEXT),
            button_pose_enter=qs_cfg.get("button_pose_enter", BUTTON_POSE_ENTER),
            button_pose_cancel=qs_cfg.get("button_pose_cancel", BUTTON_POSE_CANCEL),
        )

    def _read_jog(self, joystick) -> JogCommand:
        # Menu handling first: it decides whether this tick jogs at all.
        # Publishing pose events from here (rather than from a second
        # poll loop) keeps them edge-aligned with the jog they suppress,
        # so there is no tick where the menu is open AND the stick still
        # commands motion.
        self._poll_pose_menu(joystick)

        if self.pose_menu_open:
            # Browsing is not driving: no axes, no gripper. The mode is
            # still reported so the HUD keeps showing where jogging will
            # resume when the menu closes.
            return JogCommand(axes=np.zeros(3), mode=self.mode, gripper_delta=0.0)

        x = shape_axis(joystick.get_axis(self.axis_x), self.joystick_deadzone, self.expo)
        y = shape_axis(joystick.get_axis(self.axis_y), self.joystick_deadzone, self.expo)
        center = self._breath(joystick, self.center_sip_buttons, self.center_puff_buttons)
        side = self._breath(joystick, self.side_sip_buttons, self.side_puff_buttons)

        self._maybe_cycle_mode(joystick)

        axes = np.zeros(3)
        if self.mode in (JogMode.SHOULDER, JogMode.ELBOW):
            axes[0], axes[1] = x, y
        elif self.mode is JogMode.WRIST:
            axes[0], axes[1], axes[2] = side, x, y

        # Center breath -> gripper in every mode (see module docstring).
        return JogCommand(axes=axes, mode=self.mode, gripper_delta=center)

    def _poll_pose_menu(self, joystick) -> None:
        """Right puff opens the menu, right sip leaves it, the stick
        scrolls it, the lip switch confirms. All edge-triggered — a held
        breath or a held stick must not repeat."""
        enter = bool(joystick.get_button(self.button_pose_enter))
        cancel = bool(joystick.get_button(self.button_pose_cancel))

        # CANCEL is published even when the menu is closed: that is the
        # operator's abort for a move already running, which by then has
        # closed the menu.
        if cancel and not self._prev_pose_cancel:
            self.pose_menu_open = False
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.CANCEL))
        elif enter and not self._prev_pose_enter and not self.pose_menu_open:
            self.pose_menu_open = True
            self._menu_detent_armed = True
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.ENTER))

        self._prev_pose_enter = enter
        self._prev_pose_cancel = cancel

        if not self.pose_menu_open:
            return

        # Raw axis (not shaped): the expo curve is for velocity feel and
        # has no meaning for a discrete menu step.
        y = joystick.get_axis(self.axis_y)
        if abs(y) < MENU_RELEASE_THRESHOLD:
            self._menu_detent_armed = True
        elif self._menu_detent_armed and abs(y) >= MENU_STEP_THRESHOLD:
            self._menu_detent_armed = False
            # Stick up reads negative on this unit; up moves up the list.
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.CYCLE, delta=1 if y > 0 else -1))

        pressed = bool(joystick.get_button(self.button_mode_next))
        if pressed and not self._prev_mode_button:
            # Lip switch confirms instead of cycling jog mode while the
            # menu is open. Closing the menu here means the abort that
            # follows (right sip) applies to the MOVE, not the menu.
            self.pose_menu_open = False
            self.bus.publish(TOPIC_POSE, PoseEvent(action=PoseAction.SELECT))
        self._prev_mode_button = pressed

    @staticmethod
    def _breath(joystick, sip_buttons: tuple[int, ...], puff_buttons: tuple[int, ...]) -> float:
        """-1.0 on sip, +1.0 on puff, 0.0 if neither — or both, since
        sip+puff on the same channel simultaneously is noise, not a
        valid breath state, and must never be trusted."""
        sip = any(joystick.get_button(b) for b in sip_buttons)
        puff = any(joystick.get_button(b) for b in puff_buttons)
        if sip and puff:
            return 0.0
        if puff:
            return 1.0
        if sip:
            return -1.0
        return 0.0

    def _maybe_cycle_mode(self, joystick) -> None:
        pressed = bool(joystick.get_button(self.button_mode_next))
        if pressed and not self._prev_mode_button:
            self.mode = _MODE_ORDER[(_MODE_ORDER.index(self.mode) + 1) % len(_MODE_ORDER)]
        self._prev_mode_button = pressed
