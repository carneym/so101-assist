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
  sip = close) so grasping never requires a mode switch; the SIDE
  tubes (left/right, either one) carry the mode-specific breath
  signal (z in WRIST mode).
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
  Center buttons (0/2) form the gripper channel; left+right buttons
  (4/5 and 6/7, either tube counts) form the side channel.
- lip switch (mode-cycle) = button 1.

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

from ...bus import Bus
from ...messages import JogCommand, JogMode
from .base import PygameJoystickDevice, shape_axis

AXIS_X = 2
AXIS_Y = 1
CENTER_SIP_BUTTONS = (0,)       # gripper close
CENTER_PUFF_BUTTONS = (2,)      # gripper open
SIDE_SIP_BUTTONS = (4, 5)       # left, right tube — either counts
SIDE_PUFF_BUTTONS = (6, 7)
BUTTON_MODE_NEXT = 1            # lip switch

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
        self.mode = JogMode.SHOULDER
        self._prev_mode_button = False

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
        )

    def _read_jog(self, joystick) -> JogCommand:
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
