"""Gamepad jog via evdev/pygame. Sticks -> axes, triggers -> gripper,
face button -> mode cycle. Deadzone + expo curve applied here so
JogCommand semantics stay device-independent."""
