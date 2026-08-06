"""Standalone QuadStick smoke test — no arm hardware required.

Lists connected joysticks (use this to confirm the QuadStick enumerates
and to find its axis/button count), then streams live JogCommand
output so you can wiggle the stick / sip / puff / press the lip switch
and read off which pygame axis/button index each one is. Use the
results to override axis_x/axis_y/sip_buttons/puff_buttons/
button_mode_next in config/default.yaml if they differ from the
placeholders in so101_assist/control/inputs/quadstick.py.

If shaped output stays all-zero even while actively moving the
device, pass --raw to bypass QuadStickDevice's deadzone/mapping
entirely and print pygame's raw axis/button/hat values — this tells
you whether the device itself is reporting anything at all.

    python scripts/quadstick_test.py [--index N] [--hz 5] [--raw]
"""
from __future__ import annotations

import argparse
import time

from so101_assist.bus import TOPIC_JOG, Bus
from so101_assist.control.inputs.quadstick import QuadStickDevice


def list_joysticks() -> None:
    import pygame

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("No joysticks detected.")
        return
    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        print(
            f"[{i}] {js.get_name()}  axes={js.get_numaxes()} "
            f"buttons={js.get_numbuttons()} hats={js.get_numhats()}"
        )


def stream_raw(index: int, hz: float) -> None:
    """Print pygame's raw axis/button/hat values, no shaping or mapping."""
    import pygame

    pygame.init()
    pygame.joystick.init()
    js = pygame.joystick.Joystick(index)
    js.init()
    print(f"Raw streaming from joystick {index} at {hz} Hz — Ctrl-C to stop.")
    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(a), 2) for a in range(js.get_numaxes())]
            buttons = [b for b in range(js.get_numbuttons()) if js.get_button(b)]
            hats = [js.get_hat(h) for h in range(js.get_numhats())]
            print(f"axes={axes} buttons_down={buttons} hats={hats}")
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass


def stream(index: int, hz: float) -> None:
    bus = Bus()
    sub = bus.subscribe(TOPIC_JOG)
    device = QuadStickDevice(bus, joystick_index=index, poll_hz=hz)
    device.connect()
    print(f"Streaming from joystick {index} at {hz} Hz — Ctrl-C to stop.")
    try:
        while True:
            import pygame

            pygame.event.pump()
            cmd = device._read_jog(device._joystick)
            bus.publish(TOPIC_JOG, cmd)
            latest = sub.latest() or cmd
            print(
                f"mode={latest.mode.name:12s} axes={latest.axes.round(2)} "
                f"gripper_delta={latest.gripper_delta:+.2f}"
            )
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--hz", type=float, default=5.0, help="print rate")
    parser.add_argument("--list-only", action="store_true", help="list joysticks and exit")
    parser.add_argument(
        "--raw", action="store_true", help="print raw axis/button/hat values, bypassing mapping"
    )
    args = parser.parse_args()

    list_joysticks()
    if args.list_only:
        return
    if args.raw:
        stream_raw(args.index, args.hz)
    else:
        stream(args.index, args.hz)


if __name__ == "__main__":
    main()
