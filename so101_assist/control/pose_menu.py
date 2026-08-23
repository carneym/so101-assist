"""Operator-facing state for the named-pose menu.

Pure state: the four discrete actions a QuadStick can produce (open,
scroll, confirm, cancel) plus the move currently authorized. Kept out
of the teleop script so the interaction can be tested without a
joystick, a camera, or an arm — this is the piece that decides when
the robot starts moving on its own, which is exactly the piece that
should be provable in isolation.
"""
from __future__ import annotations

from ..arm.poses import Pose
from ..messages import PoseAction, PoseEvent


class PoseMenu:
    def __init__(self, poses: dict[str, Pose]) -> None:
        self.poses = poses
        self.names = list(poses)
        self.open = False
        self.selected = 0
        self.moving: Pose | None = None

    def handle(self, event: PoseEvent) -> Pose | None:
        """Apply one operator action. Returns a Pose when the operator
        just authorized a move to it, else None."""
        if event.action is PoseAction.ENTER:
            self.open = True
            self.moving = None      # opening the menu abandons any move
        elif event.action is PoseAction.CANCEL:
            self.open = False
            self.moving = None      # doubles as the abort for a running move
        elif event.action is PoseAction.CYCLE and self.names:
            self.selected = (self.selected + event.delta) % len(self.names)
        elif event.action is PoseAction.SELECT:
            self.open = False
            if self.names:
                self.moving = self.poses[self.names[self.selected]]
                return self.moving
        return None

    def menu_lines(self) -> list[str] | None:
        return self.names if self.open else None
