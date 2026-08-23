"""PoseMenu: the state machine that decides when the arm starts moving
on its own. Motion must follow ONLY an explicit confirm, and cancel
must reach it from every state."""
from __future__ import annotations

import numpy as np
import pytest

from so101_assist.arm.driver import JOINT_NAMES
from so101_assist.arm.poses import Pose
from so101_assist.control.pose_menu import PoseMenu
from so101_assist.messages import PoseAction, PoseEvent

N = len(JOINT_NAMES)


def event(action: PoseAction, delta: int = 0) -> PoseEvent:
    return PoseEvent(action=action, delta=delta)


@pytest.fixture
def menu() -> PoseMenu:
    return PoseMenu({
        "HOME": Pose("HOME", np.zeros(N)),
        "RAISED": Pose("RAISED", np.ones(N)),
        "EXTENDED": Pose("EXTENDED", np.full(N, 2.0)),
    })


def test_starts_closed_and_still(menu: PoseMenu):
    assert not menu.open
    assert menu.moving is None
    assert menu.menu_lines() is None


def test_enter_opens_and_lists_poses(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))

    assert menu.open
    assert menu.menu_lines() == ["HOME", "RAISED", "EXTENDED"]


def test_opening_the_menu_does_not_move_anything(menu: PoseMenu):
    assert menu.handle(event(PoseAction.ENTER)) is None
    assert menu.moving is None


def test_cycle_moves_the_selection_and_wraps(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))

    menu.handle(event(PoseAction.CYCLE, delta=1))
    assert menu.selected == 1
    menu.handle(event(PoseAction.CYCLE, delta=1))
    menu.handle(event(PoseAction.CYCLE, delta=1))
    assert menu.selected == 0          # wrapped forward

    menu.handle(event(PoseAction.CYCLE, delta=-1))
    assert menu.selected == 2          # wrapped backward


def test_cycling_never_starts_a_move(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))

    assert menu.handle(event(PoseAction.CYCLE, delta=1)) is None
    assert menu.moving is None


def test_select_authorizes_the_highlighted_pose(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))
    menu.handle(event(PoseAction.CYCLE, delta=1))

    started = menu.handle(event(PoseAction.SELECT))

    assert started is not None
    assert started.name == "RAISED"
    assert menu.moving is started
    assert not menu.open               # menu closes so the abort targets the MOVE


def test_cancel_closes_the_menu_without_moving(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))

    assert menu.handle(event(PoseAction.CANCEL)) is None
    assert not menu.open
    assert menu.moving is None


def test_cancel_aborts_a_move_in_progress(menu: PoseMenu):
    menu.handle(event(PoseAction.ENTER))
    menu.handle(event(PoseAction.SELECT))
    assert menu.moving is not None

    menu.handle(event(PoseAction.CANCEL))

    assert menu.moving is None


def test_reopening_the_menu_abandons_a_move(menu: PoseMenu):
    """Right puff during a move must not leave the arm driving toward
    the old target while the operator browses for a new one."""
    menu.handle(event(PoseAction.ENTER))
    menu.handle(event(PoseAction.SELECT))

    menu.handle(event(PoseAction.ENTER))

    assert menu.moving is None
    assert menu.open


def test_select_with_no_poses_taught_does_nothing():
    empty = PoseMenu({})
    empty.handle(event(PoseAction.ENTER))
    assert empty.menu_lines() == []          # opens, but with nothing in it

    assert empty.handle(event(PoseAction.SELECT)) is None
    assert empty.moving is None


def test_cycle_with_no_poses_does_not_crash():
    empty = PoseMenu({})
    empty.handle(event(PoseAction.ENTER))

    empty.handle(event(PoseAction.CYCLE, delta=1))

    assert empty.selected == 0
