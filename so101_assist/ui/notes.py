"""Arm state -> operator notes.

Decides WHAT the operator needs told; overlay.py decides how it looks.
Split that way so this policy is testable without OpenCV, and so the
same notes can later drive an audible announcement for an operator who
isn't watching the screen (the QuadStick user's eyes are often on the
object, not the video).

Ordering is by urgency, because the notes field is drawn top-down and
the first line is the one most likely to be read: a tripped guard
first, then a servo pushing hard, then whatever is merely limiting
motion, then steady-state information like a reached pose.
"""
from __future__ import annotations

import numpy as np

from ..arm.safety import joints_over
from .overlay import Note, NoteLevel


def arm_notes(
    *,
    joint_load: np.ndarray | None = None,
    joint_names: list[str] | None = None,
    load_warn_threshold: float | None = None,
    stopped: bool = False,
    fence_blocks: list[str] | None = None,
    limit_clips: list[str] | None = None,
    pose: str | None = None,
    pose_menu: list[str] | None = None,
    pose_selected: int = 0,
    pose_moving: str | None = None,
) -> list[Note]:
    """Build the HUD notes field for one frame.

    Every argument is optional so a caller can supply only what it
    knows — a missing reading produces no note rather than a wrong one.
    """
    notes: list[Note] = []

    if stopped:
        notes.append(Note("STOPPED - load guard tripped", NoteLevel.ALERT))

    if joint_load is not None and joint_names is not None:
        hot = joints_over(joint_load, load_warn_threshold, joint_names)
        if hot:
            peak = float(np.max(np.asarray(joint_load)))
            notes.append(Note(f"FORCE HIGH: {', '.join(hot)} ({peak:.2f})", NoteLevel.ALERT))

    # The arm moving under its own power outranks anything advisory —
    # the operator's first question is "what is it doing and how do I
    # stop it", so the abort is named in the line itself.
    if pose_moving:
        notes.append(Note(f"MOVING TO {pose_moving} - right sip aborts", NoteLevel.ALERT))

    if fence_blocks:
        notes.append(Note(f"fence: blocking {', '.join(fence_blocks)}", NoteLevel.WARN))
    if limit_clips:
        notes.append(Note(f"limit: {', '.join(limit_clips)}", NoteLevel.WARN))

    if pose_menu is not None:
        notes.append(Note("POSE MENU  (lip=go, right sip=exit)", NoteLevel.WARN))
        if not pose_menu:
            notes.append(Note("  (none taught - run teach_pose.py)", NoteLevel.INFO))
        for i, name in enumerate(pose_menu):
            marker = ">" if i == pose_selected else " "
            notes.append(Note(f" {marker} {name}", NoteLevel.INFO))
    elif pose:
        # Only while not browsing — the menu already says where we are.
        notes.append(Note(f"POSE: {pose}", NoteLevel.INFO))

    return notes
