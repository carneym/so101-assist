"""arm_notes: turns measured arm state into the HUD's notes field.

The load warning is the safety-relevant one — it must fire before the
guard trips, must name the offending joint, and must never fire on a
reading the operator can do nothing about.
"""
from __future__ import annotations

import numpy as np

from so101_assist.arm.safety import joints_over
from so101_assist.ui.notes import arm_notes
from so101_assist.ui.overlay import NoteLevel

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]


def test_no_state_produces_no_notes():
    assert arm_notes() == []


def test_load_under_threshold_is_silent():
    load = np.array([0.1, 0.2, 0.3, 0.1, 0.0])

    assert arm_notes(joint_load=load, joint_names=JOINTS, load_warn_threshold=0.675) == []


def test_high_load_warns_in_red_and_names_the_joint():
    load = np.array([0.1, 0.82, 0.3, 0.1, 0.0])

    notes = arm_notes(joint_load=load, joint_names=JOINTS, load_warn_threshold=0.675)

    assert len(notes) == 1
    assert notes[0].level is NoteLevel.ALERT       # red
    assert "shoulder_lift" in notes[0].text
    assert "0.82" in notes[0].text                 # peak value shown


def test_load_warning_lists_every_hot_joint():
    load = np.array([0.9, 0.7, 0.1, 0.1, 0.0])

    notes = arm_notes(joint_load=load, joint_names=JOINTS, load_warn_threshold=0.675)

    assert "shoulder_pan" in notes[0].text
    assert "shoulder_lift" in notes[0].text


def test_no_warn_threshold_never_warns():
    """Guard disabled and no explicit warn level -> nothing to compare
    against, so stay quiet rather than invent a threshold."""
    load = np.array([0.99, 0.99, 0.99, 0.99, 0.99])

    assert arm_notes(joint_load=load, joint_names=JOINTS, load_warn_threshold=None) == []


def test_stop_is_reported_first_and_loudest():
    load = np.array([0.9, 0.0, 0.0, 0.0, 0.0])

    notes = arm_notes(
        joint_load=load, joint_names=JOINTS, load_warn_threshold=0.675,
        stopped=True, fence_blocks=["linear"],
    )

    assert notes[0].level is NoteLevel.ALERT
    assert "STOPPED" in notes[0].text
    assert [n.level for n in notes] == [NoteLevel.ALERT, NoteLevel.ALERT, NoteLevel.WARN]


def test_fence_and_limit_are_warnings_not_alerts():
    notes = arm_notes(fence_blocks=["linear"], limit_clips=["elbow"])

    assert [n.level for n in notes] == [NoteLevel.WARN, NoteLevel.WARN]
    assert "linear" in notes[0].text
    assert "elbow" in notes[1].text


def test_pose_is_informational():
    notes = arm_notes(pose="HOME")

    assert notes[0].level is NoteLevel.INFO
    assert "HOME" in notes[0].text


def test_joints_over_ignores_disabled_threshold():
    assert joints_over(np.array([1.0, 1.0]), None, ["a", "b"]) == []
    assert joints_over(np.array([0.8, 0.1]), 0.5, ["a", "b"]) == ["a"]
