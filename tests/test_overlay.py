"""draw_detections / draw_hud: annotate a copy of the frame (same
shape, input untouched) without raising, for any number of detections
or notes."""
from __future__ import annotations

import numpy as np

from so101_assist.messages import Detection
from so101_assist.ui.overlay import (
    NOTE_COLORS,
    Note,
    NoteLevel,
    draw_detections,
    draw_hud,
)


def test_draw_detections_returns_same_shape_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(label="keys", score=0.91, box_xyxy=(10, 10, 40, 40), camera="wrist"),
        Detection(label="cup", score=0.55, box_xyxy=(50, 20, 90, 70), camera="wrist"),
    ]

    annotated = draw_detections(image, detections)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)   # boxes were actually drawn
    assert np.array_equal(image, np.zeros((100, 100, 3), dtype=np.uint8))   # input untouched


def test_draw_detections_handles_empty_list():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    annotated = draw_detections(image, [])

    assert annotated.shape == image.shape


def test_draw_hud_writes_lines_without_touching_the_input():
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    annotated = draw_hud(image, ["MODE: WRIST", "xyz: +0.100 +0.000 +0.200"])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    assert np.array_equal(image, np.zeros((200, 400, 3), dtype=np.uint8))


def test_draw_hud_with_no_lines_or_notes_is_a_noop_copy():
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    annotated = draw_hud(image, [], [])

    assert np.array_equal(annotated, image)


def test_alert_note_is_drawn_in_red():
    """The force warning must be unmistakably red — BGR (0,0,255)."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    annotated = draw_hud(image, [], [Note("FORCE HIGH: elbow (0.82)", NoteLevel.ALERT)])

    painted = annotated[np.any(annotated != 0, axis=-1)]
    assert len(painted) > 0
    # Antialiasing varies intensity, so assert the HUE: blue and green
    # stay at zero and only the red channel is ever painted.
    blue, green, red = painted[:, 0], painted[:, 1], painted[:, 2]
    assert not blue.any() and not green.any()
    assert red.max() == NOTE_COLORS[NoteLevel.ALERT][2] == 255


def test_notes_are_drawn_below_the_status_lines():
    """Notes share the status lines' column but start underneath them,
    so the field reads as one block and never overwrites the mode."""
    image = np.zeros((300, 500, 3), dtype=np.uint8)

    lines_only = draw_hud(image, ["MODE: WRIST"])
    with_note = draw_hud(image, ["MODE: WRIST"], [Note("FORCE HIGH", NoteLevel.ALERT)])

    def lowest_painted_row(img):
        rows = np.where(np.any(img != 0, axis=(1, 2)))[0]
        return rows.max()

    assert lowest_painted_row(with_note) > lowest_painted_row(lines_only)
    # the status line itself is untouched by the note
    top = lowest_painted_row(lines_only) + 1
    assert np.array_equal(with_note[:top], lines_only[:top])
