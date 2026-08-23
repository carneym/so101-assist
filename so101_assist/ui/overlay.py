"""Render annotations onto camera frames:
- numbered, labeled detection boxes/masks (matching voice "object N")
- selected target highlight + planned path projection
- BIG active-mode banner (ELBOW / HEIGHT+WRIST / GRIPPER / SHOULDER)
  and state banner — the operator must never wonder what a joystick
  deflection will do.

Only box drawing is implemented so far. Numbering is a plain
enumerate index — it becomes the stable IoU-tracker track_id once
that tracker exists (so "object two" keeps meaning the same object
across frames); target highlighting, path projection, and the mode/
state banners are built when their upstream pieces (target selection,
planner, state machine wiring) land.

cv2 is imported lazily so importing this module never needs OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..messages import Detection

BOX_COLOR = (0, 200, 255)
TEXT_COLOR = (0, 200, 255)

# HUD: fixed status lines at top-left, then a notes field beneath them.
HUD_COLOR = (0, 255, 0)
HUD_ORIGIN = (10, 30)
LINE_HEIGHT = 30
FONT_SCALE = 0.8
FONT_THICKNESS = 2


class NoteLevel(Enum):
    """How loud a note is. Colors are chosen so severity reads at a
    glance without having to parse the text."""
    INFO = auto()    # green — same weight as the status lines
    WARN = auto()    # amber — something is limiting motion
    ALERT = auto()   # red — the operator should act now


NOTE_COLORS = {
    NoteLevel.INFO: (0, 255, 0),
    NoteLevel.WARN: (0, 190, 255),
    NoteLevel.ALERT: (0, 0, 255),
}


@dataclass
class Note:
    """One line in the HUD's notes field.

    Kept deliberately dumb — a string plus a severity. Deciding WHAT
    deserves a note (high servo load, a tripped guard, a reached pose)
    belongs to the caller, so this module stays pure presentation.
    """
    text: str
    level: NoteLevel = NoteLevel.INFO


def draw_hud(
    image: np.ndarray,
    lines: list[str],
    notes: list[Note] | None = None,
) -> np.ndarray:
    """Return a copy of `image` with `lines` drawn top-left, then any
    `notes` immediately beneath them in their severity color.

    The notes field is the operator's channel for anything that isn't
    steady-state status: a servo pushing hard, a guard that tripped, a
    named pose the arm has reached. It shares the status lines' left
    margin and line spacing so it reads as one block.
    """
    import cv2

    annotated = image.copy()
    x, y = HUD_ORIGIN
    for line in lines:
        cv2.putText(
            annotated, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, HUD_COLOR, FONT_THICKNESS, cv2.LINE_AA,
        )
        y += LINE_HEIGHT
    for note in notes or []:
        cv2.putText(
            annotated, note.text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NOTE_COLORS[note.level],
            FONT_THICKNESS, cv2.LINE_AA,
        )
        y += LINE_HEIGHT
    return annotated


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of `image` with a numbered box + `label score`
    caption per detection."""
    import cv2

    annotated = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = (round(v) for v in det.box_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        caption = f"{i}: {det.label} {det.score:.2f}"
        text_y = max(0, y1 - 8)
        cv2.putText(
            annotated, caption, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return annotated
