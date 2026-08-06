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

import numpy as np

from ..messages import Detection

BOX_COLOR = (0, 200, 255)
TEXT_COLOR = (0, 200, 255)


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
