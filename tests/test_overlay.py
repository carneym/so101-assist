"""draw_detections: annotates a copy of the frame (same shape, input
untouched) without raising, for any number of detections."""
from __future__ import annotations

import numpy as np

from so101_assist.messages import Detection
from so101_assist.ui.overlay import draw_detections


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
