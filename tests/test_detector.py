"""ObjectDetector: score filtering and Detection field mapping.
DetectorNode: detections reach the bus on TOPIC_DETECTIONS, and
start()/stop() behave like CameraNode's (thread joins cleanly).
Driven with a fake YOLO-World model — no ultralytics or torch."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from so101_assist.bus import TOPIC_DETECTIONS, Bus
from so101_assist.messages import Frame
from so101_assist.perception.detector import DetectorNode, ObjectDetector


class FakeModel:
    """Stands in for an ultralytics YOLO-World model: records
    set_classes() calls and returns a canned result on predict()."""

    def __init__(self, boxes):
        self.boxes = boxes   # list of (xyxy, conf, cls)
        self.classes_set: list[str] | None = None

    def set_classes(self, prompts):
        self.classes_set = list(prompts)

    def predict(self, image, conf=0.0, verbose=False):
        xyxy = np.array([b[0] for b in self.boxes], dtype=float)
        confs = np.array([b[1] for b in self.boxes], dtype=float)
        classes = np.array([b[2] for b in self.boxes], dtype=float)
        names = {0: "keys", 1: "cup"}
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=xyxy, conf=confs, cls=classes),
            names=names,
        )
        return [result]


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_detect_filters_by_score_threshold_and_maps_fields():
    model = FakeModel(boxes=[
        ((10, 20, 30, 40), 0.9, 0),   # keys, above threshold
        ((1, 2, 3, 4), 0.1, 1),       # cup, below threshold
    ])
    detector = ObjectDetector(
        prompts=["keys", "cup"], score_threshold=0.5, model_factory=lambda: model
    )

    detections = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), camera="wrist")

    assert model.classes_set == ["keys", "cup"]
    assert len(detections) == 1
    det = detections[0]
    assert det.label == "keys"
    assert det.score == 0.9
    assert det.box_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert det.camera == "wrist"


def test_detector_node_publishes_detections_on_bus():
    model = FakeModel(boxes=[((0, 0, 1, 1), 0.8, 0)])
    detector = ObjectDetector(prompts=["keys"], score_threshold=0.3, model_factory=lambda: model)

    bus = Bus()
    sub = bus.subscribe(TOPIC_DETECTIONS)
    node = DetectorNode(bus, "frame.wrist", detector, camera="wrist", detect_hz=50.0)

    node.start()
    try:
        bus.publish("frame.wrist", Frame(camera="wrist", image=np.zeros((4, 4, 3), dtype=np.uint8)))
        detections = sub.get(timeout=2.0)
    finally:
        node.stop()

    assert len(detections) == 1
    assert detections[0].label == "keys"


def test_detector_node_exits_cleanly_on_model_load_failure():
    """A RuntimeError from the model factory (e.g. missing 'perception'
    extra) must fail the detector thread once, not retry forever."""

    def broken_factory():
        raise RuntimeError("Object detection requires the 'perception' extra")

    detector = ObjectDetector(prompts=["keys"], model_factory=broken_factory)
    bus = Bus()
    node = DetectorNode(bus, "frame.wrist", detector, detect_hz=50.0)

    node.start()
    bus.publish("frame.wrist", Frame(camera="wrist", image=np.zeros((4, 4, 3), dtype=np.uint8)))
    try:
        assert _wait_for(lambda: node._thread is not None and not node._thread.is_alive())
    finally:
        node.stop()


def test_detector_node_stop_joins_thread():
    model = FakeModel(boxes=[])
    detector = ObjectDetector(prompts=["keys"], model_factory=lambda: model)
    bus = Bus()
    node = DetectorNode(bus, "frame.wrist", detector, detect_hz=50.0)

    node.start()
    assert _wait_for(lambda: node._thread is not None and node._thread.is_alive())
    node.stop()

    assert node._thread is None
