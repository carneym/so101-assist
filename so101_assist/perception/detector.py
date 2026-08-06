"""Open-vocabulary object detection.

Default backend: YOLO-World (real-time on modest GPU / decent CPU).
Alternative: OWLv2 for better arbitrary-text grounding, slower — not
yet implemented (raises NotImplementedError if selected in config).

Two query modes:
- ambient: a standing prompt list ("keys, phone, cup, remote, ...")
  run continuously on the overhead feed for the annotated overlay
- directed: a free-form phrase from voice ("the blue keys on the left")
  run once against the current frame for grounding

This first pass only implements ambient detection (fixed prompt list,
score-thresholded boxes). Detections don't get track_ids yet — the
IoU tracker that assigns them (so the operator can say "object two"
and mean the same object across frames) lands later; `mask` (SAM 2
refinement) is also not populated here.

`ultralytics` is imported lazily inside `_load_yolo_world`, mirroring
`perception/camera.py`'s lazy `cv2` import, so importing this module —
and unit-testing detection against a fake model — never needs
`ultralytics`/`torch` installed.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..bus import TOPIC_DETECTIONS, Bus
from ..messages import Detection

DEFAULT_DETECT_HZ = 5.0
YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"


def _load_yolo_world():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Object detection requires the 'perception' extra — "
            'pip install -e ".[perception]"'
        ) from exc
    return YOLO(YOLO_WORLD_WEIGHTS)


class ObjectDetector:
    """Runs YOLO-World against a fixed ambient prompt list.

    `model_factory` exists for tests: anything callable (no args)
    returning an object with `set_classes(prompts)` and
    `predict(image, conf=..., verbose=...) -> results`, where each
    result has `.boxes` with `.xyxy`, `.conf`, and `.cls` arrays and
    a `.names` dict mapping class index -> label (this is the
    ultralytics YOLO results-object shape).
    """

    def __init__(
        self,
        prompts: list[str],
        score_threshold: float = 0.30,
        model_factory=_load_yolo_world,
    ) -> None:
        self.prompts = list(prompts)
        self.score_threshold = score_threshold
        self._model_factory = model_factory
        self._model = None

    def _model_instance(self):
        if self._model is None:
            self._model = self._model_factory()
            self._model.set_classes(self.prompts)
        return self._model

    def detect(self, image: np.ndarray, camera: str = "wrist") -> list[Detection]:
        model = self._model_instance()
        results = model.predict(image, conf=self.score_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            names = result.names
            for box_xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                score = float(conf)
                if score < self.score_threshold:
                    continue
                label = names[int(cls)]
                detections.append(
                    Detection(
                        label=label,
                        score=score,
                        box_xyxy=tuple(float(v) for v in box_xyxy),
                        camera=camera,
                    )
                )
        return detections


class DetectorNode:
    """Runs an ObjectDetector against the newest frame on `frame_topic`
    at a modest fixed rate, publishing the full detection list for
    each pass as one message on `TOPIC_DETECTIONS`.

    Deliberately decoupled from the control loop's rate — inference
    latency (tens to low-hundreds of ms) must never affect driving —
    so this runs on its own daemon thread at `detect_hz`, well below
    camera framerate, rather than processing every frame.
    """

    def __init__(
        self,
        bus: Bus,
        frame_topic: str,
        detector: ObjectDetector,
        camera: str = "wrist",
        detect_hz: float = DEFAULT_DETECT_HZ,
    ) -> None:
        self.bus = bus
        self.frame_topic = frame_topic
        self.topic = TOPIC_DETECTIONS
        self.detector = detector
        self.camera = camera
        self.detect_hz = detect_hz
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, bus: Bus, frame_topic: str, perception_cfg: dict, camera: str = "wrist") -> DetectorNode:
        """Build from the `perception` block of config/default.yaml."""
        backend = perception_cfg.get("detector", "yolo_world")
        if backend != "yolo_world":
            raise NotImplementedError(f"detector backend '{backend}' not implemented")
        detector = ObjectDetector(
            prompts=perception_cfg["ambient_prompts"],
            score_threshold=perception_cfg.get("score_threshold", 0.30),
        )
        return cls(
            bus,
            frame_topic,
            detector,
            camera=camera,
            detect_hz=perception_cfg.get("detect_hz", DEFAULT_DETECT_HZ),
        )

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="detector")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        sub = self.bus.subscribe(self.frame_topic, maxsize=1)
        period = 1.0 / self.detect_hz
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            frame = sub.latest()
            if frame is not None:
                try:
                    detections = self.detector.detect(frame.image, camera=self.camera)
                except RuntimeError as exc:
                    # Model load failed (e.g. missing 'perception' extra) —
                    # permanent, not a transient hiccup like a camera drop,
                    # so fail once cleanly instead of retracing every tick.
                    print(f"[detect] {exc}")
                    return
                self.bus.publish(self.topic, detections)
            elapsed = time.monotonic() - tick_start
            if self._stop_event.wait(max(0.0, period - elapsed)):
                return
