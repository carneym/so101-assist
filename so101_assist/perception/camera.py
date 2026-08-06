"""Camera capture threads.

One thread per camera, publishing Frame on frame.overhead / frame.wrist.
Handles reconnect on USB hiccups. Camera indices/resolutions come from
config/default.yaml.

cv2 is imported lazily inside _open_capture so importing this module —
and unit-testing the capture loop against a fake capture — never needs
OpenCV or a physical camera. On Windows the DirectShow backend is
used explicitly (default MSMF probing can take tens of seconds per
missing index).
"""
from __future__ import annotations

import logging
import sys
import threading

from ..bus import Bus
from ..messages import Frame

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 1.0


def _open_capture(index: int, width: int, height: int, fps: int):
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class CameraNode:
    """Captures one camera on its own daemon thread, publishing Frame
    on `frame.<name>`. A failed read closes and reopens the device
    (USB hiccup recovery) rather than crashing the thread; subscribers
    simply see a gap in frames, which the bus's drop-oldest semantics
    already tolerate.

    `capture_factory` exists for tests: anything callable returning an
    object with read() -> (ok, image) and release().
    """

    def __init__(
        self,
        bus: Bus,
        name: str,
        index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        capture_factory=_open_capture,
    ) -> None:
        self.bus = bus
        self.name = name
        self.topic = f"frame.{name}"
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._capture_factory = capture_factory
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, bus: Bus, name: str, cam_cfg: dict) -> CameraNode:
        """Build from one entry of the `cameras` block of config/default.yaml."""
        return cls(
            bus,
            name,
            index=cam_cfg["index"],
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
        )

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        cap = None
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    cap = self._capture_factory(self.index, self.width, self.height, self.fps)
                    if cap is None:
                        logger.warning(
                            "camera '%s' (index %d) failed to open — retrying in %.0fs",
                            self.name, self.index, RECONNECT_DELAY_S,
                        )
                        if self._stop_event.wait(RECONNECT_DELAY_S):
                            return
                        continue
                    logger.info("camera '%s' (index %d) opened", self.name, self.index)

                ok, image = cap.read()
                if not ok:
                    logger.warning("camera '%s' read failed — reconnecting", self.name)
                    cap.release()
                    cap = None
                    if self._stop_event.wait(RECONNECT_DELAY_S):
                        return
                    continue

                self.bus.publish(self.topic, Frame(camera=self.name, image=image))
        finally:
            if cap is not None:
                cap.release()
