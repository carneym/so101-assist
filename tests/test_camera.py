"""CameraNode: frames reach the bus on the right topic, and a failed
read triggers a reconnect (new capture) instead of killing the
thread. Driven with a fake capture factory — no OpenCV or camera."""
from __future__ import annotations

import time

import numpy as np

from so101_assist.bus import Bus
from so101_assist.perception.camera import CameraNode


class FakeCapture:
    def __init__(self, fail_after: int | None = None):
        self.reads = 0
        self.fail_after = fail_after
        self.released = False

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_frames_are_published_on_named_topic():
    bus = Bus()
    sub = bus.subscribe("frame.wrist")
    node = CameraNode(bus, "wrist", index=0, capture_factory=lambda *a: FakeCapture())

    node.start()
    try:
        frame = sub.get(timeout=2.0)
    finally:
        node.stop()

    assert frame.camera == "wrist"
    assert frame.image.shape == (4, 4, 3)


def test_read_failure_triggers_reconnect():
    bus = Bus()
    captures: list[FakeCapture] = []

    def factory(*args):
        cap = FakeCapture(fail_after=3)
        captures.append(cap)
        return cap

    # Speed up the reconnect wait for the test
    import so101_assist.perception.camera as camera_mod
    original_delay = camera_mod.RECONNECT_DELAY_S
    camera_mod.RECONNECT_DELAY_S = 0.01

    node = CameraNode(bus, "wrist", index=0, capture_factory=factory)
    node.start()
    try:
        assert _wait_for(lambda: len(captures) >= 2)   # reopened after failure
        assert captures[0].released                     # old capture cleaned up
    finally:
        node.stop()
        camera_mod.RECONNECT_DELAY_S = original_delay


def test_stop_releases_capture():
    bus = Bus()
    cap = FakeCapture()
    node = CameraNode(bus, "wrist", index=0, capture_factory=lambda *a: cap)

    node.start()
    assert _wait_for(lambda: cap.reads > 0)
    node.stop()

    assert cap.released
