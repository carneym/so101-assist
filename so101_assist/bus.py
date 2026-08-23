"""Minimal in-process pub/sub bus.

Nodes publish and subscribe by topic string. Each subscriber gets its
own bounded queue; slow subscribers drop the oldest message rather
than stalling publishers (fresh perception data beats stale data).

This deliberately mirrors ROS semantics so a later migration is a
mechanical rename, not a redesign.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict
from typing import Any


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, maxsize: int = 2) -> Subscription:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs[topic].append(q)
        return Subscription(q)

    def publish(self, topic: str, msg: Any) -> None:
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()      # drop oldest
                except queue.Empty:
                    pass
                q.put_nowait(msg)


class Subscription:
    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def get(self, timeout: float | None = None) -> Any:
        return self._q.get(timeout=timeout)

    def latest(self) -> Any | None:
        """Drain the queue and return only the newest message.

        Right for state-like topics (frames, jog input) where a stale
        message has no value once a newer one exists.
        """
        msg = None
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                return msg

    def drain(self) -> list[Any]:
        """Every queued message, oldest first.

        Right for EVENT-like topics, where each message is a discrete
        operator action rather than a snapshot of state: collapsing a
        burst with latest() would silently drop, say, a menu step that
        arrived in the same loop tick as the confirm that followed it.
        Subscribe such topics with a maxsize big enough to hold a tick's
        worth of events, since the queue still drops the oldest when full.
        """
        msgs = []
        while True:
            try:
                msgs.append(self._q.get_nowait())
            except queue.Empty:
                return msgs


# Canonical topic names — import these, don't use raw strings.
TOPIC_FRAME_OVERHEAD = "frame.overhead"
TOPIC_FRAME_WRIST = "frame.wrist"
TOPIC_DETECTIONS = "perception.detections"
TOPIC_TARGET = "target.selected"
TOPIC_VOICE = "voice.event"
TOPIC_JOG = "operator.jog"
TOPIC_POSE = "operator.pose"
TOPIC_ARM_STATE = "arm.state"
TOPIC_ARM_CMD = "arm.cmd"
TOPIC_STATE = "system.state"
