"""Camera smoke test — probe indices, then view a live feed.

With --list-only, probes camera indices 0..--max-index and reports
which respond and at what resolution (use this to find which index is
the wrist cam vs. a built-in webcam, then fix config/default.yaml's
`cameras` block to match). Otherwise opens the chosen camera through
CameraNode -> bus -> subscriber, i.e. the same path the real system
uses, and displays it with the measured publish rate overlaid.
Press q in the window (or Ctrl-C) to quit.

    python scripts/camera_test.py --list-only
    python scripts/camera_test.py --index 2
    python scripts/camera_test.py --name wrist        # index/size from config
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from so101_assist.bus import Bus
from so101_assist.perception.camera import CameraNode, _open_capture


def probe(max_index: int) -> None:
    for i in range(max_index + 1):
        cap = _open_capture(i, 640, 480, 30)
        if cap is None:
            print(f"[{i}] no camera")
            continue
        ok, image = cap.read()
        if ok:
            h, w = image.shape[:2]
            print(f"[{i}] OK  {w}x{h}")
        else:
            print(f"[{i}] opens but read failed")
        cap.release()


def view(bus: Bus, node: CameraNode) -> None:
    sub = bus.subscribe(node.topic)
    node.start()
    print(f"Viewing '{node.name}' (index {node.index}) — press q in the window to quit.")
    frames = 0
    fps = 0.0
    window = f"so101-assist {node.name}"
    t0 = time.monotonic()
    try:
        while True:
            frame = sub.latest()
            if frame is not None:
                frames += 1
                elapsed = time.monotonic() - t0
                if elapsed >= 1.0:
                    fps = frames / elapsed
                    frames = 0
                    t0 = time.monotonic()
                image = frame.image.copy()
                cv2.putText(
                    image, f"{node.name}  {fps:.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow(window, image)
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true", help="probe indices and exit")
    parser.add_argument("--max-index", type=int, default=4, help="highest index to probe")
    parser.add_argument("--index", type=int, help="camera index to view")
    parser.add_argument("--name", help="camera name from config (overhead / wrist)")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()

    if args.list_only:
        probe(args.max_index)
        return

    bus = Bus()
    if args.name:
        cfg = yaml.safe_load(args.config.read_text())
        node = CameraNode.from_config(bus, args.name, cfg["cameras"][args.name])
    elif args.index is not None:
        node = CameraNode(bus, f"index{args.index}", index=args.index)
    else:
        parser.error("pass --list-only, --index, or --name")
    view(bus, node)


if __name__ == "__main__":
    main()
