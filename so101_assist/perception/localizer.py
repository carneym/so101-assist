"""Pixel -> 3D point in robot base frame.

Phase 2 backend: table-plane assumption. The overhead camera is
calibrated to the table plane (calibration.py); a pixel maps to 3D by
ray-plane intersection. Valid ONLY for objects resting on the table.

Later backend: depth camera (RealSense/OAK-D) — same interface,
swap via config.
"""
from __future__ import annotations

import numpy as np


class Localizer:
    def pixel_to_robot(self, uv: np.ndarray, camera: str = "overhead") -> np.ndarray | None:
        """(u, v) -> (x, y, z) in base frame, or None if not localizable."""
        raise NotImplementedError
