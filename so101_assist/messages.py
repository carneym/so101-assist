"""Message types passed between nodes over the bus.

Every node communicates exclusively through these dataclasses.
Keeping them in one file makes the system's data flow auditable
and eases a later migration to ROS2 messages if needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


def now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------- perception

@dataclass
class Frame:
    """A single camera frame."""
    camera: str                  # "overhead" | "wrist"
    image: np.ndarray            # HxWx3 BGR
    stamp: float = field(default_factory=now)


@dataclass
class Detection:
    """One detected object in a camera frame."""
    label: str                   # open-vocab label, e.g. "keys"
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None       # HxW bool, from SAM 2
    camera: str = "overhead"
    track_id: int | None = None          # stable id shown to the operator
    stamp: float = field(default_factory=now)


@dataclass
class Target:
    """A grounded, localized target the operator has selected."""
    detection: Detection
    position_robot: np.ndarray           # (3,) xyz in robot base frame, meters
    grasp_pose: np.ndarray | None = None # (4,4) homogeneous, if planner solved one
    confidence: float = 0.0


# --------------------------------------------------------------------- voice

class VoiceCommand(Enum):
    GO = auto()
    STOP = auto()          # hard stop — always honored, highest priority
    GRIP = auto()
    RELEASE = auto()
    CANCEL = auto()
    HOME = auto()
    MODE_NEXT = auto()     # cycle jog mode (translate / height-wrist / gripper)


@dataclass
class VoiceEvent:
    """Either a discrete command or a free-form target description."""
    command: VoiceCommand | None = None
    description: str | None = None       # e.g. "the blue keys on the left"
    raw_text: str = ""
    stamp: float = field(default_factory=now)


# ----------------------------------------------------------------- operator

class JogMode(Enum):
    SHOULDER = auto()      # shoulder pan (left/right) + lift (up/down)
    ELBOW = auto()         # elbow (up/down) + shoulder pan (left/right)
    WRIST = auto()         # z via side sip/puff + wrist flex/roll


@dataclass
class JogCommand:
    """Normalized operator input, device-agnostic.

    All axes are in [-1, 1]; the shared controller scales them to
    velocity limits. Input devices (keyboard/gamepad/quadstick)
    all emit this same message. gripper_delta is mode-independent:
    the gripper channel (center sip/puff on the QuadStick) works in
    EVERY mode, so grasping never requires a mode switch.
    """
    axes: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mode: JogMode = JogMode.SHOULDER
    gripper_delta: float = 0.0
    stamp: float = field(default_factory=now)


# ---------------------------------------------------------------------- arm

@dataclass
class ArmState:
    joint_pos: np.ndarray        # (5,) radians
    joint_load: np.ndarray       # (5,) normalized servo load/current
    gripper_pos: float           # 0 closed .. 1 open
    ee_pose: np.ndarray          # (4,4) FK of end effector in base frame
    stamp: float = field(default_factory=now)


@dataclass
class CartesianVelocity:
    """Command sent to the arm controller each tick.

    `elbow` and `shoulder` are joint-space passthroughs (like `wrist`):
    positive elbow/lift rates move the end effector down/back along
    their arcs at the reference pose. The controller fence-checks each
    one's predicted EE motion since, unlike the wrist, these joints
    swing the whole arm.
    """
    linear: np.ndarray           # (3,) m/s in base frame
    wrist: np.ndarray            # (2,) rad/s flex, roll
    shoulder: np.ndarray = field(default_factory=lambda: np.zeros(2))   # (2,) rad/s pan, lift
    elbow: float = 0.0           # rad/s, direct elbow joint rate
    gripper: float = 0.0         # opening velocity
    stamp: float = field(default_factory=now)
