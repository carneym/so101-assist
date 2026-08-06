"""AprilTag-based calibration.

Provides:
- camera intrinsics (checkerboard or tag board)
- overhead-camera -> table-plane homography / extrinsics
- table -> robot-base transform (tag at a known base location, or
  touch-point registration: jog gripper tip to 3+ tags, record joints)
- wrist-camera -> gripper extrinsics (tag on table viewed from wrist)

Results are serialized to config/calibration/*.npz. Hobby servos
drift; scripts/calibrate_hand_eye.py should be re-run periodically
and takes <5 minutes by design.
"""
