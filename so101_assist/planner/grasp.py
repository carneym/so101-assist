"""Grasp pose selection under 5DOF constraints.

Given a Target (3D position + mask), choose a grasp:
- approach: top-down for flat objects (keys, phone), front for tall
- gripper roll aligned to the mask's minor axis
- verify with ik_position(); if unreachable, try alternate approach,
  then report failure to the UI ("can't reach that from here")
"""
