"""Approach path generation.

Straight-line Cartesian path: current pose -> waypoint above workspace
-> pre-grasp (offset PREGRASP_OFFSET_M back along approach axis).
The arm STOPS at pre-grasp and hands over to FINE mode; the final
descent is operator-triggered and visually servoed.
"""
PREGRASP_OFFSET_M = 0.07
