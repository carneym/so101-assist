"""Shared controller: blends autonomy with operator jog.

Modes:
- AUTON (APPROACH state): planner velocity only; any jog input scales
  it down (operator hesitation = arm slows), STOP zeroes it
- MANUAL (FINE state, default): jog input only, mapped per JogMode
- BLEND (FINE state, optional, Phase 5): manual jog plus a weak
  attraction field toward the nearest feasible grasp pose;
  gain ramps with operator inactivity, never exceeds jog authority

Output is a single CartesianVelocity on arm.cmd — safety caps are
applied downstream in controller.py, not here.
"""
