# QuadStick teleop — control map & safety settings

Current as of 2026-08-02. This documents the live behavior of
`scripts/quadstick_teleop.py` with `config/default.yaml`. **Update this
file whenever the mode mappings, caps, or safety settings change** —
the source of truth is the code/config, not this page.

New here? Start with the [getting-started tutorial](getting-started.md)
(install → calibrate → drive). This page is the control reference.

Run: `python scripts/quadstick_teleop.py --port COM3 [--debug]`

## Gripper (works in EVERY mode)

The **center sip/puff tube** opens/closes the gripper in every mode —
grasping never needs a mode switch. Puff = open, sip = close. The
**side tubes** (left/right, either counts) carry the mode-specific
breath signal (z in WRIST mode). Sip+puff on the same channel at once
is treated as noise and ignored; the center and side channels are
independent (you can puff-open the gripper while side-sipping for z).

| Control | Effect | Position range | Speed |
|---|---|---|---|
| Center puff | Open gripper | 0 (closed) … 1 (open) | full travel ~1 s |
| Center sip | Close gripper | 0 (closed) … 1 (open) | full travel ~1 s |

## Modes

Lip switch (button 1) cycles: **SHOULDER → ELBOW → WRIST → (wrap)**.
SHOULDER is the default at startup. Every mode change prints a
cheat-sheet (controls, limits, caps) to the console.

| Mode | Control | Effect on arm | Fence-checked | Position limit | Speed cap |
|---|---|---|---|---|---|
| **SHOULDER** | Stick left/right | Shoulder pan — right = arm right | ❌ (operator choice) | ±110° (URDF) | 0.4 rad/s (`max_shoulder_radps`) |
| **SHOULDER** | Stick up/down | Shoulder lift — up = arm up | ❌ (operator choice) | **±150° (config override)** | 0.4 rad/s (`max_shoulder_radps`) |
| **ELBOW** | Stick up/down | Elbow joint — up = arm up / reach out | ✅ | **±135° (config override)** | 0.5 rad/s (`max_elbow_radps`) |
| **ELBOW** | Stick left/right | Shoulder pan — right = arm right | ❌ (operator choice) | ±110° (URDF) | 0.4 rad/s (`max_shoulder_radps`) |
| **WRIST** | Side puff / sip | Up / down (Cartesian z) | ✅ | fence box | 6 cm/s (`max_linear_mps`) |
| **WRIST** | Stick left/right | Wrist roll | ❌ (small motion) | −157°…+163° (URDF) | 0.5 rad/s (`max_wrist_radps`) |
| **WRIST** | Stick up/down | Wrist flex | ❌ (small motion) | ±95° (URDF) | 0.5 rad/s (`max_wrist_radps`) |

**Shoulder pan and lift bypass the workspace fence** (used in SHOULDER
and ELBOW modes) — deliberate operator decision (2026-07-26): the
conservative fence rule blocked legitimate shoulder motion well before
the boundary. Consequence: those channels CAN drive the arm into the
table or beyond the box; protection there is the load monitor
(contact → STOP), the per-write step clamps, and the joint position
limits. Drive attentively near surfaces.

## Global safety settings (all modes, layered independently)

| Setting | Value | What it does | Where |
|---|---|---|---|
| Workspace fence | x −0.15…0.50, y ±0.45, z 0.08…0.65 m | Blocks Cartesian/elbow motion out of the box; z floor = table height. Prints `[fence] blocking: …` | `config: workspace_fence` |
| Joint limit clip | per-joint, see table | Pins joints at their operating range. Prints `[limit] joint at software limit: …` | `config: arm.joint_limits_deg` overrides URDF defaults |
| Load monitor | trip at 0.55 normalized load, 5 consecutive ticks | Sustained overload (collision/jam) → STOP and hold position, torque stays on, no auto-resume | `config: arm.load_stop_threshold` / `load_stop_ticks` |
| Deadman | 0.5 s | No fresh QuadStick input (unplugged, hung) → velocity zeroed | `--stale-s` flag |
| Driver step clamp | 5°/write per joint, 10% gripper | Hard bound on any single servo write, beneath everything above | `arm/driver.py` |
| Setpoint leash | 0.08 rad | Bounds commanded-error build-up when a joint is physically blocked | `arm/controller.py` |
| Control loop | 25 Hz | Teleop tick rate | `scripts/quadstick_teleop.py` |

## Live tuning (on the fly, while driving)

Run the tuning GUI in a second terminal alongside teleop:

```
python scripts/tuning_gui.py
```

Sliders for:

- **Servo speeds** — linear, wrist, elbow, shoulder
- **Lift torque (software)** — setpoint leash + driver step clamp. These
  work together: the effective cap on commanded position error (hence
  servo torque) is the MIN of the two — raise both to lift more.
- **Servo torque limit (firmware)** — arm and gripper caps, % of rated.
  Written to the SRAM `Torque_Limit` register: live, non-persistent
  (resets on power-cycle), no EEPROM wear. Higher = more lifting/holding
  force but more heat and stall-burnout risk; the gripper is kept
  limited (default 50%) so a stalled grasp can't cook it.
- **Load guard** — threshold + ticks, with an enable checkbox (off = no
  collision stop).
- **Joint operating range** — every joint's min/max angle (deg).
- **Workspace fence** — x/y/z min/max (m, base frame). Note pan/lift
  bypass the fence; it binds Cartesian z and the elbow.

Changes write to `config/tuning.json`; the running teleop polls that
file (~0.4 s) and applies new values to the live arm — no restart. The
override persists across runs; **Reset to config baseline** clears it
back to `config/default.yaml`. `config/tuning.json` is machine-specific
and git-ignored.

## Maintenance notes

- **Fence bounds are calibration-specific.** Recalibrating the arm
  (`scripts/calibrate_arm.py`) shifts the zero pose and moves the whole
  FK frame — re-measure the fence with `scripts/fk_live.py` (rest the
  gripper on the table, read `z`) after every recalibration or remount.
- Joint operating ranges beyond the URDF model go in
  `config: arm.joint_limits_deg` (degrees), not in code.
- To flip a control's direction, the sign lives in `jog_to_velocity()`
  in `scripts/quadstick_teleop.py` — one place per control.
