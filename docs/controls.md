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
**LEFT tube** carries the mode-specific breath signal (z in WRIST
mode). The **RIGHT tube** drives the [pose menu](#named-poses) — puff
opens it, sip exits it or aborts a pose move. Sip+puff on the same
channel at once is treated as noise and ignored; the channels are
independent (you can puff-open the gripper while left-sipping for z).

> **Changed 2026-08-23:** the two side tubes used to be interchangeable
> for z. The right tube is now the pose channel, so z is LEFT-tube only.

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
| **WRIST** | LEFT puff / sip | Up / down (Cartesian z) | ✅ | fence box | 6 cm/s (`max_linear_mps`) |
| **WRIST** | Stick left/right | Wrist roll | ❌ (small motion) | −157°…+163° (URDF) | 0.5 rad/s (`max_wrist_radps`) |
| **WRIST** | Stick up/down | Wrist flex | ❌ (small motion) | ±95° (URDF) | 0.5 rad/s (`max_wrist_radps`) |

## Keyboard controls (no QuadStick)

Teleop falls back to the keyboard automatically when no joystick is
plugged in; force it either way with `--input keyboard` / `--input
quadstick`. Keys are read from the **terminal running teleop**, which
must stay focused — it works headless over SSH and with `--no-camera`.

| Key | Effect | QuadStick equivalent |
|---|---|---|
| `w` / `s` | up / down (mode-dependent) | stick up/down |
| `a` / `d` | left / right (shoulder pan) | stick left/right |
| `r` / `f` | z up / down (WRIST mode) | left tube puff / sip |
| `o` / `c` | gripper open / close, every mode | centre puff / sip |
| `m` | next mode: SHOULDER → ELBOW → WRIST | lip switch |
| `p` | open the pose menu | right puff |
| `w` / `s`, `ENTER` | choose / go, in the menu | stick, lip switch |
| `x` | exit the menu, or abort a pose move | right sip |
| `q` | stop and release torque | — |

> **Jogging feels different here, unavoidably.** A terminal reports key
> presses but never releases, so a key counts as held until
> `key_hold_s` (0.25 s) passes without a repeat. Holding therefore
> stutters until the OS auto-repeat starts, and there is up to 0.25 s of
> coast after you let go — about 6° at the shoulder cap. Tap for small
> corrections. On X11, `xset r rate 200 30` makes holds much smoother.

This is a bring-up and fallback input. The QuadStick remains the
interface the system is designed around.

## Named poses

Taught with `scripts/teach_pose.py` (arm limp, positioned by hand) and
stored in `config/poses.json`. The standard set is **HOME** (tucked,
gripper closed), **RAISED**, and **EXTENDED**.

| Control | Effect |
|---|---|
| **Right puff** | Open the pose menu (jogging suspends — the stick stops driving the arm) |
| **Stick up/down** | Move the highlight through the list, one step per deflection |
| **Lip switch** | Go to the highlighted pose — this is what authorizes motion |
| **Right sip** | Exit the menu, or **abort a move already running** |

The menu appears both in the video overlay and the console. While a
move runs the HUD shows `MOVING TO <name> - right sip aborts` in red,
and the stick is ignored until it finishes, is aborted, or trips the
load guard.

Pose moves are slower than jogging on purpose (`arm.pose_speed_radps`,
default 0.4 rad/s) — the operator isn't steering, so it has to be slow
enough to watch and stop. They pass the load monitor and joint limits,
but **not** the workspace fence: a pose is a joint-space target that
was taught by physically placing the arm there, so teach poses inside
the reachable workspace and clear of obstacles.

When the arm is within 10° (all joints) of a taught pose, the HUD shows
`POSE: <name>`. **Re-teach poses after any recalibration** — calibration
shifts the zero pose, so old angles point somewhere else.

### Recorded paths and gestures

`teach_pose.py --record --name X` records the whole PATH you move the
arm through, not just where it ends. Playback follows that path, so the
arm goes the way you showed it — **around** whatever you went around.
The floor guard only knows about a horizontal plane; a recorded path is
the only way to avoid a wheelchair, a box, or your own body.

`teach_pose.py --gesture --name WAVE` records a **gesture** — a motion
whose point is the motion. Gestures:

- replay at the **tempo you demonstrated** (capped by
  `arm.gesture_speed_radps`), because a wave replayed at pose speed is
  just the arm leaning about
- **repeat** if you recorded a full cycle (ends where it began)
- always start at their first waypoint — a wave that begins halfway
  through isn't a wave
- never appear in the `POSE:` readout: you perform a gesture, you're
  never *in* one

Both appear in the pose menu, tagged `(path)` or `(gesture)`.

> **Speed is the open question.** `gesture_speed_radps` defaults to 1.0,
> which makes a hand-demonstrated wave replay roughly 5x slower than
> taught. Raising it makes gestures legible but moves the arm faster
> near people — tune it on hardware, and consider that a fast wrist roll
> is far less dangerous than a fast shoulder swing.

## Notes field (video overlay)

Under the mode/xyz lines, colored by urgency:

| Color | Meaning |
|---|---|
| 🔴 Red | Act now — load guard tripped, `FORCE HIGH` (a servo straining), or a pose move running |
| 🟠 Amber | Something is limiting motion — workspace fence, joint limit, or the pose menu header |
| 🟢 Green | Information — the pose list, and `POSE: <name>` |

`FORCE HIGH` fires above `arm.load_warn_threshold` (0.675 by default,
i.e. 75% of the 0.9 stop threshold) and names the straining joints plus
the peak load, so you see a servo working hard *before* the guard stops
the arm.

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
