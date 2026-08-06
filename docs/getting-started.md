# Getting started

A step-by-step walkthrough from a fresh clone to driving the SO-101 arm
with the QuadStick and a live wrist-camera view. Written for Windows +
PowerShell (the arm's serial port shows up as `COMn`); on Linux/macOS
the port is `/dev/ttyACM0` or similar and `.venv\Scripts\` becomes
`.venv/bin/`.

Each step builds on the last and is safe to stop at — nothing moves the
arm under power until Step 7.

## What you need

- The SO-101 arm connected by USB, and its wrist camera plugged in
- A QuadStick (or, for bring-up, any USB joystick)
- Python 3.10+ (3.12 tested)

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[arm,inputs,dev]"
```

The extras: `arm` (LeRobot + Feetech servo SDK), `inputs` (pygame for the
QuadStick), `dev` (pytest, ruff). Add `perception,voice` later for
Phase 2+.

> **Every terminal needs the venv active.** You'll know it is when the
> prompt starts with `(.venv)`. If a script dies with
> `ModuleNotFoundError: No module named 'so101_assist'`, the venv isn't
> active in that terminal — run `.venv\Scripts\Activate.ps1` (or call
> `& ".venv\Scripts\python.exe" <script>` directly).

Confirm the install:

```powershell
pytest -q
```

## 2. Find the arm's serial port

```powershell
python scripts/arm_test.py --list-only
```

Lists serial ports. The arm is typically a `CH343`/`USB-Enhanced-SERIAL`
device — note its `COMn` (e.g. `COM3`). Substitute your port for `COM3`
everywhere below.

## 3. Verify arm connectivity (read-only, nothing moves)

```powershell
python scripts/arm_test.py --port COM3
```

Pings all six servos and streams joint positions/loads/gripper. It fails
loudly if a servo doesn't respond (check the daisy-chain connectors).
Move the arm by hand — the numbers should change. `Ctrl-C` to stop.

## 4. Calibrate the arm

```powershell
python scripts/calibrate_arm.py --port COM3
```

Interactive and torque-off (the arm is limp — you move it by hand):

1. It disables torque, then asks you to move the arm to the **middle of
   its range** and press Enter (sets each joint's home offset).
2. Then sweep **every joint through its full range of motion** by hand
   while it records min/max. **Sweep completely** — a joint you don't
   move fully gets a truncated range that the firmware then clamps to.
3. Saves to `config/calibration/arm.json`.

Torque stays off when it finishes. If a servo write fails intermittently
(`no status packet`), just re-run — it's usually a transient bus hiccup.

## 5. Verify the calibration

```powershell
python scripts/fk_live.py --port COM3
```

Read-only, torque off. Move the arm by hand and check the printed
`ee_xyz` (end-effector position, meters) tracks reality: raise the
gripper → `z` increases; swing the base → `x`/`y` trace an arc. Rest the
gripper on the table and **note the `z` value** — you'll want it for the
workspace fence (Step 9 / tuning). If a direction is mirrored, tell the
maintainer; a joint's sign convention is off.

## 6. (Optional but recommended) First motion under power

The first script that moves the arm. Clear the workspace.

```powershell
python scripts/motion_test.py --port COM3 --gripper --delta 0.2
```

It reads the current pose, asks you to confirm the workspace is clear,
enables torque, makes a small capped move and returns, then releases
torque. Try a joint next:

```powershell
python scripts/motion_test.py --port COM3 --joint wrist_roll --delta-deg 15 --verbose
```

If it feels safe and responsive, the arm foundation is good.

## 7. Set up the QuadStick

The QuadStick must be in its **Joystick/Gamepad** output profile (via
QuadStick Configurator), **not** Mouse mode — Mouse mode moves the OS
cursor instead of reporting joystick axes. Verify Windows sees it moving
in `joy.cpl` (Test tab), then confirm the raw signals:

```powershell
python scripts/quadstick_test.py --raw
```

Move the stick, sip/puff each tube, press the lip switch, and confirm
`axes=[...]` and `buttons_down=[...]` change. The button numbers are
already mapped in config for a standard unit; if yours differ, note them
(they can be overridden under `operator.quadstick` in
`config/default.yaml`).

## 8. Find the wrist camera index

```powershell
python scripts/camera_test.py --list-only
```

Probes camera indices. You likely have more than one camera (built-in
webcam, etc.). View each to find the wrist cam:

```powershell
python scripts/camera_test.py --index 2
```

A window opens (press `q` to close) — wave a hand in front of the wrist
camera to identify it. Set its index under `cameras.wrist.index` in
`config/default.yaml` (the default is `2`).

> **OpenCV window won't open / `imshow` "not implemented"?** The `arm`
> extra pulls in `opencv-python-headless`, which shadows the GUI build.
> Fix: `pip uninstall -y opencv-python opencv-python-headless; pip install opencv-python`.

## 9. Drive it

```powershell
python scripts/quadstick_teleop.py --port COM3
```

The wrist-camera window opens with the active mode overlaid, and the
console prints a control cheat-sheet each time you change mode. It waits
for you to confirm the workspace is clear before enabling torque.

- **Lip switch** cycles modes: **SHOULDER → ELBOW → WRIST**
- **Center sip/puff** opens/closes the gripper in *every* mode
- `Ctrl-C` or `q` in the video window stops and releases torque

See [controls.md](controls.md) for the full control map, speeds, and
limits.

First-drive checks: the workspace fence (`config/default.yaml`,
`workspace_fence`) must match your table height — if the arm stops short
going down, or a control seems dead, the console will say why
(`[fence] ...`, `[limit] ...`, or `STOPPED` for a load trip). The fence
z-floor should be just below the table-touch `z` you noted in Step 5.

## 10. Fine-tune live

In a **second terminal** (venv active), while teleop runs:

```powershell
python scripts/tuning_gui.py
```

Sliders for servo speeds, lift torque (setpoint leash + step clamp), the
load guard, and every joint's min/max angle. Changes apply to the
running arm within ~0.4 s — no restart. Persists in `config/tuning.json`;
**Reset to config baseline** reverts. See
[controls.md](controls.md#live-tuning-on-the-fly-while-driving).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: so101_assist` | venv not active in that terminal — `.venv\Scripts\Activate.ps1` |
| `imshow ... not implemented` | headless OpenCV shadowing — reinstall `opencv-python` (Step 8) |
| Arm can't lift its own weight | raise setpoint leash + driver step clamp in the tuning GUI (Step 10) |
| `STOPPED (load monitor tripped?)` | collision/overload guard tripped; `Ctrl-C` and restart, or raise/disable the load guard in the tuning GUI |
| Joint stops short, `[limit]` printed | at its software operating limit — widen it in the tuning GUI or `arm.joint_limits_deg` |
| Motion stops going down, `[fence]` printed | workspace fence z-floor is above the table — re-measure with `fk_live.py` and update `workspace_fence` |
| Calibration write fails intermittently | transient servo-bus hiccup — re-run `calibrate_arm.py` |
