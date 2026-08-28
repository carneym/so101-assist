# Getting started

A step-by-step walkthrough from a fresh clone to driving the SO-101 arm
with the QuadStick and a live wrist-camera view. Written for **Linux /
Raspberry Pi** (the arm's serial port shows up as `/dev/ttyACM0` or
similar); on Windows + PowerShell the port is `COMn` and
`.venv/bin/` becomes `.venv\Scripts\` — Windows equivalents are noted
inline where they differ.

Each step builds on the last and is safe to stop at — nothing moves the
arm under power until Step 7.

## Quick start (Linux / Raspberry Pi)

Already set this machine up? Boot the Pi, plug in the arm and the
QuadStick, then:

```bash
cd ~/Projects/so101-assist && source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. **Every terminal needs
this** — `ModuleNotFoundError: No module named 'so101_assist'` always
means it's missing from that terminal.

Then drive, either via the launcher (recommended — it auto-detects the
port, pings the servos, reuses your calibration, and checks the
QuadStick before starting):

```bash
./scripts/start_teleop.sh
```

or by running teleop directly:

```bash
python scripts/quadstick_teleop.py --port /dev/ttyACM0
```

<sub>Not sure of the port? `ls /dev/ttyACM*`. Type it exactly — a trailing slash gives "Not a directory".</sub>

### Everything else you'll want

| Task | Command |
|---|---|
| Preflight only, don't drive | `./scripts/start_teleop.sh --check` |
| Drive with no video window | `./scripts/start_teleop.sh --no-camera` |
| List the serial ports | `python scripts/arm_test.py --list-only` |
| Check the arm responds (read-only) | `python scripts/arm_test.py --port /dev/ttyACM0` |
| Re-calibrate the arm | `python scripts/calibrate_arm.py --port /dev/ttyACM0` |
| Teach the standard poses | `python scripts/teach_pose.py --port /dev/ttyACM0` |
| Record a path to one pose | `python scripts/teach_pose.py --port /dev/ttyACM0 --record --name STOW` |
| Record a gesture | `python scripts/teach_pose.py --port /dev/ttyACM0 --gesture --name WAVE` |
| List what's taught | `python scripts/teach_pose.py --list` |
| Live tuning sliders (2nd terminal) | `python scripts/tuning_gui.py` |
| Run the tests | `pytest -q` |

**Driving:** lip switch cycles SHOULDER → ELBOW → WRIST · centre
sip/puff works the gripper in every mode · left tube is z in WRIST mode
· **right puff opens the pose menu** (stick up/down chooses, lip switch
goes, right sip aborts) · `Ctrl-C` or `q` in the video window stops and
releases torque. Full map in [controls.md](controls.md).

> **Release a stuck arm.** If a crash ever leaves the arm powered and
> holding:
> ```bash
> python -c "from so101_assist.arm.driver import SO101Driver; d=SO101Driver(port='/dev/ttyACM0'); d.connect(); d.torque_off(); d.disconnect()"
> ```
> If even that can't reach it, cut power at the arm's supply.

The rest of this page is the first-time setup, and the reference for
when something goes wrong.

## What you need

- The SO-101 arm connected by USB, and its wrist camera plugged in
- A QuadStick (or, for bring-up, any USB joystick)
- Python 3.10+ (3.12 tested; 3.13 also works). On a Raspberry Pi, a
  Pi 4 or Pi 5 (64-bit / aarch64 OS) is recommended.

## 0. Raspberry Pi / Linux system setup (one time)

On a fresh Raspberry Pi OS / Debian / Ubuntu box, install the system
packages the venv can't provide, then make sure your user can reach the
USB serial port and the joystick:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev python3-tk libgl1
```

- `python3-venv` / `python3-dev` — virtualenv + building any wheels.
- `python3-tk` — the live tuning GUI (`scripts/tuning_gui.py`) uses
  Tkinter; without it that one script fails to import.
- `libgl1` — OpenCV's GUI windows (`cv2.imshow`) need libGL.

**Device access — add yourself to two groups** (then log out and back
in, or reboot, for it to take effect):

```bash
sudo usermod -aG dialout,input "$USER"
```

- `dialout` — read/write the arm's `/dev/ttyACM*` serial port without
  `sudo`.
- `input` — read the joystick/QuadStick under `/dev/input/*`.

Check both took effect after re-login: `groups` should list `dialout`
and `input`.

> **Headless Pi (no monitor / SSH only)?** pygame still needs an SDL
> video driver even though this app never opens a pygame window, and
> `cv2.imshow` can't display without one. Export a dummy driver so the
> input drivers initialize:
> ```bash
> export SDL_VIDEODRIVER=dummy
> ```
> The camera-window scripts (`camera_test.py`, `quadstick_teleop.py`)
> won't show video headless — drive with a monitor attached, or use
> X-forwarding, for the steps that open a window.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
```

<sub>Windows PowerShell: `python -m venv .venv; .venv\Scripts\Activate.ps1`.</sub>

**Raspberry Pi / any non-CUDA box — install CPU-only torch first.**
LeRobot depends on `torch`, and PyPI's aarch64 `torch` wheel drags in
~450 MB of NVIDIA CUDA libraries that are useless on a Pi (and big
enough to fill a small `/tmp`). Install a CPU build in LeRobot's
supported range *before* the project, so the next step finds it
already satisfied:

```bash
pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" \
    --index-url https://download.pytorch.org/whl/cpu
```

<sub>On an x86_64 machine with a real NVIDIA GPU, skip this and let the default wheel install.</sub>

Then the project itself:

```bash
pip install -e ".[arm,inputs,dev]"
```

The extras: `arm` (LeRobot + the Feetech `scservo_sdk` servo driver, via
`lerobot[feetech]`), `inputs` (pygame for the QuadStick), `dev` (pytest,
ruff). Add `perception,voice` later for Phase 2+ (both are heavy on ARM —
expect a long build and slow runtime on a Pi).

> **`No space left on device` mid-install on a Pi?** `/tmp` is often a
> small RAM-backed `tmpfs` that a big wheel can't unpack into. Point
> pip's temp dir at real disk: `mkdir -p ~/piptmp && TMPDIR=~/piptmp pip install ...`

> **Camera window later shows `imshow` "not implemented"?** LeRobot
> depends on `opencv-python-headless`, which installs over the GUI
> `opencv-python` (they share the `cv2` package — last one installed
> wins). If a `cv2` window won't open, force the GUI build back on top —
> **run this last, after any lerobot (re)install**:
> ```bash
> pip uninstall -y opencv-python opencv-python-headless
> pip install opencv-python
> ```
> On a genuinely headless Pi you want the opposite — keep
> `opencv-python-headless` and don't use the window-opening scripts.

> **Every terminal needs the venv active.** You'll know it is when the
> prompt starts with `(.venv)`. If a script dies with
> `ModuleNotFoundError: No module named 'so101_assist'`, the venv isn't
> active in that terminal — run `source .venv/bin/activate` (or call
> `.venv/bin/python <script>` directly).

Confirm the install:

```bash
pytest -q
```

## 2. Find the arm's serial port

```bash
python scripts/arm_test.py --list-only
```

Lists serial ports. The arm is typically a `CH343`/`USB-Enhanced-SERIAL`
device — note its device path (e.g. `/dev/ttyACM0`). You can also just
`ls /dev/ttyACM*` before and after plugging it in. Substitute your port
for `/dev/ttyACM0` everywhere below.

<sub>Windows: the port is a `COMn` (e.g. `COM3`); find it in Device Manager or via this same `--list-only`.</sub>

> **`Permission denied` opening the port?** You're not in the `dialout`
> group yet, or haven't logged out/in since adding yourself — see
> Step 0.

## 3. Verify arm connectivity (read-only, nothing moves)

```bash
python scripts/arm_test.py --port /dev/ttyACM0
```

Pings all six servos and streams joint positions/loads/gripper. It fails
loudly if a servo doesn't respond (check the daisy-chain connectors).
Move the arm by hand — the numbers should change. `Ctrl-C` to stop.

## 4. Calibrate the arm

```bash
python scripts/calibrate_arm.py --port /dev/ttyACM0
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

```bash
python scripts/fk_live.py --port /dev/ttyACM0
```

Read-only, torque off. Move the arm by hand and check the printed
`ee_xyz` (end-effector position, meters) tracks reality: raise the
gripper → `z` increases; swing the base → `x`/`y` trace an arc. Rest the
gripper on the table and **note the `z` value** — you'll want it for the
workspace fence (Step 9 / tuning). If a direction is mirrored, tell the
maintainer; a joint's sign convention is off.

## 6. (Optional but recommended) First motion under power

The first script that moves the arm. Clear the workspace.

```bash
python scripts/motion_test.py --port /dev/ttyACM0 --gripper --delta 0.2
```

It reads the current pose, asks you to confirm the workspace is clear,
enables torque, makes a small capped move and returns, then releases
torque. Try a joint next:

```bash
python scripts/motion_test.py --port /dev/ttyACM0 --joint wrist_roll --delta-deg 15 --verbose
```

If it feels safe and responsive, the arm foundation is good.

## 7. Set up the QuadStick

The QuadStick must be in its **Joystick/Gamepad** output profile (via
QuadStick Configurator), **not** Mouse mode — Mouse mode moves the OS
cursor instead of reporting joystick axes. Confirm the raw signals:

```bash
python scripts/quadstick_test.py --raw
```

Move the stick, sip/puff each tube, press the lip switch, and confirm
`axes=[...]` and `buttons_down=[...]` change. The button numbers are
already mapped in config for a standard unit; if yours differ, note them
(they can be overridden under `operator.quadstick` in
`config/default.yaml`).

<sub>On Linux you can sanity-check the device exists with `ls /dev/input/js*` and, if you install `joystick`, `jstest /dev/input/js0`. On Windows, use `joy.cpl` (Test tab).</sub>

> **`no joystick at index 0`?** The device isn't visible to SDL. Check
> `ls /dev/input/js*` shows it, that you're in the `input` group
> (Step 0), and — if headless — that `SDL_VIDEODRIVER=dummy` is
> exported.

## 8. Find the wrist camera index

```bash
python scripts/camera_test.py --list-only
```

Probes camera indices. You likely have more than one camera (built-in
webcam, etc.). View each to find the wrist cam:

```bash
python scripts/camera_test.py --index 2
```

A window opens (press `q` to close) — wave a hand in front of the wrist
camera to identify it. Set its index under `cameras.wrist.index` in
`config/default.yaml` (the default is `2`).

> **`can't open camera` / black window on Linux?** V4L2 indices don't
> always match `/dev/videoN` numbers (a single camera often exposes two
> nodes). Try `v4l2-ctl --list-devices` (`sudo apt install v4l-utils`)
> to see the real capture node, and test each `--index`.

> **OpenCV window won't open / `imshow` "not implemented"?** The `arm`
> extra can pull in a headless OpenCV build that shadows the GUI one.
> Fix: `pip uninstall -y opencv-python opencv-python-headless; pip install opencv-python`.

## 9. Drive it

```bash
python scripts/quadstick_teleop.py --port /dev/ttyACM0
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

## 9b. Teach the named poses (optional)

With the arm connected, teach the poses you'll jump to from the pose
menu. Torque stays off — you position the arm by hand:

```bash
python scripts/teach_pose.py --port /dev/ttyACM0
```

It walks you through **HOME** (tucked in, gripper closed), **RAISED**,
and **EXTENDED** — move the arm into each and press Enter. Check them
with `--list`, redo one with `--name HOME`.

Then in teleop: **right puff** opens the pose menu, **stick up/down**
chooses, **lip switch** goes, **right sip** aborts. See
[controls.md](controls.md#named-poses).

> **Re-teach after any recalibration** — calibration shifts the zero
> pose, so previously taught angles point somewhere else.

## 10. Fine-tune live

In a **second terminal** (venv active), while teleop runs:

```bash
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
| `ModuleNotFoundError: so101_assist` | venv not active in that terminal — `source .venv/bin/activate` |
| `Permission denied` on `/dev/ttyACM0` | not in `dialout` group, or no re-login since adding — Step 0 |
| `no joystick at index 0` | not in `input` group, device not present, or (headless) `SDL_VIDEODRIVER` unset — Step 0 / Step 7 |
| `imshow ... not implemented` | headless OpenCV shadowing — reinstall `opencv-python` (Step 8) |
| Camera opens black / wrong device | V4L2 index ≠ `/dev/videoN`; find the real node with `v4l2-ctl --list-devices` (Step 8) |
| Arm can't lift its own weight | raise setpoint leash + driver step clamp in the tuning GUI (Step 10) |
| `STOPPED (load monitor tripped?)` | collision/overload guard tripped; `Ctrl-C` and restart, or raise/disable the load guard in the tuning GUI |
| Joint stops short, `[limit]` printed | at its software operating limit — widen it in the tuning GUI or `arm.joint_limits_deg` |
| Motion stops going down, `[fence]` printed | workspace fence z-floor is above the table — re-measure with `fk_live.py` and update `workspace_fence` |
| Calibration write fails intermittently | transient servo-bus hiccup — re-run `calibrate_arm.py` |
