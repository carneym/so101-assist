# so101-assist v0.02

A shared-autonomy interface for the SO-101 arm, designed for operation by a
quadriplegic user: annotated camera feeds, voice or automatic target selection,
autonomous approach to a pre-grasp pose, and operator fine control via
keyboard, gamepad, or QuadStick.

Design principle: **the robot handles precision, the operator handles intent.**

[THIS IS A MILESTONE COMMIT THAT INCLUDES QUADSTICK INPUT WITH ANNOTATED VIDEO FEED ONLY. NO VOICE CONTROL AT THIS POINT]

## Architecture

Nodes communicate over an in-process pub/sub bus (`so101_assist/bus.py`) using
the message types in `so101_assist/messages.py`. The interaction flow is owned
by `control/state_machine.py`: nothing moves without explicit confirmation, and
"stop"/"cancel" are honored from every state.

```
cameras ─► perception (detect / segment / localize) ─► planner ─► shared ctrl ─► arm
voice   ─► commands + target descriptions ──────────────┘              ▲
operator input (keyboard / gamepad / quadstick) ───────────────────────┘
```

- `so101_assist/arm/` — the only code that touches hardware; kinematics,
  Cartesian velocity control, and the safety layer (workspace fence, load
  monitoring, stop semantics)
- `so101_assist/perception/` — open-vocab detection (YOLO-World), SAM 2 masks,
  pixel→robot-frame localization (table-plane first, depth camera later),
  AprilTag calibration
- `so101_assist/voice/` — faster-whisper transcription; tiny exact-match
  command grammar, everything else forwarded as a target description
- `so101_assist/planner/` — grasp pose selection under the arm's 5DOF
  constraint, pre-grasp approach paths
- `so101_assist/control/` — state machine, autonomy/operator blending,
  input device drivers
- `so101_assist/ui/` — FastAPI + WebSocket video with annotation overlays,
  state and mode banners

## Build phases

1. **Jog** — `scripts/jog_test.py`: keyboard Cartesian jog, live video, safety
   layer. No AI. Milestone: it feels safe and responsive.
2. **See & approach** — overhead detection, click-to-select, table-plane
   localization, autonomous move to pre-grasp with confirm step.
3. **Voice** — target descriptions and command words replace the mouse.
4. **Accessible input** — gamepad, then QuadStick with mode switching and
   audible mode announcements.
5. **End-user iteration** — blended shared control, per-user tuning. Expect
   this phase to reshape the interaction design; that's the point.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[arm,inputs,dev]"
```

Then follow the **[getting-started tutorial](docs/getting-started.md)**
(find the port → calibrate → drive with the QuadStick), and see
**[docs/controls.md](docs/controls.md)** for the full control map. Add
the `perception,voice` extras for Phase 2+.

## Cloud policy test (pi0.5 on Modal)

`soarm-test.py` is a standalone experiment, outside the node architecture above: it
drives the arm directly from a spoken-style instruction using the LeRobot pi0.5
policy, with the forward pass running on a Modal GPU.

```bash
pip install "modal>=1.5" "lerobot[feetech]>=0.6" opencv-python pyyaml
modal setup && export HF_TOKEN=...        # the PaliGemma tokenizer repo is gated
modal run soarm-test.py --dry-run         # full loop, no motion
modal run soarm-test.py --task "pick up the glasses"
```

It reuses `config/default.yaml` (port, cameras) and `config/calibration/arm.json`
(motor calibration, and the normalization ranges handed to the policy). Note that no
pi0.5 checkpoint is trained on the SO-101, so out of the box this is an
inference-plumbing test, not a working manipulation policy — see the script's
docstring.

## Safety notes

- All motion passes through one controller with hard velocity caps.
- "stop" (voice) halts and holds from any state; torque-off is a separate
  explicit action so a held object is never dropped.
- The workspace fence zeroes outward velocity components; re-run
  `scripts/calibrate_hand_eye.py` whenever cameras or the arm are moved.
