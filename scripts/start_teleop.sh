#!/usr/bin/env bash
# One-command bring-up: boot the Pi, plug in the arm + QuadStick, run this.
#
# Does every preflight step of docs/getting-started.md that can be
# automated, then launches teleop:
#
#   1. activates the venv (no "ModuleNotFoundError: so101_assist")
#   2. auto-detects the arm's serial port (no hand-typed /dev/ttyACM0,
#      so the trailing-slash "Not a directory" mistake can't happen)
#   3. pings the servos so a dead/unpowered arm fails here, loudly,
#      instead of halfway into teleop
#   4. CALIBRATES ONLY IF NEEDED — an existing, complete calibration is
#      reused; pass --recalibrate to force a fresh one
#   5. checks the QuadStick is visible as a joystick
#   6. launches quadstick_teleop.py
#
# Anything after the script's own flags is forwarded to teleop, e.g.
#   ./scripts/start_teleop.sh --no-camera --debug
#
# Nothing here moves the arm. Teleop itself still waits for you to
# confirm the workspace is clear before it enables torque.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
CALIB="$REPO_ROOT/config/calibration/arm.json"
MOTORS='shoulder_pan shoulder_lift elbow wrist_flex wrist_roll gripper'

RECALIBRATE=0
PORT="${SO101_PORT:-}"
CHECK_ONLY=0
TELEOP_ARGS=()
# What the operator actually typed, minus --check, so the
# --check run can hand back the exact command that drives.
RERUN_ARGS=()

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recalibrate) RECALIBRATE=1; shift ;;
    --port) PORT="${2:-}"; RERUN_ARGS+=(--port "${2:-}"); shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
      echo
      echo "Usage: $0 [--port DEV] [--recalibrate] [--check] [teleop args...]"
      echo "  --port DEV      skip auto-detect, use this serial port"
      echo "  --recalibrate   force a fresh calibration even if one exists"
      echo "  --check         run preflight only, don't launch teleop"
      exit 0 ;;
    *) TELEOP_ARGS+=("$1"); RERUN_ARGS+=("$1"); shift ;;
  esac
done

bold "SO-101 teleop bring-up"

# 1 ── venv ------------------------------------------------------------
[[ -x "$PY" ]] || die "No venv at .venv — create it with:
    python3 -m venv .venv && source .venv/bin/activate && pip install -e \".[arm,inputs,dev]\""
"$PY" -c 'import so101_assist' 2>/dev/null \
  || die "venv exists but so101_assist isn't installed — run: pip install -e \".[arm,inputs,dev]\""
ok "venv ready ($("$PY" --version 2>&1))"

# 2 ── serial port -----------------------------------------------------
# Prefer real USB serial devices; the Pi's onboard /dev/ttyAMA* UART
# always enumerates and is never the arm.
if [[ -z "$PORT" ]]; then
  PORT="$("$PY" - <<'EOF'
from serial.tools import list_ports
cands = [p for p in list_ports.comports()
         if p.vid is not None or p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))]
cands = [p for p in cands if not p.device.startswith("/dev/ttyAMA")]
print(cands[0].device if len(cands) == 1 else "")
if len(cands) > 1:
    import sys
    print("AMBIGUOUS:" + ",".join(p.device for p in cands), file=sys.stderr)
EOF
)"
  PORT="${PORT//[$'\t\r\n ']/}"
fi
[[ -n "$PORT" ]] || die "Couldn't auto-detect the arm's serial port.
    Is it plugged in and powered? Check with: ls /dev/ttyACM*
    Then pass it explicitly:  $0 --port /dev/ttyACM0"
[[ -e "$PORT" ]] || die "Port '$PORT' doesn't exist (check for a stray trailing slash)."
ok "arm port: $PORT"

# 3 ── servos respond --------------------------------------------------
# Read-only: connect() pings every servo, then we disconnect.
"$PY" - "$PORT" <<'EOF' || die "Servos didn't respond on that port.
    Check the power supply is on and the daisy-chain connectors are seated."
import sys
from so101_assist.arm.driver import SO101Driver
d = SO101Driver(port=sys.argv[1])
d.connect()
d.disconnect()
EOF
ok "all six servos responding"

# 4 ── calibration (only if needed) ------------------------------------
NEED_CALIB=0
if [[ "$RECALIBRATE" == 1 ]]; then
  NEED_CALIB=1
  warn "--recalibrate given: running calibration even though one exists"
elif [[ ! -f "$CALIB" ]]; then
  NEED_CALIB=1
  warn "no calibration found at config/calibration/arm.json"
elif ! "$PY" - "$CALIB" "$MOTORS" <<'EOF' 2>/dev/null
import json, sys
data = json.load(open(sys.argv[1]))
missing = [m for m in sys.argv[2].split() if m not in data]
sys.exit(1 if missing else 0)
EOF
then
  NEED_CALIB=1
  warn "calibration at config/calibration/arm.json is incomplete"
fi

if [[ "$NEED_CALIB" == 1 ]]; then
  echo
  bold "Calibration needed — the arm will be LIMP (torque off); you move it by hand."
  echo "  1. move it to the MIDDLE of its range, press Enter"
  echo "  2. sweep EVERY joint through its FULL range (partial sweeps get clamped)"
  echo
  "$PY" scripts/calibrate_arm.py --port "$PORT" || die "Calibration failed or was cancelled."
  ok "calibration saved"
  echo
  warn "Recalibrating shifts the FK frame — the workspace_fence in"
  warn "config/default.yaml may now be wrong. Re-measure the table-touch z:"
  warn "    $PY scripts/fk_live.py --port $PORT"
else
  ok "calibration present — skipping (use --recalibrate to redo)"
fi

# 4b ── taught poses (optional) ----------------------------------------
POSE_COUNT="$("$PY" -c 'from so101_assist.arm.poses import load_poses; print(len(load_poses()))' 2>/dev/null || echo 0)"
if [[ "$POSE_COUNT" -gt 0 ]]; then
  ok "$POSE_COUNT taught pose(s) — right puff opens the menu"
else
  warn "No poses taught (optional). Teach HOME/RAISED/EXTENDED with:"
  warn "    $PY scripts/teach_pose.py --port $PORT"
fi

# 5 ── QuadStick -------------------------------------------------------
JOYSTICKS="$(SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}" "$PY" - <<'EOF' 2>/dev/null
import contextlib, io, os
with contextlib.redirect_stdout(io.StringIO()):
    import pygame
    pygame.init(); pygame.joystick.init()
    n = pygame.joystick.get_count()
    names = [pygame.joystick.Joystick(i).get_name() for i in range(n)]
print("|".join(names))
EOF
)"
if [[ -n "$JOYSTICKS" ]]; then
  ok "joystick: ${JOYSTICKS%%|*}"
else
  warn "No joystick detected. Plug in the QuadStick and set its"
  warn "QuadStick Configurator profile to Joystick/Gamepad (NOT Mouse mode)."
  warn "Teleop will fail to start without it."
fi

# 6 ── display ---------------------------------------------------------
# Teleop opens a cv2 window for the wrist camera; headless needs --no-camera.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  if [[ ! " ${TELEOP_ARGS[*]:-} " =~ " --no-camera " ]]; then
    warn "No display detected — adding --no-camera (the video window can't open)"
    TELEOP_ARGS+=(--no-camera)
  fi
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
  echo
  bold "Preflight OK — not launching teleop (--check)."
  echo "Everything above passed. To actually drive, run the same"
  echo "command again WITHOUT --check:"
  echo
  if [[ ${#RERUN_ARGS[@]} -gt 0 ]]; then
    printf '    %s %s\n\n' "$0" "${RERUN_ARGS[*]}"
  else
    printf '    %s\n\n' "$0"
  fi
  exit 0
fi

# 7 ── go --------------------------------------------------------------
echo
bold "Starting teleop on $PORT"
echo "  lip switch = cycle SHOULDER → ELBOW → WRIST"
echo "  center sip/puff = gripper (every mode)"
echo "  Ctrl-C, or 'q' in the video window = stop + release torque"
echo
exec "$PY" scripts/quadstick_teleop.py --port "$PORT" ${TELEOP_ARGS[@]+"${TELEOP_ARGS[@]}"}
