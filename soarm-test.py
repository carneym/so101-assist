"""Drive the SO-101 arm from a natural-language command, with pi0.5 inference on Modal.

    modal run soarm-test.py                       # prompts for a command, then runs
    modal run soarm-test.py --task "pick up the glasses"
    modal run soarm-test.py --dry-run             # full pipeline, arm never moves

HOW IT IS SPLIT
---------------
`modal run` executes `main()` (the @app.local_entrypoint) on THIS machine — the one
with the USB serial bus and the cameras — and only the policy forward pass happens
in the cloud:

    local: read joints + frames  ──► Modal GPU: pi0.5 → 50-step action chunk
      ▲                                                        │
      └──────────── execute the chunk on the arm ◄─────────────┘

Each round trip is one network hop (~150-400 ms) plus inference, which is why we ask
for a whole action chunk and play it out open-loop at `--fps` before re-planning.
That is exactly how pi0/pi0.5 are meant to be run; per-step remote inference would
stall the arm between every joint command.

CALIBRATION
-----------
The motor calibration in `config/calibration/arm.json` (the file this repo already
uses, see so101_assist/arm/driver.py) is reused verbatim: it is written to a
LeRobot-shaped calibration file and handed to `SO101Follower`, which pushes it into
the servos on connect. That file is also the source of the normalization statistics
sent to the policy — each joint's calibrated travel maps onto the [-1, 1] range the
model expects, so no dataset download is needed. Recalibrating the arm (or editing
that JSON) automatically updates both.

ZERO-SHOT EXPECTATIONS — PLEASE READ
------------------------------------
There is no π₀.₅ checkpoint trained on the SO-101. The public ones are
`lerobot/pi05_base` (the cross-embodiment base model, published for finetuning) and
the LIBERO simulation checkpoints; π₀.₅-DROID, the checkpoint that genuinely works
zero-shot, is a Franka + DROID-camera model distributed through openpi's GCS bucket,
not a LeRobot-loadable HF repo, and its action space is not the SO-101's.

So: this script runs π₀.₅ end-to-end against your arm and its language prompt, but
treat the motion as *unvalidated*. Start with `--dry-run`, keep `--max-relative-target`
small, and keep a hand on the power switch. To get reliable behaviour you will need to
finetune `lerobot/pi05_base` on ~50 teleop episodes of your own task (see
https://huggingface.co/docs/lerobot/pi05) and then pass `--model <your-repo-id>`.

REQUIREMENTS (local machine)
----------------------------
    pip install "modal>=1.5" "lerobot[feetech]>=0.6" opencv-python pyyaml
    modal setup
    export HF_TOKEN=...   # needed once: the PaliGemma tokenizer repo is gated
"""

# NOTE: no `from __future__ import annotations` here — Modal reads the *runtime* types of
# the class parameters below, and postponed annotations turn them into strings it rejects.
import io
import json
import os
import sys
import time
from pathlib import Path

try:
    import modal
except ModuleNotFoundError as exc:  # noqa: F841
    # --preview needs no Modal *account* and never starts a container, but this file is
    # a Modal app: the decorators below run at import. Fail with the fix, not a traceback.
    raise SystemExit(
        "soarm-test.py needs the `modal` package installed (even for --preview):\n"
        "    pip install modal"
    ) from exc

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

APP_NAME = "soarm-pi05"

# π₀.₅ weights on the Hub. `lerobot/pi05_base` is the general base checkpoint; override
# with --model once you have a finetune of your own (see the module docstring).
DEFAULT_MODEL = "lerobot/pi05_base"

# lerobot's pi0.5 processor hardcodes this tokenizer (see processor_pi05.py). It is a
# GATED repo: loading it needs an HF token belonging to an account that has accepted
# Google's Gemma licence on the model page. Checked before the weights download, because
# otherwise the failure lands several GB and many minutes later.
TOKENIZER_REPO = "google/paligemma-3b-pt-224"

# Pinned so a lerobot release can't silently change the policy/processor API underneath
# the remote half. Matches the version installed locally, so both halves share identical
# policy and normalization semantics — one less variable when something looks wrong.
LEROBOT_VERSION = "0.6.1"

# ~3.3B params in bf16 (PaliGemma 3B + the action expert) plus 10 flow-matching steps
# per call. L40S is the latency/price sweet spot; "A10G" is cheaper and ~2x slower,
# "H100" shaves another ~100 ms off each chunk.
GPU_TYPE = "L40S"

# Joint order used for the policy's state and action vectors. This is LeRobot's own
# SO-101 motor order — note `elbow_flex`, which this repo's calibration file calls
# `elbow` (remapped in _lerobot_calibration below).
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# STS3215 encoder counts. `_normalize` in lerobot's motors bus divides by resolution - 1.
ENCODER_RESOLUTION = 4096

REPO_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Modal image — everything below runs in the cloud container, never on your machine
# --------------------------------------------------------------------------------------

# The Hub cache lives on a Volume so the ~7 GB of weights are downloaded once and are
# already on disk for every later container start.
hf_cache = modal.Volume.from_name("soarm-pi05-hf-cache", create_if_missing=True)

inference_image = (
    # 3.12 is the floor: lerobot 0.6.x declares requires-python = ">=3.12", so 3.11 fails
    # the install outright. It is also the one series every Modal image-builder version
    # supports, so this cannot break on that axis either. Raise it to match your local
    # interpreter if you prefer; do not lower it.
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0")  # git: some HF download paths; glib: opencv runtime
    .pip_install(
        # [pi] adds transformers + scipy, which is what pi0/pi0.5 need on top of core lerobot.
        # Only the policy stack is installed here — no cameras, no motors, no datasets.
        f"lerobot[pi]=={LEROBOT_VERSION}",
        "pillow",  # already a lerobot dep; named because predict_chunk decodes JPEGs with it
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",  # -> the Volume, so weights survive the container
            "HF_XET_HIGH_PERFORMANCE": "1",  # fast Xet downloads (replaces HF_HUB_ENABLE_HF_TRANSFER)
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)

# Imports inside this block happen only in the container. Keeping them out of module
# scope matters: `modal run` also imports this file locally, where lerobot's policy
# stack (torch, transformers) may not be installed.
with inference_image.imports():
    import numpy as np
    import torch
    from PIL import Image as PILImage

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy


@app.cls(
    image=inference_image,
    gpu=GPU_TYPE,
    volumes={"/cache": hf_cache},
    timeout=30 * 60,
    # Keep the loaded model resident between commands: you type a new instruction, the
    # same warm container answers. Raise this if you think between prompts a lot.
    scaledown_window=5 * 60,
    # A bad token or an unaccepted licence fails identically every time, so retrying it
    # just spends GPU minutes to print the same traceback three times.
    retries=0,
    # Resolved on THIS machine when `modal run` starts, so the gated PaliGemma tokenizer
    # download inside the container uses your local HF_TOKEN. No Modal secret to create.
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class Pi05Server:
    """π₀.₅ loaded once per container, answering one action-chunk request at a time."""

    # Modal class parameters must be simple scalars, so the robot description (camera
    # keys, joint names, normalization stats) travels as a JSON blob built locally.
    model_id: str = modal.parameter(default=DEFAULT_MODEL)
    setup_json: str = modal.parameter(default="{}")

    @modal.enter()
    def load(self):
        """Cold-start work: build the policy for THIS robot's feature shapes and load weights."""
        setup = json.loads(self.setup_json)
        self.camera_keys: list[str] = setup["camera_keys"]
        stats = setup["stats"]
        state_dim = len(setup["joints"])

        # The checkpoint's own config (model sizes, chunk length, flow-matching steps),
        # then re-pointed at the SO-101's feature shapes. π₀.₅ pads state/action out to
        # 32 dims internally, so a 6-DoF arm is a legal input even for the base model.
        cfg = PreTrainedConfig.from_pretrained(self.model_id)
        cfg.pretrained_path = self.model_id
        cfg.device = "cuda"
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            **{
                f"observation.images.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in self.camera_keys
            },
        }
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,))}

        # Preflight: touch the smallest file in the gated tokenizer repo. This is the one
        # step that fails on a missing or unaccepted licence, and it costs a few KB, so
        # do it before pulling ~7 GB of weights that would then go unused.
        from huggingface_hub import hf_hub_download

        # An empty HF_TOKEN is worse than none: huggingface_hub's get_token() maps "" to
        # None, so the request goes out unauthenticated and the Hub answers 401 rather
        # than the 403 you would get from a real token lacking the licence.
        if not os.environ.get("HF_TOKEN", "").strip():
            raise RuntimeError(
                "No HF_TOKEN reached the container, so the gated tokenizer cannot be read.\n"
                "  export HF_TOKEN=hf_... in the shell you run `modal run` from, then retry.\n"
                "  (the token is forwarded from your local environment; there is no Modal secret to create)"
            )
        try:
            hf_hub_download(TOKENIZER_REPO, "tokenizer_config.json")
        except Exception as exc:
            unauthorized = "401" in str(exc)
            reason = (
                "the token was rejected — check it is a valid READ token and not expired"
                if unauthorized
                else f"the token works but has no access — accept the Gemma licence at "
                f"https://huggingface.co/{TOKENIZER_REPO}"
            )
            raise RuntimeError(
                f"Cannot read the gated tokenizer repo {TOKENIZER_REPO}: {reason}.\n"
                "  A fine-grained token also needs 'Read access to contents of all public gated repos'.\n"
                f"  Original error: {exc}"
            ) from exc
        print(f"[modal] gated tokenizer {TOKENIZER_REPO} is readable")

        print(f"[modal] loading {self.model_id} ...")
        t0 = time.perf_counter()
        self.policy = PI05Policy.from_pretrained(self.model_id, config=cfg)
        self.policy.eval()

        # Normalization/tokenization live in the processor pipeline, not the model:
        #   preprocess : rename → batch → normalize (quantiles) → discretize state →
        #                tokenize the prompt with PaliGemma's tokenizer → to CUDA
        #   postprocess: unnormalize the predicted actions → back to CPU
        # `stats` are the q01/q99 quantiles we derived from the arm's calibration.
        self.preprocess, self.postprocess = make_pre_post_processors(
            cfg,
            dataset_stats={k: {s: torch.tensor(v) for s, v in st.items()} for k, st in stats.items()},
        )
        self.chunk_size = cfg.chunk_size
        print(f"[modal] ready in {time.perf_counter() - t0:.1f}s (chunk size {self.chunk_size})")

        # Persist the freshly downloaded weights so the next cold start skips the download.
        hf_cache.commit()

    @modal.method()
    def describe(self) -> dict:
        """Cheap call used to warm the container and report what it loaded."""
        return {"model": self.model_id, "chunk_size": self.chunk_size, "cameras": self.camera_keys}

    @modal.method()
    def predict_chunk(self, state: list[float], frames: dict[str, bytes], task: str, n_steps: int) -> list[list[float]]:
        """One observation in, `n_steps` joint-position targets out.

        `frames` are JPEG bytes keyed by camera name (encoded locally — a raw 720p
        frame is ~2.7 MB, the JPEG is ~20 KB, and that difference is most of the
        round-trip time). π₀.₅ resizes-with-pad to 224x224 itself, so aspect ratio is
        preserved and we do not have to match a training resolution here.
        """
        batch: dict = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": task}
        for key, jpeg in frames.items():
            img = np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.float32) / 255.0
            # HWC -> CHW, float in [0, 1], which is the layout the policy expects.
            batch[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1)

        t0 = time.perf_counter()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(self.preprocess(batch))
            actions = self.postprocess(actions)
        print(f"[modal] '{task}' -> chunk in {(time.perf_counter() - t0) * 1e3:.0f} ms")

        # (batch, chunk, action_dim) -> the first n_steps rows of the single batch entry.
        return actions[0, :n_steps].to(torch.float32).tolist()


# --------------------------------------------------------------------------------------
# Local half: config, calibration, cameras, and the control loop
# --------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Read config/default.yaml — the same file the rest of so101_assist reads."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _lerobot_calibration(arm_json: Path, out_dir: Path, robot_id: str) -> Path:
    """Translate this repo's calibration file into the file LeRobot's SO101Follower reads.

    Same five fields per motor (id / drive_mode / homing_offset / range_min / range_max),
    with one rename: this repo calls joint 3 `elbow`, LeRobot calls it `elbow_flex`.
    """
    calib = json.loads(arm_json.read_text())
    if "elbow" in calib and "elbow_flex" not in calib:
        calib["elbow_flex"] = calib.pop("elbow")

    missing = [j for j in JOINTS if j not in calib]
    if missing:
        raise SystemExit(f"{arm_json} is missing calibration for {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{robot_id}.json"
    out_path.write_text(json.dumps({j: calib[j] for j in JOINTS}, indent=2))
    return out_path


def _stats_from_calibration(calib_path: Path) -> dict:
    """Derive the policy's normalization quantiles from the arm's calibrated travel.

    π₀.₅ normalizes state and action with quantiles: q01 -> -1, q99 -> +1. LeRobot
    reports the five arm joints in degrees measured from the centre of their calibrated
    range — `(raw - (range_min + range_max) / 2) * 360 / (resolution - 1)` — and the
    gripper on 0..100 across its own range. Mapping each joint's physical travel onto
    [-1, 1] is therefore just its half-range in those same units.
    """
    calib = json.loads(calib_path.read_text())
    lo, hi = [], []
    for joint in JOINTS:
        entry = calib[joint]
        if joint == "gripper":
            lo.append(0.0)
            hi.append(100.0)
            continue
        half_deg = (entry["range_max"] - entry["range_min"]) / 2 * 360 / (ENCODER_RESOLUTION - 1)
        lo.append(-half_deg)
        hi.append(half_deg)

    # State and action share the same space here: π₀.₅ predicts absolute joint targets.
    # q01/q99 are what π₀.₅ normalizes with; mean/std and min/max are filled in too so the
    # same stats still work if --model points at a checkpoint configured for those modes.
    mean = [(a + b) / 2 for a, b in zip(lo, hi)]
    std = [max((b - a) / 4, 1e-6) for a, b in zip(lo, hi)]
    per_feature = {"q01": lo, "q99": hi, "min": lo, "max": hi, "mean": mean, "std": std}
    return {"observation.state": dict(per_feature), "action": dict(per_feature)}


def _stats_from_dataset(repo_id: str, fallback: dict) -> dict:
    """Optional --stats-dataset path: reuse the statistics of a real SO-101 dataset.

    Preferable to the calibration-derived numbers if you have (or can point at) a
    dataset recorded on an arm like yours, because it reflects the part of the workspace
    actually used rather than the full mechanical range.
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    stats = {}
    for key in ("observation.state", "action"):
        src = meta.stats.get(key, {})
        if "q01" in src and "q99" in src:
            stats[key] = {"q01": list(map(float, src["q01"])), "q99": list(map(float, src["q99"]))}
        elif "mean" in src and "std" in src:
            # Older datasets ship mean/std only; +-2 sigma is a reasonable stand-in for
            # the 1st/99th percentiles.
            mean, std = np.asarray(src["mean"], float), np.asarray(src["std"], float)
            stats[key] = {"q01": (mean - 2 * std).tolist(), "q99": (mean + 2 * std).tolist()}
        else:
            print(f"  ! {repo_id} has no usable stats for {key}; using calibration-derived values")
            stats[key] = fallback[key]
    return stats


def _encode_frame(frame, max_width: int = 320) -> bytes:
    """Downscale (aspect preserved) and JPEG-encode one camera frame for the wire."""
    import cv2

    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    # lerobot hands us RGB; cv2.imencode expects BGR.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _camera_configs(cfg: dict) -> dict:
    """Build one LeRobot camera config per entry in the config's `cameras:` block.

    Insertion order matters downstream: π₀.₅ assigns images to camera *slots* by
    position, so the first camera here becomes the model's third-person "base" view and
    the rest become wrist views.
    """
    from lerobot.cameras.opencv import OpenCVCameraConfig

    extra = {}
    if sys.platform == "win32":
        # lerobot defaults to the ANY backend, which on Windows means MSMF — slow to probe
        # and it enumerates devices differently than DirectShow. so101_assist's own capture
        # path (perception/camera.py) forces DSHOW there, so match it.
        try:
            from lerobot.cameras import Cv2Backends

            extra["backend"] = Cv2Backends.DSHOW
        except ImportError:  # older lerobot without the backend selector
            pass

    cameras = {}
    for name, spec in cfg.get("cameras", {}).items():
        # A camera entry may name either an `index:` or a `path:`. Prefer a path on Linux:
        # each UVC camera registers two /dev/video nodes (capture + metadata), so index N
        # is not device N, and the mapping shifts on replug/reboot. A stable
        # /dev/v4l/by-id/... symlink always opens the same physical camera.
        if "path" in spec:
            target = str(spec["path"])
        elif "index" in spec:
            target = int(spec["index"])
        else:
            raise SystemExit(f"camera '{name}' needs either an `index:` or a `path:` in the config")
        # `fourcc: MJPG` matters when two USB cameras share one controller (a Pi, say):
        # raw YUYV at 640x480x30 is ~18 MB/s per camera and they will starve each other,
        # while MJPG is roughly a tenth of that.
        if spec.get("fourcc"):
            extra_cam = {**extra, "fourcc": str(spec["fourcc"])}
        else:
            extra_cam = extra
        cameras[name] = OpenCVCameraConfig(
            index_or_path=target, width=spec["width"], height=spec["height"], fps=spec["fps"], **extra_cam
        )

    if not cameras:
        raise SystemExit("No cameras in config/default.yaml — π₀.₅ needs at least one image stream.")
    return cameras


# VIDIOC_QUERYCAP = _IOR('V', 0, struct v4l2_capability). The struct is 104 bytes:
# driver[16], card[32], bus_info[32], version, capabilities, device_caps, reserved[3].
_VIDIOC_QUERYCAP = (2 << 30) | (104 << 16) | (ord("V") << 8) | 0
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_META_CAPTURE = 0x00800000
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def _v4l2_capability(device: Path):
    """Ask a V4L2 node what it is, without opening a stream.

    Returns (driver, card, caps) or None. This is the cheap, definitive classification:
    trying to *capture* from a node to find out what it is makes non-camera nodes (a
    Pi's ISP and codec blocks, for instance) block for ten seconds each in select().
    """
    import fcntl
    import struct

    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)

    driver, card, _bus, _version, caps, device_caps = struct.unpack("<16s32s32sIII", bytes(buf[:92]))
    # device_caps describes THIS node; capabilities describes the whole physical device
    # (every node it owns), so a metadata node would otherwise look like a camera.
    effective = device_caps if caps & _V4L2_CAP_DEVICE_CAPS else caps
    decode = lambda raw: raw.split(b"\x00")[0].decode("utf-8", "replace")  # noqa: E731
    return decode(driver), decode(card), effective


# VIDIOC_ENUM_FMT = _IOWR('V', 2, struct v4l2_fmtdesc)          -> 64 bytes
# VIDIOC_ENUM_FRAMESIZES = _IOWR('V', 74, struct v4l2_frmsizeenum) -> 44 bytes
_VIDIOC_ENUM_FMT = (3 << 30) | (64 << 16) | (ord("V") << 8) | 2
_VIDIOC_ENUM_FRAMESIZES = (3 << 30) | (44 << 16) | (ord("V") << 8) | 74
# VIDIOC_ENUM_FRAMEINTERVALS = _IOWR('V', 75, struct v4l2_frmivalenum) -> 52 bytes
_VIDIOC_ENUM_FRAMEINTERVALS = (3 << 30) | (52 << 16) | (ord("V") << 8) | 75
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_FRMSIZE_TYPE_DISCRETE = 1
_V4L2_FRMIVAL_TYPE_DISCRETE = 1


def _frame_rates(fd: int, pixelformat: int, width: int, height: int, limit: int = 12):
    """Discrete frame rates this format/size supports, highest first."""
    import fcntl
    import struct

    rates = []
    for index in range(limit):
        buf = bytearray(52)
        struct.pack_into("<IIII", buf, 0, index, pixelformat, width, height)
        try:
            fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMEINTERVALS, buf)
        except OSError:
            break
        interval_type = struct.unpack_from("<I", buf, 16)[0]
        if interval_type != _V4L2_FRMIVAL_TYPE_DISCRETE:
            break
        numerator, denominator = struct.unpack_from("<II", buf, 20)
        if numerator:  # an interval of n/d seconds is d/n frames per second
            rates.append(round(denominator / numerator))
    return sorted(set(rates), reverse=True)


def _v4l2_modes(device: Path, max_sizes: int = 8):
    """What this camera can be asked for: [(fourcc, [(w, h, [fps, ...]), ...]), ...].

    lerobot validates width, height AND fps on connect and raises if the camera won't
    deliver exactly what was requested, so a working config has to name a combination
    that appears here. The frame rate is per resolution and per pixel format: the same
    camera will often do 30fps as MJPG and only 10fps as raw YUYV at the same size.
    """
    import fcntl
    import struct

    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return []

    def fourcc_name(value: int) -> str:
        return "".join(chr((value >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip()

    formats = []
    try:
        for fmt_index in range(32):  # bounded: enumeration ends with EINVAL long before this
            buf = bytearray(64)
            struct.pack_into("<II", buf, 0, fmt_index, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, buf)
            except OSError:
                break  # EINVAL = no more formats
            pixelformat = struct.unpack_from("<I", buf, 44)[0]

            sizes = []
            for size_index in range(max_sizes):
                sbuf = bytearray(44)
                struct.pack_into("<II", sbuf, 0, size_index, pixelformat)
                try:
                    fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMESIZES, sbuf)
                except OSError:
                    break
                size_type, width, height = struct.unpack_from("<III", sbuf, 8)
                if size_type != _V4L2_FRMSIZE_TYPE_DISCRETE:
                    break  # continuous/stepwise: no fixed list worth printing
                sizes.append((width, height, _frame_rates(fd, pixelformat, width, height)))
            formats.append((fourcc_name(pixelformat), sizes))
    finally:
        os.close(fd)
    return formats


def _classify(device: Path):
    """(is_capture, human description) for one V4L2 node."""
    info = _v4l2_capability(device)
    if info is None:
        return False, "cannot query (permission, or not a V4L2 device)"
    driver, _card, caps = info
    if caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE):
        return True, f"video capture (driver: {driver})"
    if caps & _V4L2_CAP_META_CAPTURE:
        return False, f"metadata node (driver: {driver})"
    return False, f"not a camera (driver: {driver})"


def _v4l2_nodes(sysfs_root=Path("/sys/class/video4linux"), dev_root=Path("/dev")):
    """Enumerate this machine's V4L2 video nodes, newest kernel naming assumed.

    Returns (device_path, human_name, [stable by-id symlinks]) per node, in numeric
    order. The human name comes from sysfs, which is what tells you *which physical
    camera* a node belongs to — the thing an index can never tell you.
    """
    if not sysfs_root.is_dir():
        return []

    # Reverse /dev/v4l/by-id/* -> the /dev/videoN each symlink resolves to.
    by_id: dict[Path, list[Path]] = {}
    by_id_dir = dev_root / "v4l" / "by-id"
    if by_id_dir.is_dir():
        for link in sorted(by_id_dir.iterdir()):
            by_id.setdefault(link.resolve(), []).append(link)

    def node_number(path: Path) -> int:
        try:
            return int(path.name.removeprefix("video"))
        except ValueError:
            return 1 << 30  # unparseable names sort last rather than crashing

    nodes = []
    for entry in sorted(sysfs_root.glob("video*"), key=node_number):
        device = dev_root / entry.name
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            name = "?"
        nodes.append((device, name, by_id.get(device, [])))
    return nodes


def list_cameras() -> None:
    """Print which physical camera sits on which node, and the config line to use.

    Opens each node with the V4L2 backend *explicitly*. That matters: with OpenCV's
    default ANY backend a refused node falls through to another backend that may return
    frames from a different device entirely, which is exactly how a metadata node can
    look like a working camera.
    """
    import cv2

    nodes = _v4l2_nodes()
    if not nodes:
        print("No /sys/class/video4linux — this listing is Linux-only.")
        print("Elsewhere, use `python soarm-test.py --preview` and look at the saved images.")
        return

    print("V4L2 nodes (capability-checked, then opened with the V4L2 backend):\n")
    usable = []
    for device, name, links in nodes:
        is_capture, description = _classify(device)
        if not is_capture:
            # Don't try to read from it: a non-camera node can block for ten seconds.
            print(f"  {str(device):<16} {name[:34]:<36} {description}")
            continue

        capture = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        ok, frame = (False, None)
        if capture.isOpened():
            ok, frame = capture.read()
        capture.release()

        if ok and frame is not None:
            h, w = frame.shape[:2]
            status = f"CAPTURE {w}x{h}"
            usable.append((device, name, links))
        else:
            status = "capture device, but no frame (in use by another process?)"
        print(f"  {str(device):<16} {name[:34]:<36} {status}")
        for link in links:
            print(f"  {'':<16} stable path: {link}")
        # The config must name one of these exactly, or lerobot refuses to connect.
        for fourcc, sizes in _v4l2_modes(device):
            if sizes:
                listed = "  ".join(
                    f"{w}x{h}@{'/'.join(str(r) for r in rates) if rates else '?'}" for w, h, rates in sizes
                )
                print(f"  {'':<16} {fourcc:<5} {listed}")

    if not usable:
        print("\nNothing captured. If the arm script or another viewer is running, stop it and retry.")
        return

    print("\nPut these under `cameras:` in config/default.yaml — ORDER IS THE POLICY SLOT:")
    print("the first entry becomes the model's base/third-person view, the rest are wrist views.\n")
    for role, (device, name, links) in zip(("overhead", "wrist"), usable):
        target = links[0] if links else device
        print(f"  {role}: {{ path: {target}, width: 640, height: 480, fps: 30 }}   # {name}")
    if len(usable) > 2:
        print(f"\n({len(usable)} capture devices found; the two above are just the first two.)")


def _model_view(frame, size: int = 224):
    """Reproduce exactly what π₀.₅ sees, so you can look at it before trusting it.

    Mirrors lerobot's `resize_with_pad_torch`: scale by max(w, h) / 224 so nothing is
    cropped, then centre-pad the short axis with black. Returns the 224x224 image and
    the (width, height) actually filled by real pixels.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    ratio = max(w / size, h / size)
    rw, rh = int(w / ratio), int(h / ratio)  # int(), not round() — matches lerobot
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h = (size - rh) // 2
    pad_w = (size - rw) // 2
    out = np.zeros((size, size, 3), dtype=frame.dtype)
    out[pad_h:pad_h + rh, pad_w:pad_w + rw] = resized
    return out, (rw, rh)


def preview(config: str = "config/default.yaml", out_dir: str = "camera-preview") -> None:
    """Grab one frame per camera and write both the raw frame and the model's-eye view.

    Runs standalone (`python soarm-test.py --preview`) — no Modal account, no GPU, no
    container start, and the arm is never touched. Use it to aim the cameras first.
    """
    import cv2

    from lerobot.cameras.opencv import OpenCVCamera

    cfg = _load_yaml(REPO_ROOT / config)
    out = REPO_ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)

    for slot, (name, cam_cfg) in enumerate(_camera_configs(cfg).items()):
        role = "base / third-person" if slot == 0 else "wrist"
        # Name the physical device, not just the index — an index alone cannot tell you
        # whether you opened the camera you meant.
        target = cam_cfg.index_or_path
        device = Path(str(target)) if not isinstance(target, int) else Path(f"/dev/video{target}")
        hardware = next((n for d, n, _ in _v4l2_nodes() if d == device.resolve()), None)
        label = f"{target}" + (f"  [{hardware}]" if hardware else "")
        print(f"\n[{name}] {label} -> policy slot {slot} ({role})")
        camera = OpenCVCamera(cam_cfg)
        try:
            camera.connect()
            frame = camera.read()  # RGB, as lerobot hands it to the policy
        except Exception as exc:
            print(f"  FAILED to open: {exc}")
            print(f"    the config asks for {cam_cfg.width}x{cam_cfg.height} @ {cam_cfg.fps}fps"
                  + (f" ({cam_cfg.fourcc})" if getattr(cam_cfg, "fourcc", None) else ""))
            print("    run `python soarm-test.py --list-cameras` for the modes this camera has,")
            print("    and set width/height in config/default.yaml to one of them exactly.")
            continue
        finally:
            if camera.is_connected:
                camera.disconnect()

        h, w = frame.shape[:2]
        view, (rw, rh) = _model_view(frame)
        used = (rw * rh) / (224 * 224)
        print(f"  captured {w}x{h}  ->  {rw}x{rh} inside a 224x224 frame "
              f"({used:.0%} real pixels, {1 - used:.0%} black padding)")
        if used < 0.7:
            print("  ! a lot of the model's input is padding — a 4:3 or square mode would use more of it")

        for suffix, image in ((f"{name}_raw.png", frame), (f"{name}_model_view.png", view)):
            cv2.imwrite(str(out / suffix), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"\nwrote to {out}/ — open the *_model_view.png files: that is all π₀.₅ gets to see.")


def _build_robot(cfg: dict, port: str, calib_dir: Path, robot_id: str, max_relative_target: float,
                 max_gripper_step: float):
    """Construct the LeRobot SO101Follower from config/default.yaml + our calibration."""
    # lerobot >= 0.6 moved the SO-100/SO-101 followers into a shared module.
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # lerobot <= 0.5
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    cameras = _camera_configs(cfg)

    robot_cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        calibration_dir=calib_dir,
        cameras=cameras,
        use_degrees=True,  # matches the units our normalization stats are computed in
        # Per-command safety cap: every joint target is clipped to this far from where the
        # joint actually is, which makes the cap a SPEED limit — cap x fps deg/s. This is
        # the main guard against an out-of-distribution action chunk throwing the arm
        # around. The gripper is on its own 0..100 scale, not degrees, and needs a larger
        # allowance to close in a usable time, so it gets its own value.
        max_relative_target={
            **{joint: max_relative_target for joint in JOINTS if joint != "gripper"},
            "gripper": max_gripper_step,
        },
        # Follow the project's rule that torque-off is an explicit act, so exiting the
        # script never drops whatever the gripper is holding.
        disable_torque_on_disconnect=False,
    )
    return SO101Follower(robot_cfg), list(cameras)


def _run_episode(robot, server, task: str, cameras: list[str], limits, fps: int, exec_steps: int,
                 max_steps: int, dry_run: bool) -> None:
    """Observe → request a chunk → play it out, until the step budget or Ctrl-C."""
    period = 1.0 / fps
    step = 0

    while step < max_steps:
        obs = robot.get_observation()
        state = [float(obs[f"{joint}.pos"]) for joint in JOINTS]
        frames = {cam: _encode_frame(obs[cam]) for cam in cameras}

        t0 = time.perf_counter()
        chunk = server.predict_chunk.remote(state, frames, task, exec_steps)
        rtt = time.perf_counter() - t0
        print(f"  step {step:4d}  round trip {rtt * 1e3:5.0f} ms  executing {len(chunk)} actions")

        # Play the chunk open-loop at a fixed rate. The arm keeps moving while nothing
        # is being asked of the network, which is the whole point of chunked inference.
        deadline = time.perf_counter()
        for action in chunk:
            targets = {
                f"{joint}.pos": min(max(float(value), lo), hi)  # never command outside calibrated travel
                for joint, value, (lo, hi) in zip(JOINTS, action, limits)
            }
            if dry_run:
                if step % fps == 0:  # once a second, so the log stays readable
                    print("    " + "  ".join(f"{k.split('.')[0]}={v:7.2f}" for k, v in targets.items()))
            else:
                robot.send_action(targets)

            step += 1
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))

    print(f"  reached the {max_steps}-step budget; stopping and holding position.")


@app.local_entrypoint()
def main(
    task: str = "",
    port: str = "",
    fps: int = 30,
    exec_steps: int = 25,
    max_steps: int = 900,
    max_relative_target: float = 0.75,
    max_gripper_step: float = 3.0,
    model: str = DEFAULT_MODEL,
    stats_dataset: str = "",
    robot_id: str = "so101_assist",
    config: str = "config/default.yaml",
    calibration: str = "config/calibration/arm.json",
    dry_run: bool = False,
):
    """Runs on your machine. Owns the arm and the cameras; calls Modal for the policy.

    Args:
        task: instruction to run, e.g. "pick up the glasses". Omit to be prompted, and
            to keep being prompted for the next command after each run.
        port: serial port of the follower arm; defaults to `arm.port` in the config.
        fps: rate at which chunk actions are streamed to the servos.
        exec_steps: how many actions of each 50-step chunk to execute before re-planning.
            Fewer = more reactive and more GPU calls; more = smoother but staler.
        max_steps: hard cap on commands per instruction (900 = 30 s at 30 fps).
        max_relative_target: per-command arm-joint cap in degrees. This is a speed limit:
            the arm can move at most `max_relative_target * fps` deg/s. The default
            matches the 0.4 rad/s shoulder cap this project uses elsewhere.
        max_gripper_step: per-command gripper cap, in units of its 0..100 range.
        model: any π₀.₅ checkpoint on the Hub; point it at your finetune when you have one.
        stats_dataset: optional SO-101 LeRobot dataset whose statistics to normalize with,
            instead of the ones derived from your calibration file.
        dry_run: run the full loop but print joint targets instead of sending them.
    """
    cfg_path = REPO_ROOT / config
    calib_src = REPO_ROOT / calibration
    for path in (cfg_path, calib_src):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    cfg = _load_yaml(cfg_path)
    port = port or cfg.get("arm", {}).get("port", "/dev/ttyACM0")

    # Checked here, locally, because the alternative is a multi-minute container start
    # that loads 7 GB of weights and only then discovers it cannot read the tokenizer.
    if not os.environ.get("HF_TOKEN", "").strip():
        raise SystemExit(
            "HF_TOKEN is not set in this shell.\n"
            f"π₀.₅'s tokenizer comes from {TOKENIZER_REPO}, a gated repo, so a token is required:\n"
            f"  1. sign in at https://huggingface.co/{TOKENIZER_REPO} and accept the Gemma licence\n"
            "  2. create a READ token at https://huggingface.co/settings/tokens\n"
            "  3. export HF_TOKEN=hf_...   then re-run\n"
            "Verify it works before starting a container:\n"
            "  python -c \"from huggingface_hub import hf_hub_download as d; "
            f"print(d('{TOKENIZER_REPO}', 'tokenizer_config.json'))\""
        )

    print("=" * 78)
    print("SO-101 + pi0.5 (LeRobot policy, Modal inference)")
    print(f"  model         {model}")
    print(f"  arm port      {port}")
    print(f"  calibration   {calib_src.relative_to(REPO_ROOT)}")
    print(f"  safety        <= {max_relative_target} deg/command x {fps} fps "
          f"= {max_relative_target * fps:.0f} deg/s max joint speed"
          + ("  [DRY RUN — no motion]" if dry_run else ""))
    print(f"                gripper <= {max_gripper_step}/command ({max_gripper_step * fps:.0f} units/s of 0..100)")
    print("  NOTE: no pi0.5 checkpoint is trained on the SO-101. Motion from the base")
    print("        model is unvalidated — try --dry-run first and stay near the power switch.")
    print("=" * 78)

    # --- calibration: this repo's arm.json -> LeRobot's calibration file -------------
    calib_dir = REPO_ROOT / ".cache" / "lerobot_calibration"
    calib_path = _lerobot_calibration(calib_src, calib_dir, robot_id)
    stats = _stats_from_calibration(calib_path)
    if stats_dataset:
        print(f"  normalization from dataset {stats_dataset}")
        stats = _stats_from_dataset(stats_dataset, stats)

    # --- hardware --------------------------------------------------------------------
    robot, cameras = _build_robot(cfg, port, calib_dir, robot_id, max_relative_target, max_gripper_step)
    print(f"connecting to the arm and {len(cameras)} camera(s): {', '.join(cameras)} ...")
    # calibrate=False keeps connect() non-interactive; we push the calibration ourselves
    # rather than letting lerobot prompt for a fresh range-of-motion sweep.
    # NOTE: connect() -> configure() runs its motor writes inside `bus.torque_disabled()`,
    # which RE-ENABLES torque on exit. So the arm is live and holding position from here
    # on, in --dry-run too. That is also why send_action actually moves it.
    try:
        robot.connect(calibrate=False)
    except TypeError:  # older lerobot without the keyword
        robot.connect()
    if not robot.is_calibrated:
        print("writing calibration to the servos ...")
        robot.bus.write_calibration(robot.calibration)

    # Calibrated travel, used to clip every commanded target (belt and braces on top of
    # max_relative_target and the servos' own firmware limits).
    limits = list(zip(stats["observation.state"]["q01"], stats["observation.state"]["q99"]))

    # --- policy ----------------------------------------------------------------------
    server = Pi05Server(
        model_id=model,
        setup_json=json.dumps({"camera_keys": cameras, "joints": JOINTS, "stats": stats}),
    )
    print("starting the Modal container (first run downloads ~7 GB of weights) ...")
    info = server.describe.remote()
    print(f"policy ready: chunk size {info['chunk_size']}, cameras {info['cameras']}")

    # --- command loop ----------------------------------------------------------------
    try:
        while True:
            instruction = task or input("\ncommand (empty to quit) > ").strip()
            if not instruction:
                break
            print(f"running: {instruction!r}   (Ctrl-C to stop the arm)")
            try:
                _run_episode(robot, server, instruction, cameras, limits, fps, exec_steps,
                             max_steps, dry_run)
            except KeyboardInterrupt:
                print("\n  stopped — the arm holds its current position.")
            if task:
                break  # --task was given: run it once, then exit
    finally:
        # Leaves torque on (see disable_torque_on_disconnect above); the arm holds.
        robot.disconnect()
        print("disconnected. Torque is still enabled — power down or torque-off explicitly.")


if __name__ == "__main__":
    # `modal run` never reaches this branch; it calls main() above. Running the file with
    # plain python is reserved for --preview, which needs neither Modal nor the arm.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="save what each camera sees, and what the model sees")
    parser.add_argument("--list-cameras", action="store_true", help="show which physical camera is on which device node")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--out-dir", default="camera-preview")
    args = parser.parse_args()
    if args.list_cameras:
        list_cameras()
        raise SystemExit(0)
    if not args.preview:
        sys.exit(
            "Run this with Modal so the policy gets a GPU:\n"
            '    modal run soarm-test.py --task "pick up the glasses"\n'
            "Or check your cameras first, without a GPU or the arm:\n"
            "    python soarm-test.py --preview"
        )
    preview(config=args.config, out_dir=args.out_dir)
