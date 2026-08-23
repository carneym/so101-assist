"""Cartesian velocity controller.

Consumes CartesianVelocity (from the bus or called directly), maps it
to joint velocities, integrates to position targets, and hands them
to the driver. All motion — autonomous approach AND operator jog —
flows through this single controller, so safety limits are enforced
in exactly one place.

Velocity -> joints mapping: the linear (x/y/z) command is solved with
a damped pseudo-inverse of the position Jacobian restricted to the
FIRST THREE joints (shoulder_pan/shoulder_lift/elbow); wrist flex and
roll rates pass through directly to their joints. Solving linear
velocity over all 5 joints would recruit the wrist to translate the
end effector, fighting the operator's direct wrist commands — with
this 3+2 split, "translate" never surprises by reorienting the
gripper and vice versa. The trade-off (EE orientation drifts slightly
during pure translation) is fine for human-in-the-loop jogging.

cmd.elbow is a direct joint-rate passthrough on top of that solve
(operator preference: stick up/down in ELBOW mode drives the
elbow). Unlike the wrist channels it is fence-checked — the elbow
swings the whole forearm, so its predicted EE motion must respect the
workspace fence like any Cartesian command. Don't command nonzero
cmd.linear and cmd.elbow simultaneously from the same jog mode: the
linear solve also uses the elbow joint, and the two sum.

Setpoint integration: the controller keeps its OWN persistent joint
setpoint and integrates velocity into it (setpoint += dq*dt), rather
than re-anchoring to the measured position each tick. Anchoring to
the measurement looks equivalent but fails on real hardware: with a
per-tick step smaller than a servo's stiction/deadband (~0.03 rad on
the gravity-loaded wrist_flex, vs 0.02 rad/tick at the wrist cap and
25 Hz), the servo never moves and the error can never accumulate —
the joint is simply frozen. The persistent setpoint lets error build
until stiction breaks. SETPOINT_LEASH_RAD bounds how far the setpoint
may lead the measured position (anti-windup when a joint is
physically blocked), and stays under the driver's 5-degree per-call
clamp so writes are never silently truncated.

Safety, layered (each independent of the others):
- hard velocity caps (MAX_LINEAR_MPS / MAX_WRIST_RADPS) applied after
  any upstream blending or scaling
- workspace fence zeroes outward linear components (safety.py)
- load monitor trips -> STOP: zero velocity, hold position, torque ON
  (see safety.py for why torque stays on); resume() requires an
  explicit call plus the monitor being reset
- the setpoint leash bounds tracking error buildup; the driver's own
  per-call step clamp remains underneath as backstop
"""
from __future__ import annotations

import numpy as np

from ..messages import CartesianVelocity
from .driver import JOINT_NAMES, SO101Driver
from .kinematics import JOINT_LIMITS_RAD, fk, jacobian
from .poses import floor_guarded_step
from .safety import LoadMonitor, WorkspaceFence

# Hard caps, applied after any blending. Tune conservatively.
MAX_LINEAR_MPS = 0.06      # 6 cm/s
MAX_WRIST_RADPS = 0.5
MAX_ELBOW_RADPS = 0.5
MAX_SHOULDER_RADPS = 0.4   # slower: these joints move the entire arm

_DAMPING = 1e-3
_ELBOW = 2   # JOINT_NAMES index of the elbow joint

# How far (rad / gripper fraction) the integrated setpoint may lead the
# measured position — see "Setpoint integration" in the module docstring.
# This also caps the position error a servo sees, and therefore the
# torque it applies: too tight and a gravity-loaded joint (shoulder
# lift) can't generate enough torque to raise the arm. Loosen via
# config arm.setpoint_leash_rad if the arm can't lift its own weight.
SETPOINT_LEASH_RAD = 0.2
SETPOINT_LEASH_GRIPPER = 0.1


class CartesianController:
    def __init__(
        self,
        driver: SO101Driver,
        fence: WorkspaceFence | None = None,
        load_monitor: LoadMonitor | None = None,
        max_linear_mps: float = MAX_LINEAR_MPS,
        max_wrist_radps: float = MAX_WRIST_RADPS,
        max_elbow_radps: float = MAX_ELBOW_RADPS,
        max_shoulder_radps: float = MAX_SHOULDER_RADPS,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        setpoint_leash_rad: float = SETPOINT_LEASH_RAD,
    ) -> None:
        self.driver = driver
        self.fence = fence
        self.load_monitor = load_monitor
        self.max_linear_mps = max_linear_mps
        self.max_wrist_radps = max_wrist_radps
        self.max_elbow_radps = max_elbow_radps
        self.max_shoulder_radps = max_shoulder_radps
        self.setpoint_leash_rad = setpoint_leash_rad
        # Operating range the operator may drive — defaults to the
        # URDF-modeled limits; override per-joint (config
        # arm.joint_limits_deg) where the physical arm safely exceeds
        # the model. Self-collision beyond the model is guarded by the
        # load monitor (collision -> STOP) and the per-call step
        # clamps, not by these numbers.
        self.joint_limits = dict(JOINT_LIMITS_RAD) if joint_limits is None else dict(joint_limits)
        self.stopped = False
        self._q_setpoint: np.ndarray | None = None
        self._gripper_setpoint: float | None = None
        # Channels the fence blocked / joints pinned at their software
        # limit on the most recent tick, for UI/debug display — a
        # silently frozen joint is indistinguishable from a hardware
        # fault to the operator otherwise.
        self.last_fence_blocks: list[str] = []
        self.last_limit_clips: list[str] = []
        # End-effector position (base frame, meters) from the most recent
        # tick, for UI/tuning readout. Updated every tick INCLUDING while
        # stopped, so the readout stays live during a STOP hold.
        self.last_ee_xyz: np.ndarray | None = None
        # Measured joint position/load from the most recent tick. Kept
        # so UI can show live load without issuing its OWN read_state()
        # — a second reader at video rate would double serial traffic on
        # the servo bus and compete with the control loop for it.
        # Updated every tick INCLUDING while stopped (same as
        # last_ee_xyz), so a load warning stays visible during a hold.
        self.last_joint_pos: np.ndarray | None = None
        self.last_joint_load: np.ndarray | None = None

    def stop(self) -> None:
        """STOP: hold current position, torque stays ON."""
        self.stopped = True

    def resume(self) -> None:
        """Explicit operator action; also clears a tripped load monitor.

        Setpoints are re-seeded from the measured position on the next
        tick so resuming never commands a jump to a stale setpoint.
        """
        if self.load_monitor is not None:
            self.load_monitor.reset()
        self.stopped = False
        self._q_setpoint = None
        self._gripper_setpoint = None

    def tick(self, cmd: CartesianVelocity, dt: float) -> bool:
        """Run one control step. Returns True if motion was commanded,
        False if stopped (STOP state or load monitor trip)."""
        joint_pos, joint_load, gripper_pos = self.driver.read_state()
        self.last_ee_xyz = fk(joint_pos)[:3, 3]
        self.last_joint_pos = joint_pos
        self.last_joint_load = joint_load

        if self.load_monitor is not None and self.load_monitor.feed(joint_load):
            self.stopped = True
        if self.stopped:
            return False

        linear = np.array(cmd.linear, dtype=float)
        speed = np.linalg.norm(linear)
        if speed > self.max_linear_mps:
            linear *= self.max_linear_mps / speed
        wrist = np.clip(np.array(cmd.wrist, dtype=float), -self.max_wrist_radps, self.max_wrist_radps)
        elbow = float(np.clip(cmd.elbow, -self.max_elbow_radps, self.max_elbow_radps))
        shoulder = np.clip(
            np.array(cmd.shoulder, dtype=float), -self.max_shoulder_radps, self.max_shoulder_radps
        )

        J = jacobian(joint_pos)
        self.last_fence_blocks = []
        if self.fence is not None:
            ee_xyz = self.last_ee_xyz
            clamped_linear = self.fence.clamp(ee_xyz, linear)
            if not np.array_equal(clamped_linear, linear):
                self.last_fence_blocks.append("linear")
            linear = clamped_linear
            # The elbow joint-space command swings the whole forearm, so
            # it must not bypass the fence: if its predicted EE velocity
            # has any outward component at a fence face, zero the
            # channel entirely (conservative — no partial "sliding along
            # the fence" for joint-space motion).
            #
            # The SHOULDER channels are deliberately NOT fence-checked —
            # operator decision (2026-07-26): the any-outward-component
            # rule blocked legitimate shoulder motion well before the
            # boundary, and SHOULDER mode exists precisely as the
            # direct-drive fallback. Guarding there falls to the load
            # monitor, step clamps, and joint operating limits.
            if elbow != 0.0:
                elbow_ee_vel = J[:3, _ELBOW] * elbow
                if not np.array_equal(self.fence.clamp(ee_xyz, elbow_ee_vel), elbow_ee_vel):
                    elbow = 0.0
                    self.last_fence_blocks.append("elbow")

        # 3+2 split (see module docstring): linear via first three joints
        Jv = J[:3, :3]
        dq_linear = Jv.T @ np.linalg.solve(Jv @ Jv.T + _DAMPING * np.eye(3), linear)
        dq_linear[0] += shoulder[0]
        dq_linear[1] += shoulder[1]
        dq_linear[_ELBOW] += elbow

        dq = np.concatenate([dq_linear, wrist])

        # Persistent setpoint integration (see module docstring), leashed
        # to the measured position and clipped to joint limits.
        if self._q_setpoint is None:
            self._q_setpoint = joint_pos.copy()
        if self._gripper_setpoint is None:
            self._gripper_setpoint = gripper_pos

        q_setpoint = self._q_setpoint + dq * dt
        q_setpoint = np.clip(
            q_setpoint, joint_pos - self.setpoint_leash_rad, joint_pos + self.setpoint_leash_rad
        )
        self.last_limit_clips = [
            name
            for name, q, rate in zip(JOINT_NAMES, q_setpoint, dq)
            if rate != 0.0 and not (self.joint_limits[name][0] < q < self.joint_limits[name][1])
        ]
        q_setpoint = np.array(
            [np.clip(q, *self.joint_limits[name]) for name, q in zip(JOINT_NAMES, q_setpoint)]
        )
        gripper_setpoint = self._gripper_setpoint + cmd.gripper * dt
        gripper_setpoint = float(
            np.clip(
                gripper_setpoint,
                max(0.0, gripper_pos - SETPOINT_LEASH_GRIPPER),
                min(1.0, gripper_pos + SETPOINT_LEASH_GRIPPER),
            )
        )
        self._q_setpoint = q_setpoint
        self._gripper_setpoint = gripper_setpoint

        self.driver.write_joint_targets(q_setpoint, gripper_setpoint)
        return True

    def step_to_joints(
        self,
        target_rad: np.ndarray,
        target_gripper: float,
        max_step_rad: float,
        max_gripper_step: float,
        z_floor: float | None = None,
    ) -> bool:
        """Drive one bounded step toward a joint-space target. Returns
        True once the arm has arrived.

        This is the pose-move primitive, deliberately routed through the
        controller rather than straight to the driver so an autonomous
        move keeps the same protections a jogged one has: the load
        monitor still trips on a collision, a STOP still holds, joint
        limits still clip, and last_* stay fresh for the HUD.

        `z_floor` (meters, base frame) guards the ONE Cartesian
        constraint that matters on the way: joint-space interpolation
        sags, so a path between two poses both clear of the table can
        still drag the gripper through it. Each step is corrected to
        stay above the floor (see poses.floor_guarded_step). The rest of
        the workspace fence still does not apply — it constrains
        Cartesian velocity, while a pose is a joint-space target taught
        by physically placing the arm there.

        The floor guard is local: it cannot route around a blockage, so
        a move whose straight path is obstructed slides along the floor
        and stops making progress. Callers must watch for that stall
        instead of waiting for an arrival that never comes.

        Raises RuntimeError if called while stopped — resuming is an
        explicit operator action, never a side effect of a pose move.
        """
        joint_pos, joint_load, gripper_pos = self.driver.read_state()
        self.last_ee_xyz = fk(joint_pos)[:3, 3]
        self.last_joint_pos = joint_pos
        self.last_joint_load = joint_load

        if self.load_monitor is not None and self.load_monitor.feed(joint_load):
            self.stopped = True
        if self.stopped:
            raise RuntimeError("Arm is stopped — pose move aborted.")

        target = np.array(
            [np.clip(q, *self.joint_limits[name]) for name, q in zip(JOINT_NAMES, target_rad)]
        )
        delta = target - joint_pos
        arrived = bool(np.max(np.abs(delta)) <= max_step_rad)
        next_q = target if arrived else joint_pos + np.clip(delta, -max_step_rad, max_step_rad)

        if z_floor is not None:
            next_q = floor_guarded_step(joint_pos, next_q, z_floor)
            next_q = np.array(
                [np.clip(q, *self.joint_limits[name]) for name, q in zip(JOINT_NAMES, next_q)]
            )
            # The guard may have pushed the step off the target, so
            # arrival is decided on where we are actually going.
            arrived = bool(np.max(np.abs(target - next_q)) <= 1e-3)

        gripper_delta = float(np.clip(target_gripper - gripper_pos, -max_gripper_step, max_gripper_step))
        next_gripper = float(np.clip(gripper_pos + gripper_delta, 0.0, 1.0))

        # Keep the jog setpoints in step with where the pose move left
        # the arm, so resuming the stick afterwards doesn't snap back to
        # a setpoint from before the move.
        self._q_setpoint = next_q.copy()
        self._gripper_setpoint = next_gripper

        self.driver.write_joint_targets(next_q, next_gripper)
        return arrived
