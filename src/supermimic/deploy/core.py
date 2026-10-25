"""Hardware-agnostic policy runtime: the one code path shared by sim2sim and the real-robot runner.

Depends only on numpy (+ onnxruntime for inference) so it runs in the robot-side venv without torch/mjlab.
Everything that must match training lives here (contract verified against the training env, see
docs/DEPLOY_G1.md#deployment-conventions-identical-to-training-checked-by-the-exportparity-tests):

* 50 Hz; joint order = MJCF joint order = Unitree G1 motor index 0..28.
* obs frame (93) = [q - q0 (29), dq (29), pelvis gyro (3), pelvis projected gravity (3), last raw action (29)];
  history 10, flattened term-major, oldest -> newest; the first frame after start back-fills all 10 slots.
* goal (680) = goal table rows k+1 .. k+10 (clamped at the clip's last frame) when the robot is at frame k.
* q_target = a * scale + q0, dq_des = 0, tau_ff = 0, gains = training gains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HISTORY = 10
FUTURE = 10
NUM_JOINTS = 29
# (name, dim) of the policy observation terms, in training order.
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)^T v for a unit quaternion (w, x, y, z)."""
    w, xyz = q_wxyz[0], q_wxyz[1:]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(q_wxyz, np.array([0.0, 0.0, -1.0]))


@dataclass
class DeployConfig:
    """Robot constants dumped from the training model by supermimic.deploy.export (deploy_config.json)."""

    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    joint_lower: np.ndarray  # hard MJCF range
    joint_upper: np.ndarray
    velocity_limit: np.ndarray
    soft_limit_factor: float = 0.9
    control_dt: float = 0.02

    @classmethod
    def load(cls, path: str | Path) -> "DeployConfig":
        d = json.loads(Path(path).read_text())
        arr = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            joint_names=list(d["joint_names"]),
            default_joint_pos=arr("default_joint_pos"),
            action_scale=arr("action_scale"),
            kp=arr("kp"),
            kd=arr("kd"),
            effort_limit=arr("effort_limit"),
            joint_lower=arr("joint_lower"),
            joint_upper=arr("joint_upper"),
            velocity_limit=arr("velocity_limit"),
            soft_limit_factor=float(d.get("soft_limit_factor", 0.9)),
            control_dt=float(d.get("control_dt", 0.02)),
        )

    @property
    def soft_lower(self) -> np.ndarray:
        mid, half = 0.5 * (self.joint_lower + self.joint_upper), 0.5 * (self.joint_upper - self.joint_lower)
        return mid - self.soft_limit_factor * half

    @property
    def soft_upper(self) -> np.ndarray:
        mid, half = 0.5 * (self.joint_lower + self.joint_upper), 0.5 * (self.joint_upper - self.joint_lower)
        return mid + self.soft_limit_factor * half


@dataclass
class Clip:
    """Precomputed per-clip reference (supermimic.deploy.export goal tables)."""

    name: str
    goal: np.ndarray  # (T, 68) goal features, fp32
    joint_pos: np.ndarray  # (T, 29) reference joint positions (start / end poses)
    proj_grav: np.ndarray  # (T, 3) reference pelvis projected gravity (abort check)

    @classmethod
    def load(cls, path: str | Path) -> "Clip":
        z = np.load(path)
        return cls(name=str(z["name"]), goal=z["goal"].astype(np.float32), joint_pos=z["joint_pos"].astype(np.float64), proj_grav=z["proj_grav"].astype(np.float64))

    @property
    def num_frames(self) -> int:
        return self.goal.shape[0]


@dataclass
class SafetyCfg:
    # Training never clamps targets: 61-74% of training steps command past the 0.9 soft limit, and clamping
    # them was out of distribution in sim2sim (2.35x tracking error, falls). Off by default.
    clamp_soft_limits: bool = False
    clamp_torque: bool = True
    # MuJoCo saturates force at every 5 ms substep, while this clamps the target once per 20 ms step from the
    # sampled state (stricter as the joint moves), so allow headroom; lower it (e.g. 0.5) for first hardware runs.
    torque_clamp_factor: float = 1.5
    max_tilt_err: float = 0.35  # rad between robot and reference pelvis gravity directions
    max_vel_frac: float = 0.8  # of the joint velocity limit
    max_abs_action: float = 50.0  # raw network output; |a| this large means something is badly wrong


@dataclass
class StepResult:
    q_target: np.ndarray  # (29,) clamped joint targets to send
    action: np.ndarray  # (29,) raw network output (fed back as last action)
    frame: int  # reference frame the robot should be at after this step
    done: bool  # clip finished (reached its last frame)
    abort: str | None  # reason to drop to damping, or None
    n_clamped: int  # how many joints were clamped this step


class OnnxPolicy:
    def __init__(self, path: str | Path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # Spinning worker threads steal CPU from the 500 Hz command thread; block instead.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        out = self.sess.run(["action"], {"obs": obs[None].astype(np.float32), "goal": goal[None].astype(np.float32)})
        return out[0][0].astype(np.float64)


@dataclass
class DeployCore:
    cfg: DeployConfig
    policy: object  # callable (obs[930], goal[680]) -> action[29]
    clip: Clip
    safety: SafetyCfg = field(default_factory=SafetyCfg)

    def __post_init__(self):
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, start_frame: int = 0) -> None:
        """Call right before the first policy step; the robot is assumed to be at `start_frame`."""
        self.frame = int(start_frame)
        self.last_action = np.zeros(NUM_JOINTS)
        self.hist: np.ndarray | None = None  # (HISTORY, 93), oldest first

    def obs_frame(self, q: np.ndarray, dq: np.ndarray, gyro: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.concatenate([q - self.cfg.default_joint_pos, dq, gyro, projected_gravity(quat_wxyz), self.last_action])

    def observation(self, q, dq, gyro, quat_wxyz) -> np.ndarray:
        """Push one frame into the history and return the flattened term-major (930,) observation."""
        f = self.obs_frame(np.asarray(q, float), np.asarray(dq, float), np.asarray(gyro, float), np.asarray(quat_wxyz, float))
        if self.hist is None:
            self.hist = np.repeat(f[None], HISTORY, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], f[None]], axis=0)
        parts, o = [], 0
        for _, d in OBS_TERMS:
            parts.append(self.hist[:, o : o + d].reshape(-1))  # term-major: all 10 frames of one term
            o += d
        return np.concatenate(parts)

    def goal(self) -> np.ndarray:
        last = self.clip.num_frames - 1
        idx = np.minimum(self.frame + 1 + np.arange(FUTURE), last)
        return self.clip.goal[idx].reshape(-1)

    # ------------------------------------------------------------------ step

    def switch_clip(self, clip: "Clip", frame: int = 0) -> None:
        """Continue with another reference without resetting the observation history or the last action
        (standby -> motion -> standby hand-offs stay seamless for the policy)."""
        self.clip = clip
        self.frame = int(np.clip(frame, 0, clip.num_frames - 1))

    def step(self, q, dq, gyro, quat_wxyz, advance: bool = True) -> StepResult:
        """advance=False keeps the reference frame fixed (standby on a static reference); done stays False."""
        q, dq, gyro = np.asarray(q, float), np.asarray(dq, float), np.asarray(gyro, float)
        quat_wxyz = np.asarray(quat_wxyz, float)
        abort = self._check(q, dq, gyro, quat_wxyz)
        n = np.linalg.norm(quat_wxyz)
        obs = self.observation(q, dq, gyro, quat_wxyz / n if np.isfinite(n) and n > 0.5 else np.array([1.0, 0.0, 0.0, 0.0]))
        a = np.asarray(self.policy(obs, self.goal()), float)
        if not np.all(np.isfinite(a)):
            abort = abort or "non-finite action"
            a = np.zeros(NUM_JOINTS)
        elif np.max(np.abs(a)) > self.safety.max_abs_action:
            abort = abort or f"action magnitude {np.max(np.abs(a)):.1f}"
        q_t, n_clamped = self.targets(a, q, dq)
        self.last_action = a
        if advance:
            self.frame += 1
        done = advance and self.frame >= self.clip.num_frames - 1
        return StepResult(q_target=q_t, action=a, frame=self.frame, done=done, abort=abort, n_clamped=n_clamped)

    def targets(self, a: np.ndarray, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, int]:
        c = self.cfg
        raw = a * c.action_scale + c.default_joint_pos
        q_t = raw.copy()
        if self.safety.clamp_torque:
            # PD with dq_des = 0: tau = kp (q_t - q) - kd dq, kept within +-effort (what the sim actuator clamps).
            tau = self.safety.torque_clamp_factor * c.effort_limit
            lo = q + (-tau + c.kd * dq) / c.kp
            hi = q + (tau + c.kd * dq) / c.kp
            q_t = np.clip(q_t, lo, hi)
        if self.safety.clamp_soft_limits:
            q_t = np.clip(q_t, c.soft_lower, c.soft_upper)
        return q_t, int(np.count_nonzero(np.abs(q_t - raw) > 1e-9))

    def _check(self, q: np.ndarray, dq: np.ndarray, gyro: np.ndarray, quat_wxyz: np.ndarray) -> str | None:
        s = self.safety
        # An uninitialised IMU (all-zero quaternion) would otherwise read as perfectly upright.
        if not (np.all(np.isfinite(quat_wxyz)) and np.all(np.isfinite(gyro))):
            return "non-finite IMU"
        if abs(np.linalg.norm(quat_wxyz) - 1.0) > 0.05:
            return f"bad IMU quaternion norm {np.linalg.norm(quat_wxyz):.3f}"
        g_robot = projected_gravity(quat_wxyz / np.linalg.norm(quat_wxyz))
        g_ref = self.clip.proj_grav[min(self.frame, self.clip.num_frames - 1)]
        cosang = float(np.clip(np.dot(g_robot, g_ref) / (np.linalg.norm(g_robot) * np.linalg.norm(g_ref) + 1e-9), -1.0, 1.0))
        tilt = float(np.arccos(cosang))
        if tilt > s.max_tilt_err:
            return f"tilt error {np.degrees(tilt):.0f} deg"
        over = np.abs(dq) > s.max_vel_frac * self.cfg.velocity_limit
        if np.any(over):
            j = int(np.argmax(np.abs(dq) / self.cfg.velocity_limit))
            return f"joint velocity {self.cfg.joint_names[j]} {dq[j]:.1f} rad/s"
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return "non-finite state"
        return None


# Goal feature layout per frame (68): q-q0 [0:29] | dq [29:58] | proj grav [58:61] | ang vel body [61:64] |
# lin vel heading [64:67] | height [67].
_GOAL_RATE_COLS = np.r_[29:58, 61:67]


def static_clip(clip: Clip, frame: int) -> Clip:
    """A standing-still reference at one frame of `clip`: same pose, gravity and height, all rates zero.
    Used for policy standby before a motion starts (frame 0) and after it ends (last frame)."""
    frame = int(np.clip(frame, 0, clip.num_frames - 1))
    g = clip.goal[frame].copy()
    g[_GOAL_RATE_COLS] = 0.0
    return Clip(name=f"{clip.name}@{frame}_static", goal=np.repeat(g[None], 2, 0),
                joint_pos=np.repeat(clip.joint_pos[frame][None], 2, 0), proj_grav=np.repeat(clip.proj_grav[frame][None], 2, 0))


def interpolate_pose(q_from: np.ndarray, q_to: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Smooth (cosine) interpolation used to bring the robot to a clip's first-frame pose."""
    s = float(np.clip(t / max(duration, 1e-6), 0.0, 1.0))
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    return q_from + s * (q_to - q_from)
