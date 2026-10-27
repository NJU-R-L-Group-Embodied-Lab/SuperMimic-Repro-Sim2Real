"""Saturating-IMU observation model (DEVIATION from the paper when --env.imu-sat-prob > 0, default off).

The real G1 pelvis IMU gyro saturates at +-500 deg/s (8.727 rad/s) and, over the ~1 s of a fast spin, the onboard
attitude is a pure integrator of that clipped gyro, so the projected gravity the policy reads drifts 10-20 deg after a
yaw spin (seen on the real robot on lean35, frames 668-685 / 722-729). This module models exactly that sensor path:

    w_meas = clip(w, -limit, limit)                                     (elementwise, body frame)
    g_est <- normalize(R(-w_meas dt) g_est)                             (exact rotation of the body-frame vector)
    g_est <- normalize(g_est + k dt (g_true - g_est))                   (optional complementary correction, k >= 0)

With an unclipped gyro and k = 0 the recursion is the body-frame gravity of MuJoCo's own free-joint integration
(q' = q exp(w dt) with the post-step w), i.e. g_est == the true projected gravity up to float error.

Only the policy observation changes: the world model, the replay buffer, the critic and the rewards all keep the true
state (``base_ang_vel`` / ``projected_gravity`` are not part of the 71-d state; ``StateIO`` reads qpos/qvel).

Simulator side (``ImuSatModel``): per env and per episode, with probability ``prob`` the env uses the model and draws
k ~ U[0, accel_k_max]; g_est is propagated after every physics substep from the free-joint angular velocity (qvel[3:6],
body frame, the rate MuJoCo just integrated the orientation with) and reset to the true gravity at episode start.
Imagination side: ``supermimic.imagination.rollout.imagine`` with an ``ImuImagCfg`` (per 50 Hz step, from the
world-model predicted angular velocity, initial random tilt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, quat_from_rotvec

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Column ranges of the IMU quantities inside the 93-d policy observation (world_model.heads.OBS_TERMS).
OBS_ANG_VEL = slice(58, 61)
OBS_GRAVITY = slice(61, 64)


@dataclass
class ImuImagCfg:
    """IMU model in imagined rollouts (copied from EnvCfg by the training loop)."""

    prob: float = 0.0
    gyro_limit: float = 8.727  # rad/s (500 deg/s)
    accel_k_max: float = 0.5  # 1/s
    init_tilt_deg: float = 8.0


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.sqrt((v * v).sum(-1, keepdim=True).clamp_min(1e-12))


def clip_gyro(w: torch.Tensor, limit: float) -> torch.Tensor:
    return w.clamp(-limit, limit)


def rotate_gravity(g: torch.Tensor, w: torch.Tensor, dt: float) -> torch.Tensor:
    """Body-frame vector after the body rotated by w dt (body frame): exp(-[w]x dt) g, renormalised. Differentiable."""
    return _normalize(quat_apply(quat_from_rotvec(-w * dt), g))


def imu_gravity_step(g_est: torch.Tensor, w: torch.Tensor, g_true: torch.Tensor, k: torch.Tensor | float, limit: float, dt: float) -> torch.Tensor:
    """One IMU update: integrate the clipped gyro, then complementary-correct toward g_true with gain k [1/s]."""
    g = rotate_gravity(g_est, clip_gyro(w, limit), dt)
    if isinstance(k, torch.Tensor):
        k = k[..., None]
    return _normalize(g + (k * dt) * (g_true - g))


def random_tilt(g: torch.Tensor, max_deg: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Rotate each g (…,3) by U[0, max_deg] about a random axis perpendicular to it (= a random horizontal world axis)."""
    r = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=generator)
    axis = _normalize(torch.cross(g, r, dim=-1))
    ang = torch.rand(g.shape[:-1] + (1,), device=g.device, dtype=g.dtype, generator=generator) * math.radians(max_deg)
    return _normalize(quat_apply(quat_from_rotvec(axis * ang), g))


def imu_observation(obs: torch.Tensor, g_est: torch.Tensor, flag: torch.Tensor, limit: float) -> torch.Tensor:
    """93-d policy observation with the IMU columns of flagged rows replaced by (clipped gyro, g_est)."""
    m = flag[..., None]
    ang = torch.where(m, clip_gyro(obs[..., OBS_ANG_VEL], limit), obs[..., OBS_ANG_VEL])
    grav = torch.where(m, g_est, obs[..., OBS_GRAVITY])
    return torch.cat([obs[..., : OBS_ANG_VEL.start], ang, grav, obs[..., OBS_GRAVITY.stop :]], dim=-1)


# ---------------------------------------------------------------------------------------------------- simulator side


class ImuSatModel:
    """Per-env IMU state of the mjlab env, propagated after every physics substep (hooks ``env.sim.step``)."""

    def __init__(self, env: ManagerBasedRlEnv, prob: float, gyro_limit: float, accel_k_max: float):
        self.env = env
        n, dev = env.num_envs, env.device
        self.prob, self.limit, self.k_max = float(prob), float(gyro_limit), float(accel_k_max)
        self.dt = float(env.physics_dt)
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.k = torch.zeros(n, device=dev)
        self.g_est = torch.zeros(n, 3, device=dev)
        self.g_est[:, 2] = -1.0
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.resample(None)
        sim = env.sim
        orig_step = sim.step

        def step_with_imu(*args, **kwargs):
            out = orig_step(*args, **kwargs)
            self.substep()
            return out

        sim.step = step_with_imu

    def resample(self, env_ids) -> None:
        ids = slice(None) if env_ids is None or isinstance(env_ids, slice) else env_ids
        m = self.active[ids].shape[0]
        dev = self.active.device
        self.active[ids] = torch.rand(m, device=dev) < self.prob
        self.k[ids] = torch.rand(m, device=dev) * self.k_max
        self.fresh[ids] = True  # g_est <- true gravity at the next observation (after the reset state is written)

    def _true_gravity(self) -> torch.Tensor:
        quat = self.env.sim.data.qpos[:][:, 3:7]
        g = torch.zeros_like(quat[:, :3])
        g[:, 2] = -1.0
        return quat_apply_inverse(quat, g)

    @torch.no_grad()
    def substep(self) -> None:
        w = self.env.sim.data.qvel[:][:, 3:6]  # free-joint angular velocity, body frame (post-step)
        g_new = imu_gravity_step(self.g_est, w, self._true_gravity(), self.k, self.limit, self.dt)
        self.g_est = torch.where(self.fresh[:, None], self.g_est, g_new)

    def gravity(self, true_g: torch.Tensor) -> torch.Tensor:
        if self.fresh.any():
            self.g_est = torch.where(self.fresh[:, None], true_g, self.g_est)
            self.fresh[:] = False
        return torch.where(self.active[:, None], self.g_est, true_g)

    def gyro(self, w: torch.Tensor) -> torch.Tensor:
        return torch.where(self.active[:, None], clip_gyro(w, self.limit), w)


def get_imu_model(env: ManagerBasedRlEnv, prob: float, gyro_limit: float, accel_k_max: float) -> ImuSatModel:
    model = getattr(env, "_sm_imu_model", None)
    if model is None:
        model = ImuSatModel(env, prob, gyro_limit, accel_k_max)
        env._sm_imu_model = model
    return model


class _ImuObsTerm:
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self.model = get_imu_model(env, p["prob"], p["gyro_limit"], p["accel_k_max"])

    def reset(self, env_ids=None) -> None:
        pass


class ImuBaseAngVel(_ImuObsTerm):
    """``base_ang_vel`` observation through the IMU model (clipped gyro for flagged envs)."""

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self.model.gyro(env.scene["robot"].data.root_link_ang_vel_b)


class ImuProjectedGravity(_ImuObsTerm):
    """``projected_gravity`` observation through the IMU model (integrated g_est for flagged envs). Owns the
    per-episode resampling (one reset call per env reset)."""

    def reset(self, env_ids=None) -> None:
        self.model.resample(env_ids)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self.model.gravity(env.scene["robot"].data.projected_gravity_b)
