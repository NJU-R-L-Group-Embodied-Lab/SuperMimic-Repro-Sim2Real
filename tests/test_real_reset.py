"""Real-log resets (--env.real-reset-file / --env.real-reset-prob) and the opt-in DR events (--env.dr-push-*,
--env.dr-knee-effort-scale). All default off."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/library/lean35kfs_only.pt"
CLIP = "shanshen_duosuomao_lean35kfs_pad2"
GPU = pytest.mark.skipif(not torch.cuda.is_available() or not LIB.exists(), reason="needs CUDA + data/library/lean35kfs_only.pt")


def _proj_grav(q):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r.T @ np.array([0.0, 0.0, -1.0])


def test_off_by_default():
    from supermimic.config import EnvCfg
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg

    e = EnvCfg()
    assert e.real_reset_prob == 0.0 and e.real_reset_file is None and e.dr_push_prob == 0.0 and e.dr_knee_effort_scale is None
    a = SuperMimicEnvCfgArgs(npz_files=[])
    assert a.dr_push_prob == 0.0 and a.dr_knee_effort_scale is None
    c = MultiClipMotionCommandCfg(npz_files=[], anchor_body_name="torso_link", body_names=("pelvis",))
    assert c.real_reset_prob == 0.0 and c.real_reset_file is None
    with pytest.raises(ValueError):
        EnvCfg(real_reset_prob=0.2)


def test_build_state_foot_on_floor_and_static_contact():
    """Builder: lowest foot point at the clearance, heading = reference yaw, roll/pitch kept, and the foot-velocity
    variant leaves the stance contact point at rest."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts/data"))
    import build_real_reset_states as B
    from supermimic.data.motion_convert import load_g1_model

    m = load_g1_model()
    d = mujoco.MjData(m)
    feet = B.foot_capsules(m)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-0.3, 0.3, 29)
        dq = rng.uniform(-3, 3, 29)
        quat = B._qnorm(np.array([1.0, *rng.uniform(-0.15, 0.15, 2), rng.uniform(-1, 1)]))
        gyro = rng.uniform(-3, 3, 3)
        ref_quat = B._qnorm(np.array([np.cos(0.7), 0.0, 0.0, np.sin(0.7)]))
        st = B.build_state(m, d, feet, q, dq, quat, gyro, np.array([0.3, -0.2, 0.8]), ref_quat, np.array([0.5, 0.1, 0.2]))
        assert abs(B.yaw_of(st["root_quat"]) - 1.4) < 1e-5
        assert np.allclose(_proj_grav(st["root_quat"]), _proj_grav(quat), atol=1e-5)  # tilt unchanged by re-yaw
        assert np.allclose(st["root_pos"][:2], [0.3, -0.2])
        # Re-simulate the stored state: lowest foot point at 0.5 mm, contact-point velocity ~0 with root_lin_vel_foot.
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = st["root_pos"], st["root_quat"], q
        d.qvel[:3], d.qvel[3:6], d.qvel[6:] = st["root_lin_vel_foot"], gyro, dq
        mujoco.mj_forward(m, d)
        side = "left" if st["stance"] == 0 else "right"
        z, pt, g = B.lowest_point(m, d, feet[side])
        assert abs(z - 5e-4) < 1e-5
        other = B.lowest_point(m, d, feet["right" if side == "left" else "left"])[0]
        assert other >= z - 1e-6
        jac = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jac, None, pt, m.geom_bodyid[g])
        assert np.linalg.norm(jac @ d.qvel) < 1e-4


def _write_file(tmp_path, n=6):
    """Small synthetic reset file on the lean35kfs clip (q near the reference, distinctive gyro / velocities)."""
    pack = torch.load(LIB, map_location="cpu", weights_only=False)
    ci = list(pack["names"]).index(CLIP)
    ref = pack["ref"].float().numpy()[pack["starts"][ci]:]
    rf = np.arange(700, 700 + n)
    r = ref[rf]
    yaw = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))  # noqa: E731
    quat = np.stack([[np.cos(yaw(q) / 2), 0, 0, np.sin(yaw(q) / 2)] for q in r[:, 3:7]]).astype(np.float32)
    out = dict(
        joint_pos=r[:, 7:36].astype(np.float32), joint_vel=np.full((n, 29), 0.5, np.float32),
        root_pos=np.c_[r[:, 0:2], np.full(n, 0.78)].astype(np.float32), root_quat=quat,
        root_ang_vel_b=np.tile([0.3, -0.2, 4.0], (n, 1)).astype(np.float32),
        root_lin_vel_clip=np.tile([0.1, 0.0, 0.0], (n, 1)).astype(np.float32),
        root_lin_vel_foot=np.tile([-0.4, 0.0, 0.0], (n, 1)).astype(np.float32),
        stance=np.zeros(n, np.int8), clip=np.array(CLIP), log_frame=(rf + 1).astype(np.int32), ref_frame=rf.astype(np.int32),
        source=np.zeros(n, np.int16), source_names=np.array(["synthetic"]),
    )
    p = tmp_path / "real.npz"
    np.savez(p, **out)
    return p, out


def _env(cmd_kwargs=None, **kw):
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    args = SuperMimicEnvCfgArgs(npz_files=[str(LIB)], compact_library=True, num_envs=16, replay_capacity=64, seed=5,
                                action_noise_std=0.0, obs_noise=False, extra_command_kwargs=cmd_kwargs or {}, **kw)
    return make_env(args, device="cuda:0")


_NO_PERT = dict(real_reset_dz=(0.0, 0.0), real_reset_dvz=0.0, real_reset_tilt_deg=0.0, real_reset_joint_pos_noise=0.0, real_reset_joint_vel_noise=0.0)


@GPU
def test_real_reset_writes_state_and_command_frame(tmp_path):
    p, st = _write_file(tmp_path)
    env = _env(dict(real_reset_prob=1.0, real_reset_file=str(p), real_reset_vel="foot", **_NO_PERT))
    try:
        obs0, _ = env.reset()
        cmd = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        assert bool(cmd.real_ep.all())
        lib = cmd.library
        start = int(lib.clip_start[[c.name for c in lib.clips].index(CLIP)])
        F = cmd.frame - start - 1  # one command update after the reset
        j = (F - 700).cpu().numpy()
        assert ((j >= 0) & (j < len(st["ref_frame"]))).all(), F
        jp = robot.data.joint_pos.cpu().numpy()
        lim = robot.data.joint_pos_limits.cpu().numpy()  # real-log states: hard range, not the soft limits
        exp_q = np.clip(st["joint_pos"][j], lim[..., 0], lim[..., 1])
        assert np.allclose(jp, exp_q, atol=1e-4)
        assert np.allclose(robot.data.joint_vel.cpu().numpy(), 0.5, atol=1e-4)
        pos = (robot.data.root_link_pos_w - env.scene.env_origins).cpu().numpy()
        assert np.allclose(pos, st["root_pos"][j], atol=1e-4)
        quat = robot.data.root_link_quat_w.cpu().numpy()
        assert np.allclose(np.abs((quat * st["root_quat"][j]).sum(-1)), 1.0, atol=1e-5)
        assert np.allclose(robot.data.root_link_ang_vel_b.cpu().numpy(), [0.3, -0.2, 4.0], atol=1e-4)
        assert np.allclose(robot.data.root_link_lin_vel_w.cpu().numpy(), [-0.4, 0.0, 0.0], atol=1e-4)
        # Policy history refilled from the first post-reset observation, last action 0 (as every reset).
        obs = obs0["policy"]
        h = obs[:, : 29 * 10].reshape(16, 10, 29)
        assert torch.allclose(h, h[:, :1].expand_as(h))
        assert torch.all(env.action_manager.action == 0)
        # Sampler statistics untouched by real-start episodes.
        visits0 = cmd.bin_visits.clone()
        for _ in range(5):
            env.step(torch.zeros(16, 29, device=env.device))
        cmd._resample_command(torch.arange(16, device=env.device))
        assert torch.equal(cmd.bin_visits, visits0)
    finally:
        env.close()


@GPU
def test_real_reset_perturbation_bounds(tmp_path):
    p, st = _write_file(tmp_path)
    kw = dict(real_reset_prob=0.5, real_reset_file=str(p), real_reset_vel="mix")
    env = _env(kw)
    try:
        env.reset()
        cmd = env.command_manager.get_term("motion")
        n = 4000
        s = cmd.sample_real_states(n)
        j = s["index"].cpu().numpy()
        dz = s["root_pos"][:, 2].cpu().numpy() - st["root_pos"][j, 2]
        assert dz.min() >= -0.01 - 1e-6 and dz.max() <= 0.01 + 1e-6 and dz.std() > 0.004
        lin = s["root_lin_vel"].cpu().numpy()
        foot = s["foot_vel"].cpu().numpy()
        assert 0.4 < foot.mean() < 0.6
        assert np.allclose(lin[foot, 0], -0.4) and np.allclose(lin[~foot, 0], 0.1)
        dvz = lin[:, 2]
        assert np.abs(dvz).max() <= 0.2 + 1e-6 and dvz.std() > 0.08
        g = np.stack([_proj_grav(q) for q in s["root_quat"].cpu().numpy()])
        g0 = np.stack([_proj_grav(q) for q in st["root_quat"][j]])
        ang = np.degrees(np.arccos(np.clip((g * g0).sum(-1), -1, 1)))
        assert ang.max() <= 3.0 * math.sqrt(2) + 0.05 and ang.mean() > 1.0
        # gyro kept in the (perturbed) body frame
        from mjlab.utils.lab_api.math import quat_apply_inverse

        w_b = quat_apply_inverse(s["root_quat"], s["root_ang_vel"]).cpu().numpy()
        assert np.allclose(w_b, [0.3, -0.2, 4.0], atol=1e-4)
        dq = s["joint_pos"].cpu().numpy() - st["joint_pos"][j]
        assert np.abs(dq).max() <= 0.02 + 1e-6
        # Mixed with ordinary resets at prob 0.5.
        frac = []
        for _ in range(20):
            cmd._resample_command(torch.arange(16, device=env.device))
            frac.append(cmd.real_ep.float().mean().item())
        assert 0.3 < np.mean(frac) < 0.7
        # Forced (evaluation) resets never use real states.
        cmd.forced_frame[:] = 10
        cmd._resample_command(torch.arange(16, device=env.device))
        assert not cmd.real_ep.any() and torch.all(cmd.frame == 10)
    finally:
        env.close()


@GPU
def test_pelvis_push_window_and_knee_effort(tmp_path):
    from supermimic.envs.mdp.events import joint_ctrl_ids

    p, _ = _write_file(tmp_path)
    env = _env(dict(real_reset_prob=1.0, real_reset_file=str(p), **_NO_PERT), dr_push_prob=1.0, dr_push_frames=(690, 700),
               dr_push_steps=(100, 100), dr_push_force=(50.0, 50.0, 20.0), dr_push_torque=(5.0, 5.0, 5.0),
               dr_push_clips=("lean35",), dr_knee_effort_scale=(0.6, 0.8))
    try:
        env.reset()
        robot = env.scene["robot"]
        ev = env.event_manager.get_term_cfg("pelvis_push").func
        pel = robot.indexing.body_ids[robot.body_names.index("pelvis")]
        env.step(torch.zeros(16, 29, device=env.device))
        assert bool(ev.on.all())  # all frames 700.. are inside a window starting at <= 700 and lasting 100 steps
        xf = env.sim.data.xfrc_applied[:][:, pel].cpu()
        assert torch.allclose(xf[:, :3], ev.force.cpu()) and torch.allclose(xf[:, 3:], ev.torque.cpu())
        assert (ev.force.abs() <= torch.tensor([50.0, 50.0, 20.0], device=env.device)).all() and ev.force.abs().sum() > 0
        # Outside the window (start pushed past the clip): wrench zeroed.
        ev.start[:] = 5000
        env.step(torch.zeros(16, 29, device=env.device))
        assert not ev.on.any() and torch.all(env.sim.data.xfrc_applied[:][:, pel] == 0)
        # Knee effort: per-env scale in [0.6, 0.8] of nominal; hip roll (same actuator group) untouched.
        fr = env.sim.model.actuator_forcerange[:]
        nom = env.sim.get_default_field("actuator_forcerange").to(env.device)
        knee = joint_ctrl_ids(robot, ("left_knee_joint", "right_knee_joint"))
        hip = joint_ctrl_ids(robot, ("left_hip_roll_joint",))
        s = fr[:, knee, 1] / nom[knee, 1]
        assert s.min() >= 0.6 - 1e-5 and s.max() <= 0.8 + 1e-5 and s.std() > 0.01
        assert torch.allclose(fr[:, hip, 1], nom[hip, 1].expand(16, 1))
    finally:
        env.close()


@GPU
def test_real_reset_keeps_waist_roll_overshoot_with_wide_range(tmp_path):
    """Real states are clipped to the hard range: with --env.waist-roll-range 0.72 (soft limits nominal) a measured
    waist roll of -0.685 is written as is; reference resets still use the soft limits; nominal range clips at 0.52."""
    from supermimic.data.motion_convert import G1_JOINT_NAMES

    p, st = _write_file(tmp_path)
    z = dict(np.load(p))
    ir = G1_JOINT_NAMES.index("waist_roll_joint")
    z["joint_pos"][:, ir] = -0.685
    np.savez(p, **z)
    for rng, expect in ((0.72, -0.685), (None, -0.52)):
        env = _env(dict(real_reset_prob=1.0, real_reset_file=str(p), **_NO_PERT), waist_roll_range=rng)
        try:
            env.reset()
            robot = env.scene["robot"]
            j = [n.split("/")[-1] for n in robot.joint_names].index("waist_roll_joint")
            soft = robot.data.soft_joint_pos_limits[0, j].cpu().numpy()
            assert abs(soft[0] + 0.52 * 0.9) < 1e-4  # nominal soft limits in both cases
            assert np.allclose(robot.data.joint_pos[:, j].cpu().numpy(), expect, atol=1e-4)
            cmd = env.command_manager.get_term("motion")
            cmd.cfg.real_reset_prob = 0.0  # reference resets: soft-limit clamp unchanged
            cmd._resample_command(torch.arange(16, device=env.device))
            q = robot.data.joint_pos[:, j].cpu().numpy()
            assert (q >= soft[0] - 1e-5).all() and (q <= soft[1] + 1e-5).all()
        finally:
            env.close()
