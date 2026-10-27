"""Saturating-IMU observation model (--env.imu-sat-prob): off by default, exact gravity integration, clipping,
complementary correction, reset behaviour in the mjlab env, and the imagined-rollout path."""

import math
from pathlib import Path

import pytest
import torch

from supermimic.envs.imu_model import (
    OBS_ANG_VEL,
    OBS_GRAVITY,
    ImuImagCfg,
    clip_gyro,
    imu_gravity_step,
    imu_observation,
    random_tilt,
    rotate_gravity,
)
from supermimic.kinematics.rotations import projected_gravity, quat_from_rotvec, quat_mul, quat_normalize

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
CLIPS = ["walk1_subject1", "dance1_subject1"]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
H = 10  # policy history


def _angle(a, b):
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    return torch.arccos(c.clamp(-1, 1))


def test_off_by_default():
    from supermimic.config import EnvCfg
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.imagination.policy_trainer import PolicyTrainerCfg

    assert EnvCfg().imu_sat_prob == 0.0 and SuperMimicEnvCfgArgs(npz_files=[]).imu_sat_prob == 0.0
    assert PolicyTrainerCfg().imu_sat_prob == 0.0 and ImuImagCfg().prob == 0.0
    assert EnvCfg().imu_gyro_limit == pytest.approx(math.radians(500), abs=1e-3)
    with pytest.raises(ValueError):
        EnvCfg(imu_sat_prob=1.5)


def test_off_by_default_env_cfg_uses_plain_terms():
    from mjlab.envs import mdp as base_mdp

    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.imu_model import ImuBaseAngVel, ImuProjectedGravity
    from supermimic.envs.tracking_env_cfg import make_env_cfg

    t = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["x.npz"])).observations["policy"].terms
    assert t["base_ang_vel"].func is base_mdp.base_ang_vel and t["projected_gravity"].func is base_mdp.projected_gravity
    t = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["x.npz"], imu_sat_prob=0.5)).observations["policy"].terms
    assert t["base_ang_vel"].func is ImuBaseAngVel and t["projected_gravity"].func is ImuProjectedGravity


def test_gravity_integration_matches_reference_rotation():
    """Integrating g_b with the body rate equals the gravity of the orientation integrated as q' = q exp(w dt)."""
    torch.manual_seed(0)
    dt = 0.005
    q = quat_normalize(torch.randn(64, 4, dtype=torch.float64))
    g = projected_gravity(q)
    for _ in range(200):  # 1 s of a fast, changing rotation
        w = torch.randn(64, 3, dtype=torch.float64) * 6.0
        q = quat_normalize(quat_mul(q, quat_from_rotvec(w * dt)))
        g = rotate_gravity(g, w, dt)
    assert (g - projected_gravity(q)).abs().max() < 1e-9
    assert (g.norm(dim=-1) - 1).abs().max() < 1e-12
    # Hand case: rolling by +90 deg about body x moves body-frame gravity (0,0,-1) to (0,-1,0).
    g1 = rotate_gravity(torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64), torch.tensor([[math.pi / 2, 0, 0]], dtype=torch.float64), 1.0)
    assert torch.allclose(g1, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64), atol=1e-12)


def test_clipping_and_drift():
    lim = 8.727
    w = torch.tensor([[1.0, -20.0, 12.0], [-9.0, 0.5, -8.0]])
    assert torch.equal(clip_gyro(w, lim), torch.tensor([[1.0, -lim, lim], [-lim, 0.5, -8.0]]))
    # A pure yaw spin leaves an upright gravity unchanged even when saturated.
    g0 = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    g = imu_gravity_step(g0, torch.tensor([[0.0, 0.0, 15.0]], dtype=torch.float64), g0, 0.0, lim, 0.005)
    assert torch.allclose(g, g0)
    # Tilted body spinning at 12 rad/s about body z: the clipped integration rotates g by lim*t, not 12*t.
    tilt = math.radians(20)
    g_true = torch.tensor([[math.sin(tilt), 0.0, -math.cos(tilt)]], dtype=torch.float64)
    g_est, g_ref = g_true.clone(), g_true.clone()
    w = torch.tensor([[0.0, 0.0, 12.0]], dtype=torch.float64)
    for _ in range(100):  # 0.5 s
        g_est = imu_gravity_step(g_est, w, g_true, 0.0, lim, 0.005)
        g_ref = rotate_gravity(g_ref, w, 0.005)
    want = rotate_gravity(g_true, torch.tensor([[0.0, 0.0, lim]], dtype=torch.float64), 0.5)
    assert torch.allclose(g_est, want, atol=1e-10)
    yaw_err = 0.5 * (12.0 - lim)  # rad of missed rotation about z
    assert _angle(g_est, g_ref).item() == pytest.approx(2 * math.asin(math.sin(tilt) * math.sin(yaw_err / 2)), rel=1e-6)


def test_complementary_correction():
    g_true = torch.tensor([[0.0, 0.0, -1.0]])
    g_est = torch.tensor([[math.sin(0.3), 0.0, -math.cos(0.3)]])
    zero = torch.zeros(1, 3)
    same = imu_gravity_step(g_est, zero, g_true, torch.zeros(1), 8.727, 0.005)
    assert torch.allclose(same, g_est)  # k = 0: pure integration
    g = g_est
    for _ in range(2000):  # 10 s at k = 0.5 /s: e^-5
        g = imu_gravity_step(g, zero, g_true, torch.full((1,), 0.5), 8.727, 0.005)
    assert _angle(g, g_true).item() < 0.3 * math.exp(-5) * 1.2


def test_random_tilt_and_observation_columns():
    torch.manual_seed(1)
    g = projected_gravity(quat_normalize(torch.randn(4096, 4)))
    gt = random_tilt(g, 8.0)
    ang = torch.rad2deg(_angle(g, gt))
    assert ang.max() <= 8.0 + 1e-3 and ang.min() >= 0 and ang.mean() == pytest.approx(4.0, abs=0.3)
    assert torch.allclose(gt.norm(dim=-1), torch.ones(4096), atol=1e-5)
    obs = torch.randn(3, 93) * 20
    flag = torch.tensor([True, False, True])
    out = imu_observation(obs, gt[:3], flag, 8.727)
    assert torch.equal(out[1], obs[1])
    keep = torch.ones(93, dtype=torch.bool)
    keep[58:64] = False
    assert torch.equal(out[:, keep], obs[:, keep])
    assert torch.equal(out[[0, 2]][:, OBS_ANG_VEL], obs[[0, 2]][:, OBS_ANG_VEL].clamp(-8.727, 8.727))
    assert torch.equal(out[[0, 2]][:, OBS_GRAVITY], gt[[0, 2]])


def test_gradient_through_integration():
    w = torch.tensor([[0.5, -1.0, 9.5], [0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    g0 = torch.tensor([[0.2, 0.1, -0.97], [0.3, 0.0, -0.95]], dtype=torch.float64)
    g = imu_gravity_step(g0 / g0.norm(dim=-1, keepdim=True), w, torch.tensor([[0.0, 0, -1]] * 2, dtype=torch.float64), 0.3, 8.727, 0.02)
    g[:, 0].sum().backward()
    assert torch.isfinite(w.grad).all()
    assert w.grad[0, 2] == 0  # saturated axis: the sensor (and the gradient) does not see it
    assert w.grad[0, :2].abs().sum() > 0 and w.grad[1].abs().sum() > 0  # finite and non-zero at w = 0 too


# ---------------------------------------------------------------------------------------------- simulator (mjlab)


def _env(prob, limit, k_max, num_envs=16):
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=CLIPS)
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=num_envs, replay_capacity=64, seed=3, obs_noise=False,
                                imu_sat_prob=prob, imu_gyro_limit=limit, imu_accel_k_max=k_max)
    return make_env(args, device="cuda:0")


def _latest(obs, sl):
    """Newest frame of a 93-d term range in the term-major flattened history [A_t0..A_tH-1, B_t0..]."""
    d = sl.stop - sl.start
    return obs[:, H * sl.start + (H - 1) * d : H * sl.start + H * d]


@cuda
def test_env_unclipped_matches_true_gravity_and_resets():
    env = _env(1.0, 1e3, 0.0)
    try:
        obs, _ = env.reset()
        m = env._sm_imu_model
        robot = env.scene["robot"]
        assert m.active.all() and (m.k == 0).all()
        torch.manual_seed(0)
        max_err, n_reset = 0.0, 0
        for _ in range(150):
            obs, _, term, trunc, _ = env.step(0.8 * torch.randn(env.num_envs, 29, device=env.device))
            n_reset += int((term | trunc).sum())
            g_obs = _latest(obs["policy"], OBS_GRAVITY)
            max_err = max(max_err, (g_obs - robot.data.projected_gravity_b).abs().max().item())
            assert torch.allclose(_latest(obs["policy"], OBS_ANG_VEL), robot.data.root_link_ang_vel_b)
        print(f"\nunclipped IMU vs true gravity: max err {max_err:.2e}, {n_reset} resets")
        assert max_err < 1e-4  # per-substep integration reproduces MuJoCo's orientation (float32)
        assert n_reset > 0  # reset envs restart from the true gravity (covered by the per-step check)
    finally:
        env.close()


@cuda
def test_env_clipped_gyro_drifts_and_reset_restores():
    env = _env(1.0, 0.0, 0.0)  # gyro reads 0: g_est frozen at the episode-start gravity
    try:
        obs, _ = env.reset()
        m = env._sm_imu_model
        robot = env.scene["robot"]
        g_start = robot.data.projected_gravity_b.clone()
        assert torch.allclose(_latest(obs["policy"], OBS_GRAVITY), g_start)
        for _ in range(30):
            obs, _, term, trunc, _ = env.step(0.8 * torch.randn(env.num_envs, 29, device=env.device))
            done = term | trunc
            g_start[done] = robot.data.projected_gravity_b[done]
            assert torch.allclose(_latest(obs["policy"], OBS_GRAVITY), g_start, atol=1e-6)
            assert (_latest(obs["policy"], OBS_ANG_VEL) == 0).all()
        assert (_latest(obs["policy"], OBS_GRAVITY) - robot.data.projected_gravity_b).abs().max() > 1e-2
        # Explicit reset of a subset: those envs restart from the true gravity.
        ids = torch.tensor([0, 3], device=env.device)
        env._reset_idx(ids)
        env.sim.forward()
        g = m.gravity(robot.data.projected_gravity_b)
        assert torch.allclose(g[ids], robot.data.projected_gravity_b[ids])
    finally:
        env.close()


@cuda
def test_env_prob_flags_per_episode():
    env = _env(0.5, 0.0, 0.5, num_envs=16)
    try:
        env.reset()
        m = env._sm_imu_model
        flags = []
        for _ in range(20):
            m.resample(None)
            flags.append(m.active.float().mean().item())
            assert (m.k >= 0).all() and (m.k <= 0.5).all()
        assert 0.3 < sum(flags) / len(flags) < 0.7
        robot = env.scene["robot"]
        obs, *_ = env.step(torch.zeros(env.num_envs, 29, device=env.device))
        off = ~m.active
        assert torch.allclose(_latest(obs["policy"], OBS_GRAVITY)[off], robot.data.projected_gravity_b[off])
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- imagination


@cuda
def test_imagined_rollout_uses_imu_obs_and_keeps_true_state():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.imagination.rollout import imagine
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    env = _env(0.0, 8.727, 0.5)
    try:
        env.reset()
        for _ in range(40):
            env.step(0.3 * torch.randn(env.num_envs, 29, device=env.device))
        buf = env.recorder_manager.get_term("replay").buffer
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=env.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(WorldModelCfg(hidden=(128, 128)), heads, env.device)
        wm.requires_grad_(False)
        lib = env.command_manager.get_term("motion").library
        policy = Policy(PolicyCfg(goal_dim=680, encoder_hidden=(64, 64), decoder_hidden=(64, 64))).to(env.device)
        seen = []

        class Rec(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, obs, goal):
                seen.append(obs)
                return self.p(obs, goal)

        torch.manual_seed(0)
        start = buf.sample_starts(32, H)
        g_oldest = projected_gravity(start.hist_states[0][:, 3:7])

        # prob 1, gyro limit 0, k 0: the policy reads ang vel 0 and a constant gravity = oldest history gravity tilted
        # by <= 8 deg; the imagined states are untouched by the IMU model.
        ro = imagine(wm, Rec(policy), lib, start, 6, obs_noise=False, imu=ImuImagCfg(1.0, 0.0, 0.0, 8.0))
        for o in seen:
            g = _latest(o, OBS_GRAVITY)
            assert torch.rad2deg(_angle(g, g_oldest)).max() <= 8.0 + 1e-3
            assert torch.allclose(g, _latest(seen[0], OBS_GRAVITY), atol=1e-6)
            assert (_latest(o, OBS_ANG_VEL) == 0).all()
        assert torch.rad2deg(_angle(_latest(seen[0], OBS_GRAVITY), g_oldest)).mean() > 1.0
        # The imagined states keep the true attitude (not the frozen estimate): their gravity moves over the rollout.
        g_states = projected_gravity(ro.states[-1][:, 3:7])
        assert (g_states - _latest(seen[-1], OBS_GRAVITY)).abs().max() > 1e-3

        # prob 0 == the plain observation (true gravity of the predicted state).
        seen.clear()
        ro0 = imagine(wm, Rec(policy), lib, start, 6, obs_noise=False, imu=ImuImagCfg(0.0))
        for t, o in enumerate(seen):
            assert torch.allclose(_latest(o, OBS_GRAVITY), projected_gravity(ro0.states[t][:, 3:7]), atol=1e-6)

        # Differentiable: BPTT through the IMU path reaches the policy.
        ro = imagine(wm, policy, lib, start, 6, obs_noise=True, imu=ImuImagCfg(1.0, 8.727, 0.5, 8.0))
        ro.rewards.sum().backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    finally:
        env.close()
