"""Pelvis yaw-rate hinge (--env.yaw-rate-hinge-weight): off by default, hand computation, gradient through w_z, and
the simulator twin matches the analytic term."""

from pathlib import Path

import pytest
import torch

from supermimic.config import EnvCfg
from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms, yaw_rate_hinge_value
from supermimic.imagination.compiled import REWARD_TERM_NAMES

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def _terms(cfg, ang_b):
    n, b = ang_b.shape[0], 3
    z = torch.zeros(n, b, 3, dtype=torch.float64)
    qb = torch.zeros(n, b, 4, dtype=torch.float64)
    qb[..., 0] = 1
    a = torch.zeros(n, 29, dtype=torch.float64)
    return tracking_reward_terms(
        robot_body_pos_w=z, robot_body_quat_w=qb, robot_body_lin_vel_w=z, robot_body_ang_vel_w=z,
        ref_body_pos_w=z, ref_body_quat_w=qb, ref_body_lin_vel_w=z, ref_body_ang_vel_w=z, anchor_index=0,
        action=a, prev_action=a, joint_pos=a, soft_limits=torch.tensor([[-9.0, 9.0]] * 29, dtype=torch.float64),
        cfg=cfg, joint_vel=a, root_ang_vel_b=ang_b)


def test_registered_and_off_by_default():
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg

    assert "yaw_rate_hinge" in REWARD_TERM_NAMES
    assert TrackingRewardCfg().w_yaw_rate_hinge == 0.0 and EnvCfg().yaw_rate_hinge_weight == 0.0
    assert EnvCfg().yaw_rate_hinge_thr == 7.5
    w = torch.full((3, 3), 30.0, dtype=torch.float64)
    t = _terms(TrackingRewardCfg(), w)
    assert (t["yaw_rate_hinge"] == 0).all()
    assert "yaw_rate_hinge" not in make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["x.npz"])).rewards
    r = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["x.npz"], yaw_rate_hinge_weight=0.05)).rewards["yaw_rate_hinge"]
    assert r.weight == -0.05 and r.params == {"thr": 7.5}


@pytest.mark.parametrize("w", [0.05, -0.05])  # either sign is a penalty (-|w|)
def test_matches_hand_computation_and_gradient(w):
    cfg = TrackingRewardCfg(w_yaw_rate_hinge=w, yaw_rate_hinge_thr=7.5)
    ang = torch.zeros(4, 3, dtype=torch.float64)
    ang[0, 2] = 9.9  # real lean35 flight peak
    ang[1, 2] = -8.5
    ang[2, 2] = 7.0  # below the threshold
    ang[3, 0] = 20.0  # roll, not yaw
    ang.requires_grad_(True)
    t = _terms(cfg, ang)
    want = -abs(w) * cfg.dt * torch.tensor([2.4**2, 1.0**2, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(t["yaw_rate_hinge"], want)
    assert torch.allclose(t["total"] - t["yaw_rate_hinge"], _terms(TrackingRewardCfg(), ang.detach())["total"])
    t["yaw_rate_hinge"].sum().backward()
    g = ang.grad
    assert torch.allclose(g[0, 2], torch.tensor(-abs(w) * cfg.dt * 2 * 2.4, dtype=torch.float64))
    assert g[1, 2] > 0 and g[2].abs().sum() == 0 and g[3].abs().sum() == 0 and g[:, :2].abs().sum() == 0


def test_heads_pass_pelvis_body_rate():
    """AnalyticHeads.reward feeds s[39:42] (root angular velocity, body frame) to the term."""
    import inspect

    from supermimic.world_model import heads

    assert "root_ang_vel_b=s_next[..., 39:42]" in inspect.getsource(heads.AnalyticHeads.reward)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_sim_term_matches_analytic():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp import rewards as sm_rew
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, replay_capacity=64, seed=1, yaw_rate_hinge_weight=0.05, yaw_rate_hinge_thr=0.5), "cuda:0")
    try:
        env.reset()
        for _ in range(10):
            env.step(0.8 * torch.randn(8, 29, device=env.device))
        robot = env.scene["robot"]
        got = sm_rew.yaw_rate_hinge(env, 0.5)
        want = (robot.data.root_link_ang_vel_b[:, 2].abs() - 0.5).clamp(min=0) ** 2
        assert torch.allclose(got, want)
        assert "yaw_rate_hinge" in env.reward_manager.active_terms
    finally:
        env.close()
