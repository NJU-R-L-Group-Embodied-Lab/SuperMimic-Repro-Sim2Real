"""Joint-torque penalty (--env.joint-torque-weight): off by default, PD torque by hand, gradient through the action."""

import torch

from supermimic.config import EnvCfg
from supermimic.envs.mdp.rewards import TrackingRewardCfg, joint_torque_constants, tracking_reward_terms
from supermimic.imagination.compiled import REWARD_TERM_NAMES


def _terms(cfg, action, joint_pos, joint_vel, pd):
    n, b = action.shape[0], 3
    z = torch.zeros(n, b, 3, dtype=torch.float64)
    qb = torch.zeros(n, b, 4, dtype=torch.float64)
    qb[..., 0] = 1
    return tracking_reward_terms(
        robot_body_pos_w=z, robot_body_quat_w=qb, robot_body_lin_vel_w=z, robot_body_ang_vel_w=z,
        ref_body_pos_w=z, ref_body_quat_w=qb, ref_body_lin_vel_w=z, ref_body_ang_vel_w=z, anchor_index=0,
        action=action, prev_action=torch.zeros_like(action), joint_pos=joint_pos,
        soft_limits=torch.tensor([[-9.0, 9.0]] * 29, dtype=torch.float64), cfg=cfg, pd=pd, joint_vel=joint_vel)


def _pd():
    e = EnvCfg()
    return {k: v.double() for k, v in joint_torque_constants(e.joint_torque_jw, e.joint_torque_jw_default, "cpu").items()}


def test_term_registered():
    assert "joint_torque" in REWARD_TERM_NAMES


def test_off_by_default():
    assert TrackingRewardCfg().w_joint_torque == 0.0 and EnvCfg().joint_torque_weight == 0.0
    pd = _pd()
    a = torch.randn(4, 29, dtype=torch.float64)
    t = _terms(TrackingRewardCfg(), a, torch.zeros_like(a), torch.zeros_like(a), pd)
    assert (t["joint_torque"] == 0).all()


def test_matches_hand_computation():
    pd = _pd()
    g = torch.Generator().manual_seed(0)
    a, q, dq = (torch.randn(5, 29, generator=g, dtype=torch.float64) for _ in range(3))
    cfg = TrackingRewardCfg(w_joint_torque=-0.2)
    t = _terms(cfg, a, q, dq, pd)
    for i in range(5):
        s = 0.0
        for j in range(29):
            tau = pd["kp"][j] * (pd["q0"][j] + pd["scale"][j] * a[i, j] - q[i, j]) - pd["kd"][j] * dq[i, j]
            s += pd["jw"][j] * (tau / pd["effort"][j]) ** 2
        assert abs(t["joint_torque"][i].item() - cfg.dt * -0.2 * s.item()) < 1e-9
    base = _terms(TrackingRewardCfg(), a, q, dq, pd)
    assert torch.allclose(t["total"] - base["total"], t["joint_torque"])


def test_weights_and_gradient():
    pd = _pd()
    from supermimic.data.motion_convert import G1_JOINT_NAMES

    names = list(G1_JOINT_NAMES)
    assert pd["jw"][names.index("waist_pitch_joint")] == 2.0 and pd["jw"][names.index("left_knee_joint")] == 1.0
    assert abs(pd["jw"][names.index("left_wrist_roll_joint")] - 0.1) < 1e-6
    # waist target far past the limit (e13i commands ~2 rad vs 0.52): the term must push the action back.
    a = torch.zeros(1, 29, dtype=torch.float64)
    a[0, names.index("waist_pitch_joint")] = 2.0 / pd["scale"][names.index("waist_pitch_joint")]
    a.requires_grad_(True)
    q = pd["q0"].clone()[None]
    t = _terms(TrackingRewardCfg(w_joint_torque=-0.2), a, q, torch.zeros_like(q), pd)
    t["total"].sum().backward()
    assert a.grad[0, names.index("waist_pitch_joint")] < 0
