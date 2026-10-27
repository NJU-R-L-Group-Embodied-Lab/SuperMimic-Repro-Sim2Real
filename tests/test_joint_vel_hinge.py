"""Knee-velocity hinge penalty (--env.joint-vel-hinge-weight): off by default, hand computation, gradient through dq."""

import torch

from supermimic.config import EnvCfg
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from supermimic.imagination.compiled import REWARD_TERM_NAMES

KNEES = tuple(G1_JOINT_NAMES.index(n) for n in EnvCfg().joint_vel_hinge_joints)


def _terms(cfg, joint_vel):
    n, b = joint_vel.shape[0], 3
    z = torch.zeros(n, b, 3, dtype=torch.float64)
    qb = torch.zeros(n, b, 4, dtype=torch.float64)
    qb[..., 0] = 1
    a = torch.zeros(n, 29, dtype=torch.float64)
    return tracking_reward_terms(
        robot_body_pos_w=z, robot_body_quat_w=qb, robot_body_lin_vel_w=z, robot_body_ang_vel_w=z,
        ref_body_pos_w=z, ref_body_quat_w=qb, ref_body_lin_vel_w=z, ref_body_ang_vel_w=z, anchor_index=0,
        action=a, prev_action=a, joint_pos=a, soft_limits=torch.tensor([[-9.0, 9.0]] * 29, dtype=torch.float64),
        cfg=cfg, joint_vel=joint_vel)


def test_registered_and_off_by_default():
    assert "joint_vel_hinge" in REWARD_TERM_NAMES
    assert TrackingRewardCfg().w_joint_vel_hinge == 0.0 and EnvCfg().joint_vel_hinge_weight == 0.0
    dq = torch.full((3, 29), 30.0, dtype=torch.float64)
    assert (_terms(TrackingRewardCfg(), dq)["joint_vel_hinge"] == 0).all()
    assert G1_JOINT_NAMES[KNEES[0]] == "left_knee_joint" and G1_JOINT_NAMES[KNEES[1]] == "right_knee_joint"


def test_matches_hand_computation_and_gradient():
    cfg = TrackingRewardCfg(w_joint_vel_hinge=-0.1, joint_vel_hinge_ids=KNEES, joint_vel_hinge_thr=12.0)
    dq = torch.zeros(4, 29, dtype=torch.float64)
    dq[0, KNEES[1]] = 17.9  # knee speed seen on the real robot
    dq[1, KNEES[0]] = -14.0
    dq[2, KNEES[1]] = 11.0  # below the threshold
    dq[3, 0] = 30.0  # not a knee
    dq.requires_grad_(True)
    t = _terms(cfg, dq)["joint_vel_hinge"]
    want = torch.tensor([5.9**2, 2.0**2, 0.0, 0.0], dtype=torch.float64) * cfg.dt * -0.1
    assert torch.allclose(t, want)
    t.sum().backward()
    assert dq.grad[0, KNEES[1]] < 0 and dq.grad[1, KNEES[0]] > 0 and dq.grad[2].abs().sum() == 0 and dq.grad[3].abs().sum() == 0
