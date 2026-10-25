"""Tilt-error hinge (--env.tilt-hinge-weight): same angle as the deploy abort, off by default, smooth gradient."""

import math

import numpy as np
import torch

from supermimic.deploy.core import projected_gravity
from supermimic.envs.mdp.rewards import TrackingRewardCfg, tilt_error_angle, tracking_reward_terms
from supermimic.imagination.compiled import REWARD_TERM_NAMES


def _quat_rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def test_angle_matches_deploy():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64, 2, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    got = tilt_error_angle(torch.tensor(q[:, 0]), torch.tensor(q[:, 1])).numpy()
    for i in range(len(q)):
        gr, gm = projected_gravity(q[i, 0]), projected_gravity(q[i, 1])
        ref = np.arccos(np.clip(gr @ gm / (np.linalg.norm(gr) * np.linalg.norm(gm)), -1, 1))
        assert abs(got[i] - ref) < 1e-5


def test_direction_aware_and_yaw_invariant():
    fwd = torch.tensor([_quat_rpy(0, math.radians(26), 0.3)], dtype=torch.float64)
    back = torch.tensor([_quat_rpy(0, math.radians(-26), 1.2)], dtype=torch.float64)
    same = torch.tensor([_quat_rpy(0, math.radians(26), 2.0)], dtype=torch.float64)
    assert abs(math.degrees(tilt_error_angle(fwd, back).item()) - 52.0) < 1e-6
    assert abs(tilt_error_angle(fwd, same).item()) < 1e-5


def test_gradient_finite():
    q = torch.tensor([_quat_rpy(0.1, 0.2, 0.0)], dtype=torch.float64, requires_grad=True)
    tilt_error_angle(q, q.detach()).sum().backward()
    assert torch.isfinite(q.grad).all()
    q2 = torch.tensor([_quat_rpy(0.0, math.radians(25), 0.0)], dtype=torch.float64, requires_grad=True)
    tilt_error_angle(q2, torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)).sum().backward()
    assert q2.grad.abs().sum() > 0


def _terms(cfg, robot_q, ref_q):
    n, b = robot_q.shape[0], 3
    z = torch.zeros(n, b, 3, dtype=torch.float64)
    qb = torch.zeros(n, b, 4, dtype=torch.float64)
    qb[..., 0] = 1
    return tracking_reward_terms(
        robot_body_pos_w=z, robot_body_quat_w=qb, robot_body_lin_vel_w=z, robot_body_ang_vel_w=z,
        ref_body_pos_w=z, ref_body_quat_w=qb, ref_body_lin_vel_w=z, ref_body_ang_vel_w=z, anchor_index=0,
        action=torch.zeros(n, 29, dtype=torch.float64), prev_action=torch.zeros(n, 29, dtype=torch.float64),
        joint_pos=torch.zeros(n, 29, dtype=torch.float64), soft_limits=torch.tensor([[-1.0, 1.0]] * 29, dtype=torch.float64),
        cfg=cfg, robot_root_quat_w=robot_q, ref_root_quat_w=ref_q)


def test_hinge_term():
    up = torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float64)
    rob = torch.tensor([_quat_rpy(0, math.radians(8), 0), _quat_rpy(0, math.radians(25), 0)], dtype=torch.float64)
    off = _terms(TrackingRewardCfg(), rob, up)
    assert torch.all(off["tilt_hinge"] == 0)
    on = _terms(TrackingRewardCfg(w_tilt_hinge=-3.0, tilt_hinge_knee=0.21), rob, up)
    assert on["tilt_hinge"][0] == 0
    want = 0.02 * -3.0 * (math.radians(25) - 0.21)
    assert abs(on["tilt_hinge"][1].item() - want) < 1e-9
    assert abs((on["total"] - off["total"])[1].item() - want) < 1e-9
    assert set(REWARD_TERM_NAMES) <= set(on)
