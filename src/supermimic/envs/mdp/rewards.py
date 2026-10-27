"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_box_minus, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02
    # Waist-locked robot: squared actions at the locked slots (they move nothing). Empty = term is identically 0.
    locked_action_ids: tuple[int, ...] = ()
    w_locked_action: float = 0.0
    # DEVIATION from the paper when w_tilt_hinge != 0 (default off): -|w| * max(0, tilt_err - knee) with tilt_err the
    # angle between robot and reference pelvis projected gravity, the quantity the deploy runner aborts on (> 20 deg).
    w_tilt_hinge: float = 0.0
    tilt_hinge_knee: float = 0.21  # rad (12 deg)
    # DEVIATION from the paper when w_joint_torque != 0 (default off): -|w| * sum_j jw_j * (tau_j / effort_j)^2 with
    # tau = kp (q0 + scale a - q) - kd dq the PD torque the actuator is asked for (not clipped), aimed at the waist /
    # knee heating seen on the real robot. Needs the `pd` constants passed to tracking_reward_terms.
    w_joint_torque: float = 0.0
    # DEVIATION from the paper when w_joint_vel_hinge != 0 (default off): -|w| * sum_j max(0, |dq_j| - thr)^2 over
    # joint_vel_hinge_ids [rad/s], aimed at the knee over-speed the deploy runner aborts on (0.8 x 20 rad/s on ground).
    w_joint_vel_hinge: float = 0.0
    joint_vel_hinge_ids: tuple[int, ...] = ()
    joint_vel_hinge_thr: float = 12.0
    # DEVIATION from the paper when w_yaw_rate_hinge != 0 (default off): -|w| * max(0, |w_z| - thr)^2 with w_z the
    # pelvis body-frame z angular velocity [rad/s], aimed at the yaw spins that saturate the real gyro (500 deg/s).
    w_yaw_rate_hinge: float = 0.0
    yaw_rate_hinge_thr: float = 7.5

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tilt_error_angle(robot_root_quat_w: torch.Tensor, ref_root_quat_w: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between robot and reference pelvis projected gravity (deploy/core.py tilt check), smooth at 0."""
    g = torch.zeros_like(robot_root_quat_w[..., :3])
    g[..., 2] = -1.0
    gr = quat_apply_inverse(robot_root_quat_w, g)
    gm = quat_apply_inverse(ref_root_quat_w, g)
    cross = torch.sqrt((torch.cross(gr, gm, dim=-1) ** 2).sum(-1) + 1e-12)
    return torch.atan2(cross, (gr * gm).sum(-1))


def yaw_rate_hinge_value(root_ang_vel_b: torch.Tensor, thr: float) -> torch.Tensor:
    """max(0, |w_z| - thr)^2 of the pelvis body-frame angular velocity (…, 3)."""
    return (root_ang_vel_b[..., 2].abs() - thr).clamp(min=0.0) ** 2


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
    robot_root_quat_w: torch.Tensor | None = None,
    ref_root_quat_w: torch.Tensor | None = None,
    pd: dict[str, torch.Tensor] | None = None,  # kp, kd, effort, scale, q0, jw: (29,) each
    joint_vel: torch.Tensor | None = None,
    root_ang_vel_b: torch.Tensor | None = None,  # (N, 3) pelvis angular velocity, body frame
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)
    if cfg.locked_action_ids and cfg.w_locked_action != 0.0:
        locked_action = (action[..., list(cfg.locked_action_ids)] ** 2).sum(-1)
    else:
        locked_action = torch.zeros_like(action_rate)
    if cfg.w_tilt_hinge != 0.0 and robot_root_quat_w is not None and ref_root_quat_w is not None:
        tilt_hinge = (tilt_error_angle(robot_root_quat_w, ref_root_quat_w) - cfg.tilt_hinge_knee).clamp(min=0.0)
    else:
        tilt_hinge = torch.zeros_like(action_rate)
    if cfg.w_joint_torque != 0.0 and pd is not None and joint_vel is not None:
        tau = pd["kp"] * (pd["q0"] + pd["scale"] * action - joint_pos) - pd["kd"] * joint_vel
        joint_torque = (pd["jw"] * (tau / pd["effort"]) ** 2).sum(-1)
    else:
        joint_torque = torch.zeros_like(action_rate)
    if cfg.w_joint_vel_hinge != 0.0 and cfg.joint_vel_hinge_ids and joint_vel is not None:
        dq = joint_vel[..., list(cfg.joint_vel_hinge_ids)]
        joint_vel_hinge = ((dq.abs() - cfg.joint_vel_hinge_thr).clamp(min=0.0) ** 2).sum(-1)
    else:
        joint_vel_hinge = torch.zeros_like(action_rate)
    if cfg.w_yaw_rate_hinge != 0.0 and root_ang_vel_b is not None:
        yaw_rate_hinge = yaw_rate_hinge_value(root_ang_vel_b, cfg.yaw_rate_hinge_thr)
    else:
        yaw_rate_hinge = torch.zeros_like(action_rate)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
        "locked_action_l2": dt * cfg.w_locked_action * locked_action,
        "tilt_hinge": dt * cfg.w_tilt_hinge * tilt_hinge,
        "joint_torque": dt * cfg.w_joint_torque * joint_torque,
        "joint_vel_hinge": dt * cfg.w_joint_vel_hinge * joint_vel_hinge,
        "yaw_rate_hinge": -dt * abs(cfg.w_yaw_rate_hinge) * yaw_rate_hinge,
    }
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms


def locked_action_l2(env, action_ids: tuple[int, ...]) -> torch.Tensor:
    """Sum of squared (clean) actions at the slots of locked joints (waist-locked robot): they move nothing, so the
    only thing that keeps them small is this term (env-side twin of ``tracking_reward_terms``' locked_action_l2)."""
    return env.action_manager.action[:, list(action_ids)].square().sum(-1)


def tilt_hinge(env, command_name: str, knee: float) -> torch.Tensor:
    """max(0, tilt_err - knee) on the simulator (env-side twin of ``tracking_reward_terms``' tilt_hinge)."""
    cmd = env.command_manager.get_term(command_name)
    return (tilt_error_angle(cmd.robot_root_quat_w, cmd.root_quat_w) - knee).clamp(min=0.0)


def yaw_rate_hinge(env, thr: float) -> torch.Tensor:
    """max(0, |pelvis body-z angular velocity| - thr)^2 on the simulator (env-side twin of ``tracking_reward_terms``'
    yaw_rate_hinge; register with a negative weight)."""
    return yaw_rate_hinge_value(env.scene["robot"].data.root_link_ang_vel_b, thr)


def joint_torque_constants(jw: tuple[tuple[str, float], ...], jw_default: float, device) -> dict[str, torch.Tensor]:
    """PD constants of the joint-torque term (29-joint G1_JOINT_NAMES order) from mjlab's g1_constants."""
    from supermimic.data.motion_convert import G1_JOINT_NAMES
    from supermimic.deploy.export import constants_from_mjlab

    names = list(G1_JOINT_NAMES)
    c = constants_from_mjlab(names)
    w = dict(jw)
    assert set(w) <= set(names), sorted(set(w) - set(names))
    out = {"kp": c["kp"], "kd": c["kd"], "effort": c["effort_limit"], "scale": c["action_scale"], "q0": c["default_joint_pos"]}
    out["jw"] = [w.get(n, jw_default) for n in names]
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in out.items()}
