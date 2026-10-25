"""Push-recovery test (paper §4.3 / plan 5.2).

A synthetic static T-pose reference clip is tracked; at t=1 s a horizontal velocity
of `speed` m/s (random direction) is added to the pelvis. We report, per push speed,
the fraction of envs that fall and the time until the robot settles (root speed < 0.1 m/s
and root position error < 0.05 m for 0.5 s; criterion is ours, the paper gives none).

  uv run python scripts/eval_push.py --ckpt logs/e2/ckpt_latest.pt
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import tyro


def make_tpose_clip(out: Path, seconds: float = 20.0, fps: int = 50) -> Path:
    """Write a 50 Hz T-pose npz (standing still) via the converter's FK."""
    from supermimic.data.motion_convert import G1_JOINT_NAMES, forward_kinematics, load_g1_model, ResampledMotion

    T = int(seconds * fps)
    q = np.zeros((T, len(G1_JOINT_NAMES)))
    q[:, G1_JOINT_NAMES.index("left_shoulder_roll_joint")] = 1.5
    q[:, G1_JOINT_NAMES.index("right_shoulder_roll_joint")] = -1.5
    motion = ResampledMotion(
        fps=fps,
        root_pos=np.tile(np.array([0.0, 0.0, 0.793]), (T, 1)),
        root_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1)),
        joint_pos=q,
        root_lin_vel_w=np.zeros((T, 3)),
        root_ang_vel_w=np.zeros((T, 3)),
        joint_vel=np.zeros((T, 29)),
    )
    model = load_g1_model()
    b = forward_kinematics(model, motion)
    np.savez(
        out,
        fps=np.array([fps], dtype=np.float32),
        joint_pos=q.astype(np.float32),
        joint_vel=np.zeros((T, 29), np.float32),
        body_pos_w=b.pos_w.astype(np.float32),
        body_quat_w=b.quat_w.astype(np.float32),
        body_lin_vel_w=b.lin_vel_w.astype(np.float32),
        body_ang_vel_w=b.ang_vel_w.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(b.body_names),
        source=np.array("synthetic_tpose"),
        input_fps=np.array([fps], dtype=np.float32),
    )
    return out


@torch.no_grad()
def main(ckpt: Path, speeds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0), num_envs: int = 256, device: str = "cuda:0", push_time_s: float = 1.0, horizon_s: float = 8.0) -> None:
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    tmp = Path(tempfile.mkdtemp()) / "tpose.npz"
    make_tpose_clip(tmp)
    args = SuperMimicEnvCfgArgs(npz_files=[str(tmp)], num_envs=num_envs, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start", record_transitions=False, episode_length_s=1e5, future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False), seed=0)
    env = make_env(args, device=device)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")
    dt = env.step_dt
    for speed in speeds:
        obs, _ = env.reset()
        fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        settle_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        calm = torch.zeros(num_envs, dtype=torch.long, device=device)
        pushed = False
        for step in range(int(horizon_s / dt)):
            if not pushed and step * dt >= push_time_s:
                ang = torch.rand(num_envs, device=device) * 2 * np.pi
                v = robot.data.root_link_vel_w.clone()
                v[:, 0] += speed * torch.cos(ang)
                v[:, 1] += speed * torch.sin(ang)
                robot.write_root_link_velocity_to_sim(v)
                pushed = True
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            fallen |= term
            if pushed:
                err = torch.norm(cmd.root_pos_w - cmd.robot_root_pos_w, dim=-1)
                spd = torch.norm(robot.data.root_link_lin_vel_w, dim=-1)
                ok = (err < 0.05) & (spd < 0.1) & ~fallen
                calm = torch.where(ok, calm + 1, torch.zeros_like(calm))
                newly = (calm >= int(0.5 / dt)) & (settle_step < 0)
                settle_step[newly] = step
        settled = (settle_step >= 0) & ~fallen
        t_settle = (settle_step[settled].float() * dt - push_time_s)
        print(f"push {speed:.1f} m/s: fell {fallen.float().mean():.2%}, settled {settled.float().mean():.2%}, settle time mean {t_settle.mean().item() if settled.any() else float('nan'):.2f} s (max {t_settle.max().item() if settled.any() else float('nan'):.2f})")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
