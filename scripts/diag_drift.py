"""Drift diagnostic: how fast does a policy's global root pose drift from the reference over whole clips,
and how much of the world-frame velocity error is just that drift (yaw misalignment)?

  uv run python scripts/diag_drift.py --ckpt logs/e7/ckpt_0020000.pt --npz-dir data/npz/lafan1

Same deterministic protocol as eval_benchmark (first frame start, no noise/DR). Each clip plays once in
its own env. Reports per-clip and aggregate:
  * mean / final anchor xy error [m] and |yaw error| [deg], time until xy error > 0.3 m and |yaw| > 15 deg
  * body linear-velocity error: world frame (what the reward sees) vs after rotating the robot's velocities
    by the yaw error (drift removed) -> the residual is the genuine velocity-tracking error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import tyro
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, wrap_to_pi, yaw_quat

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None, device: str = "cuda:0", out: Path | None = None) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    n = len(files)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=n, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=device)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = -1
        dt = env.step_dt
        active = torch.ones(n, dtype=torch.bool, device=device)
        steps = torch.zeros(n, device=device)
        sum_xy = torch.zeros(n, device=device)
        sum_yaw = torch.zeros(n, device=device)
        last_xy = torch.zeros(n, device=device)
        last_yaw = torch.zeros(n, device=device)
        t_xy = torch.full((n,), -1.0, device=device)
        t_yaw = torch.full((n,), -1.0, device=device)
        sum_v_world = torch.zeros(n, device=device)
        sum_v_aligned = torch.zeros(n, device=device)
        sum_w_world = torch.zeros(n, device=device)
        sum_w_aligned = torch.zeros(n, device=device)
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        while active.any():
            # measure before stepping: reference target frame vs current robot state
            xy = (cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]).norm(dim=-1)
            yaw_err = wrap_to_pi(_yaw(cmd.anchor_quat_w) - _yaw(cmd.robot_anchor_quat_w))
            # rotate robot body velocities into the reference heading (remove yaw drift)
            q_fix = yaw_quat(quat_mul(cmd.anchor_quat_w, quat_inv(cmd.robot_anchor_quat_w)))  # (n,4)
            nb = cmd.body_lin_vel_w.shape[1]
            qf = q_fix[:, None, :].expand(n, nb, 4).reshape(-1, 4)
            v_rob = cmd.robot_body_lin_vel_w.reshape(-1, 3)
            w_rob = cmd.robot_body_ang_vel_w.reshape(-1, 3)
            v_al = quat_apply(qf, v_rob).reshape(n, nb, 3)
            w_al = quat_apply(qf, w_rob).reshape(n, nb, 3)
            ev_w = ((cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
            ev_a = ((cmd.body_lin_vel_w - v_al) ** 2).sum(-1).mean(-1)
            ew_w = ((cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)
            ew_a = ((cmd.body_ang_vel_w - w_al) ** 2).sum(-1).mean(-1)
            a = active.float()
            sum_xy += xy * a
            sum_yaw += yaw_err.abs() * a
            last_xy = torch.where(active, xy, last_xy)
            last_yaw = torch.where(active, yaw_err.abs(), last_yaw)
            t_xy = torch.where(active & (t_xy < 0) & (xy > 0.3), steps * dt, t_xy)
            t_yaw = torch.where(active & (t_yaw < 0) & (yaw_err.abs() > math.radians(15)), steps * dt, t_yaw)
            sum_v_world += ev_w * a
            sum_v_aligned += ev_a * a
            sum_w_world += ew_w * a
            sum_w_aligned += ew_a * a
            steps += a
            act = policy(obs["policy"], obs["goal"])
            obs, _, term, tout, _ = env.step(act)
            done = (term | tout) & active
            failed |= term & active
            active &= ~done
        st = steps.clamp(min=1)
        rows = []
        for i in range(n):
            rows.append(dict(
                name=lib.clips[i].name, steps=int(steps[i]), failed=bool(failed[i]),
                mean_xy=float(sum_xy[i] / st[i]), final_xy=float(last_xy[i]),
                mean_yaw_deg=math.degrees(float(sum_yaw[i] / st[i])), final_yaw_deg=math.degrees(float(last_yaw[i])),
                t_xy_gt_0p3=float(t_xy[i]), t_yaw_gt_15=float(t_yaw[i]),
                r_lin_world=float(torch.exp(-sum_v_world[i] / st[i] / 1.0)), r_lin_aligned=float(torch.exp(-sum_v_aligned[i] / st[i] / 1.0)),
                r_ang_world=float(torch.exp(-sum_w_world[i] / st[i] / 3.14**2)), r_ang_aligned=float(torch.exp(-sum_w_aligned[i] / st[i] / 3.14**2)),
            ))
        tot = steps.sum()
        agg = dict(
            clips=n, failed=int(failed.sum()),
            mean_xy=float((sum_xy).sum() / tot), mean_yaw_deg=math.degrees(float(sum_yaw.sum() / tot)),
            median_t_xy_gt_0p3=float(torch.median(t_xy[t_xy >= 0])) if (t_xy >= 0).any() else -1,
            median_t_yaw_gt_15=float(torch.median(t_yaw[t_yaw >= 0])) if (t_yaw >= 0).any() else -1,
            lin_vel_reward_world=float(torch.exp(-sum_v_world.sum() / tot)), lin_vel_reward_yaw_aligned=float(torch.exp(-sum_v_aligned.sum() / tot)),
            ang_vel_reward_world=float(torch.exp(-sum_w_world.sum() / tot / 3.14**2)), ang_vel_reward_yaw_aligned=float(torch.exp(-sum_w_aligned.sum() / tot / 3.14**2)),
        )
        print(json.dumps(agg, indent=1))
        print(f"{'clip':30s} {'steps':>6s} {'fail':>4s} {'xy_mean':>7s} {'xy_end':>6s} {'yaw_mean':>8s} {'yaw_end':>7s} {'t>0.3m':>7s} {'t>15deg':>7s} {'rlin_w':>6s} {'rlin_a':>6s}")
        for r in rows:
            print(f"{r['name']:30s} {r['steps']:6d} {str(r['failed'])[0]:>4s} {r['mean_xy']:7.2f} {r['final_xy']:6.2f} {r['mean_yaw_deg']:8.1f} {r['final_yaw_deg']:7.1f} {r['t_xy_gt_0p3']:7.1f} {r['t_yaw_gt_15']:7.1f} {r['r_lin_world']:6.2f} {r['r_lin_aligned']:6.2f}")
        out = out or ckpt.parent / f"drift_{ckpt.stem}.json"
        Path(out).write_text(json.dumps({"summary": agg, "clips": rows}, indent=1))
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
