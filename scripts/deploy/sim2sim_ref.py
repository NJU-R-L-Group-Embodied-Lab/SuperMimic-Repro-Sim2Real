"""Training-env reference rollouts for the sim2sim G1 gate (training venv: torch + mjlab, GPU, few envs).

Same deterministic protocol as scripts/render_video.py rollout() / run_benchmark (no noise, no DR, every
clip from its first frame), but additionally records the step-0 observation/goal and every action, and
can run with exact fp32 matmuls (mjlab's make_env switches TF32 on; --no-tf32 switches it back off).

  PYTHONPATH= .venv/bin/python scripts/deploy/sim2sim_ref.py --ckpt deploy_artifacts/ckpts/ckpt_0116000.pt \
      --out deploy_artifacts/reports/ref_ckpt_0116000_fp32.npz --no-tf32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint

CLIPS = (
    "neutral_idle_loop_001__A081",
    "idle_loop_002__A060",
    "shoulder_clap_004__A041_M",
    "itching_left_arm_R_003__A121",
    "walk_sideway_090_stop_003__A037",
    "wave_R_001__A431",
    "jog_sideway_045_loop_002__A025",
    "dance_hiphop_alt_stump_R_fast_001__A318_M",
)


@torch.no_grad()
def main(ckpt: Path, out: Path, tf32: bool = True, npz_dir: Path = Path("data/npz/bones_seed"), device: str = "cuda:0", max_steps: int = 2000,
         clips: tuple[str, ...] = CLIPS) -> None:
    """clips: clip names to roll out (default: the sim2sim SAFE + HARDER set), one env each, in this order."""
    CLIPS = tuple(clips)  # noqa: N806
    files = [str(npz_dir / f"{c}.npz") for c in CLIPS]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=len(files), future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n = lib.num_clips
        assert [c.name for c in lib.clips] == list(CLIPS), [c.name for c in lib.clips]
        ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[ids]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[ids]
        origins = env.scene.env_origins
        obs0, goal0 = obs["policy"].cpu().numpy(), obs["goal"].cpu().numpy()
        active = torch.ones(n, dtype=torch.bool, device=device)
        fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
        length = torch.zeros(n, dtype=torch.long, device=device)
        rq, gq, acts = [], [], []

        def snap():
            rq.append(torch.cat([cmd.robot_root_pos_w - origins, cmd.robot_root_quat_w, cmd.robot_joint_pos], -1).cpu())
            gq.append(torch.cat([cmd.root_pos_w - origins, cmd.root_quat_w, cmd.joint_pos], -1).cpu())

        snap()
        for t in range(max_steps):
            act = policy(obs["policy"], obs["goal"])
            acts.append(act.cpu())
            obs, rew, term, tout, _ = env.step(act)
            done = (term | tout) & active
            fail_step[term & active] = t + 1
            length[active & ~done] = t + 1
            snap()
            active &= ~done
            if not active.any():
                break
        rq, gq, acts = torch.stack(rq, 1).numpy(), torch.stack(gq, 1).numpy(), torch.stack(acts, 1).numpy()
        save = {"clips": np.array(CLIPS), "tf32": np.array(tf32), "obs0": obs0, "goal0": goal0, "fail_step": fail_step.cpu().numpy(), "length": length.cpu().numpy()}
        for i, c in enumerate(CLIPS):
            L = int(length[i]) + 1
            save[f"{c}/robot_q"] = rq[i, :L].astype(np.float64)
            save[f"{c}/ref_q"] = gq[i, :L].astype(np.float64)
            save[f"{c}/action"] = acts[i, : L - 1].astype(np.float64)
            print(f"{c:44s} frames {lib.clips[i].length:4d} steps {L - 1:4d} fail_step {int(fail_step[i])}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **save)
        print("wrote", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
