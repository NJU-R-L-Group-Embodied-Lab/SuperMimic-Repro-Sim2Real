"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=ghost,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
