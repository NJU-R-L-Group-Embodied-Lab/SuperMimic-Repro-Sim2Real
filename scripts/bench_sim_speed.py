"""Phase 0.3: MuJoCo Warp throughput with the real SuperMimic env config (DR, recorder on/off)."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import tyro


def main(num_envs: int = 4096, steps: int = 300, warmup: int = 50, record: bool = True, clips: int = 3) -> None:
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(Path("data/npz/lafan1"), limit=clips)
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=num_envs, record_transitions=record)
    env = make_env(args)
    env.reset()
    act = torch.zeros(num_envs, 29, device=env.device)
    for _ in range(warmup):
        env.step(act)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(0.1 * torch.randn_like(act))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = steps * num_envs / dt
    print(f"num_envs={num_envs} record={record}: {steps} steps in {dt:.2f}s -> {sps:,.0f} env-steps/s, {dt / steps * 1e3:.1f} ms per vector step")
    print(f"GPU mem allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
