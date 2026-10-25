"""Time one policy update's imagination (forward + backward) eager vs torch.compile (plan 3.6)."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import tyro


def main(batch: int = 2048, horizon: int = 32, modes: tuple[str, ...] = ("eager", "default"), reps: int = 3, clips: int = 1) -> None:
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.imagination.compiled import ImagStep
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    model = load_g1_model()
    files = discover_npz(Path("data/npz/lafan1"), limit=clips)
    q0 = torch.zeros(29, device=dev)
    lib = MotionLibrary(files, TRACKED_BODIES, q0, device=dev)
    fk = G1Kinematics(model, device=dev)
    lim = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32, device=dev)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, lim, q0)
    wm = WorldModel(WorldModelCfg(), heads, dev)
    policy = Policy(PolicyCfg()).to(dev)

    # Synthetic start batch from library reference states.
    f = torch.randint(0, lib.num_frames - 100, (batch,), device=dev)
    qpos, qvel = lib.ref_qpos_qvel(f)
    s0 = torch.cat([qpos, qvel], -1)
    start = StartBatch(
        state=s0,
        xi=torch.zeros(batch, 33, device=dev) + torch.tensor([0.75] + [0.0] * 32, device=dev),
        frame=f,
        prev_action=torch.zeros(batch, 29, device=dev),
        hist_states=s0[None].expand(10, batch, 71).contiguous(),
        hist_actions=torch.zeros(10, batch, 29, device=dev),
    )
    for mode in modes:
        step_fn = ImagStep(wm, q0, compile_mode=None if mode == "eager" else mode)
        wm.requires_grad_(False)
        times = []
        for i in range(reps + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ro = imagine(wm, policy, lib, start, horizon, step_fn=step_fn)
            J = policy_objective(ro, None, 0.99)
            (-J).backward()
            policy.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i > 0:
                times.append(dt)
            else:
                print(f"[{mode}] first call (incl. compile): {dt:.1f}s")
        print(f"[{mode}] batch={batch} H={horizon}: {sum(times)/len(times):.3f} s per imagination fwd+bwd; peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    tyro.cli(main)
