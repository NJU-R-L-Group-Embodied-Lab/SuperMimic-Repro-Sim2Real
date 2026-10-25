"""Data-parallel training helpers (plan D13 / 6.3).

Launch with torchrun, one process per GPU:
  torchrun --nproc_per_node 4 -m supermimic.scripts.train --run-name e5_ws ...

Each rank owns its own simulator (num_envs envs), replay buffer and sampler; the
world model / policy / critic are replicated and their gradients are averaged with
an explicit all-reduce after backward (no DDP wrapper, so torch.compile'd functions
stay untouched). Learning rates are scaled by sqrt(world_size) (paper: x sqrt(2) for 2 GPUs).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> tuple[int, int, int]:
    """Returns (rank, local_rank, world_size); initialises NCCL when launched by torchrun."""
    if not is_distributed():
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    pinned = os.environ.get("SM_GPU_PINNED") == "1"  # scripts/train.py set CUDA_VISIBLE_DEVICES=LOCAL_RANK
    # Pinned: each process sees exactly one GPU (cuda:0). Otherwise modulo lets a 1-GPU box smoke-test.
    local_rank = 0 if pinned else int(os.environ["LOCAL_RANK"]) % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(local_rank)
    # NCCL cannot share one GPU between ranks; fall back to gloo for single-GPU smoke tests.
    one_gpu_per_rank = pinned or torch.cuda.device_count() >= world
    backend = os.environ.get("SM_DIST_BACKEND") or ("nccl" if one_gpu_per_rank else "gloo")
    dist.init_process_group(backend, rank=rank, world_size=world)
    return rank, local_rank, world


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


@torch.no_grad()
def allreduce_grads(module: torch.nn.Module) -> None:
    """Average gradients across ranks (flattened into one buffer per dtype)."""
    if world_size() == 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


@torch.no_grad()
def broadcast_params(module: torch.nn.Module, src: int = 0) -> None:
    if world_size() == 1:
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def allreduce_mean(x: float) -> float:
    if world_size() == 1:
        return x
    t = torch.tensor([x], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t)


def finalize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
