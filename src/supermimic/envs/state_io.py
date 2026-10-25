"""Read / write the full simulator state (qpos | qvel) as one 71-d tensor.

The 71-d layout is always the 29-joint G1 one. For the waist-locked robot (``supermimic.envs.lock_waist``) the
simulator has nq = 34 / nv = 33; the missing joints read exactly 0 and are dropped on write.
"""

from __future__ import annotations

import torch

STATE_DIM = 36 + 35
NQ = 36
NV = 35


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        nq, nv = self.sim.data.qpos.shape[-1], self.sim.data.qvel.shape[-1]
        self.full = nq == NQ and nv == NV
        if not self.full:
            from supermimic.envs.lock_waist import joint_slots

            slots = joint_slots(env.scene["robot"].joint_names)
            assert nq == 7 + len(slots) and nv == 6 + len(slots), (nq, nv, len(slots))
            dev = self.origins.device
            # Columns of the 71-d state that the simulator's qpos / qvel fill.
            self.qpos_cols = torch.tensor(list(range(7)) + [7 + j for j in slots], dtype=torch.long, device=dev)
            self.qvel_cols = torch.tensor([NQ + i for i in range(6)] + [NQ + 6 + j for j in slots], dtype=torch.long, device=dev)

    def read(self) -> torch.Tensor:
        """(N, 71) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        if self.full:
            s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        else:
            qpos, qvel = self.sim.data.qpos[:], self.sim.data.qvel[:]
            s = torch.zeros(qpos.shape[0], STATE_DIM, dtype=qpos.dtype, device=qpos.device)
            s[:, self.qpos_cols] = qpos
            s[:, self.qvel_cols] = qvel
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities (locked-joint entries are ignored)."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        if self.full:
            self.sim.data.qpos[env_ids] = state[:, :NQ]
            self.sim.data.qvel[env_ids] = state[:, NQ:]
        else:
            self.sim.data.qpos[env_ids] = state[:, self.qpos_cols]
            self.sim.data.qvel[env_ids] = state[:, self.qvel_cols]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
