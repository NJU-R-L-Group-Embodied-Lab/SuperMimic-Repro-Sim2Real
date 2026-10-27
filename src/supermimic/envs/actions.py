"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).

``locked_joint_ids`` (waist-locked robot, ``supermimic.envs.lock_waist``): the entity has fewer joints than the
policy has actions; the action stays full-size (29) and the entries at the locked slots are simply not applied.

``delay_substeps_max`` (DEVIATION when > 0, default 0): per-episode command latency. Each env draws d in
{0..delay_substeps_max} physics substeps at reset; for the first d substeps of every policy step the previous
set-point is still applied (5 ms substeps: 0-20 ms for max 4), like the real runner -> bridge -> motor path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        locked = tuple(cfg.locked_joint_ids)
        self._active_slots = None
        if locked:
            n_full = self._action_dim + len(locked)
            active = [i for i in range(n_full) if i not in locked]
            assert len(active) == self._action_dim, (n_full, locked, self._action_dim)
            self._active_slots = torch.tensor(active, dtype=torch.long, device=self.device)
            self._full_dim = n_full
            self._raw_actions = torch.zeros(self.num_envs, n_full, device=self.device)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self.delay_max = int(cfg.delay_substeps_max)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_processed = torch.zeros_like(self._processed_actions)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep = 0

    @property
    def action_dim(self) -> int:
        return getattr(self, "_full_dim", None) or self._action_dim

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        if self.delay_max > 0:
            self._prev_processed = self._processed_actions.clone()
            self._substep = 0
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        executed = self._executed_raw if self._active_slots is None else self._executed_raw[:, self._active_slots]
        self._processed_actions = executed * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self.delay_max > 0:  # no stale set-point from the previous episode: a fresh env starts on its first action
            self._prev_processed[self._fresh] = self._processed_actions[self._fresh]
            self._fresh[:] = False

    def apply_actions(self) -> None:
        if self.delay_max == 0:
            return super().apply_actions()
        cur = self._processed_actions
        late = (self._substep < self._delay).unsqueeze(-1)
        self._processed_actions = torch.where(late, self._prev_processed, cur)
        try:
            super().apply_actions()
        finally:
            self._processed_actions = cur
        self._substep += 1

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0
        if self.delay_max > 0:
            ids = slice(None) if env_ids is None else env_ids
            n = self.num_envs if env_ids is None else self._delay[ids].numel()
            self._delay[ids] = torch.randint(0, self.delay_max + 1, (n,), device=self.device)
            self._fresh[ids] = True


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1
    locked_joint_ids: tuple[int, ...] = ()  # policy-action slots with no joint in the entity (ignored)
    delay_substeps_max: int = 0  # per-episode command latency drawn from {0..max} physics substeps (0 = off)

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)
