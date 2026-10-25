"""mjlab RecorderTerm that streams simulator transitions into a ReplayBuffer.

Hook usage (see ``mjlab/managers/recorder_manager.py``):

* ``record_post_reset(ids)``  fresh episode start: remember s_0, xi, target frame.
* ``record_post_step()``      for envs that did not reset: write row (s_t, a_t, a_exec, xi, r, ...),
                              then remember s_{t+1} as the next row's state.
* ``record_pre_reset(ids)``   terminal step: write the last transition row plus a *terminal* row
                              holding s_{T+1} with its termination label; qpos/qvel are final here
                              even though derived kinematics are one substep stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.sim_context import XI_DIM, SimContext
from supermimic.envs.state_io import STATE_DIM, StateIO

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ReplayRecorder(RecorderTerm):
    def __init__(self, cfg: ReplayRecorderCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.state_io = StateIO(env)
        self.ctx = SimContext(env, p["foot_geom_regex"], p.get("com_body_name", "torso_link"))
        self.command_name = p.get("command_name", "motion")
        self.action_name = p.get("action_name", "joint_pos")
        n = env.num_envs
        self.buffer = ReplayBuffer(n, int(p["capacity"]), STATE_DIM, 29, XI_DIM, env.device)
        self.prev_s = torch.zeros(n, STATE_DIM, device=env.device)
        self.prev_xi = torch.zeros(n, XI_DIM, device=env.device)
        self.prev_frame = torch.zeros(n, dtype=torch.long, device=env.device)
        self.prev_clip = torch.zeros(n, dtype=torch.long, device=env.device)
        self.ep_start = torch.ones(n, dtype=torch.bool, device=env.device)
        self.enabled = True

    # lazy handles (managers are built after the recorder in load_managers order? no: recorder is last)
    @property
    def command(self):
        return self._env.command_manager.get_term(self.command_name)

    @property
    def action_term(self):
        return self._env.action_manager.get_term(self.action_name)

    def _remember(self, ids: torch.Tensor) -> None:
        cmd = self.command
        self.prev_s[ids] = self.state_io.read()[ids]
        self.prev_xi[ids] = self.ctx.read()[ids]
        self.prev_frame[ids] = cmd.frame[ids]
        self.prev_clip[ids] = cmd.library.frame_clip[cmd.frame[ids]]

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._remember(env_ids)
        self.ep_start[env_ids] = True

    def _write_rows(self, ids: torch.Tensor, done_next: torch.Tensor) -> None:
        env = self._env
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=env.action_manager.action[ids],
            a_exec=self.action_term.executed_raw[ids],
            xi=self.prev_xi[ids],
            r=env.reward_buf[ids],
            done=done_next,
            frame=self.prev_frame[ids],
            clip=self.prev_clip[ids],
            terminal=torch.zeros_like(done_next, dtype=torch.bool),
            ep_start=self.ep_start[ids],
        )
        self.ep_start[ids] = False

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled or env_ids.numel() == 0:
            return
        env = self._env
        terminated = env.reset_terminated[env_ids].float()
        self._write_rows(env_ids, terminated)
        # Terminal row: the state reached by the last action (no transition out of it).
        cmd = self.command
        s_next = self.state_io.read()[env_ids]
        zeros_a = torch.zeros(env_ids.numel(), 29, device=env.device)
        self.buffer.add(
            env_ids,
            s=s_next,
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[env_ids],
            r=torch.zeros(env_ids.numel(), device=env.device),
            done=terminated,
            frame=cmd.frame[env_ids],
            clip=cmd.library.frame_clip[cmd.frame[env_ids]],
            terminal=torch.ones(env_ids.numel(), dtype=torch.bool, device=env.device),
            ep_start=torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device),
        )
        self.buffer.end_episode(env_ids)

    def record_post_step(self) -> None:
        if not self.enabled:
            return
        env = self._env
        mask = ~env.reset_buf
        ids = mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        self._write_rows(ids, torch.zeros(ids.numel(), device=env.device))
        self._remember(ids)


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000, "foot_geom_regex": r"^(left|right)_foot[1-7]_collision$"})
