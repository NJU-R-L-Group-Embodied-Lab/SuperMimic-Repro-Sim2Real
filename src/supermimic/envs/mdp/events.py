"""Opt-in domain-randomization events (DEVIATIONS from the paper, all default off; tracking_env_cfg.make_env_cfg).

* ``joint_effort_limit_scale`` (reset): per env and episode, the actuator force range (effort limit) of the listed
  joints is the nominal one times s ~ U[lo, hi] (``--env.dr-knee-effort-scale``). mjlab's ``dr.effort_limits`` works on
  whole actuator groups (the G1 knee shares its group with the hip roll), so this selects single joints.
* ``PelvisPushWindow`` (step): with probability ``prob`` per episode, a constant world-frame force / torque on the pelvis
  for ``steps`` policy steps starting at a reference frame drawn from ``frames`` (clip-local frame index, optionally
  only on clips whose name contains one of ``clip_names``). Targets the real-vs-sim state gap before the lean35 landing
  seen in the real deploy logs: the real robot reaches frame ~722 with a different pelvis yaw rate / waist roll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_ctrl_ids(robot, joint_names: tuple[str, ...] | list[str]) -> list[int]:
    """Global (model) actuator ids driving the given joints."""
    by_joint: dict[str, int] = {}
    for act in robot.actuators:
        for name, cid in zip(act.target_names, act.global_ctrl_ids.tolist()):
            by_joint[name.split("/")[-1]] = int(cid)
    missing = [j for j in joint_names if j not in by_joint]
    if missing:
        raise ValueError(f"no actuator for joints {missing}")
    return [by_joint[j] for j in joint_names]


@requires_model_fields("actuator_forcerange")
def joint_effort_limit_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    joint_names: tuple[str, ...],
    scale_range: tuple[float, float],
    entity_name: str = "robot",
) -> None:
    """actuator_forcerange of the listed joints := nominal * U[lo, hi] (one draw per env and joint)."""
    robot = env.scene[entity_name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    ids = torch.tensor(joint_ctrl_ids(robot, joint_names), dtype=torch.long, device=env.device)
    lo, hi = scale_range
    s = lo + (hi - lo) * torch.rand(len(env_ids), len(ids), device=env.device)
    default = env.sim.get_default_field("actuator_forcerange").to(env.device)  # (nu, 2)
    env.sim.model.actuator_forcerange[env_ids[:, None], ids[None, :]] = default[ids][None] * s[..., None]


class PelvisPushWindow:
    """Step event: per-episode pelvis wrench inside a clip-frame window (module docstring)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self.env = env
        self.robot = env.scene[p.get("entity_name", "robot")]
        self.body_ids = [self.robot.body_names.index(p.get("body_name", "pelvis"))]
        n, dev = env.num_envs, env.device
        self.prob = float(p["prob"])
        self.frames = tuple(int(x) for x in p["frames"])
        self.steps = tuple(int(x) for x in p["steps"])
        self.force_max = torch.tensor(p["force_max"], dtype=torch.float32, device=dev)
        self.torque_max = torch.tensor(p["torque_max"], dtype=torch.float32, device=dev)
        self.clip_names = tuple(p.get("clip_names", ()) or ())
        assert self.frames[0] <= self.frames[1] and 1 <= self.steps[0] <= self.steps[1], (self.frames, self.steps)
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.start = torch.zeros(n, dtype=torch.long, device=dev)
        self.dur = torch.ones(n, dtype=torch.long, device=dev)
        self.force = torch.zeros(n, 3, device=dev)
        self.torque = torch.zeros(n, 3, device=dev)
        self.on = torch.zeros(n, dtype=torch.bool, device=dev)  # wrench applied during the coming step (for tests/logs)
        self._clip_ok: torch.Tensor | None = None

    def _eligible_clips(self, lib) -> torch.Tensor:
        if self._clip_ok is None:
            ok = [not self.clip_names or any(s in c.name for s in self.clip_names) for c in lib.clips]
            if not any(ok):
                raise ValueError(f"dr push: no library clip matches {self.clip_names}")
            self._clip_ok = torch.tensor(ok, dtype=torch.bool, device=self.active.device)
        return self._clip_ok

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.env.num_envs, device=self.active.device) if env_ids is None else env_ids
        m, dev = len(ids), self.active.device
        self.active[ids] = torch.rand(m, device=dev) < self.prob
        self.start[ids] = torch.randint(self.frames[0], self.frames[1] + 1, (m,), device=dev)
        self.dur[ids] = torch.randint(self.steps[0], self.steps[1] + 1, (m,), device=dev)
        self.force[ids] = (2 * torch.rand(m, 3, device=dev) - 1) * self.force_max
        self.torque[ids] = (2 * torch.rand(m, 3, device=dev) - 1) * self.torque_max
        self.on[ids] = False
        z = torch.zeros(m, 1, 3, device=dev)
        self.robot.write_external_wrench_to_sim(z, z, env_ids=ids, body_ids=self.body_ids)

    def __call__(self, env: ManagerBasedRlEnv, env_ids, **params) -> None:
        cmd = env.command_manager.get_term(params.get("command_name", "motion"))
        lib = cmd.library
        clip = lib.frame_clip[cmd.frame]
        local = cmd.frame - lib.clip_start[clip]
        on = self.active & self._eligible_clips(lib)[clip] & (local >= self.start) & (local < self.start + self.dur)
        self.on = on
        w = on[:, None].float()
        self.robot.write_external_wrench_to_sim((self.force * w)[:, None], (self.torque * w)[:, None], body_ids=self.body_ids)
