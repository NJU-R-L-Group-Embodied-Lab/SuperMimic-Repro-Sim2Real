"""Per-environment simulation context xi (plan: 33-d = foot friction 1 + encoder bias 29 + torso COM offset 3)."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 33


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link", entity_name: str = "robot"):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.bias_slots = None
        if len(self.robot.joint_names) != 29:  # waist-locked robot: no encoder (bias 0) at the locked joints
            from supermimic.envs.lock_waist import joint_slots

            self.bias_slots = torch.tensor(joint_slots(self.robot.joint_names), dtype=torch.long, device=env.device)
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        if self.bias_slots is not None:
            from supermimic.envs.lock_waist import expand_joints

            bias = expand_joints(bias, self.bias_slots, 0.0)
        xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)
        assert xi.shape == (n, XI_DIM)
        return xi.clone()
