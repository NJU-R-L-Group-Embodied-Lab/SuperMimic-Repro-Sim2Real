from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def _slots(env: ManagerBasedRlEnv, robot) -> torch.Tensor:
    cache = getattr(env, "_sm_joint_slots", None)
    if cache is None:
        from supermimic.envs.lock_waist import joint_slots

        cache = torch.tensor(joint_slots(robot.joint_names), dtype=torch.long, device=env.device)
        env._sm_joint_slots = cache
    return cache


def joint_pos_rel_full(env: ManagerBasedRlEnv, biased: bool = True) -> torch.Tensor:
    """mjlab ``joint_pos_rel`` in the 29-joint layout: joints missing from the (waist-locked) entity read exactly 0,
    like the real lowstate (no encoder bias on them; noise is zeroed per-dimension in the term config)."""
    from supermimic.envs.lock_waist import expand_joints

    robot = env.scene["robot"]
    q = (robot.data.joint_pos_biased if biased else robot.data.joint_pos) - robot.data.default_joint_pos
    return expand_joints(q, _slots(env, robot), 0.0)


def joint_vel_rel_full(env: ManagerBasedRlEnv) -> torch.Tensor:
    """mjlab ``joint_vel_rel`` in the 29-joint layout (missing joints read exactly 0)."""
    from supermimic.envs.lock_waist import expand_joints

    robot = env.scene["robot"]
    return expand_joints(robot.data.joint_vel - robot.data.default_joint_vel, _slots(env, robot), 0.0)
