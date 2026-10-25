"""Waist-locked G1 ("g1_29dof_lock_waist_rev_1_0", mode_machine 6) for training (``--env.lock-waist``).

The real robot has waist roll / pitch (motor indices 13 / 14) mechanically locked: no motors, lowstate reports
q = dq = 0 for them. Unitree's ``g1_29dof_lock_waist_rev_1_0.xml`` is the 29-DoF model with exactly these two
joints (and their actuators) deleted, so ``waist_roll_link`` / ``torso_link`` are welded to ``waist_yaw_link`` at
q = 0 with unchanged masses and inertias. We do the same to mjlab's training MJCF (same deletion as
``scripts/deploy/sim2sim.py --lock-waist``).

The policy keeps the 29-d interface of the unlocked robot (identical to the deploy runtime):
  * actions are 29-d; entries 13/14 are ignored by the simulator (no actuator), they are still observed as the
    last action and penalised by the action-rate (and optional locked-action) terms;
  * the joint observations are 29-d with entries 13/14 exactly 0 (no encoder bias, no noise), like the real lowstate;
  * the 71-d world-model state keeps the 29-joint MuJoCo layout with q / dq of 13/14 identically 0.

The simulator entity therefore has 27 joints; the helpers below map between the entity's 27-joint order (MJCF
order without the two locked joints) and the 29-joint order used everywhere else.
"""

from __future__ import annotations

import mujoco
import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES

LOCKED_JOINT_NAMES: tuple[str, ...] = ("waist_roll_joint", "waist_pitch_joint")
LOCKED_JOINT_IDS: tuple[int, ...] = tuple(G1_JOINT_NAMES.index(n) for n in LOCKED_JOINT_NAMES)  # (13, 14)
NUM_JOINTS = len(G1_JOINT_NAMES)  # 29
ACTIVE_JOINT_IDS: tuple[int, ...] = tuple(i for i in range(NUM_JOINTS) if i not in LOCKED_JOINT_IDS)
assert LOCKED_JOINT_IDS == (13, 14)


def _short(name: str | None) -> str:
    return (name or "").split("/")[-1]


def lock_waist_spec(spec: mujoco.MjSpec | None = None) -> mujoco.MjSpec:
    """mjlab's G1 MjSpec with waist roll/pitch deleted (child bodies welded at q = 0)."""
    if spec is None:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

        spec = get_spec()
    for a in list(spec.actuators):
        if _short(a.target) in LOCKED_JOINT_NAMES:
            spec.delete(a)
    for s in list(spec.sensors):
        if _short(s.objname) in LOCKED_JOINT_NAMES:
            spec.delete(s)
    if any(_short(j.name) in LOCKED_JOINT_NAMES for j in spec.joints):
        for k in list(spec.keys):  # keyframes have the unlocked qpos size
            spec.delete(k)
    for j in list(spec.joints):
        if _short(j.name) in LOCKED_JOINT_NAMES:
            spec.delete(j)
    return spec


def get_g1_lock_waist_robot_cfg():
    """``get_g1_robot_cfg()`` with the locked model: same init state, collisions and actuators minus the waist
    roll/pitch actuator group (its two targets no longer exist)."""
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    actuators = tuple(a for a in g.G1_ARTICULATION.actuators if a is not g.G1_ACTUATOR_WAIST)
    assert len(actuators) == len(g.G1_ARTICULATION.actuators) - 1
    assert set(g.G1_ACTUATOR_WAIST.target_names_expr) == set(LOCKED_JOINT_NAMES)
    return EntityCfg(
        init_state=g.KNEES_BENT_KEYFRAME,
        collisions=(g.FULL_COLLISION,),
        spec_fn=lock_waist_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=g.G1_ARTICULATION.soft_joint_pos_limit_factor),
    )


def joint_slots(entity_joint_names) -> list[int]:
    """Position in the 29-joint order of each entity joint (entity joint names may carry a prefix)."""
    names = [_short(n) for n in entity_joint_names]
    slots = [G1_JOINT_NAMES.index(n) for n in names]
    assert slots == sorted(slots), "entity joint order must follow the MJCF order"
    return slots


def is_locked_entity(robot) -> bool:
    return len(robot.joint_names) != NUM_JOINTS


def expand_joints(x: torch.Tensor, slots: list[int] | torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """(..., n_entity) -> (..., 29), locked slots set to ``fill``."""
    if x.shape[-1] == NUM_JOINTS:
        return x
    out = torch.full((*x.shape[:-1], NUM_JOINTS), fill, dtype=x.dtype, device=x.device)
    out[..., slots] = x
    return out


def full_joint_constants(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """29-joint default joint positions (29,) and soft limits (29, 2) of a (possibly locked) entity.

    Locked joints: default 0 (the lock pose) and the 29-DoF model's soft limits (0.9 x range); q = 0 is inside, so
    they never contribute to the joint-limit penalty.
    """
    q0 = robot.data.default_joint_pos[0].detach().clone()
    soft = robot.data.soft_joint_pos_limits[0].detach().clone()
    if not is_locked_entity(robot):
        return q0, soft
    from supermimic.data.motion_convert import load_g1_model

    m = load_g1_model()
    rng = torch.as_tensor(m.jnt_range[1:], dtype=soft.dtype, device=soft.device)  # skip the free joint; 29 x 2
    factor = float(robot.cfg.articulation.soft_joint_pos_limit_factor)
    mid, half = rng.mean(-1), 0.5 * (rng[:, 1] - rng[:, 0])
    soft_full = torch.stack([mid - factor * half, mid + factor * half], dim=-1)
    slots = joint_slots(robot.joint_names)
    soft_full[slots] = soft
    return expand_joints(q0, slots, 0.0), soft_full
