"""Real-G1 waist roll/pitch coupling as a MuJoCo physics constraint (opt-in: ``--env.waist-coupling`` in training,
``scripts/deploy/sim2sim.py --waist-coupling`` / ``SM_WAIST_COUPLING=k,c`` in the sim2sim gates).

On the real G1 the waist roll and pitch are driven through a parallel linkage, so the reachable |roll| shrinks with
|pitch|. The MJCFs (mjlab g1.xml, Unitree g1_29dof.xml) model them as independent joints with box ranges +-0.52, so a
sim policy can use roll -0.53 at pitch 0.53, which the real robot cannot reach (seen on the real robot, lean35 frames 702-711:
pitch 0.53, roll stalled at -0.17..-0.22 against a -0.95..-1.65 target -> missing torso roll -> tilt abort).

Constraint (diamond envelope, intersected with the unchanged +-0.52 joint boxes):
    |roll| + k |pitch| <= c
implemented with two limited fixed tendons  t+ = roll + k pitch,  t- = roll - k pitch,  range [-c, c].

Fit (2026-10-26, all real deploy logs deploy_logs/real/*/*.npz, policy-state samples, 28 runs / 28 798 samples):
k = 1.4, c = 0.95. Every sample with |pitch| <= 0.55 is inside (the 8 outside ones are at |pitch| > 0.55, beyond the
sim pitch range); limit 0.39 / 0.32 / 0.21 at |pitch| 0.40 / 0.45 / 0.53 vs observed max |roll| 0.35 / 0.29 / 0.22.
The fit is symmetric in the pitch sign, the data is not: at pitch > 0 the hull is c ~ 0.92 (c = 0.95 leaves ~0.03 rad of
roll slack where lean35 fails); at pitch < 0 the observed |roll| is smaller (<= 0.27 above |pitch| 0.2, few samples), so
the envelope is loose there. Only joint positions are limited: the linkage's shared motor torque is not modelled.

Only ``mujoco`` is imported here, so the robot-side venv (sim2sim) can use it as well.
"""

from __future__ import annotations

import os

import mujoco

WAIST_COUPLING_K = 1.4
WAIST_COUPLING_C = 0.95
ROLL, PITCH = "waist_roll_joint", "waist_pitch_joint"
TENDON_NAMES = ("waist_coupling_pos", "waist_coupling_neg")  # roll + k pitch, roll - k pitch
# Stiffer than MuJoCo's default limit (solref 0.02, solimp 0.9/0.95). Quasi-static (joint pushed slowly into the limit by
# 50 Nm of roll drive) the penetration drops from ~0.03-0.05 rad with the defaults to <= 0.003 rad with these. Dynamic
# transients are larger: in the 1 kHz sim2sim lean35 runs the envelope is briefly exceeded by up to ~0.05 rad.
# (timeconst 0.01 = MuJoCo's 2 dt floor at the 5 ms training step.)
SOLREF_LIMIT = (0.01, 1.0)
SOLIMP_LIMIT = (0.95, 0.99, 0.001, 0.5, 2.0)
ENV_VAR = "SM_WAIST_COUPLING"  # "k,c" (or "1" for the defaults); unset / "" / "0" = off


def _short(name: str | None) -> str:
    return (name or "").split("/")[-1]


def add_waist_coupling(spec: mujoco.MjSpec, k: float = WAIST_COUPLING_K, c: float = WAIST_COUPLING_C) -> mujoco.MjSpec:
    """Add the two limited fixed tendons to ``spec`` (in place; also returned). Joint names may carry a prefix."""
    if k < 0 or c <= 0:
        raise ValueError(f"waist coupling needs k >= 0 and c > 0, got k={k} c={c}")
    names = {_short(j.name): j.name for j in spec.joints}
    if ROLL not in names or PITCH not in names:
        raise ValueError("waist coupling needs waist_roll_joint and waist_pitch_joint (not compatible with --lock-waist)")
    prefix = names[ROLL][: -len(ROLL)]
    for tname, sgn in zip(TENDON_NAMES, (1.0, -1.0)):
        t = spec.add_tendon()
        t.name = prefix + tname
        t.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        t.range = [-c, c]
        t.solref_limit = list(SOLREF_LIMIT)
        t.solimp_limit = list(SOLIMP_LIMIT)
        t.wrap_joint(names[ROLL], 1.0)
        t.wrap_joint(names[PITCH], sgn * k)
    return spec


def coupling_from_env() -> tuple[float, float] | None:
    """(k, c) from ``SM_WAIST_COUPLING`` ("k,c" or "1" = defaults), None when unset/off."""
    v = os.environ.get(ENV_VAR, "").strip()
    if v.lower() in ("", "0", "off", "false", "no"):
        return None
    if v.lower() in ("1", "on", "true", "yes"):
        return WAIST_COUPLING_K, WAIST_COUPLING_C
    parts = v.split(",")
    try:
        if len(parts) != 2:
            raise ValueError
        k, c = (float(x) for x in parts)
    except ValueError:
        raise ValueError(f"{ENV_VAR}={v!r}: expected 'k,c' (e.g. '{WAIST_COUPLING_K},{WAIST_COUPLING_C}'), '1'/'on' for the defaults, "
                         "or unset/'0'/'off'") from None
    if k < 0 or c <= 0:
        raise ValueError(f"{ENV_VAR}={v!r}: needs k >= 0 and c > 0")
    return k, c


def resolve_coupling(env_cfg: dict | None = None) -> tuple[float, float] | None:
    """(k, c) for an evaluation of a checkpoint: ``SM_WAIST_COUPLING`` if set, else the checkpoint's training setting
    (``cfg["env"]`` dict with waist_coupling / _k / _c), else None (uncoupled)."""
    wc = coupling_from_env()
    if wc is None and env_cfg and env_cfg.get("waist_coupling", False):
        wc = (float(env_cfg.get("waist_coupling_k", WAIST_COUPLING_K)), float(env_cfg.get("waist_coupling_c", WAIST_COUPLING_C)))
    return wc


def env_args(wc: tuple[float, float] | None) -> dict:
    """SuperMimicEnvCfgArgs keyword arguments for (k, c) / None; empty when off (so the default args are untouched)."""
    return {} if wc is None else {"waist_coupling": True, "waist_coupling_k": wc[0], "waist_coupling_c": wc[1]}


def envelope(pitch, k: float = WAIST_COUPLING_K, c: float = WAIST_COUPLING_C, joint_limit: float = 0.52):
    """Max |roll| reachable at ``pitch`` under the coupling and the joint box."""
    import numpy as np

    return np.minimum(joint_limit, c - k * np.abs(pitch))
