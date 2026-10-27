"""Real-G1 waist roll joint model (opt-in): wider joint range and extra waist-roll damping / friction.

Training: ``--env.waist-roll-range 0.72`` (range), ``--env.dr-waist-roll-damping LO HI`` / ``--env.dr-waist-roll-friction
LO HI`` (per-episode DR, under ``--env.domain-randomization``). sim2sim gates: ``--waist-roll-range 0.72
--waist-roll-damping D --waist-roll-friction F`` or ``SM_WAIST_ROLL="range=0.72,damping=1.8"``.

Why. Both MJCFs (mjlab g1.xml / the exported g1_train_scene.xml, Unitree g1_29dof.xml) limit waist_roll_joint to
+-0.52, but the real G1 goes past it: max |roll| over the POLICY steps of the three lean35 runs (2026-10-26) is
run A -0.670 / +0.701, run B -0.604 / +0.682, run C -0.685 (frame 714). In the run C abort the roll
overshot to -0.69 at frame 714 and was still at -0.16 at 722 (target +0.66 at 718); in sim it stops at -0.52. The
roll/pitch coupling |roll| + 1.4 |pitch| <= 0.95 (supermimic.deploy.waist_coupling) is unchanged and independent of
this: it only binds the 0.72 box at |pitch| > 0.164.

Fit of the waist-roll tracking (q vs q_target, all POLICY steps of the 3 runs):
  * per-joint 2nd-order model qdd = A (q_t - q) - B dq + c, exact 20 ms discretisation: real A 45-60 / B 12-14
    (bandwidth 0.67-0.99 Hz, t63 250-320 ms, pooled 0.81 Hz / 287 ms); sim nominal (training MJCF and Unitree gate
    model, same policy and clip) A 24-25 / B 7.4-7.7 (0.72-0.73 Hz, t63 ~367 ms). No input delay in either (fitted 0 ms).
  * whole-body open-loop replay of the real q_target from the real measured states (1098 windows of 100 ms / 1090 of
    200 ms, Unitree gate model, coupling on, range 0.72): waist-roll error is lowest with kp x1.0 (0.85-1.15 equal),
    kd x2-2.5 (rmse 0.0389 -> 0.0345 at 100 ms, 0.0675 -> 0.0585 at 200 ms), frictionloss 2-4 Nm (0.0346-0.0358),
    armature 0.05 (0.0369). A first-order target lag makes it worse monotonically (10 ms 0.0390, 80 ms 0.0452), lower
    kp too (x0.7 0.0425, x0.5 0.0508).
  So the real waist roll is not slower than sim's on average; it is more damped (about 2x the 1.81 Nm s/rad PD kd) and
  it overshoots the +-0.52 sim stop. Extra joint damping (dof_damping, = kd x (1 + damping / 1.81)) and frictionloss
  are the DR knobs; ranges that cover the fit: damping U[0, 3.6] Nm s/rad (kd x1..x3), friction U[0, 2] Nm.

Only ``mujoco`` is imported here, so the robot-side venv (sim2sim) can use it as well.
"""

from __future__ import annotations

import os

import mujoco

ROLL = "waist_roll_joint"
NOMINAL_RANGE = 0.52  # both MJCFs
REAL_RANGE = 0.72  # covers the real max |roll| 0.70 (run A frame 689)
NOMINAL_KD = 1.814445686584846  # training PD kd of waist_roll (2 x 5020 actuator), for reporting the kd-equivalent scale
ENV_VAR = "SM_WAIST_ROLL"  # "range=0.72,damping=1.8,friction=2"; unset / "" / "0" = off
KEYS = ("range", "damping", "friction")


def _short(name: str | None) -> str:
    return (name or "").split("/")[-1]


def set_waist_roll_range(spec: mujoco.MjSpec, half_range: float) -> mujoco.MjSpec:
    """waist_roll_joint range -> [-half_range, half_range] in ``spec`` (in place; also returned). Names may carry a prefix."""
    if not half_range > 0:
        raise ValueError(f"waist roll range must be > 0, got {half_range}")
    js = [j for j in spec.joints if _short(j.name) == ROLL]
    if len(js) != 1:
        raise ValueError(f"expected one {ROLL} in the spec, found {len(js)} (not compatible with --lock-waist)")
    js[0].range = [-half_range, half_range]
    js[0].limited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def roll_dof(m: mujoco.MjModel) -> int:
    ids = [j for j in range(m.njnt) if _short(m.joint(j).name) == ROLL]
    if len(ids) != 1:
        raise ValueError(f"expected one {ROLL} in the model, found {len(ids)}")
    return int(m.jnt_dofadr[ids[0]])


def apply_actuator_model(m: mujoco.MjModel, damping: float = 0.0, friction: float | None = None) -> mujoco.MjModel:
    """Compiled model, in place: waist-roll dof_damping += damping [Nm s/rad]; frictionloss = friction [Nm] if given."""
    if damping < 0 or (friction is not None and friction < 0):
        raise ValueError(f"waist roll damping / friction must be >= 0, got {damping} / {friction}")
    a = roll_dof(m)
    m.dof_damping[a] += damping
    if friction is not None:
        m.dof_frictionloss[a] = friction
    return m


def waist_roll_from_env() -> dict | None:
    """{"range": float | None, "damping": float, "friction": float | None} from ``SM_WAIST_ROLL``, None when unset/off."""
    v = os.environ.get(ENV_VAR, "").strip()
    if v.lower() in ("", "0", "off", "false", "no"):
        return None
    out: dict = {"range": None, "damping": 0.0, "friction": None}
    try:
        for part in v.split(","):
            k, x = part.split("=")
            k = k.strip()
            if k not in KEYS:
                raise ValueError
            out[k] = float(x)
    except ValueError:
        raise ValueError(f"{ENV_VAR}={v!r}: expected comma-separated key=value with keys {KEYS} (e.g. 'range=0.72,damping=1.8'), "
                         "or unset/'0'/'off'") from None
    if (out["range"] is not None and out["range"] <= 0) or out["damping"] < 0 or (out["friction"] is not None and out["friction"] < 0):
        raise ValueError(f"{ENV_VAR}={v!r}: needs range > 0, damping >= 0, friction >= 0")
    return out


def env_string(range_: float | None, damping: float = 0.0, friction: float | None = None) -> str:
    """``SM_WAIST_ROLL`` value for these settings ("" when all off)."""
    parts = ([f"range={range_}"] if range_ is not None else []) + ([f"damping={damping}"] if damping else []) + (
        [f"friction={friction}"] if friction is not None else [])
    return ",".join(parts)


def resolve_range(env_cfg: dict | None = None) -> float | None:
    """Waist-roll range for an evaluation of a checkpoint: ``SM_WAIST_ROLL`` range if set, else the checkpoint's
    training setting (``cfg["env"]["waist_roll_range"]``), else None (MJCF nominal +-0.52)."""
    wr = waist_roll_from_env()
    if wr is not None and wr["range"] is not None:
        return wr["range"]
    r = (env_cfg or {}).get("waist_roll_range")
    return None if r is None else float(r)


def soft_limits(half_range: float, factor: float = 0.9) -> tuple[float, float]:
    return -factor * half_range, factor * half_range
