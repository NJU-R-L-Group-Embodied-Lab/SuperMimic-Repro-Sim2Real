"""Sim2sim gates for the G1 deployment of the BONES-SEED tracking policy (CPU MuJoCo + onnxruntime, no torch).

Runs in the robot-side deploy venv ($G1_DEPLOY_HOME/env.sh, python3.10); every policy-related computation goes
through supermimic.deploy.core.DeployCore (imported by putting <repo>/src on sys.path).

Gates (round 2: core defaults soft-limit clamp off, torque clamp 1.5 x effort)
  g1  same-model: dumped training MJCF (position actuators = training PD), dt 0.005, policy every 4 substeps,
      benchmark init (clip frame 0 incl. reference velocities). Compared with training-env rollouts written by
      scripts/deploy/sim2sim_ref.py (training venv) to deploy_artifacts/reports/sim2sim/ref_<ckpt>_<tf32|no-tf32>.npz.
      Variants: torque clamp factor 1.5 / 1.0 / 0.5, no clamps, soft clamp on (reference).
  g2  realistic: Unitree's official G1 29-DoF MJCF (torque motors); PD computed here at 1 kHz from the latest
      (delayed) q_target with training kp/kd, clipped to the effort limit; robot starts at rest on the ground in
      the clip's frame-0 pose. Variants (G2_VARIANTS): core defaults (full sweep latency/kp/mass/friction for the
      recommended ckpt), 0.5 s settle, targets clipped to the hard joint range, hip-roll effort 88 Nm, torque clamp
      factor 1.0 / 0.5 / off.
  g3  transitions (runner protocol): q0 stand -> 4 s cosine interpolation to frame 0 (stand gains) -> 1 s HOLD ->
      POLICY (training gains) -> END blend (1 s ramp to stand gains / last-frame pose, effort clamp) -> 2 s hold;
      gantry spring on the pelvis; plus staged abort thresholds (air / feet / slack) on the SAFE clips.
  margins  |a| and |dq|/velocity_limit along the fp32 training-env reference rollouts.
  g5  timing of DeployCore.step (incl. ONNX), single thread.
  compare  Unitree MJCF vs training MJCF: masses, inertias, armature, damping, friction loss, ranges, effort.
  report   deploy_artifacts/reports/sim2sim_<ckpt>.json -> deploy_artifacts/reports/SIM2SIM.md
  standby  runner policy standby (STANDBY_PRE -> motion -> STANDBY_POST without support), Unitree MJCF, slack/ground profiles,
      latency 0/10/20 ms x mu 0.6/1.0 x kp 0.85/1.0/1.15 x torso +0/+1.8 kg -> sim2sim_standby_<ckpt>.json / SIM2SIM_standby_<ckpt>.md
      python scripts/deploy/sim2sim.py --cmd standby --ckpt ckpt_0116000 --workers 16
      R1+X is pressed at --standby-pre-s and the motion starts only once the runner's check passes (STANDBY_PRE >= 3 s, last 3 s
      tilt err < 8 deg and max|dq| < 1 rad/s; --standby-r1x-*), re-pressed for up to --standby-r1x-cap-s, else no motion ('never').
  --lock-waist  "G1 29dof with waist locked" (the real robot: motors 13/14 absent, lowstate q = dq = 0): waist_roll/pitch
      joints, actuators and sensors are deleted from BOTH MJCFs (bodies welded at q = 0, as Unitree's
      g1_29dof_lock_waist.xml), q = dq = 0 reported to DeployCore, actions 13/14 computed but never applied.
      Applies to g1/g2/g3 (results -> sim2sim_lw_<ckpt>.json).
  lw / lw_robust  locked vs unlocked twins on the same grid (G1-style training MJCF + G2 nominal latency 0/10/20 ms,
      mu 0.6/1.0; lw_robust: kp 0.85/1.15, +1.8 kg) -> deploy_artifacts/reports/sim2sim_lockwaist.json
      python scripts/deploy/sim2sim.py --cmd lw --ckpt ckpt_0116000 ckpt_0100000 ckpt_0140000 ckpt_0144000 --workers 16

  source $G1_DEPLOY_HOME/env.sh   # default ~/workspace/g1_deploy
  python scripts/deploy/sim2sim.py --cmd all --workers 10

Env: G1_UNITREE_XML (Unitree G1 29-DoF scene; default $G1_DEPLOY_HOME/unitree_mujoco/unitree_robots/g1/scene_29dof.xml,
G1_DEPLOY_HOME default ~/workspace/g1_deploy).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from supermimic.deploy.core import Clip, DeployConfig, DeployCore, OnnxPolicy, SafetyCfg, interpolate_pose, projected_gravity  # noqa: E402

ART = REPO / "deploy_artifacts"
REPORTS = ART / "reports"
TRAIN_XML = ART / "models" / "g1_train_scene.xml"
UNITREE_XML = Path(os.environ.get("G1_UNITREE_XML", Path(os.environ.get("G1_DEPLOY_HOME", Path.home() / "workspace/g1_deploy")).expanduser()
                                   / "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"))
CKPTS = ("ckpt_0100000", "ckpt_0116000")
SAFE = ("neutral_idle_loop_001__A081", "idle_loop_002__A060", "shoulder_clap_004__A041_M", "itching_left_arm_R_003__A121", "walk_sideway_090_stop_003__A037")
HARDER = ("wave_R_001__A431", "jog_sideway_045_loop_002__A025", "dance_hiphop_alt_stump_R_fast_001__A318_M")
CLIPS = SAFE + HARDER
IDLE = "neutral_idle_loop_001__A081"

FALL_HEIGHT = 0.5  # pelvis height [m]
FALL_TILT = np.radians(float(os.environ.get("SM_FALL_TILT_DEG", "60")))  # override only for diagnostics
TRAIN_HEIGHT_DEV, TRAIN_TILT_DEV = 0.25, 0.8  # training terminations (pelvis height / gravity-z vs reference)
DIVERGE_Q_RMS, DIVERGE_XY = 0.05, 0.10  # sim2sim vs training-env robot trajectories

# Stand gains (order hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | waist | arms)
STAND_LEG_KP, STAND_LEG_KD = (100, 100, 100, 150, 40, 40), (2, 2, 2, 4, 2, 2)
STAND_WAIST_KP, STAND_WAIST_KD = 200.0, 5.0
STAND_ARM_KP, STAND_ARM_KD = 40.0, 10.0

G2_LATENCY_MS = (0, 5, 10, 20)
G2_KP_SCALE = (0.85, 1.0, 1.15)
G2_EXTRA_MASS = (0.0, 1.8)
G2_FRICTION = (0.6, 1.0)
BAND_K, BAND_D = 2000.0, 200.0  # G3 gantry spring on the pelvis [N/m], damper [N s/m]
AIR_K, AIR_D, AIR_CLEARANCE = 20000.0, 1500.0, 0.05  # hung from the torso COM, feet 5 cm above the ground
# Staged runner abort thresholds (max_vel_frac, max_abs_action): stages 5.3/5.4 (air, feet on ground + gantry), 5.5+ (gantry slack)
STAGES = {"air/feet": (0.3, 10.0), "slack": (0.5, 15.0)}
# "G1 29dof with waist locked": waist roll/pitch (motors 13/14) are mechanically locked; the real lowstate reports
# q = dq = 0 for them and no motor answers. --lock-waist removes these joints (and their actuators/sensors) from
# the MJCF, which welds torso to pelvis at q = 0 exactly as Unitree's g1_29dof_lock_waist.xml does.
LOCKED_WAIST = ("waist_roll_joint", "waist_pitch_joint")
WRP = (13, 14)  # motor indices of the locked joints


def load_model(xml: Path, lock: bool = False) -> mujoco.MjModel:
    """MJCF -> MjModel; with lock=True, LOCKED_WAIST joints are deleted (child bodies welded at q = 0)."""
    if not lock:
        return mujoco.MjModel.from_xml_path(str(xml))
    spec = mujoco.MjSpec.from_file(str(xml))
    for a in list(spec.actuators):
        if short(a.target) in LOCKED_WAIST:
            spec.delete(a)
    for s in list(spec.sensors):
        if short(s.objname) in LOCKED_WAIST:
            spec.delete(s)
    for k in list(spec.keys):  # keyframes have the unlocked qpos size
        spec.delete(k)
    for j in list(spec.joints):
        if short(j.name) in LOCKED_WAIST:
            spec.delete(j)
    m = spec.compile()
    left = [short(m.joint(i).name) for i in range(m.njnt)]
    assert not set(LOCKED_WAIST) & set(left) and m.nu == 27, (left, m.nu)
    return m


# ============================================================================================ helpers


def short(name: str) -> str:
    return name.split("/")[-1]


@lru_cache(maxsize=None)
def load_cfg(ckpt: str) -> DeployConfig:
    return DeployConfig.load(ART / ckpt / "deploy_config.json")


@lru_cache(maxsize=None)
def load_clip(name: str) -> Clip:
    return Clip.load(ART / "clips" / f"{name}.npz")


@lru_cache(maxsize=None)
def load_clip_raw(name: str) -> dict:
    with np.load(ART / "clips" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def load_policy(ckpt: str) -> OnnxPolicy:
    return OnnxPolicy(ART / ckpt / "policy.onnx", threads=1)


class Recorder:
    """Wraps the policy to keep the (obs, goal) the core fed it (for the step-1 comparison)."""

    def __init__(self, policy):
        self.policy, self.obs, self.goal = policy, None, None

    def __call__(self, obs, goal):
        self.obs, self.goal = obs.copy(), goal.copy()
        return self.policy(obs, goal)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def tilt_of(quat_wxyz) -> float:
    g = projected_gravity(np.asarray(quat_wxyz, float))
    return float(np.arccos(np.clip(-g[2], -1.0, 1.0)))


class Robot:
    """MuJoCo model + the state extraction a real G1 gives us (motor-order q/dq, pelvis IMU gyro + quaternion)."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str]):
        self.m, self.d = model, mujoco.MjData(model)
        m = model
        names = [short(m.joint(i).name) for i in range(m.njnt)]
        # Joints missing from the model are locked (see LOCKED_WAIST): reported as q = dq = 0, no torque applied.
        self.nj = len(joint_names)
        self.locked = np.array([n not in names for n in joint_names])
        self.live = ~self.locked
        assert all(joint_names[i] in LOCKED_WAIST for i in np.nonzero(self.locked)[0]), "unexpected missing joints"
        self.jid = np.array([names.index(n) for n in joint_names if n in names])
        assert all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in self.jid)
        self.qadr, self.dadr = m.jnt_qposadr[self.jid], m.jnt_dofadr[self.jid]
        free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1
        self.fq, self.fv = m.jnt_qposadr[free[0]], m.jnt_dofadr[free[0]]
        self.pelvis = m.jnt_bodyid[free[0]]
        assert short(m.body(self.pelvis).name) == "pelvis"
        act_j = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
        self.act = np.array([act_j.index(j) for j in self.jid])  # actuator index per live motor-order joint
        self.range = np.tile([-np.inf, np.inf], (self.nj, 1))
        self.range[self.live] = m.jnt_range[self.jid]

    # -------------------------------------------------------------- motor-order (29) accessors; locked joints read 0
    def jq(self, qpos=None) -> np.ndarray:
        q = np.zeros(self.nj)
        q[self.live] = (self.d.qpos if qpos is None else qpos)[self.qadr]
        return q

    def jdq(self) -> np.ndarray:
        dq = np.zeros(self.nj)
        dq[self.live] = self.d.qvel[self.dadr]
        return dq

    def set_ctrl(self, u) -> None:
        self.d.ctrl[self.act] = np.asarray(u)[self.live]  # locked joints: nothing to actuate

    # -------------------------------------------------------------- state as the robot reports it
    def state(self):
        d = self.d
        q, dq = self.jq(), self.jdq()
        quat = d.qpos[self.fq + 3 : self.fq + 7].copy()
        gyro = d.qvel[self.fv + 3 : self.fv + 6].copy()  # free joint: angular velocity in the body (pelvis) frame
        return q, dq, gyro, quat

    def gyro_check(self) -> float:
        """|qvel[3:6] - pelvis angular velocity in pelvis frame (mj_objectVelocity, local)|."""
        d2 = copy.copy(self.d)  # after mj_step cvel is stale (pre-integration); recompute on a copy
        mujoco.mj_forward(self.m, d2)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d2, mujoco.mjtObj.mjOBJ_XBODY, self.pelvis, v, 1)  # XBODY = body frame (BODY = inertial frame)
        return float(np.abs(v[:3] - d2.qvel[self.fv + 3 : self.fv + 6]).max())

    def root(self):
        return self.d.qpos[self.fq : self.fq + 3].copy(), self.d.qpos[self.fq + 3 : self.fq + 7].copy()

    def set_state(self, root_pos, root_quat, q, root_lin_w=None, root_ang_w=None, dq=None):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.fq : self.fq + 3] = root_pos
        d.qpos[self.fq + 3 : self.fq + 7] = root_quat
        d.qpos[self.qadr] = np.asarray(q)[self.live]
        if root_lin_w is not None:
            d.qvel[self.fv : self.fv + 3] = root_lin_w  # free joint linear velocity: world frame
        if root_ang_w is not None:
            d.qvel[self.fv + 3 : self.fv + 6] = quat_to_mat(root_quat).T @ np.asarray(root_ang_w)  # body frame
        if dq is not None:
            d.qvel[self.dadr] = np.asarray(dq)[self.live]
        mujoco.mj_forward(self.m, d)

    def foot_clearance(self) -> float:
        """Lowest point of the ankle-roll-link collision geoms (z)."""
        m, d = self.m, self.d
        low = np.inf
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if "ankle_roll" not in short(m.body(m.geom_bodyid[g]).name):
                continue
            r = m.geom_size[g, 0]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                low = min(low, d.geom_xpos[g, 2] - r)
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
                low = min(low, d.geom_xpos[g, 2] - abs(ax[2]) - r)
        return float(low)

    def place_on_ground(self) -> float:
        dz = -self.foot_clearance()
        self.d.qpos[self.fq + 2] += dz
        mujoco.mj_forward(self.m, self.d)
        return dz


def rms(x) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def rms_ex_wrp(err) -> float:
    """Joint RMSE over the 27 joints that are not WRP (waist roll/pitch), for locked vs unlocked comparisons."""
    e = np.asarray(err)
    return rms(np.delete(e, list(WRP), axis=1)) if e.size else float("nan")


def with_effort(cfg: DeployConfig, hip_roll_effort: float | None) -> DeployConfig:
    """Deploy config with the hip-roll effort limit replaced (used by the core's torque clamp and the simulated motor)."""
    if hip_roll_effort is None:
        return cfg
    eff = cfg.effort_limit.copy()
    eff[[i for i, n in enumerate(cfg.joint_names) if "hip_roll" in n]] = hip_roll_effort
    return dataclasses.replace(cfg, effort_limit=eff)


def clip_segments(raw: dict) -> list[tuple[str, int, int]] | None:
    """Frame segments of a combo clip (segment_* keys written by make_combo_clip.py, copied by the exporter), else None."""
    if "segment_names" not in raw:
        return None
    return [(str(n), int(a), int(b)) for n, a, b in zip(raw["segment_names"], raw["segment_start"], raw["segment_end"])]


def safety(o: dict) -> SafetyCfg:
    """SafetyCfg from the run options; anything not given keeps the current core default."""
    kw = {}
    for k_opt, k_cfg in (("soft", "clamp_soft_limits"), ("torque", "clamp_torque"), ("factor", "torque_clamp_factor"), ("max_vel_frac", "max_vel_frac"), ("max_abs_action", "max_abs_action")):
        if o.get(k_opt) is not None:
            kw[k_cfg] = o[k_opt]
    return SafetyCfg(**kw)


class StepStats:
    """Per policy step: raw action size, sampled |dq|/velocity_limit (what the core's abort check sees), clamped joints."""

    def __init__(self, cfg: DeployConfig, segments: list[tuple[str, int, int]] | None = None):
        self.cfg = cfg
        self.amax, self.vfrac, self.vjoint, self.clamped = [], [], [], []
        self.segments = segments  # (name, start, end) frame ranges of a combo clip (make_combo_clip.py), else None
        self.frame, self.terr, self.jmse = [], [], []  # per policy step (mark): reference frame at the step, tilt error [rad], joint MSE
        self.wrist = [i for i, n in enumerate(cfg.joint_names) if "wrist" in n]
        self.a_wrp, self.tau_wrp = [], []  # raw actions 13/14 and |kp (q_t - q) - kd dq| / effort the PD would ask of them

    def add(self, dq, r, q=None) -> None:
        c = self.cfg
        v = np.abs(np.asarray(dq)) / c.velocity_limit
        w = list(WRP)
        self.a_wrp.append(np.asarray(r.action)[w])
        if q is not None:
            self.tau_wrp.append(np.abs(c.kp[w] * (r.q_target[w] - np.asarray(q)[w]) - c.kd[w] * np.asarray(dq)[w]) / c.effort_limit[w])
        self.amax.append(float(np.abs(r.action).max()))
        self.vfrac.append(float(v.max()))
        self.vjoint.append(int(np.argmax(v)))
        raw = r.action * c.action_scale + c.default_joint_pos
        self.clamped.append(np.abs(r.q_target - raw) > 1e-9)

    def mark(self, frame: int, tilt_err: float, jmse: float | None = None) -> None:
        """Once per policy step (with add): the reference frame the step started at, tilt error vs the reference, joint MSE."""
        self.frame.append(int(frame))
        self.terr.append(float(tilt_err))
        self.jmse.append(np.nan if jmse is None else float(jmse))

    def segment_summary(self) -> dict:
        n = min(len(self.amax), len(self.frame))
        a, v, f = np.array(self.amax[:n]), np.array(self.vfrac[:n]), np.array(self.frame[:n])
        te, je, vj = np.array(self.terr[:n]), np.array(self.jmse[:n]), np.array(self.vjoint[:n])
        out = {}
        for name, s0, s1 in self.segments:
            m = (f >= s0) & (f < s1)
            if not m.any():
                out[name] = {"steps": 0}
                continue
            i = int(np.argmax(np.where(m, v, -1)))
            out[name] = {"steps": int(m.sum()), "max_abs_a": float(a[m].max()), "max_vfrac": float(v[m].max()), "max_vfrac_joint": self.cfg.joint_names[vj[i]],
                         "max_tilt_err_deg": float(np.degrees(te[m].max())),
                         "rmse": float(np.sqrt(np.nanmean(je[m]))) if np.isfinite(je[m]).any() else None,
                         "stage_trips": {s: int(((v[m] > vt) | (a[m] > at)).sum()) for s, (vt, at) in STAGES.items()}}
        return out

    def summary(self) -> dict:
        if not self.amax:
            return {}
        a, v, cl = np.array(self.amax), np.array(self.vfrac), np.array(self.clamped)
        names = self.cfg.joint_names
        per = cl.mean(0)
        return {
            "max_abs_a": float(a.max()), "p99_abs_a": float(np.percentile(a, 99)),
            "n_a_gt10": int((a > 10).sum()), "n_a_gt15": int((a > 15).sum()), "policy_steps": int(len(a)),
            "max_vfrac": float(v.max()), "max_vfrac_joint": names[self.vjoint[int(np.argmax(v))]],
            "stage_trips": {s: int(((v > vt) | (a > at)).sum()) for s, (vt, at) in STAGES.items()},
            "n_clamped_mean": float(cl.sum(1).mean()), "frac_steps_any_clamp": float(cl.any(1).mean()),
            "wrist_clamp_frac_any": float(cl[:, self.wrist].any(1).mean()), "wrist_clamp_frac_max_joint": float(per[self.wrist].max()),
            "clamp_frac_by_joint": {names[j]: round(float(per[j]), 4) for j in np.nonzero(per)[0]},
            **self._wrp(),
            **({"segments": self.segment_summary()} if self.segments and self.frame else {}),
        }

    def _wrp(self) -> dict:
        aw = np.abs(np.array(self.a_wrp))
        out = {"wrp_abs_a_max": aw.max(0).tolist(), "wrp_abs_a_mean": aw.mean(0).tolist(), "wrp_abs_a_p99": np.percentile(aw, 99, axis=0).tolist(),
               "wrp_q_target_max_rad": (aw.max(0) * self.cfg.action_scale[list(WRP)]).tolist()}
        if self.tau_wrp:
            t = np.array(self.tau_wrp)
            out |= {"wrp_tau_req_ratio_max": t.max(0).tolist(), "wrp_frac_tau_req_gt1": (t > 1.0).mean(0).tolist()}
        return out


# ============================================================================================ G1


def g1_run(ckpt: str, clip_name: str, o: dict) -> dict:
    cfg, clip, raw = load_cfg(ckpt), load_clip(clip_name), load_clip_raw(clip_name)
    m = load_model(TRAIN_XML, o.get("lock", False))
    assert abs(m.opt.timestep - 0.005) < 1e-12
    rob = Robot(m, cfg.joint_names)
    # Actuator gains in the dumped model must be the deploy-config gains.
    kp_m, kd_m = m.actuator_gainprm[rob.act, 0], -m.actuator_biasprm[rob.act, 2]
    gain_err = float(max(np.abs(kp_m - cfg.kp[rob.live]).max(), np.abs(kd_m - cfg.kd[rob.live]).max()))
    q0 = np.clip(raw["joint_pos"][0].astype(np.float64), cfg.soft_lower, cfg.soft_upper)  # as _write_reference_state_to_sim
    rob.set_state(raw["root_pos"][0], raw["root_quat_wxyz"][0], q0, raw["root_lin_vel_w"][0], raw["root_ang_vel_w"][0], raw["joint_vel"][0])
    rec = Recorder(load_policy(ckpt))
    core = DeployCore(cfg, rec, clip, safety(o))
    core.reset(0)
    stats = StepStats(cfg, clip_segments(raw))
    T = clip.num_frames
    qs, acts, aborts, gyro_err = [], [], [], 0.0
    qs.append(rob.d.qpos.copy())
    fail_step, obs0, goal0, max_tilt, max_tilt_err = -1, None, None, 0.0, 0.0
    for k in range(T + 5):
        q, dq, gyro, quat = rob.state()
        gyro_err = max(gyro_err, rob.gyro_check())
        r = core.step(q, dq, gyro, quat)
        stats.add(dq, r, q)
        if k == 0:
            obs0, goal0 = rec.obs, rec.goal
        acts.append(r.action)
        if r.abort:
            aborts.append((k, r.abort))
        rob.set_ctrl(r.q_target)
        for _ in range(4):
            mujoco.mj_step(m, rob.d)
        qs.append(rob.d.qpos.copy())
        f = min(r.frame, T - 1)
        z = rob.d.qpos[rob.fq + 2]
        gz_ref = clip.proj_grav[f][2]
        gz = projected_gravity(rob.d.qpos[rob.fq + 3 : rob.fq + 7])[2]
        max_tilt = max(max_tilt, tilt_of(rob.d.qpos[rob.fq + 3 : rob.fq + 7]))
        g_r = projected_gravity(rob.d.qpos[rob.fq + 3 : rob.fq + 7])
        te = float(np.arccos(np.clip(np.dot(g_r, clip.proj_grav[f]) / (np.linalg.norm(clip.proj_grav[f]) + 1e-9), -1, 1)))
        max_tilt_err = max(max_tilt_err, te)
        stats.mark(r.frame - 1, te, float(np.mean((rob.jq() - clip.joint_pos[f]) ** 2)))
        if abs(raw["root_pos"][f][2] - z) > TRAIN_HEIGHT_DEV or abs(gz_ref - gz) > TRAIN_TILT_DEV:
            fail_step = k + 1
            break
        if r.done:
            break
    qs = np.array(qs)
    n = len(qs)
    jq = np.array([rob.jq(x) for x in qs])
    jerr = jq - clip.joint_pos[np.minimum(np.arange(n), T - 1)]
    return {
        "clip": clip_name, "steps": n - 1, "fail_step": fail_step, "gain_err": gain_err, "gyro_qvel_vs_objvel": gyro_err,
        "track_rmse": rms(jerr[1:]), "track_rmse_ex_wrp": rms_ex_wrp(jerr[1:]), "max_tilt_deg": float(np.degrees(max_tilt)),
        "max_tilt_err_deg": float(np.degrees(max_tilt_err)), "aborts": aborts[:5], "n_aborts": len(aborts), "stats": stats.summary(),
        "_qpos": qs, "_act": np.array(acts), "_obs0": obs0, "_goal0": goal0, "_jq": jq,
    }


def g1_gate(ckpt: str, o: dict, clips=CLIPS, safe=SAFE, ref_prefix: str = "") -> dict:
    """clips / safe / ref_prefix: clip set, the clips the pass rule covers, reference file ref_<prefix><ckpt>_<p>.npz."""
    refs = {}
    for p in ("no-tf32", "tf32"):
        f = REPORTS / "sim2sim" / f"ref_{ref_prefix}{ckpt}_{p}.npz"
        if f.exists():
            refs[p] = dict(np.load(f))
    if "no-tf32" not in refs:
        raise FileNotFoundError(f"run scripts/deploy/sim2sim_ref.py first ({ckpt})")
    ref = refs["no-tf32"]
    clip_order = [str(c) for c in ref["clips"]]
    rows = []
    for c in clips:
        i = clip_order.index(c)
        s = g1_run(ckpt, c, o)
        rq = ref[f"{c}/robot_q"]
        clip = load_clip(c)
        T = clip.num_frames
        env_err = rq[:, 7:] - clip.joint_pos[np.minimum(np.arange(len(rq)), T - 1)]  # robot_q[:, 7:] = motor order
        env_rmse = rms(env_err[1:])
        env_fail = int(ref["fail_step"][i])
        sq = s["_qpos"]
        L = min(len(sq), len(rq))
        dq_rms = np.sqrt(np.mean((s["_jq"][:L] - rq[:L, 7:]) ** 2, axis=1))
        dxy = np.linalg.norm(sq[:L, :2] - rq[:L, :2], axis=1)
        bad = np.where((dq_rms > DIVERGE_Q_RMS) | (dxy > DIVERGE_XY))[0]
        div = int(bad[0]) if len(bad) else -1
        row = {
            "clip": c, "safe": c in safe, "frames": T,
            "env_fail_step": env_fail, "s2s_fail_step": s["fail_step"],
            "fail_agree": (env_fail >= 0) == (s["fail_step"] >= 0),
            "env_rmse": env_rmse, "s2s_rmse": s["track_rmse"], "rmse_ratio": s["track_rmse"] / env_rmse,
            "diverge_step": div, "diverge_s": div * 0.02 if div >= 0 else None,
            "traj_q_rms_mean": float(dq_rms.mean()), "traj_xy_final": float(dxy[-1]),
            "action_step1_absdiff": float(np.abs(s["_act"][0] - ref[f"{c}/action"][0]).max()),
            "n_aborts": s["n_aborts"], "aborts": s["aborts"], "stats": s["stats"],
            "gyro_qvel_vs_objvel": s["gyro_qvel_vs_objvel"], "gain_err": s["gain_err"],
        }
        row["obs_step1_absdiff"] = float(np.abs(s["_obs0"] - ref["obs0"][i]).max())
        row["goal_step1_absdiff"] = float(np.abs(s["_goal0"] - ref["goal0"][i]).max())
        if "tf32" in refs:  # intrinsic sensitivity: same env, TF32 vs fp32 matmuls
            r2 = refs["tf32"][f"{c}/robot_q"]
            L2 = min(len(r2), len(rq))
            d2 = np.sqrt(np.mean((r2[:L2, 7:] - rq[:L2, 7:]) ** 2, axis=1))
            dx2 = np.linalg.norm(r2[:L2, :2] - rq[:L2, :2], axis=1)
            b2 = np.where((d2 > DIVERGE_Q_RMS) | (dx2 > DIVERGE_XY))[0]
            row["env_tf32_vs_fp32_diverge_step"] = int(b2[0]) if len(b2) else -1
        rows.append(row)
        print(f"  G1 {ckpt} {o} {c[:30]:30s} fail env {env_fail:4d} s2s {s['fail_step']:4d} | rmse env {env_rmse:.4f} s2s {s['track_rmse']:.4f} "
              f"({row['rmse_ratio']:.2f}x) | diverge {div} | clamp {s['stats']['n_clamped_mean']:.2f} wrist {s['stats']['wrist_clamp_frac_any']:.3f}", flush=True)
    safe = [r for r in rows if r["safe"]]
    passed = all(r["fail_agree"] for r in safe) and all(r["rmse_ratio"] <= 1.2 for r in safe)
    return {"opts": o, "rows": rows, "pass": passed,
            "criteria": "SAFE clips: same fail outcome as the fp32 training env and s2s joint RMSE <= 1.2 x env RMSE"}


# ============================================================================================ G2 / G3 physics


@lru_cache(maxsize=None)
def unitree_base(lock: bool = False) -> mujoco.MjModel:
    m = load_model(UNITREE_XML, lock)
    m.opt.timestep = 0.001
    return m


def unitree_model(extra_mass: float, friction: float | None, lock: bool = False) -> mujoco.MjModel:
    m = copy.deepcopy(unitree_base(lock))
    if extra_mass:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        m0 = m.body_mass[b]
        m.body_mass[b] = m0 + extra_mass
        m.body_inertia[b] *= (m0 + extra_mass) / m0
    if friction is not None:
        m.geom_friction[:, 0] = friction  # MuJoCo combines pair friction with max -> set every geom
    return m


class PDSim:
    """1 kHz PD on a torque-motor model; targets (and optional gains) become active `latency_ms` after they were sent.

    The PD torque is clipped to `effort` (the motor limit we assume); MuJoCo additionally clips it to the Unitree MJCF
    ctrlrange (counted in n_model_sat)."""

    def __init__(self, rob: Robot, cfg: DeployConfig, effort: np.ndarray):
        self.rob, self.cfg, self.effort = rob, cfg, effort
        self.kp, self.kd = cfg.kp.copy(), cfg.kd.copy()
        self.q_t = None
        self.pending: list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        self.ms = 0
        self.tau_prev = np.zeros(29)
        self.log_tau_ratio, self.log_tau = [], []
        self.ctrlrange = np.full(29, np.inf)
        self.ctrlrange[rob.live] = rob.m.actuator_ctrlrange[rob.act, 1]
        self.n_model_sat = np.zeros(29, int)
        self.vmax = 0.0  # max |dq| / velocity_limit at 1 kHz (recorded steps)

    def command(self, q_t, latency_ms: int, kp=None, kd=None):
        self.pending.append((self.ms + latency_ms, np.asarray(q_t, float).copy(), None if kp is None else np.asarray(kp, float).copy(), None if kd is None else np.asarray(kd, float).copy()))

    def step_ms(self, record=True):
        while self.pending and self.pending[0][0] <= self.ms:
            _, self.q_t, kp, kd = self.pending.pop(0)
            if kp is not None:
                self.kp, self.kd = kp, kd
        d = self.rob.d
        q, dq = self.rob.jq(), self.rob.jdq()
        tau_d = self.kp * (self.q_t - q) - self.kd * dq
        tau_d[self.rob.locked] = 0.0  # locked joints: no motor, no torque
        tau = np.clip(tau_d, -self.effort, self.effort)
        self.rob.set_ctrl(tau)
        if record:
            self.log_tau_ratio.append(np.abs(tau_d) / self.effort)
            self.log_tau.append(tau)
            self.n_model_sat += np.abs(tau) > self.ctrlrange + 1e-9
            self.vmax = max(self.vmax, float((np.abs(dq) / self.cfg.velocity_limit).max()))
        self.tau_prev = tau
        mujoco.mj_step(self.rob.m, d)
        self.ms += 1


def stand_gains(cfg: DeployConfig) -> tuple[np.ndarray, np.ndarray]:
    kp, kd = np.zeros(29), np.zeros(29)
    for i, n in enumerate(cfg.joint_names):
        if n.startswith(("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")):
            j = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll").index(n.split("_", 1)[1].replace("_joint", ""))
            kp[i], kd[i] = STAND_LEG_KP[j], STAND_LEG_KD[j]
        elif n.startswith("waist"):
            kp[i], kd[i] = STAND_WAIST_KP, STAND_WAIST_KD
        else:
            kp[i], kd[i] = STAND_ARM_KP, STAND_ARM_KD
    return kp, kd


def fell(rob: Robot) -> bool:
    return rob.d.qpos[rob.fq + 2] < FALL_HEIGHT or tilt_of(rob.d.qpos[rob.fq + 3 : rob.fq + 7]) > FALL_TILT


def g2_run(o: dict) -> dict:
    ckpt, clip_name, lat, kps, mass, fric = o["ckpt"], o["clip"], o["lat"], o["kps"], o["mass"], o["fric"]
    settle_ms = o.get("settle_ms", 0)
    cfg = with_effort(load_cfg(ckpt), o.get("hr_effort"))
    clip, raw = load_clip(clip_name), load_clip_raw(clip_name)
    m = unitree_model(mass, fric, o.get("lock", False))
    rob = Robot(m, cfg.joint_names)
    pose0 = np.clip(clip.joint_pos[0], cfg.soft_lower, cfg.soft_upper)
    rob.set_state(raw["root_pos"][0], raw["root_quat_wxyz"][0], pose0)
    dz = rob.place_on_ground()
    sim = PDSim(rob, cfg, cfg.effort_limit)
    kp = cfg.kp * kps
    sim.command(pose0, 0, kp, cfg.kd.copy())
    for _ in range(settle_ms):  # optional settle holding the frame-0 pose with the policy gains
        sim.step_ms(record=False)
    settle_fall = fell(rob)
    tilt_at_start = float(np.degrees(tilt_of(rob.d.qpos[rob.fq + 3 : rob.fq + 7])))
    core = DeployCore(cfg, load_policy(ckpt), clip, safety(o))
    core.reset(0)
    stats = StepStats(cfg, clip_segments(raw))
    T = clip.num_frames
    fall_step, aborts, jerr, lim_hits, steps, n_hard = -1, [], [], 0, 0, 0
    max_tilt, min_h, train_term, max_tilt_err = 0.0, 9.0, -1, 0.0
    near = 0.01
    while True:
        q, dq, gyro, quat = rob.state()
        r = core.step(q, dq, gyro, quat)
        stats.add(dq, r, q)
        q_send = r.q_target
        if o.get("hard_clip"):  # what a motor firmware that clips position targets to the joint range would do
            qc = np.clip(q_send, cfg.joint_lower, cfg.joint_upper)
            n_hard += int(np.any(qc != q_send))
            q_send = qc
        sim.command(q_send, lat)
        if r.abort:
            aborts.append((steps, r.abort))
        for _ in range(20):
            sim.step_ms()
        steps += 1
        q = rob.jq()
        f = min(r.frame, T - 1)
        jerr.append(q - clip.joint_pos[f])
        lim_hits += int(np.any((q < rob.range[:, 0] + near) | (q > rob.range[:, 1] - near)))
        quat = rob.d.qpos[rob.fq + 3 : rob.fq + 7]
        max_tilt, min_h = max(max_tilt, tilt_of(quat)), min(min_h, float(rob.d.qpos[rob.fq + 2]))
        g_r, g_ref = projected_gravity(quat), clip.proj_grav[f]
        te = float(np.arccos(np.clip(np.dot(g_r, g_ref) / (np.linalg.norm(g_r) * np.linalg.norm(g_ref) + 1e-9), -1, 1)))
        max_tilt_err = max(max_tilt_err, te)
        stats.mark(r.frame - 1, te, float(np.mean(jerr[-1] ** 2)))
        if train_term < 0:
            gz, gz_ref = projected_gravity(quat)[2], clip.proj_grav[f][2]
            if abs(raw["root_pos"][f][2] - rob.d.qpos[rob.fq + 2]) > TRAIN_HEIGHT_DEV or abs(gz - gz_ref) > TRAIN_TILT_DEV:
                train_term = steps
        if fell(rob):
            fall_step = steps
            break
        if r.done:
            break
    disp = rob.d.qpos[rob.fq : rob.fq + 2] - (raw["root_pos"][0][:2])  # pelvis xy travelled (start = frame-0 root xy)
    ref_disp = raw["root_pos"][min(steps, T - 1)][:2] - raw["root_pos"][0][:2]
    ratio = np.array(sim.log_tau_ratio)
    tau = np.abs(np.array(sim.log_tau))
    w = tau[lat : lat + 20]  # first policy target active (recording starts at the first policy step)
    first_abort = aborts[0] if aborts else None
    abort_fp = bool(aborts) and fall_step < 0  # we keep running the policy after an abort; FP = no fall followed
    worst_j = int(np.argmax(ratio.max(0)))
    return {
        **{k: v for k, v in o.items()}, "tilt_at_policy_start_deg": tilt_at_start,
        "place_dz": dz, "settle_fall": settle_fall, "steps": steps, "frames": T, "completed": fall_step < 0 and steps >= T - 1,
        "fall": fall_step >= 0, "fall_step": fall_step, "train_term_step": train_term,
        "max_tilt_deg": float(np.degrees(max_tilt)), "max_tilt_err_deg": float(np.degrees(max_tilt_err)), "min_height": min_h,
        "n_abort_steps": len(aborts), "first_abort": first_abort, "abort_reasons": sorted({" ".join(a[1].split(" ")[:2]) for a in aborts}),
        "abort_fp": abort_fp,
        "max_tau_ratio": float(ratio.max()), "worst_joint": cfg.joint_names[worst_j],
        "frac_tau_gt_0p9": float(np.mean(ratio.max(1) > 0.9)),
        "first_step_tau_max_Nm": float(w.max()), "first_step_tau_joint": cfg.joint_names[int(np.argmax(w.max(0)))],
        "first_step_tau_ratio_max": float((w / cfg.effort_limit).max()),
        "model_sat_ms": {cfg.joint_names[j]: int(sim.n_model_sat[j]) for j in np.nonzero(sim.n_model_sat)[0]},
        "joint_limit_hit_steps": lim_hits, "hard_clip_steps": n_hard, "vfrac_1khz_max": sim.vmax,
        "track_rmse": rms(np.array(jerr)), "track_rmse_ex_wrp": rms_ex_wrp(np.array(jerr)), "stats": stats.summary(),
        "disp_xy": disp.tolist(), "ref_disp_xy": ref_disp.tolist(), "disp_err_m": float(np.linalg.norm(disp - ref_disp)),
    }


FULL = {"clips": CLIPS, "lat": G2_LATENCY_MS, "kps": G2_KP_SCALE, "mass": G2_EXTRA_MASS, "fric": G2_FRICTION}
NOM = {"clips": CLIPS, "lat": G2_LATENCY_MS, "kps": (1.0,), "mass": (0.0,), "fric": G2_FRICTION}
REC = "ckpt_0116000"
# key -> (description, run options, {ckpt: grid}); every variant also gets the SAFE-nominal pass rule
G2_VARIANTS = {
    "g2": ("core defaults (soft clamp off, torque clamp 1.5x effort), policy starts at rest", {}, {REC: FULL, "ckpt_0100000": NOM}),
    "g2_settle500": ("core defaults, 0.5 s settle holding frame 0 with training gains before the policy (as originally specified)", {"settle_ms": 500}, {REC: NOM, "ckpt_0100000": NOM}),
    "g2_hardclip": ("core defaults + targets clipped to the HARD joint range (firmware-clip hypothesis)", {"hard_clip": True},
                    {REC: {**NOM, "clips": SAFE, "lat": (0, 20)}}),
    "g2_hr88": ("core defaults, hip-roll effort 88 Nm (core torque clamp and simulated motor limit)", {"hr_effort": 88.0}, {REC: {**NOM, "kps": (1.0, 0.85)}}),
    "g2_f1.0": ("torque clamp factor 1.0", {"factor": 1.0}, {REC: NOM}),
    "g2_f0.5": ("torque clamp factor 0.5 (first hardware stages)", {"factor": 0.5}, {REC: NOM}),
    "g2_noclamp": ("no torque clamp (motor saturation only)", {"torque": False}, {REC: NOM}),
}


def g2_criteria(rows: list[dict]) -> dict:
    nom = [r for r in rows if r["kps"] == 1.0 and r["mass"] == 0.0 and r["clip"] in SAFE]
    crit = {
        "safe_nominal_falls": sum(r["fall"] for r in nom), "safe_nominal_runs": len(nom),
        "safe_nominal_abort_fp": sum(r["abort_fp"] for r in nom),
        "safe_nominal_min_frac_tau_ok": float(min(1 - r["frac_tau_gt_0p9"] for r in nom)) if nom else None,
    }
    crit["pass"] = bool(nom) and crit["safe_nominal_falls"] == 0 and crit["safe_nominal_abort_fp"] == 0 and crit["safe_nominal_min_frac_tau_ok"] >= 0.99
    return crit


def g2_all(ckpts, workers: int, keys, lock: bool = False) -> dict[str, dict]:
    jobs = []
    for key in keys:
        _, opts, grids = G2_VARIANTS[key]
        for c, g in grids.items():
            if c not in ckpts:
                continue
            jobs += [{"key": key, **opts, "lock": lock, "ckpt": c, "clip": cl, "lat": la, "kps": k, "mass": mm, "fric": fr}
                     for cl in g["clips"] for la in g["lat"] for k in g["kps"] for mm in g["mass"] for fr in g["fric"]]
    t0 = time.time()
    res = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(g2_run, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  G2 {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    out: dict[str, dict] = {c: {} for c in ckpts}
    for key in keys:
        for c in ckpts:
            rows = sorted([r for r in res if r["key"] == key and r["ckpt"] == c], key=lambda r: (CLIPS.index(r["clip"]), r["lat"], r["kps"], r["mass"], r["fric"]))
            if rows:
                out[c][key] = {"desc": G2_VARIANTS[key][0], "rows": rows, "criteria": g2_criteria(rows),
                               "pass_rule": "SAFE clips, kp 1.0, +0 kg, all simulated latencies, both frictions: 0 falls, 0 abort false positives, |tau|<=0.9 limit in >=99% of 1 kHz steps"}
                print(f"  {key} {c}: {out[c][key]['criteria']}", flush=True)
    return out


# ============================================================================================ G3


def g3_run(o: dict) -> dict:
    """q0 stand -> 4 s cosine interpolation to frame 0 (stand gains) -> 1 s HOLD -> POLICY (training gains, first target
    and gains switch together) -> END.

    support: "none" | "band" (gantry spring on the pelvis until the policy starts, re-engaged at END)
             | "feet" (pelvis band on throughout, feet loaded) | "air" (torso hung, feet ~5 cm above the ground, throughout)
    end:     "blend" (runner fix: from the last target with training gains, `blend_s` ramp of kp/kd to stand gains and of
             the target to the soft-clipped last-frame pose, target re-clamped to |kp(t)(q_cmd - q)| <= effort every
             `blend_dt_ms`, then stand-gain hold of that pose with the same clamp)
             | "switch_pose" (round-1 sim2sim: stand gains + soft-clipped last-frame pose at once)
             | "switch_last" (round-1 runner: stand gains at the last policy target at once)
    """
    ckpt, lat = o["ckpt"], o["lat"]
    clip_name = o.get("clip", IDLE)
    support, end = o.get("support", "band"), o.get("end", "blend")
    end_support = o.get("end_support", support != "none")
    blend_s, blend_dt, shape = o.get("blend_s", 1.0), o.get("blend_dt_ms", 2), o.get("blend_shape", "linear")
    cfg, clip = load_cfg(ckpt), load_clip(clip_name)
    m = unitree_model(0.0, None, o.get("lock", False))
    rob = Robot(m, cfg.joint_names)
    q0 = cfg.default_joint_pos.copy()
    rob.set_state(np.array([0.0, 0.0, 0.8]), np.array([1.0, 0, 0, 0]), q0)
    dz = rob.place_on_ground()
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    weight = float(m.body_subtreemass[0] * 9.81)
    if support == "air":
        rob.d.qpos[rob.fq + 2] += AIR_CLEARANCE
        mujoco.mj_forward(m, rob.d)
        band = {"on": True, "body": torso, "k": AIR_K, "d": AIR_D, "anchor": rob.d.xipos[torso].copy() + np.array([0, 0, weight / AIR_K])}
    else:
        band = {"on": support in ("band", "feet"), "body": rob.pelvis, "k": BAND_K, "d": BAND_D, "anchor": rob.d.qpos[rob.fq : rob.fq + 3].copy()}
    sim = PDSim(rob, cfg, cfg.effort_limit)
    skp, skd = stand_gains(cfg)
    pose0 = np.clip(clip.joint_pos[0], cfg.soft_lower, cfg.soft_upper)
    upright = True
    max_tilt, min_h = 0.0, 9.0
    tau_hist = []
    phase, fall_phase, tilt_end, fall_ms = "interp", None, {}, None
    v6 = np.zeros(6)

    def band_pv():
        if band["body"] == rob.pelvis:
            return rob.d.qpos[rob.fq : rob.fq + 3], rob.d.qvel[rob.fv : rob.fv + 3]
        mujoco.mj_objectVelocity(m, rob.d, mujoco.mjtObj.mjOBJ_BODY, band["body"], v6, 0)  # COM frame, world orientation
        return rob.d.xipos[band["body"]], v6[3:]

    def run_ms(n):
        nonlocal upright, max_tilt, min_h, fall_phase, fall_ms
        for _ in range(n):
            rob.d.xfrc_applied[:, :3] = 0.0
            if band["on"]:  # gantry: 3-D spring-damper toward the anchor, applied at the body's COM
                p, v = band_pv()
                rob.d.xfrc_applied[band["body"], :3] = band["k"] * (band["anchor"] - p) - band["d"] * v
            sim.step_ms()
            if fall_ms is None and fell(rob):
                fall_ms = sim.ms
            tau_hist.append(sim.tau_prev.copy())
        quat = rob.d.qpos[rob.fq + 3 : rob.fq + 7]
        max_tilt, min_h = max(max_tilt, tilt_of(quat)), min(min_h, float(rob.d.qpos[rob.fq + 2]))
        tilt_end[phase] = float(np.degrees(tilt_of(quat)))
        if fell(rob) and upright:
            upright, fall_phase = False, phase

    # stand -> interpolate 4 s -> HOLD 1 s (stand gains, 50 Hz target updates)
    sim.command(q0, 0, skp, skd)
    for k in range(200):
        sim.command(interpolate_pose(q0, pose0, (k + 1) * 0.02, 4.0), 0)
        run_ms(20)
    phase = "hold"
    n_ramp = int(round(o.get("hold_ramp_s", 0.0) / 0.02))  # optional: ramp stand -> training gains over the end of HOLD
    for k in range(50):
        if k >= 50 - n_ramp:
            s = (k - (50 - n_ramp) + 1) / n_ramp
            sim.command(pose0, 0, (1 - s) * skp + s * cfg.kp, (1 - s) * skd + s * cfg.kd)
        else:
            sim.command(pose0, 0)
        run_ms(20)
    phase = "policy"
    if support == "band":
        band["on"] = False  # gantry slackened when the policy takes over
    s_start = sim.ms
    core = DeployCore(cfg, load_policy(ckpt), clip, safety(o))
    core.reset(0)
    stats = StepStats(cfg, clip_segments(load_clip_raw(clip_name)))
    T = clip.num_frames
    aborts, steps, wraps = [], 0, 0
    loop_s = o.get("loop_s")
    total = int(round(loop_s / 0.02)) if loop_s else None
    q_last = pose0
    max_tilt_err = 0.0
    p0 = rob.d.qpos[rob.fq : rob.fq + 2].copy()
    while upright:
        q, dq, gyro, quat = rob.state()
        g_ref = clip.proj_grav[min(core.frame, T - 1)]  # what DeployCore's tilt abort compares with
        te = float(np.arccos(np.clip(np.dot(projected_gravity(quat), g_ref) / (np.linalg.norm(g_ref) + 1e-9), -1, 1)))
        max_tilt_err = max(max_tilt_err, te)
        stats.mark(core.frame, te)
        r = core.step(q, dq, gyro, quat)
        stats.add(dq, r, q)
        if steps == 0:
            sim.command(r.q_target, lat, cfg.kp, cfg.kd)  # gains switch together with the first policy target
        else:
            sim.command(r.q_target, lat)
        q_last = r.q_target.copy()
        if r.abort:
            aborts.append((steps, r.abort))
        run_ms(20)
        steps += 1
        if total is not None and steps >= total:
            break
        if r.done:
            if total is None:
                break
            wraps += 1
            core.frame = 0  # loop clip: keep history / last action, restart the goal table
    s_end = sim.ms
    policy_disp = (rob.d.qpos[rob.fq : rob.fq + 2] - p0).tolist()
    # ---- END
    phase = "end"
    if end_support:
        band["on"] = True
        if band["body"] == rob.pelvis:
            band["anchor"][:2] = rob.d.qpos[rob.fq : rob.fq + 2]
    else:
        band["on"] = False
    hold = np.clip(clip.joint_pos[T - 1], cfg.soft_lower, cfg.soft_upper)
    eff = cfg.effort_limit
    blend_end_ms = s_end
    if upright:
        if end == "blend":
            n = int(round(blend_s * 1000 / blend_dt))
            for i in range(1, n + 1):
                s = i / n
                sq = s if shape == "linear" else 0.5 - 0.5 * np.cos(np.pi * s)
                kp_t, kd_t = (1 - s) * cfg.kp + s * skp, (1 - s) * cfg.kd + s * skd
                q = rob.jq()
                q_cmd = np.clip(q_last + sq * (hold - q_last), q - eff / kp_t, q + eff / kp_t)
                sim.command(q_cmd, o.get("blend_lat", lat), kp_t, kd_t)
                run_ms(blend_dt)
                if not upright:
                    break
            blend_end_ms = sim.ms
            phase = "end_hold"
            for _ in range(100):  # 2 s stand-gain hold, same effort clamp at 50 Hz
                if not upright:
                    break
                q = rob.jq()
                sim.command(np.clip(hold, q - eff / skp, q + eff / skp), o.get("blend_lat", lat))
                run_ms(20)
        else:
            sim.pending.clear()
            sim.command(hold if end == "switch_pose" else q_last, 0, skp, skd)
            phase = "end_hold"
            run_ms(3000)
    tau = np.array(tau_hist)
    jump = np.abs(np.diff(tau, axis=0, prepend=tau[:1]))

    def win(a, b):
        a, b = max(a, 0), min(b, len(jump))
        if b <= a:
            return None, ""
        seg = jump[a:b]
        return float(seg.max()), cfg.joint_names[int(np.argmax(seg.max(0)))]

    j_start, jn_start = win(s_start - 1, s_start + lat + 40)
    # END window starts after the last policy target became active (sent at s_end - 20, active lat ms later), except for the
    # immediate switches, which drop that pending target. tau index t = torque applied in ms t; jump[t] = |tau[t] - tau[t-1]|.
    e0 = s_end if end != "blend" else max(s_end, s_end - 20 + lat + 1)
    bl = o.get("blend_lat", lat)
    j_end, jn_end = win(e0, s_end + max(lat, bl) + 40)
    j_blend, jn_blend = win(e0, blend_end_ms + bl + 20)
    j_after, jn_after = win(blend_end_ms + bl + 20, len(jump))
    steady = jump[s_start + 500 : s_end - 100] if s_end - s_start > 700 else jump[s_start:s_end]
    w = np.abs(tau[s_start + lat : s_start + lat + 20])
    pre = np.abs(tau[s_start - 20 : s_start])
    eb = np.abs(tau[s_end:blend_end_ms + lat]) if blend_end_ms > s_end else np.zeros((1, 29))
    return {
        **o, "place_dz": dz, "clip": clip_name, "support": support, "end": end, "end_support": end_support,
        "upright": upright, "fall_phase": fall_phase, "fall_time_s": fall_ms / 1000 if fall_ms else None,
        "tilt_at_phase_end_deg": tilt_end, "max_tilt_deg": float(np.degrees(max_tilt)), "min_height": min_h,
        "policy_steps": steps, "clip_wraps": wraps, "n_abort_steps": len(aborts), "first_abort": aborts[0] if aborts else None,
        "max_tilt_err_policy_deg": float(np.degrees(max_tilt_err)), "policy_disp_xy": policy_disp, "completed": upright and steps >= T - 1,
        "max_tau_jump_interp_Nm_per_ms": float(jump[1:s_start].max()) if s_start > 1 else None,
        "jump_policy_start": j_start, "jump_policy_start_joint": jn_start,
        "jump_end_switch": j_end, "jump_end_switch_joint": jn_end,
        "jump_end_blend_whole": j_blend, "jump_end_blend_whole_joint": jn_blend,
        "jump_after_blend": j_after, "jump_after_blend_joint": jn_after,
        "max_tau_jump_policy_steady_Nm_per_ms": float(steady.max()) if len(steady) else None,
        "p99_tau_jump_policy_steady_Nm_per_ms": float(np.percentile(steady.max(1), 99)) if len(steady) else None,
        "hold_tau_max_Nm": float(pre.max()) if len(pre) else None,
        "first_step_tau_max_Nm": float(w.max()) if len(w) else None, "first_step_tau_joint": cfg.joint_names[int(np.argmax(w.max(0)))] if len(w) else "",
        "first_step_tau_ratio_max": float((w / eff).max()) if len(w) else None,
        "max_tau_ratio_policy": float((np.abs(tau[s_start:s_end]) / eff).max()) if s_end > s_start else None,
        "max_tau_ratio_end_blend": float((eb / eff).max()), "vfrac_1khz_max_all": sim.vmax,
        "stats": stats.summary(),
    }


def g3_jobs(ckpts, rec_only: bool = True) -> list[dict]:
    jobs = []
    for c in ckpts:
        if rec_only and c != REC:
            continue
        b = {"ckpt": c}
        for f in (1.5, 1.0, 0.5):
            for la in (0, 20):
                jobs.append({**b, "tag": "main", "factor": f, "lat": la})
        for la in (0, 20):
            jobs.append({**b, "tag": "cosine", "factor": 1.5, "lat": la, "blend_shape": "cosine"})
            jobs.append({**b, "tag": "switch_pose", "factor": 1.5, "lat": la, "end": "switch_pose"})
            jobs.append({**b, "tag": "switch_last", "factor": 1.5, "lat": la, "end": "switch_last"})
            jobs.append({**b, "tag": "loop60", "factor": 1.5, "lat": la, "loop_s": 60.0})
        jobs.append({**b, "tag": "blend_unsupported_end", "factor": 1.5, "lat": 0, "end_support": False})
        for la in (0, 20):  # proposal (not in the runner): ramp stand -> training gains during the last 1 s of HOLD
            jobs.append({**b, "tag": "start_ramp", "factor": 1.5, "lat": la, "hold_ramp_s": 1.0})
        # END-blend commands come from the 500 Hz command thread (fresh state, ~2 ms), not the 20 ms-delayed policy path
        for la in (0, 20):
            jobs.append({**b, "tag": "blend_lat2", "factor": 1.5, "lat": la, "blend_lat": 2})
        jobs.append({**b, "tag": "blend_lat2_air", "clip": "idle_loop_002__A060", "support": "air", "factor": 0.5, "lat": 20, "blend_lat": 2})
        # staged abort thresholds, with the abort config actually in the core
        for clip in SAFE:
            for la in (0, 20):
                for sup, stage in (("air", "air/feet"), ("feet", "air/feet"), ("band", "slack")):
                    vt, at = STAGES[stage]
                    for f in ((0.5, 1.5) if la == 0 else (0.5,)):
                        jobs.append({**b, "tag": f"stage_{sup}", "clip": clip, "support": sup, "factor": f, "lat": la, "max_vel_frac": vt, "max_abs_action": at})
    return jobs


def g3_gate(ckpts, workers: int, lock: bool = False) -> dict[str, list]:
    jobs = [{**j, "lock": lock} for j in g3_jobs(ckpts, rec_only=not lock)]  # locked: every ckpt given
    t0 = time.time()
    res = []
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(g3_run, jobs, chunksize=1):
            res.append(r)
            print(f"  G3 [{time.time() - t0:.0f} s] {r['tag']} {r['clip'][:20]} {r['support']} f{r['factor']} lat {r['lat']} upright {r['upright']} ({r['fall_phase']}) tilt {r['max_tilt_deg']:.1f} "
                  f"start {fmt(r['jump_policy_start'], 1)} ({r['jump_policy_start_joint']}) end {fmt(r['jump_end_switch'], 1)} ({r['jump_end_switch_joint']}) blend {fmt(r['jump_end_blend_whole'], 1)} "
                  f"first tau {fmt(r['first_step_tau_max_Nm'], 1)} aborts {r['n_abort_steps']} {r['first_abort']}", flush=True)
    key = lambda r: (r["tag"], r["clip"], r["support"], -r["factor"], r["lat"])  # noqa: E731
    return {c: sorted([r for r in res if r["ckpt"] == c], key=key) for c in ckpts}


# ============================================================================================ locked waist (G1 29dof, waist locked)
# Same policy (29 actions), model with waist roll/pitch welded at 0 (load_model(lock=True)); actions 13/14 are computed by
# the core as usual but never applied; q = dq = 0 is reported for them, as the real lowstate does. Every locked run has an
# unlocked twin with identical settings, so RMSE ratios compare like with like (RMSE over the other 27 joints).

LW_CKPTS = ("ckpt_0116000", "ckpt_0100000", "ckpt_0140000", "ckpt_0144000")
LW_LAT = (0, 10, 20)
LW_ROBUST = ((0.85, 0.0), (1.15, 0.0), (1.0, 1.8), (0.85, 1.8), (1.15, 1.8))  # (kp scale, extra torso mass)


def lw_g1_run(o: dict) -> dict:
    s = g1_run(o["ckpt"], o["clip"], o)
    return {**{k: v for k, v in s.items() if not k.startswith("_")}, "ckpt": o["ckpt"], "lock": o["lock"],
            "fall": s["fail_step"] >= 0, "abort_fp": s["n_aborts"] > 0 and s["fail_step"] < 0}


def lw_eval(ckpts, workers: int, phase: str) -> dict:
    g1, g2 = [], []
    for c in ckpts:
        for lock in (False, True):
            if phase == "nominal":
                g1 += [{"ckpt": c, "clip": cl, "lock": lock} for cl in CLIPS]
                g2 += [{"key": "lw", "ckpt": c, "clip": cl, "lat": la, "kps": 1.0, "mass": 0.0, "fric": fr, "lock": lock}
                       for cl in CLIPS for la in LW_LAT for fr in G2_FRICTION]
            else:
                g2 += [{"key": "lw", "ckpt": c, "clip": cl, "lat": la, "kps": k, "mass": mm, "fric": fr, "lock": lock}
                       for (k, mm) in LW_ROBUST for cl in CLIPS for la in LW_LAT for fr in G2_FRICTION]
    t0 = time.time()
    with Pool(workers) as p:
        r1 = list(p.imap_unordered(lw_g1_run, g1, chunksize=1)) if g1 else []
        print(f"  LW G1 {len(r1)} runs ({time.time() - t0:.0f} s)", flush=True)
        r2 = []
        for i, r in enumerate(p.imap_unordered(g2_run, g2, chunksize=1)):
            r2.append(r)
            if (i + 1) % 100 == 0:
                print(f"  LW G2 {i + 1}/{len(g2)} ({time.time() - t0:.0f} s)", flush=True)
    return {"g1": r1, "g2": r2}


def lw_summary(rows: list[dict], clips=SAFE) -> dict:
    """Aggregate over rows of `clips`: falls, abort false positives, RMSE (27 joints), max tilt, torque saturation, |a| 13/14."""
    rs = [r for r in rows if r["clip"] in clips]
    if not rs:
        return {}
    st = [r["stats"] for r in rs]
    out = {"runs": len(rs), "falls": sum(r["fall"] for r in rs), "abort_fp": sum(r["abort_fp"] for r in rs),
           "rmse_ex_wrp": _mean([r["track_rmse_ex_wrp"] for r in rs]), "rmse_all": _mean([r["track_rmse"] for r in rs]),
           "max_tilt_deg": max(r["max_tilt_deg"] for r in rs), "max_tilt_err_deg": max(r["max_tilt_err_deg"] for r in rs),
           "wrp_abs_a_max": max(max(s["wrp_abs_a_max"]) for s in st), "wrp_abs_a_mean": _mean([float(np.mean(s["wrp_abs_a_mean"])) for s in st]),
           "max_abs_a": max(s["max_abs_a"] for s in st)}
    if "frac_tau_gt_0p9" in rs[0]:
        out |= {"max_tau_ratio": max(r["max_tau_ratio"] for r in rs), "min_frac_tau_ok": min(1 - r["frac_tau_gt_0p9"] for r in rs),
                "model_sat_ms": sum(sum(r["model_sat_ms"].values()) for r in rs)}
    return out


# ============================================================================================ abort margins on the training env


def ref_margins(ckpt: str) -> dict:
    """|a| and finite-difference |dq|/velocity_limit along the deterministic fp32 training-env rollouts (sim2sim_ref.py)."""
    f = REPORTS / "sim2sim" / f"ref_{ckpt}_no-tf32.npz"
    if not f.exists():
        return {}
    ref, cfg = dict(np.load(f)), load_cfg(ckpt)
    rows = []
    for c in CLIPS:
        a = np.abs(ref[f"{c}/action"]).max(1)
        q = ref[f"{c}/robot_q"][:, 7:]
        v = (np.abs(np.diff(q, axis=0)) / 0.02 / cfg.velocity_limit).max(1)
        rows.append({"clip": c, "safe": c in SAFE, "steps": int(len(a)), "max_abs_a": float(a.max()), "n_a_gt10": int((a > 10).sum()), "n_a_gt15": int((a > 15).sum()),
                     "max_vfrac_fd": float(v.max()), "stage_trips": {s: int(((v[: len(a)] > vt) | (a > at)).sum()) for s, (vt, at) in STAGES.items()}})
    return {"rows": rows, "note": "dq from 50 Hz finite differences of the env joint positions (a lower bound on the sampled |dq|)"}


# ============================================================================================ arbitrary clip sets (--clips)
# Gates for clips outside the built-in SAFE/HARDER set, results in their own report (sim2sim_<tag>_<ckpt>.json, SIM2SIM_<tag>.md):
#   g1   training MJCF vs training-env rollouts ref_<tag>_<ckpt>_<tf32|no-tf32>.npz (sim2sim_ref.py --clips ... --out ...),
#        core defaults and the runner's 'slack' profile
#   g2   Unitree MJCF, at rest on the ground, latency 0/10/20 ms x mu 0.6/1.0, nominal + CLIPSET_ROBUST (kp scale, torso mass),
#        for each runner stage profile in PROFILES
#   g3   runner protocol: stand -> 4 s interp -> 1 s HOLD under the gantry -> POLICY (gantry slack) -> END blend (commands at 2 ms)
#        under the gantry, latency 0/10/20 ms, each profile
# Pass rule per clip (gate_clip): G1 same fail outcome as the env, RMSE <= 1.2 x env; 'slack' G2 nominal + G3: 0 falls, 0 aborts;
# slack margins (threshold / max seen over G2 nominal + G3, sampled 50 Hz state as the core checks) >= 1.2 for |a|, dq/vlim and
# the core tilt-error abort.

# Runner --stage profiles (scripts/deploy/g1_runner.py STAGES; 'ground' = DeployCore defaults); keep in sync.
PROFILES = {"slack": {"max_vel_frac": 0.5, "max_abs_action": 15.0, "factor": 1.0},
            "ground": {"max_vel_frac": 0.8, "max_abs_action": 50.0, "factor": 1.5}}
CLIPSET_LAT = (0, 10, 20)
STAGE_OF_PROFILE = {"slack": "slack"}  # STAGES entry with the same thresholds as the profile (trip counts per segment)
CLIPSET_ROBUST = ((1.0, 0.0), (0.85, 0.0), (1.15, 0.0), (1.0, 1.8), (0.85, 1.8), (1.15, 1.8))  # (kp scale, extra torso mass); first = nominal
MARGIN_MIN = 1.2


def clipset_gates(ckpt: str, clips, tag: str, workers: int, cmds: set, res: dict) -> dict:
    clips = tuple(clips)
    res.update({"ckpt": ckpt, "clips": list(clips), "tag": tag, "profiles": PROFILES, "tilt_abort_deg": float(np.degrees(SafetyCfg().max_tilt_err))})
    if "g1" in cmds:
        for prof in ("ground", "slack"):
            o = {k: v for k, v in PROFILES[prof].items()}
            res[f"g1_{prof}"] = g1_gate(ckpt, o, clips, clips, f"{tag}_")
    jobs2, jobs3 = [], []
    for prof, po in PROFILES.items():
        for cl in clips:
            for la in CLIPSET_LAT:
                if "g2" in cmds:
                    jobs2 += [{"key": prof, **po, "ckpt": ckpt, "clip": cl, "lat": la, "kps": k, "mass": mm, "fric": fr}
                              for (k, mm) in CLIPSET_ROBUST for fr in G2_FRICTION]
                if "g3" in cmds:
                    jobs3.append({"tag": prof, **po, "ckpt": ckpt, "clip": cl, "lat": la, "support": "band", "end": "blend", "blend_lat": 2})
    t0 = time.time()
    with Pool(workers) as p:
        if jobs3:
            r3 = list(p.imap_unordered(g3_run, jobs3, chunksize=1))
            res["g3"] = sorted(r3, key=lambda r: (r["tag"], clips.index(r["clip"]), r["lat"]))
            print(f"  clipset G3 {len(r3)} runs ({time.time() - t0:.0f} s)", flush=True)
        if jobs2:
            r2 = []
            for i, r in enumerate(p.imap_unordered(g2_run, jobs2, chunksize=1)):
                r2.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  clipset G2 {i + 1}/{len(jobs2)} ({time.time() - t0:.0f} s)", flush=True)
            res["g2"] = sorted(r2, key=lambda r: (r["key"], clips.index(r["clip"]), -r["kps"], r["mass"], r["lat"], r["fric"]))
    if any(k in res for k in ("g1_slack", "g2", "g3")):
        res["summary"] = {cl: clip_summary(res, cl) for cl in clips}
    return res


def _max(xs):
    xs = [x for x in xs if x is not None]
    return float(max(xs)) if xs else float("nan")


def clip_summary(res: dict, cl: str) -> dict:
    out: dict = {}
    raw = load_clip_raw(cl)
    v = np.linalg.norm(raw["root_lin_vel_w"][:, :2], axis=1)
    d = raw["root_pos"][-1, :2] - raw["root_pos"][0, :2]
    qw = raw["root_quat_wxyz"].astype(np.float64)
    yaw = np.arctan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))
    c0, s0 = np.cos(yaw[0]), np.sin(yaw[0])
    d_h = [float(c0 * d[0] + s0 * d[1]), float(-s0 * d[0] + c0 * d[1])]  # in the frame-0 heading frame: forward, left
    out["clip_info"] = {"frames": int(len(v)), "duration_s": round(len(v) / 50.0, 2), "disp_xy_m": d.tolist(), "dist_m": float(np.linalg.norm(d)),
                        "disp_heading_fwd_left_m": d_h, "yaw_change_deg": float(np.degrees(np.angle(np.exp(1j * (yaw[-1] - yaw[0]))))),
                        "path_m": float(np.linalg.norm(np.diff(raw["root_pos"][:, :2], axis=0), axis=1).sum()), "peak_speed_mps": float(v.max()),
                        "start_speed_mps": float(v[0]), "end_speed_mps": float(v[-1]), "max_joint_speed_rad_s": float(np.abs(raw["joint_vel"]).max())}
    for prof in PROFILES:
        g1 = res.get(f"g1_{prof}")
        if g1:
            x = next(r for r in g1["rows"] if r["clip"] == cl)
            out[f"g1_{prof}"] = {k: x[k] for k in ("env_fail_step", "s2s_fail_step", "fail_agree", "env_rmse", "s2s_rmse", "rmse_ratio", "diverge_s", "n_aborts")} | {
                "max_abs_a": x["stats"]["max_abs_a"], "max_vfrac": x["stats"]["max_vfrac"]}
        g2 = [r for r in res.get("g2", []) if r["clip"] == cl and r["key"] == prof]
        nom = [r for r in g2 if r["kps"] == 1.0 and r["mass"] == 0.0]
        rob = [r for r in g2 if not (r["kps"] == 1.0 and r["mass"] == 0.0)]
        g3 = [r for r in res.get("g3", []) if r["clip"] == cl and r["tag"] == prof]
        po = PROFILES[prof]
        if nom:
            st = [r["stats"] for r in nom]
            out[f"g2_{prof}"] = {
                "runs": len(nom), "falls": sum(r["fall"] for r in nom), "abort_runs": sum(r["n_abort_steps"] > 0 for r in nom), "abort_fp": sum(r["abort_fp"] for r in nom),
                "abort_reasons": sorted({a for r in nom for a in r["abort_reasons"]}),
                "rmse_mean": _mean([r["track_rmse"] for r in nom]), "rmse_max": _max([r["track_rmse"] for r in nom]),
                "max_abs_a": _max([s["max_abs_a"] for s in st]), "max_vfrac": _max([s["max_vfrac"] for s in st]),
                "max_vfrac_joint": max(st, key=lambda s: s["max_vfrac"])["max_vfrac_joint"], "max_vfrac_1khz": _max([r["vfrac_1khz_max"] for r in nom]),
                "max_tilt_err_deg": _max([r["max_tilt_err_deg"] for r in nom]), "max_tilt_deg": _max([r["max_tilt_deg"] for r in nom]),
                "max_tau_ratio": _max([r["max_tau_ratio"] for r in nom]), "min_frac_tau_ok": float(min(1 - r["frac_tau_gt_0p9"] for r in nom)),
                "model_sat_ms": sum(sum(r["model_sat_ms"].values()) for r in nom), "first_step_tau_max_Nm": _max([r["first_step_tau_max_Nm"] for r in nom]),
                "disp_err_m_max": _max([r["disp_err_m"] for r in nom]), "disp_x_mean": _mean([r["disp_xy"][0] for r in nom]), "disp_y_mean": _mean([r["disp_xy"][1] for r in nom]),
                "train_term_runs": sum(r["train_term_step"] >= 0 for r in nom),
            }
        if rob:
            out[f"g2r_{prof}"] = {f"kp{k}_m{mm}": {"falls": sum(r["fall"] for r in rob if r["kps"] == k and r["mass"] == mm),
                                                   "abort_runs": sum(r["n_abort_steps"] > 0 for r in rob if r["kps"] == k and r["mass"] == mm),
                                                   "runs": sum(1 for r in rob if r["kps"] == k and r["mass"] == mm),
                                                   "max_abs_a": _max([r["stats"]["max_abs_a"] for r in rob if r["kps"] == k and r["mass"] == mm]),
                                                   "max_vfrac": _max([r["stats"]["max_vfrac"] for r in rob if r["kps"] == k and r["mass"] == mm]),
                                                   "max_tilt_err_deg": _max([r["max_tilt_err_deg"] for r in rob if r["kps"] == k and r["mass"] == mm])}
                                  for (k, mm) in CLIPSET_ROBUST[1:]}
        if g3:
            st = [r["stats"] for r in g3]
            out[f"g3_{prof}"] = {
                "runs": len(g3), "upright": sum(r["upright"] for r in g3), "completed": sum(r["completed"] for r in g3), "abort_runs": sum(r["n_abort_steps"] > 0 for r in g3),
                "first_aborts": [r["first_abort"] for r in g3 if r["first_abort"]],
                "max_abs_a": _max([s["max_abs_a"] for s in st]), "max_vfrac": _max([s["max_vfrac"] for s in st]), "max_vfrac_1khz": _max([r["vfrac_1khz_max_all"] for r in g3]),
                "max_tilt_err_deg": _max([r["max_tilt_err_policy_deg"] for r in g3]), "max_tilt_deg": _max([r["max_tilt_deg"] for r in g3]),
                "jump_policy_start": _max([r["jump_policy_start"] for r in g3]), "jump_end_blend": _max([r["jump_end_blend_whole"] for r in g3]),
                "max_tau_ratio_policy": _max([r["max_tau_ratio_policy"] for r in g3]), "max_tau_ratio_end_blend": _max([r["max_tau_ratio_end_blend"] for r in g3]),
                "tilt_end_hold_deg": _max([r["tilt_at_phase_end_deg"].get("end_hold") for r in g3]),
                "policy_disp_x": _mean([r["policy_disp_xy"][0] for r in g3]), "policy_disp_y": _mean([r["policy_disp_xy"][1] for r in g3]),
            }
        src = [out[k] for k in (f"g2_{prof}", f"g3_{prof}") if k in out]
        if src:
            a, vf, te = _max([x["max_abs_a"] for x in src]), _max([x["max_vfrac"] for x in src]), _max([x["max_tilt_err_deg"] for x in src])
            tl = float(np.degrees(SafetyCfg().max_tilt_err))
            out[f"margin_{prof}"] = {"max_abs_a": a, "max_vfrac": vf, "max_tilt_err_deg": te,
                                     "a": po["max_abs_action"] / a, "dq": po["max_vel_frac"] / vf, "tilt": tl / te, "source": "G2 nominal (kp 1.0, +0 kg, lat 0-20 ms, mu 0.6/1.0) + G3"}
    segs = clip_segments(raw)
    if segs:  # combo clip: per-segment maxima over the same run sets as the clip-level numbers
        for prof in PROFILES:
            sets = {"g1": [x for x in (res.get(f"g1_{prof}") or {}).get("rows", []) if x["clip"] == cl],
                    "g2_nominal": [r for r in res.get("g2", []) if r["clip"] == cl and r["key"] == prof and r["kps"] == 1.0 and r["mass"] == 0.0],
                    "g2_robust": [r for r in res.get("g2", []) if r["clip"] == cl and r["key"] == prof and not (r["kps"] == 1.0 and r["mass"] == 0.0)],
                    "g3": [r for r in res.get("g3", []) if r["clip"] == cl and r["tag"] == prof]}
            sets["g2_nominal+g3"] = sets["g2_nominal"] + sets["g3"]
            seg_out = {}
            for name, s0, s1 in segs:
                seg_out[name] = {"frames": [s0, s1]}
                for k, rows in sets.items():
                    ss = [r["stats"]["segments"][name] for r in rows if "segments" in r["stats"] and r["stats"]["segments"][name].get("steps")]
                    if not ss:
                        continue
                    seg_out[name][k] = {"runs": len(ss), "max_abs_a": _max([x["max_abs_a"] for x in ss]), "max_vfrac": _max([x["max_vfrac"] for x in ss]),
                                        "max_vfrac_joint": max(ss, key=lambda x: x["max_vfrac"])["max_vfrac_joint"],
                                        "max_tilt_err_deg": _max([x["max_tilt_err_deg"] for x in ss]), "rmse_max": _max([x["rmse"] for x in ss]),
                                        "rmse_mean": _mean([x["rmse"] for x in ss if x["rmse"] is not None]),
                                        "stage_trips": sum(x["stage_trips"][STAGE_OF_PROFILE[prof]] for x in ss) if prof in STAGE_OF_PROFILE else None}
            out[f"segments_{prof}"] = seg_out
    out["gate"] = gate_clip(out)
    return out


def gate_clip(s: dict) -> dict:
    g = {}
    if "g1_slack" in s:
        x = s["g1_slack"]
        y = s.get("g1_ground", x)
        g["g1"] = all(z["fail_agree"] and z["s2s_fail_step"] < 0 and z["rmse_ratio"] <= 1.2 for z in (x, y))
    if "g2_slack" in s:
        g["g2_slack_nominal"] = s["g2_slack"]["falls"] == 0 and s["g2_slack"]["abort_runs"] == 0
    if "g3_slack" in s:
        g["g3_slack"] = s["g3_slack"]["upright"] == s["g3_slack"]["runs"] and s["g3_slack"]["abort_runs"] == 0 and s["g3_slack"]["completed"] == s["g3_slack"]["runs"]
    if "margin_slack" in s:
        m = s["margin_slack"]
        g["margins_slack"] = min(m["a"], m["dq"], m["tilt"]) >= MARGIN_MIN
        g["min_margin_slack"] = min(m["a"], m["dq"], m["tilt"])
    g["pass"] = all(v for k, v in g.items() if k != "min_margin_slack") and len(g) >= 4
    return g


def write_clipset_md(res: dict) -> str:
    clips, ck = res["clips"], res["ckpt"]
    S = res["summary"]
    L = [f"# Sim2sim gates on a custom clip set ({res['tag']}, {ck})", "",
         f"Generated by `scripts/deploy/sim2sim.py --clips ... --tag {res['tag']}` on {time.strftime('%Y-%m-%d %H:%M')}; raw numbers in `sim2sim_{res['tag']}_{ck}.json`. Simulation only.",
         f"Profiles (runner --stage): {json.dumps(PROFILES)}; tilt-error abort {res['tilt_abort_deg']:.0f} deg in every profile.", "",
         "## Clips", "", "| clip | frames | duration [s] | net disp fwd / left (frame-0 heading) [m] | yaw change [deg] | path [m] | peak pelvis speed [m/s] | start / end speed [m/s] | max joint speed [rad/s] |", "|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        i = S[c]["clip_info"]
        L.append(f"| {c} | {i['frames']} | {i['duration_s']:.2f} | {i['disp_heading_fwd_left_m'][0]:.2f} / {i['disp_heading_fwd_left_m'][1]:.2f} | {i['yaw_change_deg']:.0f} | {i['path_m']:.2f} | {i['peak_speed_mps']:.2f} | {i['start_speed_mps']:.3f} / {i['end_speed_mps']:.3f} | {i['max_joint_speed_rad_s']:.1f} |")
    L += ["", "## Gate summary", "", f"Pass: G1 (both profiles) same fail outcome as the env, no fall, RMSE <= 1.2x env; slack G2 nominal (lat 0/10/20, mu 0.6/1.0) and G3 (lat 0/10/20): 0 falls, 0 aborts; slack margins >= {MARGIN_MIN}x for |a|, dq/vlim, tilt error.", "",
          "| clip | G1 | G2 slack nominal | G3 slack | margins slack | min margin | PASS |", "|---|---|---|---|---|---|---|"]
    for c in clips:
        g = S[c]["gate"]
        L.append(f"| {c} | {fmt(g.get('g1'))} | {fmt(g.get('g2_slack_nominal'))} | {fmt(g.get('g3_slack'))} | {fmt(g.get('margins_slack'))} | {fmt(g.get('min_margin_slack'), 2)} | **{'PASS' if g['pass'] else 'FAIL'}** |")
    for prof in ("ground", "slack"):
        if not any(f"g1_{prof}" in S[c] for c in clips):
            continue
        L += ["", f"## G1 same-model ({prof} profile)", "", "| clip | env fail | s2s fail | env RMSE | s2s RMSE | ratio | diverge [s] | aborts | max abs a | max dq/vlim |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in clips:
            x = S[c][f"g1_{prof}"]
            L.append(f"| {c} | {x['env_fail_step']} | {x['s2s_fail_step']} | {x['env_rmse']:.4f} | {x['s2s_rmse']:.4f} | {x['rmse_ratio']:.2f} | {fmt(x['diverge_s'], 2)} | {x['n_aborts']} | {x['max_abs_a']:.2f} | {x['max_vfrac']:.3f} |")
    for prof in ("slack", "ground"):
        if not any(f"g2_{prof}" in S[c] for c in clips):
            continue
        L += ["", f"## G2 nominal, {prof} profile (Unitree MJCF, at rest on the ground, kp 1.0, +0 kg, lat 0/10/20 ms x mu 0.6/1.0)", "",
              "| clip | falls / runs | abort runs | RMSE mean / max | max abs a | max dq/vlim 50 Hz (joint) | 1 kHz | max tilt err [deg] | max tilt [deg] | max tau/limit | min frac tau<=0.9 | pelvis disp x / y [m] | max |disp - ref| [m] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in clips:
            x = S[c][f"g2_{prof}"]
            L.append(f"| {c} | {x['falls']}/{x['runs']} | {x['abort_runs']} {x['abort_reasons'] or ''} | {x['rmse_mean']:.4f} / {x['rmse_max']:.4f} | {x['max_abs_a']:.2f} | {x['max_vfrac']:.3f} ({x['max_vfrac_joint']}) | {x['max_vfrac_1khz']:.3f} | "
                     f"{x['max_tilt_err_deg']:.1f} | {x['max_tilt_deg']:.1f} | {x['max_tau_ratio']:.2f} | {x['min_frac_tau_ok']:.4f} | {x['disp_x_mean']:.2f} / {x['disp_y_mean']:.2f} | {x['disp_err_m_max']:.2f} |")
    for prof in ("slack", "ground"):
        if not any(f"g2r_{prof}" in S[c] for c in clips):
            continue
        keys = [f"kp{k}_m{mm}" for (k, mm) in CLIPSET_ROBUST[1:]]
        L += ["", f"## G2 robustness, {prof} profile: falls / abort runs / runs (lat 0/10/20 x mu 0.6/1.0); worst |a|, dq/vlim, tilt err", "",
              "| clip | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for c in clips:
            r = S[c][f"g2r_{prof}"]
            L.append(f"| {c} | " + " | ".join(f"{r[k]['falls']}/{r[k]['abort_runs']}/{r[k]['runs']}; {r[k]['max_abs_a']:.1f}, {r[k]['max_vfrac']:.2f}, {r[k]['max_tilt_err_deg']:.1f} deg" for k in keys) + " |")
    for prof in ("slack", "ground"):
        if not any(f"g3_{prof}" in S[c] for c in clips):
            continue
        L += ["", f"## G3 transitions, {prof} profile (stand -> 4 s interp -> 1 s HOLD under gantry -> POLICY gantry slack -> 1 s END blend under gantry, lat 0/10/20 ms)", "",
              "| clip | upright / completed / runs | abort runs | max abs a | max dq/vlim 50 Hz | 1 kHz | max tilt err [deg] | max tilt [deg] | jump at POLICY start [Nm/ms] | max jump END blend | max tau/limit policy / blend | tilt after END hold [deg] | pelvis disp x / y [m] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in clips:
            x = S[c][f"g3_{prof}"]
            L.append(f"| {c} | {x['upright']}/{x['completed']}/{x['runs']} | {x['abort_runs']} {x['first_aborts'][:1] or ''} | {x['max_abs_a']:.2f} | {x['max_vfrac']:.3f} | {x['max_vfrac_1khz']:.3f} | {x['max_tilt_err_deg']:.1f} | "
                     f"{x['max_tilt_deg']:.1f} | {x['jump_policy_start']:.1f} | {x['jump_end_blend']:.2f} | {x['max_tau_ratio_policy']:.2f} / {x['max_tau_ratio_end_blend']:.2f} | {fmt(x['tilt_end_hold_deg'], 1)} | {x['policy_disp_x']:.2f} / {x['policy_disp_y']:.2f} |")
    for prof in ("slack", "ground"):
        for c in clips:
            sg = S[c].get(f"segments_{prof}")
            if not sg:
                continue
            L += ["", f"## Per segment, {c}, {prof} profile (max over runs; G2 nominal = kp 1.0, +0 kg, lat 0/10/20 x mu 0.6/1.0; robust = kp 0.85/1.15, +1.8 kg)", "",
                  "| segment | frames | G2 nominal + G3: max abs a | max dq/vlim (joint) | max tilt err [deg] | G2 nominal RMSE mean / max | G2 robust: max abs a | max dq/vlim | max tilt err [deg] | G1 max abs a / dq/vlim / tilt err / RMSE |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for name, x in sg.items():
                a, n, rb, g1 = x.get("g2_nominal+g3"), x.get("g2_nominal"), x.get("g2_robust"), x.get("g1")
                L.append(f"| {name} | {x['frames'][0]}-{x['frames'][1] - 1} | " + (f"{a['max_abs_a']:.2f} | {a['max_vfrac']:.3f} ({a['max_vfrac_joint']}) | {a['max_tilt_err_deg']:.1f} | " if a else "- | - | - | ")
                         + (f"{n['rmse_mean']:.4f} / {n['rmse_max']:.4f} | " if n else "- | ")
                         + (f"{rb['max_abs_a']:.2f} | {rb['max_vfrac']:.3f} | {rb['max_tilt_err_deg']:.1f} | " if rb else "- | - | - | ")
                         + (f"{g1['max_abs_a']:.2f} / {g1['max_vfrac']:.3f} / {g1['max_tilt_err_deg']:.1f} / {g1['rmse_max']:.4f} |" if g1 else "- |"))
    L += ["", "## Abort-threshold margins (threshold / max seen over G2 nominal + G3; 50 Hz sampled state as the core checks)", "",
          "| clip | profile | max abs a | margin a | max dq/vlim | margin dq | max tilt err [deg] | margin tilt (20 deg) |", "|---|---|---|---|---|---|---|---|"]
    for c in clips:
        for prof in ("slack", "ground"):
            m = S[c].get(f"margin_{prof}")
            if m:
                L.append(f"| {c} | {prof} | {m['max_abs_a']:.2f} | {m['a']:.2f}x | {m['max_vfrac']:.3f} | {m['dq']:.2f}x | {m['max_tilt_err_deg']:.1f} | {m['tilt']:.2f}x |")
    return "\n".join(L) + "\n"


# ============================================================================================ policy standby (runner STANDBY_PRE / STANDBY_POST)
# Runner protocol since 2026-10-24: HOLD (stand gains, gantry) -[R1+A]-> STANDBY_PRE: DeployCore on static_clip(clip, 0),
# step(advance=False), training gains -[R1+X]-> POLICY: switch_clip(clip, 0) (history kept) -> clip done -> STANDBY_POST:
# switch_clip(static_clip(clip, last)), step(advance=False). Gate: Unitree MJCF, gantry (pelvis spring) only until the
# policy takes over, then NO support for STANDBY_PRE (pre_s), the motion and STANDBY_POST (post_s).

STANDBY_CLIPS = (IDLE, "walk_forward_stop_001__A026", "combo_itch_clap_fwd_side", "ib_dodge_back_L_001__A437_M")
STANDBY_LAT = (0, 10, 20)
STANDBY_KP = (0.85, 1.0, 1.15)
STANDBY_MASS = (0.0, 1.8)
STANDBY_R1X_TILT_DEG, STANDBY_R1X_DQ = 8.0, 1.0  # runner R1+X check defaults (--standby-max-tilt-deg / --standby-max-dq)
STANDBY_R1X_WIN_S, STANDBY_R1X_MIN_S, STANDBY_R1X_CAP_S = 3.0, 3.0, 30.0  # runner --standby-window-s / --standby-min-s; retry cap
STANDBY_SMOOTH = 0.5  # pass: hand-off torque jump <= this x the HOLD->STANDBY_PRE jump of the same run


def _yaw(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def standby_run(o: dict) -> dict:
    from supermimic.deploy.core import static_clip

    ckpt, clip_name, lat, kps, mass, fric = o["ckpt"], o["clip"], o["lat"], o["kps"], o["mass"], o["fric"]
    pre_s, post_s = o.get("pre_s", 60.0), o.get("post_s", 30.0)
    # R1+X: the operator presses at pre_s; the motion starts at the first step at which the runner's check passes (re-pressing
    # until it does), at most r1x_cap_s later; otherwise the motion never starts (the runner keeps refusing)
    r1x_win = max(1, int(round(o.get("r1x_win_s", STANDBY_R1X_WIN_S) / 0.02)))
    r1x_min = int(round(o.get("r1x_min_s", STANDBY_R1X_MIN_S) / 0.02))
    r1x_cap = int(round(o.get("r1x_cap_s", STANDBY_R1X_CAP_S) / 0.02))
    cfg, clip = load_cfg(ckpt), load_clip(clip_name)
    m = unitree_model(mass, fric, o.get("lock", False))
    rob = Robot(m, cfg.joint_names)
    q0 = cfg.default_joint_pos.copy()
    rob.set_state(np.array([0.0, 0.0, 0.8]), np.array([1.0, 0, 0, 0]), q0)
    rob.place_on_ground()
    band = {"on": True, "anchor": rob.d.qpos[rob.fq : rob.fq + 3].copy()}
    sim = PDSim(rob, cfg, cfg.effort_limit)
    skp, skd = stand_gains(cfg)
    skp, kp_pol = skp * kps, cfg.kp * kps  # motor gain error scales every gain we send (kd as in G2: unscaled)
    pose0 = np.clip(clip.joint_pos[0], cfg.soft_lower, cfg.soft_upper)
    tau_hist = []
    st = {"phase": "interp", "upright": True, "fall_phase": None, "fall_s": None}

    def run_ms(n):
        for _ in range(n):
            rob.d.xfrc_applied[:, :3] = 0.0
            if band["on"]:
                rob.d.xfrc_applied[rob.pelvis, :3] = BAND_K * (band["anchor"] - rob.d.qpos[rob.fq : rob.fq + 3]) - BAND_D * rob.d.qvel[rob.fv : rob.fv + 3]
            sim.step_ms()
            tau_hist.append(sim.tau_prev.copy())
            if st["upright"] and fell(rob):
                st["upright"], st["fall_phase"], st["fall_s"] = False, st["phase"], sim.ms / 1000

    sim.command(q0, 0, skp, skd)
    for k in range(200):  # 4 s interpolation, stand gains, gantry
        sim.command(interpolate_pose(q0, pose0, (k + 1) * 0.02, 4.0), 0)
        run_ms(20)
    st["phase"] = "hold"
    for _ in range(50):
        sim.command(pose0, 0)
        run_ms(20)
    band["on"] = False  # gantry slack from the moment the policy takes over
    core = DeployCore(cfg, load_policy(ckpt), static_clip(clip, 0), safety(o))
    core.reset(0)
    ph: dict[str, dict] = {}
    marks = {}
    first = [True]
    vlim = cfg.velocity_limit

    def r1x_ok(rec, k):  # runner standby_allowed() on the first k standby steps (tilt err / max|dq| over the last window)
        if k < max(r1x_min, 1) or k < r1x_win:
            return False
        return (np.degrees(max(rec["te"][k - r1x_win : k])) < STANDBY_R1X_TILT_DEG and max(rec["dq"][k - r1x_win : k]) < STANDBY_R1X_DQ)

    def policy_phase(name, n_steps, advance):
        marks[name] = sim.ms
        st["phase"] = name
        p0 = rob.d.qpos[rob.fq : rob.fq + 2].copy()
        y0 = _yaw(rob.d.qpos[rob.fq + 3 : rob.fq + 7])
        rec = {"te": [], "a": [], "vf": [], "dq": [], "dxy": [], "dyaw": [], "aborts": [], "xy": [], "ok": []}
        pre = name == "standby_pre"
        k = 0
        while st["upright"]:
            if pre:
                if k >= n_steps and rec["ok"] and rec["ok"][-1]:
                    break  # R1+X accepted
                if k >= n_steps + r1x_cap:
                    break  # never accepted within the cap
            elif n_steps is not None and k >= n_steps:
                break
            q, dq, gyro, quat = rob.state()
            g_ref = core.clip.proj_grav[min(core.frame, core.clip.num_frames - 1)]
            te = float(np.arccos(np.clip(np.dot(projected_gravity(quat), g_ref) / (np.linalg.norm(g_ref) + 1e-9), -1, 1)))
            r = core.step(q, dq, gyro, quat, advance=advance)
            if first[0]:
                sim.command(r.q_target, lat, kp_pol, cfg.kd)  # gains switch together with the first policy target
                first[0] = False
            else:
                sim.command(r.q_target, lat)
            if r.abort:
                rec["aborts"].append((k, r.abort))
            rec["te"].append(te)
            rec["a"].append(float(np.abs(r.action).max()))
            rec["vf"].append(float((np.abs(dq) / vlim)[rob.live].max()))
            rec["dq"].append(float(np.abs(dq)[rob.live].max()))
            run_ms(20)
            k += 1
            xy = rob.d.qpos[rob.fq : rob.fq + 2]
            rec["dxy"].append(float(np.linalg.norm(xy - p0)))
            rec["xy"].append(xy.copy())
            dy = float(np.degrees(np.angle(np.exp(1j * (_yaw(rob.d.qpos[rob.fq + 3 : rob.fq + 7]) - y0)))))
            prev = rec["dyaw"][-1] if rec["dyaw"] else 0.0
            rec["dyaw"].append(prev + (dy - prev + 180.0) % 360.0 - 180.0)  # unwrapped
            if pre:
                rec["ok"].append(r1x_ok(rec, k))
            if advance and r.done:
                break
        te, a, vf, dqm = (np.array(rec[x]) for x in ("te", "a", "vf", "dq"))
        out = {"steps": k, "aborts": len(rec["aborts"]), "first_abort": rec["aborts"][0] if rec["aborts"] else None}
        if k:
            out |= {"max_tilt_err_deg": float(np.degrees(te.max())), "mean_tilt_err_deg": float(np.degrees(te.mean())),
                    "max_abs_a": float(a.max()), "max_vfrac": float(vf.max()), "max_dq": float(dqm.max()),
                    "drift_xy_max_m": float(max(rec["dxy"])), "drift_xy_final_m": rec["dxy"][-1],
                    "drift_yaw_max_deg": float(np.abs(rec["dyaw"]).max()), "drift_yaw_final_deg": rec["dyaw"][-1]}
            if pre:
                ok = np.array(rec["ok"])
                started = bool(st["upright"] and ok[-1] and k >= n_steps)
                # moving at the motion start (review definition): pelvis speed over the last 2 s > 2 cm/s or yaw rate > 2 deg/s
                mv_v = float(np.linalg.norm(rec["xy"][-1] - rec["xy"][-101]) / 2.0) if k > 100 else None
                mv_w = float(abs(rec["dyaw"][-1] - rec["dyaw"][-101]) / 2.0) if k > 100 else None
                out |= {"r1x_ok_first_s": float((np.argmax(ok) + 1) * 0.02) if ok.any() else None,
                        "r1x_ok_frac_after_min": float(ok[r1x_min:].mean()) if k > r1x_min else None, "r1x_ok_at_end": bool(ok[-1]),
                        "r1x_started": started, "r1x_never": bool(st["upright"] and not started),
                        "r1x_start_s": float(k * 0.02) if started else None,
                        "r1x_wait_s": float(k * 0.02 - pre_s) if started else None,
                        "start_speed_mps": mv_v, "start_yaw_rate_dps": mv_w,
                        "start_moving": bool(started and mv_v is not None and (mv_v > 0.02 or mv_w > 2.0))}
            if name in ("standby_pre", "standby_post") and k > 250:  # settled part: after the first 5 s
                out |= {"settled_max_tilt_err_deg": float(np.degrees(te[250:].max())), "settled_max_dq": float(dqm[250:].max()),
                        "settled_drift_xy_m": float(rec["dxy"][-1] - rec["dxy"][249]) if len(rec["dxy"]) > 250 else None}
        ph[name] = out

    policy_phase("standby_pre", int(round(pre_s / 0.02)), False)
    if st["upright"] and ph["standby_pre"].get("r1x_started"):
        core.switch_clip(clip, 0)
        policy_phase("policy", None, True)
    if st["upright"] and "policy" in ph:  # no STANDBY_POST when R1+X was never accepted (the motion did not run)
        core.switch_clip(static_clip(clip, clip.num_frames - 1), 0)
        policy_phase("standby_post", int(round(post_s / 0.02)), False)
    end_ms = sim.ms
    tau = np.array(tau_hist)
    jump = np.abs(np.diff(tau, axis=0, prepend=tau[:1]))

    def win(a, b):
        a, b = max(a, 0), min(b, len(jump))
        if b <= a:
            return None, ""
        seg = jump[a:b]
        return float(seg.max()), cfg.joint_names[int(np.argmax(seg.max(0)))]

    def steady(a, b):
        a, b = max(a, 0), min(b, len(jump))
        if b - a < 50:
            return None, None
        x = jump[a:b].max(1)
        return float(np.percentile(x, 99)), float(x.max())

    J = {}
    for key, nm in (("hold_to_standby", "standby_pre"), ("standby_to_policy", "policy"), ("policy_to_post", "standby_post")):
        if nm in marks:
            J[key], J[key + "_joint"] = win(marks[nm] - 1, marks[nm] + lat + 40)
    pre_end = marks.get("policy", end_ms)
    J["standby_pre_steady_p99"], J["standby_pre_steady_max"] = steady(marks["standby_pre"] + 1000, pre_end)
    if "policy" in marks:
        J["policy_steady_p99"], J["policy_steady_max"] = steady(marks["policy"] + lat + 60, marks.get("standby_post", end_ms))
    if "standby_post" in marks:
        J["standby_post_steady_p99"], J["standby_post_steady_max"] = steady(marks["standby_post"] + 1000, end_ms)
    h = J.get("hold_to_standby") or float("nan")
    smooth = all(J.get(k) is not None and J[k] <= STANDBY_SMOOTH * h for k in ("standby_to_policy", "policy_to_post"))
    n_ab = sum(p["aborts"] for p in ph.values())
    sb_te = max([ph[p].get("max_tilt_err_deg", np.inf) for p in ("standby_pre", "standby_post") if p in ph] or [np.inf])
    completed = st["upright"] and "standby_post" in ph and ph["standby_post"]["steps"] >= int(round(post_s / 0.02))
    sp = ph.get("standby_pre", {})
    return {**o, "upright": st["upright"], "fall_phase": st["fall_phase"], "fall_s": st["fall_s"], "completed": completed,
            "aborts": n_ab, "phases": ph, "jumps": J, "smooth": smooth, "standby_max_tilt_err_deg": sb_te,
            "r1x_never": bool(sp.get("r1x_never", False)), "r1x_wait_s": sp.get("r1x_wait_s"), "start_moving": bool(sp.get("start_moving", False)),
            "pass": bool(completed and n_ab == 0 and sb_te < STANDBY_R1X_TILT_DEG and smooth)}


def standby_gate(ckpt: str, clips, workers: int, pre_s: float, post_s: float, lock: bool = False, grid: dict | None = None,
                 r1x: dict | None = None) -> dict:
    grid = {"profiles": tuple(PROFILES), "lat": STANDBY_LAT, "fric": G2_FRICTION, "kps": STANDBY_KP, "mass": STANDBY_MASS, **(grid or {})}
    r1x = {"win_s": STANDBY_R1X_WIN_S, "min_s": STANDBY_R1X_MIN_S, "cap_s": STANDBY_R1X_CAP_S, **(r1x or {})}
    jobs = [{"tag": prof, **PROFILES[prof], "ckpt": ckpt, "clip": cl, "lat": la, "kps": k, "mass": mm, "fric": fr, "pre_s": pre_s, "post_s": post_s, "lock": lock,
             "r1x_win_s": r1x["win_s"], "r1x_min_s": r1x["min_s"], "r1x_cap_s": r1x["cap_s"]}
            for prof in grid["profiles"] for cl in clips for la in grid["lat"] for fr in grid["fric"] for k in grid["kps"] for mm in grid["mass"]]
    t0 = time.time()
    res = []
    with Pool(min(workers, len(jobs))) as p:
        for i, r in enumerate(p.imap_unordered(standby_run, jobs, chunksize=1)):
            res.append(r)
            ph = r["phases"]
            if not r["pass"] or (i + 1) % 20 == 0:
                print(f"  standby [{time.time() - t0:.0f} s] {i + 1}/{len(jobs)} {r['tag']} {r['clip'][:22]} lat {r['lat']} mu {r['fric']} kp {r['kps']} +{r['mass']} kg: "
                      f"pass {r['pass']} r1x never/wait/moving {r['r1x_never']}/{fmt(r['r1x_wait_s'], 2)}/{r['start_moving']} upright {r['upright']} ({r['fall_phase']}) aborts {r['aborts']} sb tilt err {fmt(r['standby_max_tilt_err_deg'], 1)} "
                      f"jumps {fmt(r['jumps'].get('hold_to_standby'), 1)}/{fmt(r['jumps'].get('standby_to_policy'), 2)}/{fmt(r['jumps'].get('policy_to_post'), 2)}", flush=True)
    res.sort(key=lambda r: (r["tag"], clips.index(r["clip"]), r["lat"], r["fric"], r["kps"], r["mass"]))
    return {"ckpt": ckpt, "clips": list(clips), "pre_s": pre_s, "post_s": post_s, "profiles": {k: PROFILES[k] for k in grid["profiles"]}, "rows": res, "r1x": r1x,
            "grid": {k: grid[k] for k in ("lat", "fric", "kps", "mass")},
            "pass_rule": f"0 falls, 0 aborts, standby tilt err < {STANDBY_R1X_TILT_DEG} deg, hand-off jumps STANDBY->POLICY and POLICY->STANDBY_POST "
                         f"<= {STANDBY_SMOOTH} x HOLD->STANDBY_PRE jump of the same run"}


def standby_summary(rows: list[dict]) -> dict:
    def mx(path):
        v = []
        for r in rows:
            x = r
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
                if x is None:
                    break
            if isinstance(x, (int, float)) and x is not None:
                v.append(abs(x))
        return float(max(v)) if v else None

    def mn(path):
        v = [r["phases"]["standby_pre"].get(path) for r in rows if "standby_pre" in r["phases"]]
        v = [x for x in v if x is not None]
        return float(max(v)) if v else None

    return {
        "runs": len(rows), "pass": sum(r["pass"] for r in rows), "falls": sum(not r["upright"] for r in rows),
        "fall_phases": sorted({r["fall_phase"] for r in rows if r["fall_phase"]}), "abort_runs": sum(r["aborts"] > 0 for r in rows),
        "first_aborts": [(p, v["first_abort"]) for r in rows for p, v in r["phases"].items() if v.get("first_abort")][:5],
        **{f"{p}_{k}": mx(("phases", p, k)) for p in ("standby_pre", "standby_post") for k in
           ("max_tilt_err_deg", "drift_xy_max_m", "drift_yaw_max_deg", "max_abs_a", "max_vfrac", "max_dq", "settled_max_tilt_err_deg",
            "settled_max_dq", "settled_drift_xy_m")},
        "policy_max_tilt_err_deg": mx(("phases", "policy", "max_tilt_err_deg")), "policy_max_abs_a": mx(("phases", "policy", "max_abs_a")),
        "policy_max_vfrac": mx(("phases", "policy", "max_vfrac")),
        "r1x_ok_first_s_max": mn("r1x_ok_first_s"),
        "r1x_ok_at_end_all": all(r["phases"].get("standby_pre", {}).get("r1x_ok_at_end", False) for r in rows if r["upright"] or r["fall_phase"] != "standby_pre"),
        "r1x_never": sum(bool(r.get("r1x_never")) for r in rows),
        "r1x_wait_max_s": (max(w) if (w := [r["r1x_wait_s"] for r in rows if r.get("r1x_wait_s") is not None]) else None),
        "r1x_wait_n_delayed": sum((r.get("r1x_wait_s") or 0) > 0 for r in rows),
        "start_moving": sum(bool(r.get("start_moving")) for r in rows),
        **{f"jump_{k}": mx(("jumps", k)) for k in ("hold_to_standby", "standby_to_policy", "policy_to_post", "standby_pre_steady_p99",
                                                   "standby_pre_steady_max", "policy_steady_p99", "policy_steady_max", "standby_post_steady_p99", "standby_post_steady_max")},
        "jump_hold_to_standby_min": float(min(r["jumps"]["hold_to_standby"] for r in rows if r["jumps"].get("hold_to_standby") is not None)),
        "max_ratio_handoff_vs_hold": (max(v) if (v := [max(r["jumps"]["standby_to_policy"], r["jumps"]["policy_to_post"]) / r["jumps"]["hold_to_standby"]
                                                         for r in rows if r["jumps"].get("hold_to_standby") and r["jumps"].get("standby_to_policy") is not None
                                                         and r["jumps"].get("policy_to_post") is not None]) else None),
    }


def write_standby_md(res: dict) -> str:
    rows = res["rows"]
    L = [f"# Policy standby gate ({res['ckpt']})", "",
         f"Unitree MJCF, 1 kHz PD; gantry until the policy takes over, then NO support: STANDBY_PRE {res['pre_s']:.0f} s -> motion -> "
         f"STANDBY_POST {res['post_s']:.0f} s. Grid per clip and profile: latency {list(res['grid']['lat'])} ms x mu {list(res['grid']['fric'])} x "
         f"kp {list(res['grid']['kps'])} x torso +{list(res['grid']['mass'])} kg = {len(res['grid']['lat']) * len(res['grid']['fric']) * len(res['grid']['kps']) * len(res['grid']['mass'])} runs.",
         f"Pass rule: {res['pass_rule']}.",
         f"R1+X check applied (runner defaults): pressed at {res['pre_s']:.0f} s, motion starts at the first step at which STANDBY_PRE >= "
         f"{res.get('r1x', {}).get('min_s', '?')} s and over the last {res.get('r1x', {}).get('win_s', '?')} s tilt err < {STANDBY_R1X_TILT_DEG} deg and max|dq| < "
         f"{STANDBY_R1X_DQ} rad/s; never within {res.get('r1x', {}).get('cap_s', '?')} s -> no motion (counted as 'never'). 'moving at start': pelvis speed over "
         "the last 2 s > 2 cm/s or yaw rate > 2 deg/s when the motion started.", "",
         "| clip | profile | pass | R1+X never / delayed (max wait) / moving at start | falls (phase) | abort runs | PRE tilt err max | PRE xy drift max | PRE yaw drift max | PRE max abs a | PRE dq/vlim | POST tilt err max | POST xy drift max | POST yaw drift max | POST max abs a | POST dq/vlim | R1+X ok after (max) | jump HOLD->SB | SB->POLICY | POLICY->POST | max ratio |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cl in res["clips"]:
        for prof in res["profiles"]:
            sel = [r for r in rows if r["clip"] == cl and r["tag"] == prof]
            if not sel:
                continue
            s = standby_summary(sel)
            L.append(f"| {cl} | {prof} | {s['pass']}/{s['runs']} | {s['r1x_never']} / {s['r1x_wait_n_delayed']} ({fmt(s['r1x_wait_max_s'], 2)} s) / {s['start_moving']} | {s['falls']} {s['fall_phases'] or ''} | {s['abort_runs']} | "
                     f"{fmt(s['standby_pre_max_tilt_err_deg'], 2)} deg | {fmt((s['standby_pre_drift_xy_max_m'] or 0) * 100, 1)} cm | {fmt(s['standby_pre_drift_yaw_max_deg'], 1)} deg | "
                     f"{fmt(s['standby_pre_max_abs_a'], 2)} | {fmt(s['standby_pre_max_vfrac'], 3)} | "
                     f"{fmt(s['standby_post_max_tilt_err_deg'], 2)} deg | {fmt((s['standby_post_drift_xy_max_m'] or 0) * 100, 1)} cm | {fmt(s['standby_post_drift_yaw_max_deg'], 1)} deg | "
                     f"{fmt(s['standby_post_max_abs_a'], 2)} | {fmt(s['standby_post_max_vfrac'], 3)} | {fmt(s['r1x_ok_first_s_max'], 2)} s | "
                     f"{fmt(s['jump_hold_to_standby_min'], 1)}-{fmt(s['jump_hold_to_standby'], 1)} | {fmt(s['jump_standby_to_policy'], 2)} | {fmt(s['jump_policy_to_post'], 2)} | "
                     f"{fmt(s['max_ratio_handoff_vs_hold'], 3)} |")
    L += ["", "Jumps: max over joints of |tau(t) - tau(t - 1 ms)| [Nm/ms] in the window [switch - 1 ms, switch + latency + 40 ms]; max over the runs "
          "(HOLD->SB: min-max). Steady baselines (max over runs of the per-run p99 / max):", ""]
    s = standby_summary(rows)
    L.append(f"- STANDBY_PRE (after 1 s): p99 {fmt(s['jump_standby_pre_steady_p99'], 2)}, max {fmt(s['jump_standby_pre_steady_max'], 2)}; POLICY: p99 "
             f"{fmt(s['jump_policy_steady_p99'], 2)}, max {fmt(s['jump_policy_steady_max'], 2)}; STANDBY_POST (after 1 s): p99 {fmt(s['jump_standby_post_steady_p99'], 2)}, "
             f"max {fmt(s['jump_standby_post_steady_max'], 2)} Nm/ms.")
    L.append(f"- Settled standby (after the first 5 s): PRE tilt err max {fmt(s['standby_pre_settled_max_tilt_err_deg'], 2)} deg, max|dq| {fmt(s['standby_pre_settled_max_dq'], 2)} rad/s, "
             f"xy drift over the rest {fmt((s['standby_pre_settled_drift_xy_m'] or 0) * 100, 1)} cm; POST tilt err max {fmt(s['standby_post_settled_max_tilt_err_deg'], 2)} deg, "
             f"max|dq| {fmt(s['standby_post_settled_max_dq'], 2)} rad/s, xy drift {fmt((s['standby_post_settled_drift_xy_m'] or 0) * 100, 1)} cm.")
    fails = [r for r in rows if not r["pass"]]
    if fails:
        L += ["", "## Failing runs", "", "| clip | profile | lat | mu | kp | mass | R1+X never / wait / moving | upright (phase) | aborts | first abort | standby tilt err | jumps HOLD->SB / SB->POL / POL->POST |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in fails:
            fa = next((f"{p}: {v['first_abort']}" for p, v in r["phases"].items() if v.get("first_abort")), "")
            J = r["jumps"]
            L.append(f"| {r['clip']} | {r['tag']} | {r['lat']} | {r['fric']} | {r['kps']} | {r['mass']} | {r.get('r1x_never')} / {fmt(r.get('r1x_wait_s'), 2)} / {r.get('start_moving')} | {r['upright']} ({r['fall_phase']}) | {r['aborts']} | {fa} | "
                     f"{fmt(r['standby_max_tilt_err_deg'], 2)} | {fmt(J.get('hold_to_standby'), 2)} / {fmt(J.get('standby_to_policy'), 2)} / {fmt(J.get('policy_to_post'), 2)} |")
    return "\n".join(L) + "\n"


# ============================================================================================ G5


def g5_gate(ckpt: str, n: int = 5000) -> dict:
    cfg = load_cfg(ckpt)
    clip = load_clip(IDLE)
    raw = load_clip_raw(IDLE)
    pol = load_policy(ckpt)
    core = DeployCore(cfg, pol, clip)
    core.reset(0)
    rng = np.random.default_rng(0)
    T = clip.num_frames
    lat, lat_onnx = [], []
    for k in range(n + 100):
        f = k % (T - 1)
        if f == 0:
            core.reset(0)
        q = raw["joint_pos"][f] + 0.01 * rng.standard_normal(29)
        dq = raw["joint_vel"][f] + 0.1 * rng.standard_normal(29)
        gyro = 0.05 * rng.standard_normal(3)
        quat = raw["root_quat_wxyz"][f]
        t0 = time.perf_counter()
        core.step(q, dq, gyro, quat)
        t1 = time.perf_counter()
        if k >= 100:
            lat.append(t1 - t0)
    obs, goal = np.zeros(930, np.float32), np.zeros(680, np.float32)
    for k in range(n + 100):
        t0 = time.perf_counter()
        pol(obs, goal)
        if k >= 100:
            lat_onnx.append(time.perf_counter() - t0)
    ms = lambda a, p: float(np.percentile(np.array(a) * 1e3, p))  # noqa: E731
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    import onnxruntime

    return {"n": n, "threads": 1, "step_p50_ms": ms(lat, 50), "step_p99_ms": ms(lat, 99), "step_max_ms": float(np.max(lat) * 1e3),
            "onnx_p50_ms": ms(lat_onnx, 50), "onnx_p99_ms": ms(lat_onnx, 99), "onnx_max_ms": float(np.max(lat_onnx) * 1e3),
            "loadavg": load, "cpu": _cpu_name(), "onnxruntime": onnxruntime.__version__, "budget_ms": 20.0,
            "note": "measured while training run e11 (1 busy CPU core + GPU) was running on the same PC"}


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


# ============================================================================================ model comparison


def compare_models(ckpt: str = CKPTS[1]) -> dict:
    cfg = load_cfg(ckpt)
    a = mujoco.MjModel.from_xml_path(str(TRAIN_XML))
    b = mujoco.MjModel.from_xml_path(str(UNITREE_XML))
    ra, rb = Robot(a, cfg.joint_names), Robot(b, cfg.joint_names)
    joints = []
    for i, n in enumerate(cfg.joint_names):
        ja, jb = ra.jid[i], rb.jid[i]
        da, db = a.jnt_dofadr[ja], b.jnt_dofadr[jb]
        joints.append({
            "joint": n,
            "armature": [float(a.dof_armature[da]), float(b.dof_armature[db])],
            "damping": [float(a.dof_damping[da]), float(b.dof_damping[db])],
            "frictionloss": [float(a.dof_frictionloss[da]), float(b.dof_frictionloss[db])],
            "range": [a.jnt_range[ja].tolist(), b.jnt_range[jb].tolist()],
            "effort": [float(a.actuator_forcerange[ra.act[i], 1]), float(b.actuator_ctrlrange[rb.act[i], 1]), float(cfg.effort_limit[i])],
        })
    bodies = []
    names_b = {short(b.body(i).name): i for i in range(b.nbody)}
    for i in range(a.nbody):
        n = short(a.body(i).name)
        if n in names_b and a.body_mass[i] > 0:
            j = names_b[n]
            bodies.append({"body": n, "mass": [float(a.body_mass[i]), float(b.body_mass[j])], "ipos": [a.body_ipos[i].tolist(), b.body_ipos[j].tolist()],
                           "inertia": [a.body_inertia[i].tolist(), b.body_inertia[j].tolist()]})
    only_b = sorted(set(names_b) - {short(a.body(i).name) for i in range(a.nbody)})
    pa, pb = Robot(a, cfg.joint_names), Robot(b, cfg.joint_names)
    pa.set_state([0, 0, 1.0], [1, 0, 0, 0], cfg.default_joint_pos)
    pb.set_state([0, 0, 1.0], [1, 0, 0, 0], cfg.default_joint_pos)
    stand_h = (1.0 - pa.foot_clearance(), 1.0 - pb.foot_clearance())
    big = [r for r in bodies if abs(r["mass"][0] - r["mass"][1]) > 0.01]
    # IMU check on the Unitree model: imu_gyro / imu_quat sensors (site on the pelvis origin) vs qvel[3:6] / qpos[3:7]
    rng = np.random.default_rng(0)
    imu_err = [0.0, 0.0]
    for _ in range(20):
        q = rng.normal(size=4)
        rb.set_state([0, 0, 1.0], q / np.linalg.norm(q), cfg.default_joint_pos, rng.normal(size=3), rng.normal(size=3), rng.normal(size=29))
        _, _, gyro, quat = rb.state()
        sg = b.sensor("imu_gyro"); sq = b.sensor("imu_quat")
        g_s = rb.d.sensordata[sg.adr[0] : sg.adr[0] + 3]
        q_s = rb.d.sensordata[sq.adr[0] : sq.adr[0] + 4]
        imu_err[0] = max(imu_err[0], float(np.abs(g_s - gyro).max()))
        imu_err[1] = max(imu_err[1], float(min(np.abs(q_s - quat).max(), np.abs(q_s + quat).max())))
    return {
        "total_mass": [float(a.body_subtreemass[0]), float(b.body_subtreemass[0])],
        "q0_pelvis_height_feet_on_ground": stand_h,
        "unitree_imu_vs_qvel_qpos_maxabs": {"gyro": imu_err[0], "quat": imu_err[1]},
        "opt": {"train": {"timestep": a.opt.timestep, "integrator": int(a.opt.integrator), "iterations": int(a.opt.iterations)},
                "unitree": {"timestep_xml": b.opt.timestep, "timestep_used": 0.001, "integrator": int(b.opt.integrator), "iterations": int(b.opt.iterations)}},
        "bodies_mass_differs": big, "bodies_only_in_unitree": only_b, "joints": joints,
        "max_range_diff": float(max(np.abs(np.array(j["range"][0]) - np.array(j["range"][1])).max() for j in joints)),
        "effort_mismatch": [j for j in joints if abs(j["effort"][1] - j["effort"][2]) > 1e-6],
        "bodies": bodies,
    }



# ============================================================================================ report


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


G1_VARIANTS = {  # key -> (description, safety options)
    "g1": ("core defaults: soft clamp off, torque clamp 1.5x", {}),
    "g1_f1.0": ("torque clamp 1.0x", {"factor": 1.0}),
    "g1_f0.5": ("torque clamp 0.5x", {"factor": 0.5}),
    "g1_noclamp": ("no clamps", {"torque": False}),
    "g1_soft": ("soft-limit clamp on + torque clamp 1.5x (round-1 default, for reference)", {"soft": True}),
}


def g3_ok(rows) -> bool:
    return all(x["upright"] and x["n_abort_steps"] == 0 for x in rows)


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def g2_summary_row(rows: list[dict], clips) -> dict:
    rs = [x for x in rows if x["clip"] in clips and x["kps"] == 1.0 and x["mass"] == 0.0]
    if not rs:
        return {}
    st = [x["stats"] for x in rs]
    return {
        "runs": len(rs), "falls": sum(x["fall"] for x in rs), "abort_fp": sum(x["abort_fp"] for x in rs),
        "rmse": _mean([x["track_rmse"] for x in rs if not x["fall"]]),
        "first_tau_max": max(x["first_step_tau_max_Nm"] for x in rs), "first_tau_joint": max(rs, key=lambda x: x["first_step_tau_max_Nm"])["first_step_tau_joint"],
        "first_tau_ratio": max(x["first_step_tau_ratio_max"] for x in rs),
        "wrist_any": _mean([s["wrist_clamp_frac_any"] for s in st]), "wrist_max_joint": max(s["wrist_clamp_frac_max_joint"] for s in st),
        "n_clamped": _mean([s["n_clamped_mean"] for s in st]), "frac_tau_ok_min": min(1 - x["frac_tau_gt_0p9"] for x in rs),
        "max_tilt": max(x["max_tilt_deg"] for x in rs),
    }


def write_md(results: dict[str, dict], cmp: dict | None) -> str:
    ck = list(results)
    L = ["# Sim2sim gates: G1 29-DoF, BONES-SEED tracking policy e11 (round 2: new core defaults)", "",
         f"Generated by `scripts/deploy/sim2sim.py` on {time.strftime('%Y-%m-%d %H:%M')}; raw numbers in `sim2sim_<ckpt>.json` and `sim2sim_model_compare.json`. "
         "Core defaults now: soft-limit clamp off, torque clamp on at 1.5 x effort, IMU validity abort. Simulation only.", ""]
    L += ["## Summary", "", "| gate | variant | " + " | ".join(ck) + " |", "|---|---|" + "---|" * len(ck)]
    for key, (desc, _) in G1_VARIANTS.items():
        L.append(f"| G1 same-model | {desc} | " + " | ".join(("PASS" if r[key]["pass"] else "FAIL") if r.get(key) else "not run" for r in results.values()) + " |")
    for key, (desc, _, _) in G2_VARIANTS.items():
        L.append(f"| G2 realistic | {desc} | " + " | ".join(("PASS" if r[key]["criteria"]["pass"] else "FAIL") if r.get(key) else "not run" for r in results.values()) + " |")
    for tag, desc in (("main", "HOLD (stand gains, gantry) -> POLICY -> 1 s END blend under gantry, factors 0.5/1.0/1.5, latency 0/20 ms"),
                      ("loop60", "60 s idle loop (keep history) + END blend"),
                      ("start_ramp", "proposal: 1 s stand->training gain ramp at the end of HOLD, then POLICY + END blend"),
                      ("switch_pose", "round-1 end: stand gains + last-frame pose at once (reference)"),
                      ("switch_last", "round-1 runner end: stand gains at the last policy target (reference)")):
        L.append(f"| G3 transitions | {desc} | " + " | ".join(
            (("PASS" if g3_ok(sel) else "FAIL") if (sel := [x for x in r["g3"] if x["tag"] == tag]) else "not run") if r.get("g3") else "not run" for r in results.values()) + " |")
    L.append("| G5 DeployCore.step p50 / p99 | 1 thread | " + " | ".join(f"{r['g5']['step_p50_ms']:.2f} / {r['g5']['step_p99_ms']:.2f} ms" if r.get("g5") else "not run" for r in results.values()) + " |")
    L.append("")
    notes = REPORTS / "sim2sim" / "notes.md"  # hand-written findings / recommendation, kept across regenerations
    if notes.exists():
        L += [notes.read_text().rstrip(), ""]
    for c, r in results.items():
        L += [f"## {c}", ""]
        # ---------------- G1
        for key, (desc, _) in G1_VARIANTS.items():
            if not r.get(key):
                continue
            g = r[key]
            L += [f"### G1 same-model, {desc}", "",
                  "Training MJCF on CPU MuJoCo vs the training env (GPU, fp32 matmuls). Fail = training terminations.", "",
                  "| clip | safe | env fail | s2s fail | agree | env RMSE | s2s RMSE | ratio | diverge [s] | env TF32-vs-fp32 diverge [s] | act1 diff | n_clamped mean | wrist clamp frac | max abs a |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for x in g["rows"]:
                tdiv = x.get("env_tf32_vs_fp32_diverge_step")
                s = x["stats"]
                L.append(f"| {x['clip']} | {fmt(x['safe'])} | {x['env_fail_step']} | {x['s2s_fail_step']} | {fmt(x['fail_agree'])} | {x['env_rmse']:.4f} | {x['s2s_rmse']:.4f} | {x['rmse_ratio']:.2f} | "
                         f"{fmt(x['diverge_s'], 2)} | {fmt(tdiv * 0.02 if tdiv is not None and tdiv >= 0 else None, 2)} | {x['action_step1_absdiff']:.1e} | {s['n_clamped_mean']:.2f} | {s['wrist_clamp_frac_any']:.3f} | {s['max_abs_a']:.2f} |")
            L += ["", f"Gate: **{'PASS' if g['pass'] else 'FAIL'}** ({g['criteria']}). obs1/goal1 max diff {max(x['obs_step1_absdiff'] for x in g['rows']):.1e} / {max(x['goal_step1_absdiff'] for x in g['rows']):.1e}.", ""]
        # ---------------- G2
        for key, (desc, _, _) in G2_VARIANTS.items():
            if not r.get(key):
                continue
            g = r[key]
            rows = g["rows"]
            lats = sorted({x["lat"] for x in rows})
            kpss = sorted({x["kps"] for x in rows}, reverse=True)
            L += [f"### G2 realistic, {desc}", "", "Unitree MJCF, 1 kHz PD (training kp/kd), robot on the ground at rest in the frame-0 pose.", ""]
            for kps in kpss:
                L += [f"kp x{kps}, +0 kg:", "",
                      "| clip | safe | mu | " + " | ".join(f"lat {la}" for la in lats) + " | max tilt [deg] | max tau/limit | frac tau>0.9 | runs w/ abort (FP) | first-step max tau [Nm] (joint) | wrist clamp frac | n_clamped | RMSE |",
                      "|---|---|---|" + "---|" * len(lats) + "---|---|---|---|---|---|---|---|"]
                for clip in CLIPS:
                    for fr in G2_FRICTION:
                        rs = [x for x in rows if x["clip"] == clip and x["kps"] == kps and x["mass"] == 0.0 and x["fric"] == fr]
                        if not rs:
                            continue
                        by = {x["lat"]: x for x in rs}
                        cell = lambda x: ("FALL@" + str(x["fall_step"])) if x["fall"] else "ok"  # noqa: E731
                        ok = [x["track_rmse"] for x in rs if not x["fall"]]
                        top = max(rs, key=lambda x: x["first_step_tau_max_Nm"])
                        L.append(f"| {clip} | {fmt(clip in SAFE)} | {fr} | " + " | ".join(cell(by[la]) if la in by else "-" for la in lats) +
                                 f" | {max(x['max_tilt_deg'] for x in rs):.1f} | {max(x['max_tau_ratio'] for x in rs):.2f} | {max(x['frac_tau_gt_0p9'] for x in rs):.4f} | "
                                 f"{sum(x['n_abort_steps'] > 0 for x in rs)} ({sum(x['abort_fp'] for x in rs)}) | {top['first_step_tau_max_Nm']:.1f} ({top['first_step_tau_joint']}) | "
                                 f"{np.mean([x['stats']['wrist_clamp_frac_any'] for x in rs]):.3f} | {np.mean([x['stats']['n_clamped_mean'] for x in rs]):.2f} | {(np.mean(ok) if ok else float('nan')):.4f} |")
                L.append("")
            if len({x["mass"] for x in rows}) > 1:
                L += ["Falls / runs over all latencies x friction {0.6, 1.0}:", "",
                      "| clip | " + " | ".join(f"kp x{k}, +{m} kg" for k in G2_KP_SCALE for m in G2_EXTRA_MASS) + " |", "|---|" + "---|" * 6]
                for clip in CLIPS:
                    cells = []
                    for k in G2_KP_SCALE:
                        for mm in G2_EXTRA_MASS:
                            rs = [x for x in rows if x["clip"] == clip and x["kps"] == k and x["mass"] == mm]
                            cells.append(f"{sum(x['fall'] for x in rs)}/{len(rs)}")
                    L.append(f"| {clip} | " + " | ".join(cells) + " |")
                L.append("")
            reasons, sat = {}, {}
            for x in rows:
                for a in x["abort_reasons"]:
                    reasons[a] = reasons.get(a, 0) + 1
                for j, v in x["model_sat_ms"].items():
                    sat[j] = sat.get(j, 0) + v
            extra = ""
            if key == "g2_hardclip":
                extra = f" Steps with at least one target changed by the hard-range clip: {sum(x['hard_clip_steps'] for x in rows)} of {sum(x['steps'] for x in rows)}."
            L += [f"Criteria: {json.dumps(g['criteria'])} -> **{'PASS' if g['criteria']['pass'] else 'FAIL'}**.{extra}", "",
                  f"Abort reasons (runs): {reasons or 'none'}. ms where the PD torque exceeded the Unitree MJCF ctrlrange (MuJoCo clipped it): {sat or 'none'}.", ""]
        # ---------------- torque clamp factor comparison
        fk = [k for k in ("g2_f0.5", "g2_f1.0", "g2", "g2_noclamp") if r.get(k)]
        if len(fk) > 1:
            L += ["### Torque clamp factor (G2 nominal: kp 1.0, +0 kg, latency 0-20 ms, mu 0.6/1.0)", "",
                  "| variant | clips | falls / runs | abort FP | RMSE (no-fall runs) | first-step max tau [Nm] (joint) | first-step tau/limit | wrist clamp frac (any wrist / worst wrist joint) | n_clamped mean | min frac tau<=0.9 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for k in fk:
                for nm, cl in (("SAFE", SAFE), ("HARDER", HARDER)):
                    s = g2_summary_row(r[k]["rows"], cl)
                    if s:
                        L.append(f"| {r[k]['desc'][:60]} | {nm} | {s['falls']}/{s['runs']} | {s['abort_fp']} | {s['rmse']:.4f} | {s['first_tau_max']:.1f} ({s['first_tau_joint']}) | {s['first_tau_ratio']:.2f} | "
                                 f"{s['wrist_any']:.3f} / {s['wrist_max_joint']:.3f} | {s['n_clamped']:.2f} | {s['frac_tau_ok_min']:.4f} |")
            L.append("")
        # ---------------- G3
        if r.get("g3"):
            main = [x for x in r["g3"] if not x["tag"].startswith("stage_")]
            L += ["### G3 transitions (Unitree MJCF, nominal)", "",
                  "q0 stand -> 4 s cosine interpolation (stand gains) -> 1 s HOLD -> POLICY (first target + training gains together) -> END. "
                  "Support 'band' = pelvis spring-damper gantry until the policy starts, re-engaged at END. END 'blend' = runner fix (1 s linear ramp of kp/kd to stand gains and of the target "
                  "from the last policy target to the soft-clipped last-frame pose, |kp(t)(q_cmd-q)| <= effort every 2 ms, then 2 s stand hold). Torque jumps are max |tau(t) - tau(t-1 ms)| over all joints.", "",
                  "| run | factor | latency | upright | max tilt [deg] | aborts | HOLD max tau [Nm] | first-step max tau [Nm] (joint) | jump at POLICY start [Nm/ms] (joint) | jump at END switch (joint) | max jump during END blend (joint) | jump after blend | policy steady max / p99 | max tau/limit in blend |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for x in main:
                run = x["tag"] + (f" ({x['clip_wraps']} wraps)" if x.get("loop_s") else "") + ("" if x["end_support"] else ", END unsupported") + \
                    (f", {x['clip'][:16]} {x['support']}" if x["clip"] != IDLE or x["support"] != "band" else "") + (f", blend cmd latency {x['blend_lat']} ms" if x.get("blend_lat") is not None else "")
                L.append(f"| {run} | {x['factor']} | {x['lat']} ms | {fmt(x['upright'])}{'' if x['upright'] else ' (fell in ' + str(x['fall_phase']) + ' at ' + fmt(x.get('fall_time_s'), 2) + ' s)'} | {x['max_tilt_deg']:.1f} | {x['n_abort_steps']} | "
                         f"{fmt(x['hold_tau_max_Nm'], 1)} | {fmt(x['first_step_tau_max_Nm'], 1)} ({x['first_step_tau_joint']}) | {fmt(x['jump_policy_start'], 1)} ({x['jump_policy_start_joint']}) | "
                         f"{fmt(x['jump_end_switch'], 1)} ({x['jump_end_switch_joint']}) | {fmt(x['jump_end_blend_whole'], 1)} ({x['jump_end_blend_whole_joint']}) | {fmt(x['jump_after_blend'], 1)} | "
                         f"{fmt(x['max_tau_jump_policy_steady_Nm_per_ms'], 1)} / {fmt(x['p99_tau_jump_policy_steady_Nm_per_ms'], 2)} | {fmt(x['max_tau_ratio_end_blend'], 2)} |")
            L.append("")
            stage = [x for x in r["g3"] if x["tag"].startswith("stage_")]
            if stage:
                L += ["### Staged abort thresholds (abort config set in the core; policy keeps running after an abort so trips are counted)", "",
                      f"Stages: {json.dumps({k: {'max_vel_frac': v[0], 'max_abs_action': v[1]} for k, v in STAGES.items()})}. 'air' = hung from the torso, feet {AIR_CLEARANCE * 100:.0f} cm above the ground; "
                      "'feet' = pelvis gantry on throughout, feet loaded; 'band' (slack) = gantry released at policy start. |dq| is the 50 Hz state the core checks; '1 kHz' is the max over all simulated ms.", "",
                      "| stage | clip | factor | lat | upright | abort steps (first) | max abs a | steps a>10 / a>15 | max dq/vlim (joint) | 1 kHz max dq/vlim | margin a | margin dq |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for x in stage:
                    s = x["stats"]
                    vt, at = x["max_vel_frac"], x["max_abs_action"]
                    L.append(f"| {x['support']} | {x['clip']} | {x['factor']} | {x['lat']} | {fmt(x['upright'])} | {x['n_abort_steps']} ({x['first_abort'][1] if x['first_abort'] else '-'}) | {s['max_abs_a']:.2f} | "
                             f"{s['n_a_gt10']} / {s['n_a_gt15']} | {s['max_vfrac']:.3f} ({s['max_vfrac_joint']}) | {x['vfrac_1khz_max_all']:.3f} | {at / s['max_abs_a']:.2f}x | {vt / s['max_vfrac']:.2f}x |")
                L.append("")
        if r.get("g2") or r.get("ref_margins"):
            L += ["### Abort-threshold margins on the unsupported runs", "",
                  "| source | clips | runs | max abs a | steps a>10 / a>15 (of all steps) | max sampled dq/vlim (joint) | max 1 kHz dq/vlim | trips air/feet (0.3/10) | trips slack (0.5/15) |", "|---|---|---|---|---|---|---|---|---|"]
            for key in ("g2", "g2_f0.5"):
                if not r.get(key):
                    continue
                for nm, cl in (("SAFE", SAFE), ("HARDER", HARDER)):
                    rs = [x for x in r[key]["rows"] if x["clip"] in cl and x["kps"] == 1.0 and x["mass"] == 0.0]
                    if not rs:
                        continue
                    st = [x["stats"] for x in rs]
                    top = max(st, key=lambda s: s["max_vfrac"])
                    L.append(f"| G2 {key} nominal | {nm} | {len(rs)} | {max(s['max_abs_a'] for s in st):.2f} | {sum(s['n_a_gt10'] for s in st)} / {sum(s['n_a_gt15'] for s in st)} (of {sum(s['policy_steps'] for s in st)}) | "
                             f"{top['max_vfrac']:.3f} ({top['max_vfrac_joint']}) | {max(x['vfrac_1khz_max'] for x in rs):.3f} | {sum(s['stage_trips']['air/feet'] for s in st)} | {sum(s['stage_trips']['slack'] for s in st)} |")
            if r.get("ref_margins"):
                for nm, flag in (("SAFE", True), ("HARDER", False)):
                    rs = [x for x in r["ref_margins"]["rows"] if x["safe"] == flag]
                    L.append(f"| training env fp32 (finite-diff dq) | {nm} | {len(rs)} | {max(x['max_abs_a'] for x in rs):.2f} | {sum(x['n_a_gt10'] for x in rs)} / {sum(x['n_a_gt15'] for x in rs)} (of {sum(x['steps'] for x in rs)}) | "
                             f"{max(x['max_vfrac_fd'] for x in rs):.3f} | - | {sum(x['stage_trips']['air/feet'] for x in rs)} | {sum(x['stage_trips']['slack'] for x in rs)} |")
            L.append("")
        if r.get("g5"):
            t = r["g5"]
            L += ["### G5 timing", "", "| n | DeployCore.step p50 | p99 | max | ONNX only p50 | p99 | CPU | onnxruntime | load avg |", "|---|---|---|---|---|---|---|---|---|",
                  f"| {t['n']} | {t['step_p50_ms']:.3f} ms | {t['step_p99_ms']:.3f} ms | {t['step_max_ms']:.2f} ms | {t['onnx_p50_ms']:.3f} ms | {t['onnx_p99_ms']:.3f} ms | {t['cpu']} | {t['onnxruntime']} | "
                  f"{', '.join(f'{v:.1f}' for v in t['loadavg']) if t['loadavg'] else '-'} |", "", t["note"] + ".", ""]
    if cmp:
        L += ["## Unitree MJCF vs training MJCF", "",
              f"Total mass: training {cmp['total_mass'][0]:.3f} kg, Unitree {cmp['total_mass'][1]:.3f} kg. "
              f"q0 pelvis height with the feet on the ground: training {cmp['q0_pelvis_height_feet_on_ground'][0]:.4f} m, Unitree {cmp['q0_pelvis_height_feet_on_ground'][1]:.4f} m. "
              f"Unitree IMU sensors vs the state extraction (20 random states): |imu_gyro - qvel[3:6]| max {cmp['unitree_imu_vs_qvel_qpos_maxabs']['gyro']:.1e}, |imu_quat - qpos[3:7]| max {cmp['unitree_imu_vs_qvel_qpos_maxabs']['quat']:.1e}. "
              f"Max joint range difference: {cmp['max_range_diff']:.2e} rad. Bodies only in the Unitree model: {cmp['bodies_only_in_unitree'] or 'none'}.", "",
              "| body | mass train [kg] | mass Unitree [kg] | COM train | COM Unitree |", "|---|---|---|---|---|"]
        for x in cmp["bodies_mass_differs"]:
            L.append(f"| {x['body']} | {x['mass'][0]:.4f} | {x['mass'][1]:.4f} | {[round(v, 4) for v in x['ipos'][0]]} | {[round(v, 4) for v in x['ipos'][1]]} |")
        L += ["", "| joint | armature train / Unitree | damping | frictionloss | effort train / Unitree / deploy config |", "|---|---|---|---|---|"]
        for j in cmp["joints"]:
            L.append(f"| {j['joint']} | {j['armature'][0]:.4f} / {j['armature'][1]:.4f} | {j['damping'][0]:.3f} / {j['damping'][1]:.3f} | {j['frictionloss'][0]:.3f} / {j['frictionloss'][1]:.3f} | "
                     f"{j['effort'][0]:.0f} / {j['effort'][1]:.0f} / {j['effort'][2]:.0f} |")
        L.append("")
    return "\n".join(L)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="all", help="g1 | g2 | g3 | g5 | margins | compare | report | all (comma-separated) | standby (alone)")
    ap.add_argument("--ckpt", nargs="+", default=list(CKPTS))
    ap.add_argument("--g2", nargs="+", default=list(G2_VARIANTS), help="G2 variant keys")
    ap.add_argument("--g1", nargs="+", default=None, help="G1 variant keys (default: all for the recommended ckpt, g1/g1_noclamp otherwise)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lock-waist", action="store_true",
                    help="G1 29dof with waist roll/pitch locked at 0: joints removed from both MJCFs, q = dq = 0 reported, actions 13/14 not "
                         "applied; g1/g2/g3 results go to sim2sim_lw_<ckpt>.json. --cmd lw / lw_robust run the locked-vs-unlocked comparison "
                         "(sim2sim_lockwaist.json)")
    ap.add_argument("--clips", nargs="+", default=None,
                    help="run the clip-set gates (g1/g2/g3 with the runner's slack/ground profiles, report) on these clips instead of the "
                         "built-in set; results -> sim2sim_<tag>_<ckpt>.json + SIM2SIM_<tag>.md, G1 refs ref_<tag>_<ckpt>_<p>.npz")
    ap.add_argument("--tag", default="clips", help="name of the --clips report / reference files")
    ap.add_argument("--standby-pre-s", type=float, default=60.0, help="--cmd standby: unsupported STANDBY_PRE duration")
    ap.add_argument("--standby-post-s", type=float, default=30.0, help="--cmd standby: unsupported STANDBY_POST duration")
    ap.add_argument("--standby-json", type=Path, default=None, help="--cmd standby_report: rewrite the .md of this standby json")
    ap.add_argument("--standby-profiles", nargs="+", default=None, choices=list(PROFILES), help="--cmd standby: grid override (name the report with --tag)")
    ap.add_argument("--standby-lat", nargs="+", type=int, default=None)
    ap.add_argument("--standby-kps", nargs="+", type=float, default=None)
    ap.add_argument("--standby-mass", nargs="+", type=float, default=None)
    ap.add_argument("--standby-fric", nargs="+", type=float, default=None)
    ap.add_argument("--standby-r1x-window-s", type=float, default=STANDBY_R1X_WIN_S, help="--cmd standby: R1+X check window (runner --standby-window-s)")
    ap.add_argument("--standby-r1x-min-s", type=float, default=STANDBY_R1X_MIN_S, help="--cmd standby: R1+X minimum standby (runner --standby-min-s)")
    ap.add_argument("--standby-r1x-cap-s", type=float, default=STANDBY_R1X_CAP_S, help="--cmd standby: re-press R1+X for at most this long after --standby-pre-s")
    a = ap.parse_args()
    cmds = set(a.cmd.split(","))
    if "standby_report" in cmds:
        res = json.loads(a.standby_json.read_text())
        md = a.standby_json.with_name(a.standby_json.name.replace("sim2sim_", "SIM2SIM_")).with_suffix(".md")
        md.write_text(write_standby_md(res))
        print(md.read_text())
        return
    if "standby" in cmds:
        REPORTS.mkdir(parents=True, exist_ok=True)
        clips = tuple(a.clips or STANDBY_CLIPS)
        for c in a.ckpt:
            grid = {k: tuple(v) for k, v in (("profiles", a.standby_profiles), ("lat", a.standby_lat), ("kps", a.standby_kps),
                                             ("mass", a.standby_mass), ("fric", a.standby_fric)) if v}
            res = standby_gate(c, clips, a.workers, a.standby_pre_s, a.standby_post_s, a.lock_waist, grid,
                               {"win_s": a.standby_r1x_window_s, "min_s": a.standby_r1x_min_s, "cap_s": a.standby_r1x_cap_s})
            tag = "standby" + ("" if clips == STANDBY_CLIPS and not grid and a.tag == "clips" else f"_{a.tag}") + ("_lw" if a.lock_waist else "")
            f = REPORTS / f"sim2sim_{tag}_{c}.json"
            f.write_text(json.dumps(strip(res), indent=1))
            md = REPORTS / f"SIM2SIM_{tag}_{c}.md"
            md.write_text(write_standby_md(res))
            print(f"  wrote {f}\n  wrote {md}")
            print(md.read_text())
        return
    if "all" in cmds:
        cmds = {"g1", "g2", "g3", "g5", "margins", "compare", "report"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    if a.clips:
        for c in a.ckpt:
            f = REPORTS / f"sim2sim_{a.tag}_{c}.json"
            res = json.loads(f.read_text()) if f.exists() else {}
            if res.get("clips") not in (None, list(a.clips)):
                raise SystemExit(f"{f} holds clips {res['clips']}; use another --tag")
            res = clipset_gates(c, a.clips, a.tag, a.workers, cmds, res)
            f.write_text(json.dumps(strip(res), indent=1))
            print(f"  wrote {f}")
            if "report" in cmds and "summary" in res:
                md = REPORTS / f"SIM2SIM_{a.tag}_{c}.md"
                md.write_text(write_clipset_md(res))
                print(f"  wrote {md}")
        return

    def path(c):
        return REPORTS / (f"sim2sim_lw_{c}.json" if a.lock_waist else f"sim2sim_{c}.json")

    results = {c: (json.loads(path(c).read_text()) if path(c).exists() else {"ckpt": c}) for c in a.ckpt}

    def save():
        for c, r in results.items():
            path(c).write_text(json.dumps(strip(r), indent=1))

    if "compare" in cmds:
        cmp = compare_models()
        (REPORTS / "sim2sim_model_compare.json").write_text(json.dumps(strip(cmp), indent=1))
        print(f"  compare: total mass train {cmp['total_mass'][0]:.3f} Unitree {cmp['total_mass'][1]:.3f}; effort mismatch {[j['joint'] for j in cmp['effort_mismatch']]}")
    if "margins" in cmds:
        for c in a.ckpt:
            results[c]["ref_margins"] = ref_margins(c)
        save()
    if "g5" in cmds:
        for c in a.ckpt:
            results[c]["g5"] = g5_gate(c)
            print(f"  G5 {c}: {results[c]['g5']}", flush=True)
        save()
    if "g1" in cmds:
        for c in a.ckpt:
            for key in a.g1 or (list(G1_VARIANTS) if c == REC else ["g1", "g1_noclamp"]):
                results[c][key] = g1_gate(c, {**G1_VARIANTS[key][1], "lock": a.lock_waist})
            save()
    if "g2" in cmds:
        for c, v in g2_all(a.ckpt, a.workers, a.g2, a.lock_waist).items():
            results[c].update(v)
        save()
    if "g3" in cmds:
        for c, v in g3_gate(a.ckpt, a.workers, a.lock_waist).items():
            if v:
                results[c]["g3"] = v
        save()
    for phase in ("nominal", "robust"):
        if ("lw" if phase == "nominal" else "lw_robust") not in cmds:
            continue
        lwf = REPORTS / "sim2sim_lockwaist.json"
        data = json.loads(lwf.read_text()) if lwf.exists() else {}
        res = lw_eval(a.ckpt, a.workers, phase)
        for c in a.ckpt:
            for lock in (False, True):
                k = f"{phase}/{c}/{'locked' if lock else 'unlocked'}"
                data[k] = {"g1": [r for r in res["g1"] if r["ckpt"] == c and r["lock"] == lock],
                           "g2": sorted([r for r in res["g2"] if r["ckpt"] == c and r["lock"] == lock],
                                        key=lambda r: (CLIPS.index(r["clip"]), r["lat"], r["kps"], r["mass"], r["fric"]))}
                for g in ("g1", "g2"):
                    for cl, name in ((SAFE, "SAFE"), (HARDER, "HARDER")):
                        sm = lw_summary(data[k][g], cl)
                        if sm:
                            print(f"  {k} {g} {name}: " + " ".join(f"{kk} {fmt(v)}" for kk, v in sm.items()), flush=True)
        lwf.write_text(json.dumps(strip(data), indent=1))
    if "report" in cmds and a.lock_waist:
        print("  report: not written for --lock-waist (see docs/SIM2SIM.md#sim2sim-evaluation-of-the-waist-locked-g1)")
    elif "report" in cmds:
        cmpf = REPORTS / "sim2sim_model_compare.json"
        allres = {c: json.loads(path(c).read_text()) for c in CKPTS if path(c).exists()}
        md = write_md(allres, json.loads(cmpf.read_text()) if cmpf.exists() else None)
        (REPORTS / "SIM2SIM.md").write_text(md)
        print(f"  wrote {REPORTS / 'SIM2SIM.md'}")


if __name__ == "__main__":
    main()
