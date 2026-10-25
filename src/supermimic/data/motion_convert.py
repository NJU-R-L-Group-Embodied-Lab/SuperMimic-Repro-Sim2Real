"""CSV (G1 retarget) -> mjlab-compatible motion npz, batch friendly.

Conventions mirror ``mjlab/scripts/csv_to_npz.py`` (commit 8bb2bc9) exactly, but
everything is vectorised and the forward kinematics run on CPU MuJoCo instead of a
GPU Simulation so that hundreds of thousands of clips can be converted in parallel.

CSV layout (no header, one row per frame):
  cols 0:3   root position (m)
  cols 3:7   root quaternion in **xyzw** order (converted to wxyz here)
  cols 7:36  29 joint angles in :data:`G1_JOINT_NAMES` order

Output npz keys (same as mjlab ``MotionLoader`` expects):
  fps, joint_pos (T,29), joint_vel (T,29), body_pos_w (T,30,3), body_quat_w (T,30,4) wxyz,
  body_lin_vel_w (T,30,3), body_ang_vel_w (T,30,3)
plus extras: joint_names, body_names, source (str), input_fps.

Velocity definitions (as in mjlab):
  * joint_vel and root linear velocity: ``np.gradient`` of the resampled signal.
  * root angular velocity: central SO(3) difference of consecutive quaternions,
    expressed in the **world** frame, first/last sample repeated.
  * body link velocities: from MuJoCo ``cvel`` with
    ``lin_w = cvel_lin - ang x (subtree_com[root] - xpos)``, ``ang_w = cvel_ang``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
NUM_JOINTS = len(G1_JOINT_NAMES)


def g1_xml_path() -> Path:
    import mjlab

    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def load_g1_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Standalone CPU MjModel of the G1. Asserts the joint order matches the CSV order."""
    model = mujoco.MjModel.from_xml_path(str(xml_path or g1_xml_path()))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
    assert tuple(names[1:]) == G1_JOINT_NAMES, f"joint order mismatch: {names[1:]}"
    assert model.nq == 7 + NUM_JOINTS and model.nv == 6 + NUM_JOINTS
    return model


# ----------------------------------------------------------------------------- quaternions
# All quaternions below are wxyz (MuJoCo / mjlab convention).


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation vector (angle * axis) of wxyz quaternions, numerically stable near identity."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    scale = np.where(s > 1e-9, angle / np.where(s > 1e-9, s, 1.0), 2.0)
    return v * scale[..., None]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the body frame: R(q)^T v."""
    r = Rotation.from_quat(quat_wxyz_to_xyzw(q))
    return r.inv().apply(v)


# ----------------------------------------------------------------------------- resampling


@dataclass
class ResampledMotion:
    fps: float
    root_pos: np.ndarray  # (T,3)
    root_quat: np.ndarray  # (T,4) wxyz
    joint_pos: np.ndarray  # (T,29)
    root_lin_vel_w: np.ndarray  # (T,3) world
    root_ang_vel_w: np.ndarray  # (T,3) world
    joint_vel: np.ndarray  # (T,29)

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])


BONES_HEADER_PREFIX = "Frame,root_translateX"


def csv_format(path: str | Path) -> str:
    """'lafan1' (no header: pos m, quat xyzw, joints rad) or 'bones' (header: Frame, pos cm, Euler xyz deg, joints deg)."""
    with open(path) as f:
        first = f.readline()
    return "bones" if first.startswith(BONES_HEADER_PREFIX) else "lafan1"


def load_csv(path: str | Path) -> np.ndarray:
    """Load a G1 retarget CSV in either format. Returns float64 (N, 36) in the LAFAN1 layout:
    root position (m), root quaternion **xyzw**, 29 joint angles (rad) in G1_JOINT_NAMES order."""
    import pandas as pd

    if csv_format(path) == "bones":
        # BONES-SEED G1 CSV (same parsing as SONIC's convert_soma_csv_to_motion_lib.py):
        # centimeters -> metres, Euler 'xyz' degrees -> quaternion, joint degrees -> radians.
        df = pd.read_csv(path)
        cols = list(df.columns)
        joint_cols = [c for c in cols if c.endswith("_joint_dof")]
        expected = [f"{n}_dof" for n in G1_JOINT_NAMES]
        if joint_cols != expected:
            raise ValueError(f"{path}: joint columns differ from G1 order: {joint_cols[:5]}...")
        pos = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy(np.float64) / 100.0
        eul = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(np.float64)
        quat_xyzw = Rotation.from_euler("xyz", eul, degrees=True).as_quat()
        joints = np.deg2rad(df[joint_cols].to_numpy(np.float64))
        arr = np.concatenate([pos, quat_xyzw, joints], axis=1)
    else:
        arr = pd.read_csv(path, header=None, dtype=np.float64).to_numpy()
    if arr.ndim != 2 or arr.shape[1] != 7 + NUM_JOINTS:
        raise ValueError(f"{path}: expected (N, {7 + NUM_JOINTS}) got {arr.shape}")
    return arr


def resample_motion(raw: np.ndarray, input_fps: float, output_fps: float) -> ResampledMotion:
    """Resample a raw CSV array to ``output_fps`` and compute velocities (mjlab conventions)."""
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    n_in = raw.shape[0]
    duration = (n_in - 1) * input_dt
    # Include the end point when it lands on the output grid (same tolerance as mjlab).
    n_out = math.floor(duration / output_dt + 1e-6) + 1
    times = np.arange(n_out, dtype=np.float64) * output_dt
    phase = times / duration if duration > 0 else np.zeros_like(times)
    idx0 = np.floor(phase * (n_in - 1)).astype(np.int64)
    idx0 = np.clip(idx0, 0, n_in - 1)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = (phase * (n_in - 1) - idx0)[:, None]

    pos_in = raw[:, 0:3]
    quat_in = quat_xyzw_to_wxyz(raw[:, 3:7])
    quat_in = quat_in / np.linalg.norm(quat_in, axis=-1, keepdims=True)
    dof_in = raw[:, 7:]

    root_pos = pos_in[idx0] * (1 - blend) + pos_in[idx1] * blend
    joint_pos = dof_in[idx0] * (1 - blend) + dof_in[idx1] * blend

    # Vectorised slerp between the two bracketing input quaternions (shortest path).
    q0, q1 = quat_in[idx0], quat_in[idx1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    lin = dot > 0.9995
    w0 = np.where(lin, 1 - blend, np.sin((1 - blend) * theta) / np.where(lin, 1.0, sin_theta))
    w1 = np.where(lin, blend, np.sin(blend * theta) / np.where(lin, 1.0, sin_theta))
    root_quat = w0 * q0 + w1 * q1
    root_quat = root_quat / np.linalg.norm(root_quat, axis=-1, keepdims=True)

    # Velocities.
    if n_out >= 2:
        root_lin_vel = np.gradient(root_pos, output_dt, axis=0)
        joint_vel = np.gradient(joint_pos, output_dt, axis=0)
    else:
        root_lin_vel = np.zeros_like(root_pos)
        joint_vel = np.zeros_like(joint_pos)
    root_ang_vel = _so3_derivative(root_quat, output_dt)

    return ResampledMotion(
        fps=output_fps,
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_vel=joint_vel,
    )


def _so3_derivative(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity by central differences; first/last sample repeated."""
    n = q.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=q.dtype)
    q_prev, q_next = q[:-2], q[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = rotvec_from_quat(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


# ----------------------------------------------------------------------------- forward kinematics


@dataclass
class BodyStates:
    body_names: list[str]
    pos_w: np.ndarray  # (T, nb, 3) link frame origin
    quat_w: np.ndarray  # (T, nb, 4) wxyz
    lin_vel_w: np.ndarray  # (T, nb, 3) link frame origin velocity, world
    ang_vel_w: np.ndarray  # (T, nb, 3) world


def forward_kinematics(model: mujoco.MjModel, motion: ResampledMotion) -> BodyStates:
    """Body link poses and velocities for every frame via CPU MuJoCo (no contacts)."""
    data = mujoco.MjData(model)
    T = motion.num_frames
    nb = model.nbody - 1  # exclude world
    pos_w = np.empty((T, nb, 3))
    quat_w = np.empty((T, nb, 4))
    lin_w = np.empty((T, nb, 3))
    ang_w = np.empty((T, nb, 3))
    root_id = model.jnt_bodyid[0]

    # Free-joint qvel: world linear velocity, **body-frame** angular velocity.
    ang_vel_b = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)

    qpos = np.zeros(model.nq)
    qvel = np.zeros(model.nv)
    for t in range(T):
        qpos[0:3] = motion.root_pos[t]
        qpos[3:7] = motion.root_quat[t]
        qpos[7:] = motion.joint_pos[t]
        qvel[0:3] = motion.root_lin_vel_w[t]
        qvel[3:6] = ang_vel_b[t]
        qvel[6:] = motion.joint_vel[t]
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        pos_w[t] = data.xpos[1:]
        quat_w[t] = data.xquat[1:]
        cvel = data.cvel[1:]
        offset = data.subtree_com[root_id][None, :] - data.xpos[1:]
        lin_w[t] = cvel[:, 3:6] - np.cross(cvel[:, 0:3], offset)
        ang_w[t] = cvel[:, 0:3]

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    return BodyStates(names, pos_w, quat_w, lin_w, ang_w)


# ----------------------------------------------------------------------------- end to end


def convert_csv_to_npz(
    csv_path: str | Path,
    out_path: str | Path,
    input_fps: float,
    output_fps: float = 50.0,
    model: mujoco.MjModel | None = None,
) -> dict:
    """Convert one CSV to an mjlab-compatible npz. Returns a small summary dict."""
    model = model or load_g1_model()
    raw = load_csv(csv_path)
    motion = resample_motion(raw, input_fps, output_fps)
    bodies = forward_kinematics(model, motion)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        fps=np.array([output_fps], dtype=np.float32),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(bodies.body_names),
        source=np.array(str(csv_path)),
        input_fps=np.array([input_fps], dtype=np.float32),
    )
    return {
        "source": str(csv_path),
        "out": str(out_path),
        "input_frames": int(raw.shape[0]),
        "output_frames": motion.num_frames,
        "duration_s": (motion.num_frames - 1) / output_fps,
    }
