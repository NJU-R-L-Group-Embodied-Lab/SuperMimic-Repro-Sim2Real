"""Shared Unitree G1 29-DoF helpers for the video2motion tools (MuJoCo + NumPy only).

The tools run in the GMR environment, which does not have this repository's ``supermimic`` package installed,
so the joint order and model loading are repeated here (they must match
``supermimic.data.motion_convert.G1_JOINT_NAMES`` / ``load_g1_model``).

Robot model resolution (first match wins):
  1. an explicit ``--robot_xml`` argument;
  2. the ``G1_XML`` environment variable;
  3. mjlab's G1 MJCF (``mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml``) if ``mjlab`` is importable.
Print the path used by this repository with
  uv run python -c "from supermimic.data.motion_convert import g1_xml_path; print(g1_xml_path())"

CSV row layout (36 columns, 30 fps): root pos xyz [m], root quat **xyzw**, 29 joint angles [rad] in G1_JOINT_NAMES order.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

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
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
CSV_COLS = 7 + len(G1_JOINT_NAMES)  # 36
FPS = 30


def resolve_robot_xml(path: str | None = None) -> str:
    """Path of the G1 29-DoF MJCF used for joint limits, foot contact geometry and rendering."""
    path = path or os.environ.get("G1_XML")
    if path:
        if not Path(path).is_file():
            raise SystemExit(f"robot xml not found: {path}")
        return str(path)
    try:
        import mjlab  # noqa: PLC0415

        p = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
        if p.is_file():
            return str(p)
    except ImportError:
        pass
    raise SystemExit(
        "no G1 MJCF found: pass --robot_xml or set G1_XML (e.g. the mjlab g1.xml used by this repository, see "
        "docs/VIDEO_TO_MOTION.md#environment-setup)"
    )


def _add_floor(spec: mujoco.MjSpec) -> None:
    """Checker floor + brighter headlight for preview renders."""
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.add_texture(
        name="v2m_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.75, 0.75, 0.75],
        rgb2=[0.65, 0.65, 0.65],
        width=512,
        height=512,
    )
    mat = spec.add_material(name="v2m_grid", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "v2m_grid"
    spec.worldbody.add_geom(
        name="v2m_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        material="v2m_grid",
        contype=0,
        conaffinity=0,
    )


def load_g1(robot_xml: str | None = None, floor: bool = False) -> mujoco.MjModel:
    """Compile the G1 MJCF (optionally with a preview floor) and check the joint order against the CSV layout."""
    spec = mujoco.MjSpec.from_file(resolve_robot_xml(robot_xml))
    if floor:
        _add_floor(spec)
    m = spec.compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or tuple(names[1:]) != G1_JOINT_NAMES or m.nq != CSV_COLS:
        raise SystemExit(f"robot xml is not a G1 29-DoF model in Unitree joint order: {names}")
    return m


def joint_limits(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    return m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()


def foot_contact_points(m: mujoco.MjModel) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Per foot: (body id, sphere centres in the foot-body frame (k,3), radii (k,)).

    The feet touch the ground through their collision geoms (meshes are visual only). Spheres are used as is
    (Unitree URDF-derived models: 4 sole spheres per foot); a capsule is represented by the two spheres at its
    segment ends, which gives its exact lowest point (mjlab's G1: 7 sole capsules per foot).
    """
    out = []
    for name in FOOT_BODIES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        centres, radii = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            pos, r = m.geom_pos[g].copy(), float(m.geom_size[g][0])
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                centres.append(pos)
                radii.append(r)
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, m.geom_quat[g])
                axis = rot.reshape(3, 3)[:, 2] * m.geom_size[g][1]
                centres += [pos - axis, pos + axis]
                radii += [r, r]
        if not centres:
            raise SystemExit(f"robot xml has no sphere/capsule collision geoms on {name}")
        out.append((b, np.array(centres), np.array(radii)))
    return out


def set_row(m: mujoco.MjModel, d: mujoco.MjData, row: np.ndarray) -> None:
    """Load one CSV row (quat xyzw) into qpos (quat wxyz) and run forward kinematics."""
    d.qpos[:3] = row[:3]
    d.qpos[3:7] = row[[6, 3, 4, 5]]
    d.qpos[7:] = row[7:]
    mujoco.mj_kinematics(m, d)


def sole_z(d: mujoco.MjData, feet) -> list[float]:
    """Lowest collision point (world z) of each foot for the kinematics currently in ``d``."""
    zs = []
    for b, c, r in feet:
        z = d.xpos[b][2] + c @ d.xmat[b].reshape(3, 3)[2] - r
        zs.append(float(z.min()))
    return zs


def yaw_xyzw(q: np.ndarray) -> np.ndarray:
    """Heading (rad, about world z) of xyzw quaternions."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def hemisphere_continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive rows are in the same hemisphere (same rotations)."""
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def save_csv(path: str | Path, rows: np.ndarray) -> None:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != CSV_COLS or not np.isfinite(rows).all():
        raise ValueError(f"bad motion array {rows.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%.6f")
