"""Resampler and CPU-FK conventions must match mjlab's csv_to_npz exactly."""

from pathlib import Path

import numpy as np
import pytest

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    forward_kinematics,
    load_csv,
    load_g1_model,
    quat_rotate_inverse,
    resample_motion,
)

ROOT = Path(__file__).resolve().parents[1]
LAFAN = ROOT / "data/raw/lafan1/g1"
CLIP = LAFAN / "walk1_subject1.csv"

pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="LAFAN1 G1 CSVs not downloaded")


def _signed_quat_close(q, q_ref, atol):
    sign = np.sign(np.sum(q * q_ref, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    np.testing.assert_allclose(q * sign, q_ref, atol=atol)


def test_resample_matches_mjlab_motion_loader():
    """Positions / joints / their velocities match mjlab's per-frame reference implementation.

    Rotations are checked against scipy's Slerp instead: mjlab's float32 ``quat_slerp`` has an
    early-return (``|dot - 1| < 4 eps -> return q1``) that fires for nearly identical consecutive
    mocap frames and returns the *first* keyframe regardless of the blend, which lags the
    resampled orientation by up to one input frame (observed max quaternion error 1e-3 and
    angular-velocity error 0.06 rad/s on walk1_subject1). Our vectorised nlerp/slerp agrees
    with scipy to 1e-9.
    """
    from mjlab.scripts.csv_to_npz import MotionLoader  # per-frame reference implementation
    from scipy.spatial.transform import Rotation, Slerp

    n_rows = 400
    ref = MotionLoader(str(CLIP), input_fps=30, output_fps=50, device="cpu", line_range=(1, n_rows))
    raw = load_csv(CLIP)[:n_rows]
    ours = resample_motion(raw, 30, 50)

    assert ours.num_frames == ref.output_frames
    np.testing.assert_allclose(ours.root_pos, ref.motion_base_poss.numpy(), atol=1e-5)
    np.testing.assert_allclose(ours.joint_pos, ref.motion_dof_poss.numpy(), atol=1e-5)
    np.testing.assert_allclose(ours.root_lin_vel_w, ref.motion_base_lin_vels.numpy(), atol=2e-4)
    np.testing.assert_allclose(ours.joint_vel, ref.motion_dof_vels.numpy(), atol=2e-4)

    key_times = np.arange(n_rows) / 30.0
    slerp = Slerp(key_times, Rotation.from_quat(raw[:, 3:7]))  # scipy wants xyzw
    times = np.minimum(np.arange(ours.num_frames) / 50.0, key_times[-1])
    gt = slerp(times).as_quat()[:, [3, 0, 1, 2]]
    _signed_quat_close(ours.root_quat, gt, atol=1e-6)  # nlerp branch for near-identical keys
    # Angular velocity: our central difference of the scipy-exact rotations.
    from supermimic.data.motion_convert import _so3_derivative

    np.testing.assert_allclose(ours.root_ang_vel_w, _so3_derivative(gt, 0.02), atol=1e-4)


def test_fk_root_velocities_round_trip():
    """Root link velocity from cvel must equal the velocity we wrote (as mjlab asserts)."""
    model = load_g1_model()
    motion = resample_motion(load_csv(CLIP)[:200], 30, 50)
    bodies = forward_kinematics(model, motion)
    T = motion.num_frames
    assert bodies.pos_w.shape == (T, 30, 3) and bodies.quat_w.shape == (T, 30, 4)
    assert bodies.body_names[0] == "pelvis" and len(bodies.body_names) == 30
    np.testing.assert_allclose(np.linalg.norm(bodies.quat_w, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(bodies.pos_w[:, 0], motion.root_pos, atol=1e-9)
    _signed_quat_close(bodies.quat_w[:, 0], motion.root_quat, atol=1e-9)
    np.testing.assert_allclose(bodies.lin_vel_w[:, 0], motion.root_lin_vel_w, atol=1e-7)
    np.testing.assert_allclose(bodies.ang_vel_w[:, 0], motion.root_ang_vel_w, atol=1e-7)


def test_fk_matches_mj_forward_and_finite_difference():
    """Kinematics-only FK equals full mj_forward, and body velocities match position differences."""
    import mujoco

    model = load_g1_model()
    motion = resample_motion(load_csv(CLIP)[:200], 30, 50)
    bodies = forward_kinematics(model, motion)
    data = mujoco.MjData(model)
    t = 50
    data.qpos[:3] = motion.root_pos[t]
    data.qpos[3:7] = motion.root_quat[t]
    data.qpos[7:] = motion.joint_pos[t]
    data.qvel[:3] = motion.root_lin_vel_w[t]
    data.qvel[3:6] = quat_rotate_inverse(motion.root_quat[t : t + 1], motion.root_ang_vel_w[t : t + 1])[0]
    data.qvel[6:] = motion.joint_vel[t]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[1:], bodies.pos_w[t], atol=1e-9)
    # mj_objectVelocity gives the 6D velocity at the body frame origin in world coordinates.
    for b in (1, 10, 20, 30):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)  # frame origin, not COM
        np.testing.assert_allclose(res[3:], bodies.lin_vel_w[t, b - 1], atol=1e-8)
        np.testing.assert_allclose(res[:3], bodies.ang_vel_w[t, b - 1], atol=1e-8)
    # Finite difference of link positions ~ link linear velocity (central difference, interior frames).
    fd = (bodies.pos_w[2:] - bodies.pos_w[:-2]) / (2 * 0.02)
    err = np.abs(fd - bodies.lin_vel_w[1:-1]).max()
    assert err < 0.15, err  # 30 fps source resampled to 50 Hz: small but nonzero discretisation error


def test_joint_names_order():
    assert len(G1_JOINT_NAMES) == 29 and G1_JOINT_NAMES[0] == "left_hip_pitch_joint"
