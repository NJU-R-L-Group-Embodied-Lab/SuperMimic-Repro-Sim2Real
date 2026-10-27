"""supermimic.deploy.imu_model: saturating gyro + gyro-only attitude (sim2sim --imu-model / SM_IMU_MODEL=1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from supermimic.deploy.imu_model import ENV_VAR, GYRO_LIMIT, ImuModel, angle_deg, gravity_b, imu_model_from_env, quat_exp, quat_mul

REPO = Path(__file__).resolve().parents[1]


def _rx(a):  # world-from-body quaternion of a roll a
    return np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])


def _true_after(q0, w, t, dt):
    q = q0.copy()
    for _ in range(int(round(t / dt))):
        q = quat_mul(q, quat_exp(w, dt))
    return q / np.linalg.norm(q)


def test_limit_is_500_deg_per_s():
    assert GYRO_LIMIT == pytest.approx(8.7266, abs=1e-4)


def test_gravity_matches_core_projected_gravity():
    core = pytest.importorskip("supermimic.deploy.core")
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        np.testing.assert_allclose(gravity_b(q), core.projected_gravity(q), atol=1e-12)


def test_below_limit_is_exact_and_lossless():
    q0 = _rx(0.4)
    imu = ImuModel(q0)
    w = np.array([1.0, -2.0, 6.0])  # every axis below the rail
    for _ in range(500):
        imu.integrate(w, 0.001)
    qt = _true_after(q0, w, 0.5, 0.001)
    g, q = imu.observe(w, qt, frame=25)
    np.testing.assert_allclose(g, w)
    assert angle_deg(gravity_b(q), gravity_b(qt)) < 1e-6
    s = imu.summary()
    assert s["imu_model"] and s["imu_sat_z_frames"] == 0 and s["imu_rot_lost_z_rad"] == 0.0 and s["imu_rot_lost_rad"] == 0.0
    assert s["imu_grav_err_max_deg"] < 1e-6


def test_saturated_yaw_spin_loses_rotation_and_tilts_gravity():
    roll, wz, T, dt = np.radians(30.0), 12.0, 0.5, 0.001
    q0 = _rx(roll)
    imu = ImuModel(q0)
    w = np.array([0.0, 0.0, wz])
    for k in range(int(T / dt)):
        if k % 20 == 0:
            g, _ = imu.observe(w, _true_after(q0, w, k * dt, dt), frame=100 + k // 20)
            assert g[2] == pytest.approx(GYRO_LIMIT)  # reported gyro on the rail
        imu.integrate(w, dt)
    qt = _true_after(q0, w, T, dt)
    imu.observe(w, qt, frame=125)
    s = imu.summary()
    assert s["imu_sat_z_frames"] == 26 and s["imu_sat_z_frame_list"][:2] == [100, 101]
    assert s["imu_rot_lost_z_rad"] == pytest.approx((wz - GYRO_LIMIT) * T, rel=1e-9)
    assert s["imu_rot_lost_rad"] == pytest.approx(s["imu_rot_lost_z_rad"], rel=1e-9)
    # closed form: the estimate is the same spin at the clipped rate; gravity error = angle between the two body-frame
    # gravity vectors (the horizontal part, sin(roll), rotated by the lost yaw)
    q_est = _true_after(q0, np.array([0.0, 0.0, GYRO_LIMIT]), T, dt)
    expect = angle_deg(gravity_b(q_est), gravity_b(qt))
    assert expect > 20.0
    assert s["imu_grav_err_max_deg"] == pytest.approx(expect, abs=1e-6)
    assert s["imu_grav_err_max_frame"] == 125
    np.testing.assert_allclose(imu.q, q_est, atol=1e-9)


def test_per_axis_clip():
    imu = ImuModel([1.0, 0, 0, 0])
    g, _ = imu.observe(np.array([-20.0, 3.0, 9.0]), [1.0, 0, 0, 0])
    np.testing.assert_allclose(g, [-GYRO_LIMIT, 3.0, GYRO_LIMIT])


def test_env_flag(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert not imu_model_from_env()
    for v, on in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv(ENV_VAR, v)
        assert imu_model_from_env() is on


def test_sim2sim_make_imu_default_off(monkeypatch):
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location("sim2sim_deploy_imu_under_test", REPO / "scripts/deploy/sim2sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Rob:
        def root(self):
            return np.zeros(3), _rx(0.1)

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert mod.make_imu(Rob(), {}) is None
    assert isinstance(mod.make_imu(Rob(), {"imu_model": True}), ImuModel)
    monkeypatch.setenv(ENV_VAR, "1")
    imu = mod.make_imu(Rob(), {})
    np.testing.assert_allclose(imu.q, _rx(0.1))
    assert mod.make_imu(Rob(), {"imu_model": False}) is None
