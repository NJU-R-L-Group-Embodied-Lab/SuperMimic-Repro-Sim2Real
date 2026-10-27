"""Helpers of scripts/deploy/spin_feasibility.py: dense QP, difference operators, pelvis yaw rates under a heading offset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/deploy/spin_feasibility.py"


@pytest.fixture(scope="module")
def sf():
    spec = importlib.util.spec_from_file_location("spin_feasibility", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qp_box_and_equality(sf):
    # min |x - c|^2 s.t. x <= 1, x0 + x1 = 0.5
    c = np.array([2.0, -3.0, 0.5])
    H = np.eye(3)
    x, res = sf.solve_qp(H, np.zeros(3), np.eye(3), np.ones(3), np.array([[1.0, 1.0, 0.0]]), np.array([0.5]), f=-2 * c)
    assert res.success
    assert np.allclose(x, [1.0, -0.5, 0.5], atol=1e-6)


def test_difference_operators(sf):
    x = np.array([1.0, 2.0, 4.0])
    G = sf.fwd_diff_operator(3)
    assert np.allclose(G @ x, [1.0, 1.0, 2.0, -4.0])  # zero padded on both sides
    D = sf.d2_operator(3)
    assert np.allclose(D @ x, [1.0, 0.0, 1.0, -6.0, 4.0])


def test_heading_offset_adds_to_body_yaw_rate(sf):
    fps, T = 50.0, 40
    t = np.arange(T) / fps
    yaw = 2.0 * t  # 2 rad/s about z, pelvis upright
    q = sf.quat_z(yaw)
    wc, wf = sf.body_z_rates(q, 1 / fps)
    assert np.allclose(wc, 2.0) and np.allclose(wf, 2.0)
    psi = 0.5 * t  # +0.5 rad/s heading offset
    q2 = sf.quat_mul(sf.quat_z(psi), q)
    wc2, wf2 = sf.body_z_rates(q2, 1 / fps)
    assert np.allclose(wc2, 2.5) and np.allclose(wf2, 2.5)
