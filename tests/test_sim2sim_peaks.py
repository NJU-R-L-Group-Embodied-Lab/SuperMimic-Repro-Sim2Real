"""sim2sim.PeakTracker: 1 kHz joint-speed peaks (knee_peak_1khz / peak_joint_1khz) and the 1 kHz |dq| abort count."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[1]


def _sim2sim():
    spec = importlib.util.spec_from_file_location("sim2sim_under_test", REPO / "scripts/deploy/sim2sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAMES = ["left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint"]


def test_peaks_track_max_abs_and_frame():
    PeakTracker = _sim2sim().PeakTracker
    t = PeakTracker(NAMES)
    # 50 Hz policy steps of 20 ms each; the knee spikes between two policy samples (ms 1013 of step 50)
    for ms in range(2000):
        f = ms // 20
        dq = np.array([1.0, -3.0, 2.0])
        if ms == 1013:
            dq[1] = -15.3
        if ms == 1500:
            dq[2] = 20.0  # fastest joint overall is not a knee
        t.add(dq, f, ms)
    s = t.summary()
    assert s["knee_peak_1khz"] == pytest.approx(15.3)
    assert s["knee_peak_1khz_joint"] == "right_knee_joint"
    assert s["knee_peak_1khz_frame"] == 50
    assert s["peak_joint_1khz"] == ["left_shoulder_pitch_joint", 20.0, 75]
    assert s["dq_peak_1khz"]["left_knee_joint"] == [1.0, 0]
    assert s["vel_over_1khz_ms"] == 0 and s["first_vel_over_1khz"] is None  # no threshold given
    # the 50 Hz samples (first ms of every step) never see the spike: the 1 kHz peak is strictly larger
    assert max(abs(-3.0), 1.0) < s["knee_peak_1khz"]


def test_over_threshold_counts_like_core_check():
    PeakTracker = _sim2sim().PeakTracker
    vlim = np.array([20.0, 20.0, 37.0])
    t = PeakTracker(NAMES, thr=0.8 * vlim)  # DeployCore: |dq| > max_vel_frac * velocity_limit
    t.add([0.0, 16.0, 0.0], 10, 200)  # == threshold: not over (core uses a strict >)
    assert t.n_over == 0 and t.step_over is None
    t.add([0.0, -16.5, 0.0], 11, 221)
    t.add([16.2, 0.0, 30.0], 12, 240)  # both over; the larger fraction of its limit is reported (core argmax |dq|/vlim)
    assert t.n_over == 2
    assert t.first_over == (221, 11, "right_knee_joint", -16.5)
    assert t.step_over == t.first_over
    t.step_over = None
    t.add([16.2, 0.0, 30.0], 12, 241)
    assert t.step_over[2] == "left_shoulder_pitch_joint"  # 30/29.6 > 16.2/16
    t.reset()
    assert t.n_over == 0 and t.first_over is None and t.summary(per_joint=False)["knee_peak_1khz"] == 0.0
