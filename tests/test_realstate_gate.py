"""scripts/deploy/realstate_gate.py: waist-roll start clipping, grid / subsample, RS summary and pass rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _rs():
    spec = importlib.util.spec_from_file_location("realstate_gate_under_test", REPO / "scripts/deploy/realstate_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q(roll, pitch):
    q = np.zeros(29)
    q[13], q[14] = roll, pitch
    return q


def test_roll_inside_range_untouched():
    rs = _rs()
    q, c = rs.clip_waist_roll(_q(-0.3, 0.1), (-0.52, 0.52))
    assert not c and q[13] == -0.3


def test_roll_beyond_mjcf_limit_is_clipped_and_counted():
    rs = _rs()
    q0 = _q(-0.69, 0.11)  # waist roll seen on the real robot (lean35kfs clip, frame 714)
    q, c = rs.clip_waist_roll(q0, (-0.52, 0.52))
    assert c and -0.52 < q[13] < -0.51 and q[14] == 0.11
    assert np.array_equal(np.delete(q, 13), np.delete(q0, 13))  # only the roll changes


def test_widened_range_keeps_real_roll():
    rs = _rs()
    q, c = rs.clip_waist_roll(_q(-0.69, 0.11), (-0.75, 0.75))
    assert not c and q[13] == -0.69


def test_coupling_envelope_tightens_the_limit():
    rs = _rs()
    lo, hi = rs.roll_limit(0.45, (-0.52, 0.52), (1.4, 0.95))
    assert np.isclose(hi, 0.95 - 1.4 * 0.45 - rs.ROLL_EPS) and np.isclose(lo, -hi)
    q, c = rs.clip_waist_roll(_q(-0.45, 0.45), (-0.52, 0.52), (1.4, 0.95))
    assert c and abs(q[13]) + 1.4 * 0.45 <= 0.95
    assert rs.roll_limit(0.9, (-0.52, 0.52), (1.4, 0.95)) == (0.0, 0.0)  # envelope closed -> roll 0


def test_full_grid_size_and_subsample_keeps_anchors():
    rs = _rs()
    full = rs.build_grid(3)
    assert len(full) == 3 * 6 * 3 * 3 * 2 * 2 * 2
    sub = rs.build_grid(3, max_runs=360)
    assert 0 < len(sub) <= 360 and len(sub) == len({tuple(sorted(j.items())) for j in sub})
    groups = {(j["log"], j["f0"], j["vel"]) for j in sub}
    assert len(groups) == 3 * 6 * 2  # every start / velocity estimate kept
    for g in groups:
        cells = {(j["lat"], j["kps"], j["mass"], j["fric"]) for j in sub if (j["log"], j["f0"], j["vel"]) == g}
        assert set(rs.ANCHORS) <= cells
    assert rs.build_grid(1, max_runs=10_000) == rs.build_grid(1)


def _row(peak, abort=None, fall=False, clipped=False):
    return {"knee_peak_1khz": peak, "first_abort": abort, "fall": fall, "roll_clipped": clipped}


def test_summary_pass_rule():
    rs = _rs()
    ok = [_row(p) for p in np.linspace(8, 12.5, 40)]
    s = rs.summarize(ok)
    assert s["pass"] and s["knee_aborts"] == 0 and s["knee_peak_p95"] <= 13
    bad = ok + [_row(17.3, [734, "joint velocity right_knee_joint 17.3 rad/s"])]
    s = rs.summarize(bad)
    assert not s["pass"] and s["knee_aborts"] == 1 and s["knee_peak_over16"] == 1
    hi = [_row(p) for p in np.linspace(10, 15, 40)]
    assert not rs.summarize(hi)["pass"]  # no abort, p95 > 13
    t = rs.summarize(ok + [_row(5.0, [716, "tilt error 25 deg"], fall=True, clipped=True)])
    assert t["tilt_aborts"] == 1 and t["knee_aborts"] == 0 and t["falls"] == 1 and t["roll_clipped"] == 1
    assert not rs.is_knee_abort("joint velocity left_hip_yaw_joint 33.0 rad/s")
