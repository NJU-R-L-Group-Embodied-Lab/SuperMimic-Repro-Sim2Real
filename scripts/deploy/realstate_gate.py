"""RS gate: counterfactual sim2sim runs started from REAL measured robot states (landing-reflex knee over-speed check).

Why: an earlier checkpoint of the lean35kfs lineage (before the published e24_0355000) aborted on the real G1 at lean35kfs
frame 734 (right knee 17.3 rad/s). The frame-0 G2 gate reproduces the
landing reflex but peaks at only 12.5 rad/s, because the real robot reaches the landing through a different upstream state
(deeper left crouch 702-706, waist roll overshoot to -0.69, faster pelvis yaw). Initialising sim from the real logged states
at frames 712-722 gives 31/72 knee aborts vs 3-14/72 from sim's own states, so this gate checks a checkpoint from there.

Per start (real log x start frame) the Unitree MJCF robot (sim2sim.unitree_model, 1 kHz PD, sim2sim.PDSim) is set to:
  * joints q / dq = the logged lowstate row of that frame (DeployCore frame convention: core.reset(f0 - 1), the first step
    returns frame f0), pelvis quaternion = logged (IMU) quaternion, pelvis angular velocity = logged gyro;
  * height: the stance foot's lowest collision point on the floor (--stance left|right|auto);
  * pelvis linear velocity: 'foot' = the one that makes the stance foot contact point stationary, 'clip' = the clip's root
    velocity rotated by the real-vs-clip yaw difference (both, by default);
  * policy observation history (10 frames) and last action = the logged rows before f0 (what the real policy saw);
  * waist roll clipped into the model's range (and, with the coupling on, into |roll| + k |pitch| <= c): the real robot
    reaches |roll| 0.6-0.7 at frames 712-716, beyond the +-0.52 MJCF limit; such starts are counted (roll_clipped).
Then the deploy core runs the policy (IMU model + waist coupling on by default) up to --f-end over a grid of
latency x kp scale x torso mass x friction x velocity estimate.

Reported (RS line): knee aborts = first core abort is a knee joint-velocity abort (|dq| > 0.8 x 20 = 16 rad/s on the 50 Hz
sample, what the real runner checks); median / p95 / max of the per-run 1 kHz knee speed peak (both knees, frames >=
--peak-from, up to and including the first-abort step: after an abort the real robot is in damping); tilt aborts; falls.
PASS = 0 knee aborts and p95 <= 13 rad/s.

  source $G1_DEPLOY_HOME/env.sh   # default ~/workspace/g1_deploy
  python scripts/deploy/realstate_gate.py --artifacts deploy_artifacts/<ckpt> \\
      --clip deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz \\
      --real-logs deploy_logs/real/<run_a> deploy_logs/real/<run_b>   # runner logs of real lean35kf(s) runs
  -> deploy_artifacts/reports/realstate_<tag>.json and the RS summary line on stdout.

Waist roll range: by default the range of the gate model (whatever load_model builds, so a widened MJCF / load_model
option is picked up automatically). --waist-roll-range R (or SM_WAIST_ROLL_RANGE=R) sets waist_roll_joint's range to +-R in
the gate model for this gate only (opt-in; default off).
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

POLICY_STATE = 3  # runner state code of POLICY rows in the real logs
KNEE_ABORT = 16.0  # rad/s = DeployCore max_vel_frac 0.8 x knee velocity limit 20 (reported for the 1 kHz peaks)
P95_MAX = 13.0  # rad/s, pass criterion on the p95 of the 1 kHz knee peaks
WAIST_ROLL, WAIST_PITCH = 13, 14  # motor order
ROLL_EPS = 1e-3  # clip strictly inside the limit (no active limit constraint impulse at t = 0)
WAIST_ROLL_RANGE_ENV = "SM_WAIST_ROLL_RANGE"

FRAMES = (712, 714, 716, 718, 720, 722)
LATS = (0, 10, 20)
KPS = (0.85, 1.0, 1.15)
MASSES = (0.0, 1.8)
FRICS = (0.6, 1.0)
VELS = ("foot", "clip")
ANCHORS = ((10, 0.85, 1.8, 0.6), (0, 1.0, 0.0, 1.0))  # worst frame-0 gate cell, nominal: always kept when subsampling


# ============================================================================================ pure helpers (unit-tested)


def roll_limit(pitch: float, roll_range: tuple[float, float], coupling: tuple[float, float] | None) -> tuple[float, float]:
    """Admissible waist roll interval at `pitch`: joint range, intersected with |roll| + k |pitch| <= c when coupled."""
    lo, hi = float(roll_range[0]) + ROLL_EPS, float(roll_range[1]) - ROLL_EPS
    if coupling is not None:
        k, c = coupling
        e = max(c - k * abs(float(pitch)) - ROLL_EPS, 0.0)
        lo, hi = max(lo, -e), min(hi, e)
    return lo, hi


def clip_waist_roll(q, roll_range, coupling=None) -> tuple[np.ndarray, bool]:
    """Real start pose -> pose with the waist roll clipped into the model's admissible range; (q, clipped?)."""
    q = np.asarray(q, float).copy()
    lo, hi = roll_limit(q[WAIST_PITCH], roll_range, coupling)
    r = float(np.clip(q[WAIST_ROLL], lo, hi))
    clipped = r != q[WAIST_ROLL]
    q[WAIST_ROLL] = r
    return q, bool(clipped)


def build_grid(n_logs: int, frames=FRAMES, lats=LATS, kps=KPS, masses=MASSES, frics=FRICS, vels=VELS, max_runs: int = 0) -> list[dict]:
    """Full grid, or (max_runs > 0 and smaller than the grid) a deterministic stratified subsample: every (log, frame, vel)
    group keeps the ANCHORS cells that are in the grid plus an evenly rotated share of the other physics cells."""
    cells = list(itertools.product(lats, kps, masses, frics))
    groups = list(itertools.product(range(n_logs), frames, vels))
    full = [(g, c) for g in groups for c in cells]
    if max_runs <= 0 or max_runs >= len(full):
        sel = full
    else:
        anchors = [c for c in cells if c in ANCHORS]
        rest = [c for c in cells if c not in ANCHORS]
        per = max(max_runs // len(groups) - len(anchors), 0)
        sel = []
        for gi, g in enumerate(groups):
            pick = [rest[(gi * per + j) % len(rest)] for j in range(min(per, len(rest)))] if rest else []
            sel += [(g, c) for c in anchors + sorted(set(pick), key=pick.index)]
    return [{"log": g[0], "f0": int(g[1]), "vel": g[2], "lat": int(c[0]), "kps": float(c[1]), "mass": float(c[2]), "fric": float(c[3])}
            for g, c in sel]


def is_knee_abort(reason) -> bool:
    return bool(reason) and str(reason).startswith("joint velocity") and "knee" in str(reason)


def is_tilt_abort(reason) -> bool:
    return bool(reason) and str(reason).startswith("tilt error")


def summarize(rows: list[dict], p95_max: float = P95_MAX) -> dict:
    """RS statistics over run rows (keys first_abort [frame, reason] | None, knee_peak_1khz, fall, roll_clipped)."""
    n = len(rows)
    pk = np.array([r["knee_peak_1khz"] for r in rows], float)
    ka = sum(is_knee_abort(r["first_abort"][1]) for r in rows if r["first_abort"])
    ta = sum(is_tilt_abort(r["first_abort"][1]) for r in rows if r["first_abort"])
    s = {"runs": n, "knee_aborts": int(ka), "tilt_aborts": int(ta), "falls": int(sum(bool(r["fall"]) for r in rows)),
         "roll_clipped": int(sum(bool(r["roll_clipped"]) for r in rows)),
         "knee_peak_median": float(np.median(pk)) if n else float("nan"),
         "knee_peak_p95": float(np.percentile(pk, 95)) if n else float("nan"),
         "knee_peak_max": float(pk.max()) if n else float("nan"), "knee_peak_over16": int((pk > KNEE_ABORT).sum())}
    s["pass"] = bool(n and ka == 0 and s["knee_peak_p95"] <= p95_max)
    return s


# ============================================================================================ real logs


def load_real_log(d: str) -> dict:
    """Run npz of a deploy_logs/real/<run> directory (or an npz path): the one with the most POLICY rows."""
    p = Path(d)
    files = [p] if p.suffix == ".npz" else sorted(Path(x) for x in glob.glob(str(p / "run_*.npz")))
    if not files:
        raise SystemExit(f"no run_*.npz in {d}")
    best = None
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            n = int((z["state"] == POLICY_STATE).sum())
            if best is None or n > best[0]:
                best = (n, f, {k: z[k] for k in ("state", "frame", "q", "dq", "gyro", "quat", "action", "q_target")})
    _, f, R = best
    R["name"], R["path"] = (p.name if p.suffix != ".npz" else p.parent.name), str(f)
    return R


def log_row(R: dict, f: int) -> int | None:
    P = np.nonzero(R["state"] == POLICY_STATE)[0]
    hit = P[R["frame"][P] == f]
    return int(hit[0]) if len(hit) else None


# ============================================================================================ worker

_W: dict = {}


def _init_worker(o: dict) -> None:
    import sim2sim as S
    from supermimic.deploy.core import Clip, DeployConfig, OnnxPolicy

    art = Path(o["artifacts"])
    _W.update(S=S, o=o, cfg=DeployConfig.load(art / "deploy_config.json"), pol=OnnxPolicy(art / "policy.onnx", threads=1),
              clip=Clip.load(o["clip"]), logs=[load_real_log(d) for d in o["real_logs"]])
    with np.load(o["clip"]) as z:
        _W["raw"] = {k: z[k] for k in z.files}


def _nq(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _foot_geoms(S, m, side):
    import mujoco

    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    return b, [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _foot_low(S, rob, side):
    b, gs = _foot_geoms(S, rob.m, side)
    z = [rob.d.geom_xpos[g, 2] - rob.m.geom_size[g, 0] for g in gs]
    k = int(np.argmin(z))
    return z[k], gs[k], b


def make_model(S, mass: float, fric: float, roll_range_override: float | None):
    import mujoco

    m = S.unitree_model(mass, fric)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_roll_joint")
    if roll_range_override:
        m.jnt_range[j] = [-roll_range_override, roll_range_override]
    return m, tuple(float(x) for x in m.jnt_range[j])


def run_one(job: dict) -> dict:
    import mujoco

    from supermimic.deploy.core import HISTORY, DeployCore, SafetyCfg
    from supermimic.deploy.imu_model import ImuModel
    from supermimic.deploy.waist_coupling import coupling_from_env

    S, o, cfg, clip, raw = _W["S"], _W["o"], _W["cfg"], _W["clip"], _W["raw"]
    R = _W["logs"][job["log"]]
    f0, lat = job["f0"], job["lat"]
    m, roll_range = make_model(S, job["mass"], job["fric"], o.get("waist_roll_range"))
    rob = S.Robot(m, cfg.joint_names)
    sim = S.PDSim(rob, cfg, cfg.effort_limit)
    kp = cfg.kp * job["kps"]
    i = log_row(R, f0)
    if i is None or i - HISTORY - 1 < 0 or not np.all(R["state"][i - HISTORY - 1 : i + 1] == POLICY_STATE):
        return {**job, "log_name": R["name"], "skipped": "start frame / history not in the log's POLICY rows"}
    # ---- start state from the real row
    q, roll_clipped = clip_waist_roll(R["q"][i], roll_range, coupling_from_env())
    dq, quat = R["dq"][i].copy(), _nq(R["quat"][i])
    rob.set_state(np.r_[0.0, 0.0, raw["root_pos"][f0][2]], quat, q, dq=dq)
    d = rob.d
    d.qvel[rob.fv + 3 : rob.fv + 6] = R["gyro"][i]  # free joint angular velocity: body frame, as the gyro
    mujoco.mj_forward(m, d)
    stance = o["stance"]
    if stance == "auto":
        stance = min(("left", "right"), key=lambda s: _foot_low(S, rob, s)[0])
    z, g, b = _foot_low(S, rob, stance)
    d.qpos[rob.fq + 2] -= z + 0.0005
    mujoco.mj_forward(m, d)
    if job["vel"] == "foot":
        jp = np.zeros((3, m.nv))
        pt = d.geom_xpos[g].copy()
        pt[2] -= m.geom_size[g, 0]
        mujoco.mj_jac(m, d, jp, None, pt, b)
        d.qvel[rob.fv : rob.fv + 3] = 0.0
        d.qvel[rob.fv : rob.fv + 3] = -(jp @ d.qvel)
    else:
        dy = S._yaw(quat) - S._yaw(raw["root_quat_wxyz"][f0])
        c, s = np.cos(dy), np.sin(dy)
        v = raw["root_lin_vel_w"][f0]
        d.qvel[rob.fv : rob.fv + 3] = [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
    mujoco.mj_forward(m, d)
    # ---- deploy core with the real observation history
    saf = SafetyCfg()
    core = DeployCore(cfg, _W["pol"], clip, saf)
    core.reset(f0 - 1)
    H = []
    for r in range(i - HISTORY, i):
        core.last_action = R["action"][r - 1]
        H.append(core.obs_frame(R["q"][r], R["dq"][r], R["gyro"][r], _nq(R["quat"][r])))
    core.hist = np.array(H)
    core.last_action = R["action"][i - 1].copy()
    sim.command(R["q_target"][i - 1], 0, kp, cfg.kd.copy())
    sim.imu = ImuModel(rob.root()[1]) if o["imu_model"] else None
    kn = [rob.dadr[np.nonzero(rob.live)[0].tolist().index(j)] for j, n in enumerate(cfg.joint_names) if "knee" in n]
    first_abort, fall, peak, peak_frame, peak_all, steps = None, False, 0.0, -1, 0.0, 0
    while True:
        qq, dqq, gyro, qu = rob.state()
        if sim.imu is not None:
            gyro, qu = sim.imu.observe(gyro, qu, core.frame)
        r = core.step(qq, dqq, gyro, qu)
        sim.command(r.q_target, lat)
        sim.frame = r.frame - 1
        k1 = 0.0
        for _ in range(20):
            sim.step_ms()
            k1 = max(k1, float(np.abs(d.qvel[kn]).max()))
        steps += 1
        if first_abort is None:
            peak_all = max(peak_all, k1)
            if r.frame >= o["peak_from"] and k1 > peak:
                peak, peak_frame = k1, int(r.frame)
            if r.abort:
                first_abort = [int(r.frame), r.abort]
        if S.fell(rob):
            fall = True
            break
        if r.done or r.frame >= o["f_end"]:
            break
    if peak_frame < 0:  # aborted / ended before --peak-from: whole run up to the abort
        peak = peak_all
    return {**job, "log_name": R["name"], "stance": stance, "roll_clipped": roll_clipped, "roll_range": roll_range,
            "first_abort": first_abort, "fall": fall, "steps": steps, "knee_peak_1khz": peak, "knee_peak_frame": peak_frame}


# ============================================================================================ main


def fmt_line(tag: str, s: dict, per_log: dict, extra: str = "") -> str:
    logs = "; ".join(f"{k} knee {v['knee_aborts']}/{v['runs']} p95 {v['knee_peak_p95']:.1f}" for k, v in per_log.items())
    return (f"RS {tag} {'PASS' if s['pass'] else 'FAIL'} knee aborts {s['knee_aborts']}/{s['runs']} | knee 1k peak median "
            f"{s['knee_peak_median']:.1f} p95 {s['knee_peak_p95']:.1f} max {s['knee_peak_max']:.1f} (>16: {s['knee_peak_over16']}) | "
            f"tilt aborts {s['tilt_aborts']} | falls {s['falls']} | roll clipped {s['roll_clipped']} | {logs}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", required=True, help="deploy_artifacts/<tag> (deploy_config.json + policy.onnx)")
    ap.add_argument("--clip", required=True, help="clip npz (deploy_artifacts/clips/<name>.npz) or a clip name")
    ap.add_argument("--real-logs", nargs="+", required=True, help="deploy_logs/real/<run> directories (or run npz files)")
    ap.add_argument("--frames", nargs="+", type=int, default=list(FRAMES))
    ap.add_argument("--lat", nargs="+", type=int, default=list(LATS))
    ap.add_argument("--kps", nargs="+", type=float, default=list(KPS))
    ap.add_argument("--mass", nargs="+", type=float, default=list(MASSES))
    ap.add_argument("--fric", nargs="+", type=float, default=list(FRICS))
    ap.add_argument("--vel", nargs="+", default=list(VELS), choices=list(VELS))
    ap.add_argument("--max-runs", type=int, default=0, help="stratified subsample of the grid to about this many runs (0 = full)")
    ap.add_argument("--f-end", type=int, default=760)
    ap.add_argument("--peak-from", type=int, default=724, help="first frame of the knee-peak window (landing 728-734)")
    ap.add_argument("--stance", default="left", choices=("left", "right", "auto"), help="foot put on the floor at the start")
    ap.add_argument("--waist-coupling", default=os.environ.get("SM_WAIST_COUPLING") or "1.4,0.95",
                    help="k,c of the waist coupling (default 1.4,0.95 = on; '0' = off)")
    ap.add_argument("--no-imu-model", action="store_true", help="true gyro / quaternion to the policy (default: real IMU model)")
    ap.add_argument("--waist-roll-range", type=float, default=float(os.environ.get(WAIST_ROLL_RANGE_ENV) or 0) or None,
                    help=f"opt-in: waist_roll_joint range +-R in the gate model (default: the model's own; or {WAIST_ROLL_RANGE_ENV})")
    ap.add_argument("--p95-max", type=float, default=P95_MAX)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--tag", default=None, help="report tag (default: the artifacts directory name)")
    ap.add_argument("--out", type=Path, default=None, help="json (default deploy_artifacts/reports/realstate_<tag>.json)")
    a = ap.parse_args()

    os.environ["SM_WAIST_COUPLING"] = a.waist_coupling  # read by sim2sim.load_model in the (forked) workers
    from supermimic.deploy.waist_coupling import coupling_from_env

    wc = coupling_from_env()
    clip = a.clip if a.clip.endswith(".npz") else str(REPO / "deploy_artifacts/clips" / f"{a.clip}.npz")
    tag = a.tag or Path(a.artifacts).name
    real_logs = [str(Path(x)) for x in a.real_logs]
    names = [load_real_log(d)["name"] for d in real_logs]
    jobs = build_grid(len(real_logs), a.frames, a.lat, a.kps, a.mass, a.fric, a.vel, a.max_runs)
    o = {"artifacts": a.artifacts, "clip": clip, "real_logs": real_logs, "stance": a.stance, "imu_model": not a.no_imu_model,
         "waist_roll_range": a.waist_roll_range, "f_end": a.f_end, "peak_from": a.peak_from}
    t0 = time.time()
    print(f"RS {tag}: {len(jobs)} runs, {a.workers} workers, logs {names}, waist coupling {wc}, imu model {o['imu_model']}", flush=True)
    rows = []
    with get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(o,)) as p:
        for k, r in enumerate(p.imap_unordered(run_one, jobs, chunksize=1)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    skipped = [r for r in rows if r.get("skipped")]
    rows = sorted([r for r in rows if not r.get("skipped")], key=lambda r: (r["log"], r["f0"], r["vel"], r["lat"], r["kps"], r["mass"], r["fric"]))
    s = summarize(rows, a.p95_max)
    per_log = {n: summarize([r for r in rows if r["log"] == i], a.p95_max) for i, n in enumerate(names)}
    per_frame = {f: summarize([r for r in rows if r["f0"] == f], a.p95_max) for f in a.frames}
    rr = sorted({tuple(r["roll_range"]) for r in rows})
    states = sorted({(r["log"], r["f0"]) for r in rows if r["roll_clipped"]})
    elapsed = time.time() - t0
    out = a.out or REPO / "deploy_artifacts/reports" / f"realstate_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": tag, "artifacts": a.artifacts, "clip": clip, "real_logs": real_logs, "log_names": names,
               "options": {**o, "waist_coupling": wc, "frames": a.frames, "lat": a.lat, "kps": a.kps, "mass": a.mass, "fric": a.fric,
                           "vel": a.vel, "max_runs": a.max_runs, "p95_max": a.p95_max},
               "criteria": f"PASS = 0 knee joint-velocity aborts (50 Hz core check, > {KNEE_ABORT:.0f} rad/s) and p95 of the 1 kHz knee "
                           f"peak <= {a.p95_max:.0f} rad/s", "summary": s, "per_log": per_log,
               "per_frame": {str(k): v for k, v in per_frame.items()}, "roll_range": rr,
               "roll_clipped_states": [[names[i], f] for i, f in states], "skipped": skipped, "elapsed_s": elapsed, "rows": rows},
              open(out, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    for f, v in per_frame.items():
        print(f"  f0 {f}: knee aborts {v['knee_aborts']}/{v['runs']} p95 {v['knee_peak_p95']:.1f} tilt {v['tilt_aborts']}")
    extra = f" | roll range {rr} clipped states {len(states)}" + (f" | skipped {len(skipped)}" if skipped else "") + f" | {elapsed:.0f} s"
    print(fmt_line(tag, s, per_log, extra), flush=True)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
