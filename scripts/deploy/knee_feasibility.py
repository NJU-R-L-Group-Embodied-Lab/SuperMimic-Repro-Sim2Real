"""Feasibility edit of a 50 Hz source clip: cap the knee speed and keep the knees off the mechanical stop, same timing.

  uv run python scripts/deploy/knee_feasibility.py \
      --src data/npz/shanshen_lean35_src/shanshen_duosuomao_lean35.npz \
      --dst data/npz/shanshen_lean35kf_src/shanshen_duosuomao_lean35kf.npz \
      --frame-offset 149 --floor-windows 535 555 0.15

Then build the padded reference with make_combo_clip.py exactly as the original (FK, velocities, transitions).

Method (per knee, joints 3 / 9 in SDK order; frame count and timing unchanged):
  * violations: frames whose forward difference |q[t+1] - q[t]| * fps exceeds --vel-cap (so the linearly interpolated
    1 kHz reference and the np.gradient joint_vel both stay under it), or q < floor (--floor everywhere, raised to the
    --floor-windows values inside those windows; windows given in padded-clip frames, frame = clip frame + offset);
  * every group of violations, dilated by --pad frames, is re-solved as a small QP over the deviation d = x - q:
    min sum a_t d_t^2 + beta sum (second difference of d)^2 (d = 0 outside the window, so d and its slope go to zero at
    the edges: C1 blend), s.t. |x[t+1] - x[t]| <= vel_cap / fps and x >= floor; a_t is large where the foot of that
    leg is on the ground (original foot z < contact_z), so the change goes into the swing phase; the window grows
    until the QP is feasible;
  * stance compensation: where the edited leg's foot is on the ground (smooth weight from the original foot height),
    hip pitch and ankle pitch of that leg are re-solved (Gauss-Newton on MuJoCo FK, root unchanged) to bring the
    ankle_roll_link position / orientation back to the original;
  * joint_vel (np.gradient, as the converter) and every body state (forward_kinematics, converter velocity conventions)
    are recomputed; frames whose joint state is unchanged keep their stored values bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.optimize import minimize

from supermimic.data.motion_convert import G1_JOINT_NAMES, ResampledMotion, _so3_derivative, forward_kinematics, load_g1_model

KNEES = {"left": 3, "right": 9}
HIP_PITCH = {"left": 0, "right": 6}
ANKLE_PITCH = {"left": 4, "right": 10}
FOOT = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}


def solve_window(q: np.ndarray, lo: int, hi: int, step: float, floor: np.ndarray, a: np.ndarray, beta: float):
    """Deviation over frames lo..hi (inclusive); None if infeasible."""
    n = hi - lo + 1
    T = len(q)
    # second-difference operator on frames lo-1 .. hi+1 (d = 0 outside lo..hi)
    rows = []
    for t in range(lo - 1, hi + 2):
        r = np.zeros(n)
        for k, c in ((t - 1, 1.0), (t, -2.0), (t + 1, 1.0)):
            if lo <= k <= hi:
                r[k - lo] = c
        rows.append(r)
    D = np.array(rows)
    H = np.diag(a[lo : hi + 1]) + beta * D.T @ D
    # x[t+1] - x[t] for t = lo-1 .. hi, x = q + d
    G = np.zeros((n + 1, n))
    for i, t in enumerate(range(lo - 1, hi + 1)):
        if lo <= t + 1 <= hi:
            G[i, t + 1 - lo] += 1.0
        if lo <= t <= hi:
            G[i, t - lo] -= 1.0
    g0 = q[lo : hi + 2] - q[lo - 1 : hi + 1]
    assert lo >= 1 and hi + 1 < T
    cons = [
        {"type": "ineq", "fun": lambda d: step - (g0 + G @ d), "jac": lambda d: -G},
        {"type": "ineq", "fun": lambda d: step + (g0 + G @ d), "jac": lambda d: G},
        {"type": "ineq", "fun": lambda d: q[lo : hi + 1] + d - floor[lo : hi + 1], "jac": lambda d: np.eye(n)},
    ]
    d0 = np.maximum(floor[lo : hi + 1] - q[lo : hi + 1], 0.0)
    res = minimize(lambda d: d @ H @ d, d0, jac=lambda d: 2 * H @ d, constraints=cons, method="SLSQP",
                   options={"maxiter": 2000, "ftol": 1e-14})
    d = res.x
    ok = res.success and np.all(np.abs(g0 + G @ d) <= step + 1e-7) and np.all(q[lo : hi + 1] + d >= floor[lo : hi + 1] - 1e-7)
    return d if ok else None


def foot_pose(model, data, root_pos, root_quat, qj, foot_id):
    data.qpos[0:3], data.qpos[3:7], data.qpos[7:] = root_pos, root_quat, qj
    mujoco.mj_kinematics(model, data)
    return data.xpos[foot_id].copy(), data.xmat[foot_id].reshape(3, 3).copy()


def main(src: Path, dst: Path, vel_cap: float = 10.9, floor: float = 0.052, floor_windows: tuple[float, ...] = (535, 555, 0.15),
         frame_offset: int = 149, pad: int = 10, beta: float = 30.0, contact_z: tuple[float, float] = (0.08, 0.12),
         contact_weight: float = 200.0, keep_edges: int = 2, report: Path | None = None) -> None:
    """vel_cap: forward-difference cap [rad/s] (task limit 11); floor: knee floor everywhere [rad] (task limit 0.05);
    floor_windows: (start, end, floor) triples in padded frames (inclusive); keep_edges: first/last clip frames never
    edited (the pad transitions are built from frame 0 / T-1, its joint_vel seam from frame 1 / T-2)."""
    d = dict(np.load(src, allow_pickle=True))
    fps = float(d["fps"][0])
    dt = 1.0 / fps
    step = vel_cap * dt
    q0 = d["joint_pos"].astype(np.float64)
    q = q0.copy()
    T = len(q)
    bn = [str(b) for b in d["body_names"]]
    r = bn.index("pelvis")
    root_pos = d["body_pos_w"][:, r].astype(np.float64)
    root_quat = d["body_quat_w"][:, r].astype(np.float64)
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    model = load_g1_model()
    data = mujoco.MjData(model)
    foot_z = {s: d["body_pos_w"][:, bn.index(FOOT[s]), 2].astype(np.float64) for s in KNEES}
    # contact weight 1 on the ground .. 0 in the air (smoothstep on the original foot height)
    u = {s: np.clip((contact_z[1] - foot_z[s]) / (contact_z[1] - contact_z[0]), 0, 1) for s in KNEES}
    wc = {s: u[s] * u[s] * (3 - 2 * u[s]) for s in KNEES}

    fl = np.full(T, floor)
    fw = np.asarray(floor_windows, float).reshape(-1, 3)
    for a_, b_, f_ in fw:
        fl[max(0, int(a_) - frame_offset) : min(T, int(b_) - frame_offset + 1)] = np.maximum(fl[max(0, int(a_) - frame_offset) : min(T, int(b_) - frame_offset + 1)], f_)
    rep = {"src": str(src), "dst": str(dst), "vel_cap_fwd_diff": vel_cap, "floor": floor, "floor_windows": fw.tolist(),
           "frame_offset": frame_offset, "windows": []}
    lo_min, hi_max = keep_edges, T - 1 - keep_edges
    for side, j in KNEES.items():
        a = 1.0 + contact_weight * wc[side]
        for _ in range(20):
            x = q[:, j]
            bad = np.zeros(T, bool)
            over = np.abs(np.diff(x)) > step + 1e-6
            bad[:-1] |= over
            bad[1:] |= over
            bad |= x < fl - 1e-6
            if not bad.any():
                break
            idx = np.nonzero(bad)[0]
            # first group (gap <= 2 * pad joins)
            g_lo = g_hi = idx[0]
            for t in idx[1:]:
                if t - g_hi <= 2 * pad:
                    g_hi = t
                else:
                    break
            for extra in range(0, 60, 2):
                lo, hi = max(lo_min, g_lo - pad - extra), min(hi_max, g_hi + pad + extra)
                dev = solve_window(x, lo, hi, step, fl, a, beta)
                if dev is not None:
                    break
            else:
                raise SystemExit(f"{side} knee: no feasible window around frames {g_lo + frame_offset}-{g_hi + frame_offset}")
            q[lo : hi + 1, j] = x[lo : hi + 1] + dev
            ch = np.nonzero(np.abs(dev) > 1e-9)[0]
            w = {"joint": G1_JOINT_NAMES[j], "violations_pad": [int(g_lo + frame_offset), int(g_hi + frame_offset)],
                 "window_pad": [int(lo + frame_offset), int(hi + frame_offset)],
                 "changed_pad": [int(lo + ch[0] + frame_offset), int(lo + ch[-1] + frame_offset)] if len(ch) else None,
                 "max_abs_dq": float(np.abs(dev).max()),
                 "peak_speed_before": float(np.abs(np.diff(q0[lo - 1 : hi + 2, j])).max() * fps),
                 "peak_speed_after": float(np.abs(np.diff(q[lo - 1 : hi + 2, j])).max() * fps),
                 "min_q_before": float(q0[lo : hi + 1, j].min()), "min_q_after": float(q[lo : hi + 1, j].min())}
            rep["windows"].append(w)
            print(f"{side} knee: violations {w['violations_pad']} window {w['window_pad']} changed {w['changed_pad']} "
                  f"max|dq| {w['max_abs_dq']:.3f} speed {w['peak_speed_before']:.2f} -> {w['peak_speed_after']:.2f} "
                  f"min q {w['min_q_before']:.3f} -> {w['min_q_after']:.3f}")
        else:
            raise SystemExit(f"{side} knee: not converged")

    # ---------------------------------------------------------------- stance compensation (hip / ankle pitch)
    rep["stance_ik"] = {}
    for side, j in KNEES.items():
        frames = np.nonzero((np.abs(q[:, j] - q0[:, j]) > 1e-9) & (wc[side] > 0))[0]
        fid = model.body(FOOT[side]).id
        cols = [6 + HIP_PITCH[side], 6 + ANKLE_PITCH[side]]
        jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        corr = np.zeros((T, 2))
        for t in frames:
            p_ref, R_ref = foot_pose(model, data, root_pos[t], root_quat[t], q0[t], fid)
            qt = q[t].copy()
            for _ in range(20):
                p, R = foot_pose(model, data, root_pos[t], root_quat[t], qt, fid)
                e_p = p_ref - p
                Re = R_ref @ R.T
                e_r = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                mujoco.mj_comPos(model, data)
                mujoco.mj_jacBody(model, data, jac_p, jac_r, fid)
                Jx = np.vstack([jac_p[:, cols], 0.1 * jac_r[:, cols]])
                ex = np.concatenate([e_p, 0.1 * e_r])
                dqi = np.linalg.lstsq(Jx, ex, rcond=None)[0]
                qt[HIP_PITCH[side]] += dqi[0]
                qt[ANKLE_PITCH[side]] += dqi[1]
                if np.abs(dqi).max() < 1e-9:
                    break
            corr[t] = [qt[HIP_PITCH[side]] - q[t, HIP_PITCH[side]], qt[ANKLE_PITCH[side]] - q[t, ANKLE_PITCH[side]]]
        corr *= wc[side][:, None]
        q[:, HIP_PITCH[side]] += corr[:, 0]
        q[:, ANKLE_PITCH[side]] += corr[:, 1]
        rep["stance_ik"][side] = {"frames": len(frames), "max_abs_hip_pitch_corr": float(np.abs(corr[:, 0]).max()),
                                  "max_abs_ankle_pitch_corr": float(np.abs(corr[:, 1]).max())}
        print(f"{side} stance IK: {len(frames)} frames, max |d hip_pitch| {np.abs(corr[:, 0]).max():.4f}, "
              f"max |d ankle_pitch| {np.abs(corr[:, 1]).max():.4f}")

    # ---------------------------------------------------------------- recompute derived quantities
    q32 = q.astype(np.float32)
    jp_changed = np.any(q32 != d["joint_pos"], axis=1)
    jv_new = np.gradient(q, dt, axis=0)
    jv_changed = jp_changed.copy()
    jv_changed[1:] |= jp_changed[:-1]
    jv_changed[:-1] |= jp_changed[1:]
    motion = ResampledMotion(fps=fps, root_pos=root_pos, root_quat=root_quat, joint_pos=q,
                             root_lin_vel_w=np.gradient(root_pos, dt, axis=0), root_ang_vel_w=_so3_derivative(root_quat, dt),
                             joint_vel=jv_new)
    bodies = forward_kinematics(model, motion)
    assert bodies.body_names == bn
    out = dict(d)
    out["joint_pos"] = np.where(jp_changed[:, None], q32, d["joint_pos"])
    out["joint_vel"] = np.where(jv_changed[:, None], jv_new.astype(np.float32), d["joint_vel"])
    for key, val in (("body_pos_w", bodies.pos_w), ("body_quat_w", bodies.quat_w), ("body_lin_vel_w", bodies.lin_vel_w),
                     ("body_ang_vel_w", bodies.ang_vel_w)):
        out[key] = np.where(jv_changed[:, None, None], val.astype(np.float32), d[key])
        out[key][:, r] = d[key][:, r]  # root unchanged: keep the stored pelvis state bit for bit (FK only rounds it)
    out["source"] = np.array(f"knee_feasibility({d['source']}; cap {vel_cap} rad/s fwd diff, floor {floor}, windows {fw.tolist()})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)

    # ---------------------------------------------------------------- verification
    ch = np.nonzero(jv_changed)[0]
    rep["changed_frames_pad"] = [int(ch.min() + frame_offset), int(ch.max() + frame_offset)] if len(ch) else None
    rep["changed_frame_count"] = int(jv_changed.sum())
    fv = {}
    for side in KNEES:
        fid = bn.index(FOOT[side])
        dp = np.linalg.norm(out["body_pos_w"][:, fid].astype(np.float64) - d["body_pos_w"][:, fid], axis=-1)
        on = foot_z[side] < contact_z[0]
        fv[side] = {"max_drift_on_ground_m": float(dp[on].max()), "max_drift_any_m": float(dp.max()),
                    "frame_of_max_on_ground_pad": int(np.nonzero(on)[0][np.argmax(dp[on])] + frame_offset)}
    rep["foot_drift"] = fv
    jv = out["joint_vel"]
    rep["knee_speed"] = {s: {"grad_before": float(np.abs(d["joint_vel"][:, j]).max()), "grad_after": float(np.abs(jv[:, j]).max()),
                             "fwd_before": float(np.abs(np.diff(q0[:, j])).max() * fps),
                             "fwd_after": float(np.abs(np.diff(out["joint_pos"][:, j].astype(np.float64))).max() * fps),
                             "min_before": float(q0[:, j].min()), "min_after": float(out["joint_pos"][:, j].min())} for s, j in KNEES.items()}
    print(json.dumps({k: rep[k] for k in ("changed_frames_pad", "changed_frame_count", "foot_drift", "knee_speed")}, indent=1))
    print(f"wrote {dst}")
    if report is not None:
        report.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
