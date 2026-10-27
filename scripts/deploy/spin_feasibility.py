"""Feasibility edit of a 50 Hz source clip: slow the pelvis yaw spins and make the right-foot landing absorbable,
same frame count and timing.

  uv run python scripts/deploy/spin_feasibility.py edit \
      --src data/npz/shanshen_lean35kf_src/shanshen_duosuomao_lean35kf.npz \
      --dst data/npz/shanshen_lean35kfs_src/shanshen_duosuomao_lean35kfs.npz --frame-offset 149 \
      --report data/npz/shanshen_lean35kfs_src/spin_feasibility.json

Then build the padded reference with make_combo_clip.py exactly as the original (stand_idle + clip + stand_idle,
--blend-s 2.0 --end-settle-s 1.0) and check it against the old padded clip:

  uv run python scripts/deploy/spin_feasibility.py verify \
      --old data/npz/shanshen_lean35kf_pad/shanshen_duosuomao_lean35kf_pad2.npz \
      --new data/npz/shanshen_lean35kfs_pad/shanshen_duosuomao_lean35kfs_pad2.npz

All frames below are padded-clip frames (frame = source frame + --frame-offset).

Edit B (right-foot landing, window --land-window, default 722..792): new right ankle_roll_link targets, pelvis/root
untouched (only the heading offset of edit A applies to it):
  * orientation: blend (geodesic, smoothstep) from the original into a flat sole (world roll = pitch = 0, yaw = the
    original foot yaw) over --flat-in, flat through the stance (touchdown .. lift-off), blend back over --flat-out;
  * landing point: the original foot xy at touchdown moved toward the point under the right hip until the flat-foot
    IK gives the knee --knee-td at touchdown (pre-bend) with the ankle at --stance-z (this clip's local stance level);
  * xy: a QP over the offset from the original path (C1 zero at the window edges, second-difference smoothing) with
    the foot at the landing point at touchdown and the horizontal speed (forward difference) <= --v-td around
    touchdown and <= --v-stance until lift-off; z: same QP with z = --stance-z from touchdown to lift-off, z >=
    --stance-z in the window and |vz| <= --vz-td just before touchdown.
Edit A (spins, window --spin-window, default 628..792): a heading offset dpsi(t) about world z through the pelvis
origin (root position unchanged), zero outside the window, minimal and smooth (QP: sum dpsi^2 + beta sum (second
difference)^2 with dpsi = 0 outside), s.t. the pelvis body-z yaw rate (central difference as stored, and forward
difference) <= --yaw-rate-cap; the feet (ankle_roll_link world pose, edit-B targets for the right foot) and the
torso_link world orientation are kept by re-solving the 12 leg joints and the 3 waist joints (Gauss-Newton on MuJoCo
FK); the joint constraints (soft range or no worse than the original, waist coupling |roll| + k |pitch| <= c or no
worse than the original, hip-yaw / waist-yaw / knee forward-difference speed caps, knee floor, other leg joints <=
max(10, original peak)) are linearised with the IK sensitivities dq/dpsi and the QP is re-solved until the exact
re-solved trajectory satisfies everything.

Derived fields: joint_vel (np.gradient), pelvis angular velocity (central SO(3) difference) and every body state
(forward_kinematics, converter conventions) are recomputed on the changed frames (+-1); frames whose joint / root state
is unchanged keep their stored values bit for bit; the pelvis position is never changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.optimize import linprog, minimize

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    ResampledMotion,
    _so3_derivative,
    forward_kinematics,
    load_g1_model,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    rotvec_from_quat,
)

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LEG_COLS = {FEET[0]: np.arange(0, 6), FEET[1]: np.arange(6, 12)}
WAIST = np.arange(12, 15)
EDIT_J = np.arange(15)  # joints re-solved: legs + waist
HIP_YAW, KNEE, WAIST_YAW, WAIST_ROLL, WAIST_PITCH = (2, 8), (3, 9), 12, 13, 14
R_HIP_PITCH, R_KNEE, R_ANKLE_P, R_ANKLE_R = 6, 9, 10, 11


# ----------------------------------------------------------------------------- small helpers


def quat_z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return np.stack([np.cos(0.5 * a), 0 * a, 0 * a, np.sin(0.5 * a)], -1)


def vee_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return rotvec_from_quat(q[None])[0]


def exp_so3(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def yaw_of_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def body_z_rates(rq: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis body-z yaw rate: (central as stored by the converter (T,), forward body-frame increment (T-1,))."""
    wc = quat_rotate_inverse(rq, _so3_derivative(rq, dt))[:, 2]
    wf = rotvec_from_quat(quat_mul(quat_conjugate(rq[:-1]), rq[1:]))[:, 2] / dt
    return wc, wf


def solve_qp(H: np.ndarray, x0: np.ndarray, A_ub: np.ndarray | None = None, b_ub: np.ndarray | None = None,
             A_eq: np.ndarray | None = None, b_eq: np.ndarray | None = None, f: np.ndarray | None = None,
             eps: float = 1e-9, max_iter: int = 50000):
    """min x^T H x + f^T x  s.t. A_ub x <= b_ub, A_eq x = b_eq: OSQP-style ADMM (dense, adaptive rho), then an
    active-set polish (equality-constrained KKT on the active rows) kept when it is feasible."""
    from scipy.linalg import cho_factor, cho_solve

    n = len(x0)
    f = np.zeros(n) if f is None else f
    P, q = 2.0 * H, f
    blocks, lo_, hi_ = [], [], []
    if A_ub is not None and len(A_ub):
        blocks.append(A_ub), lo_.append(np.full(len(b_ub), -np.inf)), hi_.append(b_ub)  # noqa: E702
    if A_eq is not None and len(A_eq):
        blocks.append(A_eq), lo_.append(b_eq), hi_.append(b_eq)  # noqa: E702
    if not blocks:
        x = np.linalg.solve(P, -q)
        return x, _Res(True, "unconstrained", 0, 0.0)
    A, lo, hi = np.vstack(blocks), np.concatenate(lo_), np.concatenate(hi_)
    # row scaling
    rn = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    A, lo, hi = A / rn[:, None], lo / rn, hi / rn
    eq = lo == hi
    sigma, alpha = 1e-6, 1.6
    rho = 0.1 * max(1.0, float(np.abs(P).max()))

    def fact(rho):
        rv = np.where(eq, 1e3 * rho, rho)
        return rv, cho_factor(P + sigma * np.eye(n) + A.T @ (rv[:, None] * A))

    rv, cho = fact(rho)
    x = x0.astype(float).copy()
    z = np.clip(A @ x, lo, hi)
    y = np.zeros(len(A))
    it = 0
    for it in range(max_iter):
        xt = cho_solve(cho, sigma * x - q + A.T @ (rv * z - y))
        zt = A @ xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        zn = np.clip(zr + y / rv, lo, hi)
        y = y + rv * (zr - zn)
        z = zn
        if it % 25 == 0:
            Ax = A @ x
            rp_, rd_ = np.abs(Ax - z).max(), np.abs(P @ x + q + A.T @ y).max()
            sp = max(np.abs(Ax).max(), np.abs(z).max(), 1e-12)
            sd = max(np.abs(P @ x).max(), np.abs(A.T @ y).max(), np.abs(q).max(), 1e-12)
            if rp_ < eps * (1 + sp) and rd_ < eps * (1 + sd):
                break
            if it % 200 == 0 and it > 0:
                ratio = np.sqrt((rp_ / sp) / max(rd_ / sd, 1e-30))
                if ratio > 5 or ratio < 0.2:
                    rho = float(np.clip(rho * ratio, 1e-6, 1e8))
                    rv, cho = fact(rho)
    # polish: active rows (y != 0) as equalities
    act = np.abs(y) > 1e-9 * max(1.0, np.abs(y).max())
    if act.any():
        Aa = A[act]
        ba = np.where(y[act] > 0, hi[act], lo[act])
        K = np.block([[P + 1e-12 * np.eye(n), Aa.T], [Aa, -1e-12 * np.eye(int(act.sum()))]])
        try:
            sol = np.linalg.lstsq(K, np.concatenate([-q, ba]), rcond=None)[0]
            xp = sol[:n]
            Axp = A @ xp
            if np.all(Axp <= hi + 1e-9) and np.all(Axp >= lo - 1e-9):
                x = xp
        except np.linalg.LinAlgError:
            pass
    Ax = A @ x
    viol = float(max(np.max(Ax - hi), np.max(lo - Ax), 0.0))
    return x, _Res(viol < 1e-6, f"admm {it} iters, max viol {viol:.1e}", it, viol)


class _Res:
    def __init__(self, success, message, nit, viol):
        self.success, self.message, self.nit, self.viol = success, message, nit, viol


def d2_operator(n: int) -> np.ndarray:
    """Second differences on frames -1 .. n (x = 0 outside 0..n-1): (n+2, n)."""
    D = np.zeros((n + 2, n))
    for i, t in enumerate(range(-1, n + 1)):
        for k, c in ((t - 1, 1.0), (t, -2.0), (t + 1, 1.0)):
            if 0 <= k < n:
                D[i, k] = c
    return D


def fwd_diff_operator(n: int) -> np.ndarray:
    """x[t+1] - x[t] for t = -1 .. n-1 (x = 0 outside): (n+1, n)."""
    G = np.zeros((n + 1, n))
    for i, t in enumerate(range(-1, n)):
        if t + 1 < n:
            G[i, t + 1] += 1.0
        if t >= 0:
            G[i, t] -= 1.0
    return G


# ----------------------------------------------------------------------------- kinematics


class Kin:
    def __init__(self):
        self.m = load_g1_model()
        self.d = mujoco.MjData(self.m)
        self.fid = {b: self.m.body(b).id for b in FEET}
        self.tid = self.m.body("torso_link").id
        self.hip_r = self.m.body("right_hip_pitch_link").id
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.sole = [i for i in range(self.m.ngeom) if self.m.geom(i).name.startswith("right_foot")]

    def fk(self, rp, rq, qj):
        self.d.qpos[0:3], self.d.qpos[3:7], self.d.qpos[7:] = rp, rq, qj
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def pose(self, b):
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def targets(self, rp, rq, qj):
        self.fk(rp, rq, qj)
        return {**{b: self.pose(self.fid[b]) for b in FEET}, "torso": self.pose(self.tid)[1]}

    def ik(self, rp, rq, q_init, tgt, chains=(FEET[0], FEET[1], "torso"), iters=40, tol=1e-11):
        """Re-solve legs (6 DoF foot pose each) and waist (3 DoF torso orientation), root fixed. Returns (q, err)."""
        q = q_init.copy()
        for _ in range(iters):
            self.fk(rp, rq, q)
            step = 0.0
            for c in chains:
                if c == "torso":
                    R = self.d.xmat[self.tid].reshape(3, 3)
                    e = vee_log(tgt["torso"] @ R.T)
                    mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.tid)
                    J = self.jr[:, 6 + WAIST]
                    cols = WAIST
                else:
                    p, R = self.pose(self.fid[c])
                    e = np.concatenate([tgt[c][0] - p, vee_log(tgt[c][1] @ R.T)])
                    mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.fid[c])
                    cols = LEG_COLS[c]
                    J = np.vstack([self.jp[:, 6 + cols], self.jr[:, 6 + cols]])
                dq = np.linalg.solve(J.T @ J + 1e-10 * np.eye(len(cols)), J.T @ e)
                q[cols] += dq
                step = max(step, float(np.abs(dq).max()))
            if step < tol:
                break
        self.fk(rp, rq, q)
        err = 0.0
        for c in chains:
            if c == "torso":
                err = max(err, float(np.linalg.norm(vee_log(tgt["torso"] @ self.d.xmat[self.tid].reshape(3, 3).T))))
            else:
                p, R = self.pose(self.fid[c])
                err = max(err, float(np.linalg.norm(tgt[c][0] - p)), float(np.linalg.norm(vee_log(tgt[c][1] @ R.T))))
        return q, err

    def ik_leg_bounded(self, rp, rq, q_init, foot, p_t, R_t, lo, hi, w_rot=0.05, iters=300):
        """One leg, joints boxed to [lo, hi] (6,): position exact if reachable, orientation weighted least squares
        (active-set Gauss-Newton). Returns (q, pos err, rot err)."""
        cols = LEG_COLS[foot]
        q = q_init.copy()
        q[cols] = np.clip(q[cols], lo, hi)
        W = np.diag([1.0, 1.0, 1.0, w_rot, w_rot, w_rot])
        fid = self.fid[foot]
        for _ in range(iters):
            self.fk(rp, rq, q)
            p, R = self.pose(fid)
            e = np.concatenate([p_t - p, vee_log(R_t @ R.T)])
            mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, fid)
            J = np.vstack([self.jp[:, 6 + cols], self.jr[:, 6 + cols]])
            g = J.T @ W @ e  # descent direction of 0.5 e^T W e in q
            x = q[cols]
            free = ~(((x <= lo + 1e-12) & (g < 0)) | ((x >= hi - 1e-12) & (g > 0)))
            dq = np.zeros(6)
            Jf = J[:, free]
            dq[free] = np.linalg.solve(Jf.T @ W @ Jf + 1e-9 * np.eye(int(free.sum())), Jf.T @ W @ e)
            x_new = np.clip(x + dq, lo, hi)
            q[cols] = x_new
            if np.abs(x_new - x).max() < 1e-12:
                break
        self.fk(rp, rq, q)
        p, R = self.pose(fid)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(vee_log(R_t @ R.T)))

    def sole_min_z(self):
        zs = []
        for g in self.sole:
            c, ax = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3)[:, 2]
            h, r = self.m.geom_size[g, 1], self.m.geom_size[g, 0]
            zs += [c[2] + h * ax[2] - r, c[2] - h * ax[2] - r]
        return min(zs)


# ----------------------------------------------------------------------------- edit B: right-foot landing targets


def landing_targets(kin: Kin, rp, rq, q0, tg0, fps, off, win, touch, lift, flat_in, flat_out, stance_z, knee_td,
                    v_td, v_stance, vz_td, beta, leg_lo, leg_hi, knee_from, c_stance, cap_from, v_ramp=6.0, n_ramp=8):
    """New right-foot targets (pos (T,3), rot (T,3,3)) on the source frames; also returns info."""
    T = len(q0)
    rf = FEET[1]
    P0 = np.array([tg0[t][rf][0] for t in range(T)])
    R0 = np.array([tg0[t][rf][1] for t in range(T)])
    s = lambda f: int(f) - off  # noqa: E731  padded -> source
    w0, w1, td, lo = s(win[0]), s(win[1]), s(touch), s(lift)
    # ---- orientation: blend to a flat sole (yaw of the original foot) and back
    yaw = np.unwrap(np.array([yaw_of_R(R) for R in R0]))
    wgt = np.zeros(T)
    t_ = np.arange(T)
    wgt = np.maximum(wgt, smoothstep((t_ - s(flat_in[0])) / (s(flat_in[1]) - s(flat_in[0]))))
    wgt = np.minimum(wgt, 1.0 - smoothstep((t_ - s(flat_out[0])) / (s(flat_out[1]) - s(flat_out[0]))))
    wgt[: w0 + 1] = 0.0
    wgt[w1:] = 0.0
    R_new = R0.copy()
    for t in range(w0, w1 + 1):
        Rf = rot_z(yaw[t])
        R_new[t] = exp_so3(wgt[t] * vee_log(Rf @ R0[t].T)) @ R0[t]
    # ---- landing point: from the original xy toward the point under the right hip until knee(td) = knee_td
    cols = LEG_COLS[rf]
    kin.fk(rp[td], rq[td], q0[td])
    hip = kin.d.xpos[kin.hip_r].copy()

    def leg_ik(t, p_t, R_t, q_init):
        """Bounded IK, multi-start (warm start, original frame, bent-knee variants); best = smallest weighted error,
        ties broken by closeness to the warm start."""
        best = None
        for qs in (q_init, q0[t]):
            for kn in (None, 0.8, 1.4):
                qi = np.where(np.isin(np.arange(29), cols), qs, q0[t])
                if kn is not None:
                    qi = qi.copy()
                    qi[R_KNEE] = kn
                r = kin.ik_leg_bounded(rp[t], rq[t], qi, rf, p_t, R_t, leg_lo[t], leg_hi[t])
                cost = 10 * r[1] + r[2] + 1e-3 * float(np.abs(r[0][cols] - q_init[cols]).sum())
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, r)
        return best[1]

    def knee_at(frac):
        xy = (1 - frac) * P0[td, :2] + frac * hip[:2]
        qi = q0[td].copy()
        qi[R_KNEE] = max(qi[R_KNEE], 0.6)
        q, ep, _ = leg_ik(td, np.array([xy[0], xy[1], stance_z]), rot_z(yaw[td]), qi)
        return (q[R_KNEE] if ep < 1e-6 else -1.0), xy

    fr_lo, fr_hi = 0.0, 1.0
    if knee_at(fr_lo)[0] >= knee_td:
        frac = 0.0
    else:
        for _ in range(40):
            mid = 0.5 * (fr_lo + fr_hi)
            if knee_at(mid)[0] < knee_td:
                fr_lo = mid
            else:
                fr_hi = mid
        frac = fr_hi
    knee_land, L = knee_at(frac)
    # ---- xy / z QPs on frames w0+1 .. w1-1 (offset e = new - original, zero outside -> C1 blend at the edges);
    # solved in millimetres for conditioning
    a, b = w0 + 1, w1 - 1
    n = b - a + 1
    D = d2_operator(n)
    G = fwd_diff_operator(n)  # rows: t = a-1 .. b
    H = np.eye(n) + beta * D.T @ D
    rows_t = np.arange(a - 1, b + 1)
    # xy: variables [ex (n), ey (n)]; speed cap |v| <= cap via 16-gon (linear): (v . u_k) <= cap * cos(pi/16)
    caps = np.full(len(rows_t), np.inf)
    cf = s(cap_from)
    caps[(rows_t >= cf) & (rows_t <= lo)] = v_stance
    caps[(rows_t >= td - 4) & (rows_t < td + 4)] = v_td
    ramp = (rows_t >= cf - n_ramp) & (rows_t < cf)  # gradual deceleration into the cap
    caps[ramp] = v_stance + (v_ramp - v_stance) * (cf - rows_t[ramp]) / n_ramp
    A_ub, b_ub = [], []
    vel0 = np.diff(P0[a - 1 : b + 2, :2], axis=0) * 1e3  # original x[t+1] - x[t], t = a-1 .. b [mm]
    angs = np.arange(16) * 2 * np.pi / 16
    for i in np.nonzero(np.isfinite(caps))[0]:
        for th in angs:
            u = np.array([np.cos(th), np.sin(th)])
            row = np.zeros(2 * n)
            row[:n] = u[0] * G[i]
            row[n:] = u[1] * G[i]
            A_ub.append(row)
            b_ub.append(caps[i] * 1e3 / fps * np.cos(np.pi / 16) - u @ vel0[i])
    A_eq = np.zeros((2, 2 * n))
    A_eq[0, td - a], A_eq[1, n + td - a] = 1.0, 1.0
    b_eq = (L - P0[td, :2]) * 1e3
    # stance (touchdown .. lift-off): also penalise the absolute new velocity (as slow as the caps allow)
    Hxy = np.kron(np.eye(2), H)
    f_xy = np.zeros(2 * n)
    st = np.nonzero((rows_t >= td) & (rows_t < lo))[0]
    Gs = G[st]
    for k in range(2):
        sl = slice(k * n, (k + 1) * n)
        Hxy[sl, sl] += c_stance * Gs.T @ Gs
        f_xy[sl] += 2 * c_stance * Gs.T @ vel0[st, k]
    exy, res_xy = solve_qp(Hxy, np.zeros(2 * n), np.array(A_ub), np.array(b_ub), A_eq, b_eq, f_xy)
    exy = exy * 1e-3
    P_new = P0.copy()
    P_new[a : b + 1, 0] += exy[:n]
    P_new[a : b + 1, 1] += exy[n:]
    # z: z = stance_z on [td, lo], z >= stance_z on the window, |vz| <= vz_td on [td-4, td); before touchdown z
    # follows a knee pre-bend profile: knee ramps (smoothstep) from its value at knee_from to knee_td at touchdown
    z0 = P0[a : b + 1, 2] * 1e3
    tt = np.arange(a, b + 1)
    eqs = np.nonzero((tt >= td) & (tt <= lo))[0]
    A_eq_z = np.eye(n)[eqs]
    b_eq_z = stance_z * 1e3 - z0[eqs]
    A_z = [-np.eye(n)]
    b_z = [z0 - stance_z * 1e3]
    vz0 = np.diff(P0[a - 1 : b + 2, 2]) * 1e3
    near = np.nonzero((rows_t >= td - 4) & (rows_t < td))[0]
    A_z += [G[near], -G[near]]
    b_z += [vz_td * 1e3 / fps - vz0[near], vz_td * 1e3 / fps + vz0[near]]
    A_z, b_z = np.vstack(A_z), np.concatenate(b_z)
    ez, res_z = solve_qp(H, np.maximum(stance_z * 1e3 - z0, 0.0), A_z, b_z, A_eq_z, b_eq_z)
    P_new[a : b + 1, 2] += ez * 1e-3
    k0 = s(knee_from)
    q_k0, _, _ = leg_ik(k0, P_new[k0], R_new[k0], q0[k0])
    kn0 = float(q_k0[R_KNEE])
    z_tgt = P_new[:, 2].copy()
    q_w = q_k0.copy()
    for t in range(k0, td):
        k_star = kn0 + (knee_td - kn0) * float(smoothstep(np.array((t - k0) / (td - k0))))
        # continuation scan from high z (bent knee) down to stance_z: knee decreases as the foot goes down
        zs = np.arange(max(P_new[t, 2], stance_z) + 0.25, stance_z - 1e-9, -0.004)
        qq = q_w.copy()
        ks = []
        for z in zs:
            qq, ep, _ = leg_ik(t, np.array([P_new[t, 0], P_new[t, 1], z]), R_new[t], qq)
            ks.append(qq[R_KNEE])
        ks = np.array(ks)
        idx = np.nonzero((ks[:-1] >= k_star) & (ks[1:] < k_star))[0]
        if len(idx):
            i = idx[np.argmin(np.abs(zs[idx] - P_new[t, 2]))]  # crossing closest to the current height
            u = (ks[i] - k_star) / max(ks[i] - ks[i + 1], 1e-12)
            z_tgt[t] = zs[i] + u * (zs[i + 1] - zs[i])
        else:
            z_tgt[t] = stance_z if ks[-1] >= k_star else zs[0]
        q_w = leg_ik(t, np.array([P_new[t, 0], P_new[t, 1], z_tgt[t]]), R_new[t], q_w)[0]
    kk = np.nonzero((tt >= k0) & (tt < td))[0]
    Hk = H.copy()
    fk_ = np.zeros(n)
    c_k = 200.0
    zt = (z_tgt[a : b + 1] * 1e3 - z0)  # target offset
    Hk[kk, kk] += c_k
    fk_[kk] = -2 * c_k * zt[kk]
    ez2, res_z2 = solve_qp(Hk, ez, A_z, b_z, A_eq_z, b_eq_z, fk_)
    P_new[a : b + 1, 2] = (z0 + ez2) * 1e-3
    # ---- achievable targets: bounded IK on the window (warm start), targets := achieved FK poses
    qB = q0.copy()
    perr, rerr = 0.0, []
    q_w = q0[a - 1].copy()
    for t in range(a, b + 1):
        q_w[:] = np.where(np.isin(np.arange(29), cols), q_w, q0[t])
        qq, ep, er = leg_ik(t, P_new[t], R_new[t], q_w)
        qB[t] = qq
        q_w = qq.copy()
        perr = max(perr, ep)
        if ep > 1e-6:
            print(f"  bounded IK: frame {t + off} position error {ep:.4f} m")
        rerr.append(er)
        kin.fk(rp[t], rq[t], qq)
        P_new[t], R_new[t] = kin.pose(kin.fid[rf])
    info = {"landing_frac_toward_hip": frac, "landing_xy": L.tolist(), "landing_xy_orig": P0[td, :2].tolist(),
            "landing_shift_m": float(np.linalg.norm(L - P0[td, :2])), "knee_at_landing_ik": float(knee_land),
            "knee_at_knee_from": kn0, "qp_xy_ok": bool(res_xy.success), "qp_z_ok": bool(res_z.success and res_z2.success),
            "window_src": [a, b], "bounded_ik_max_pos_err_m": perr,
            "bounded_ik_rot_dev_from_flat_target_deg": {int(t + off): round(float(np.degrees(x)), 2) for t, x in zip(range(a, b + 1), rerr) if x > 1e-4}}
    return P_new, R_new, qB, info


# ----------------------------------------------------------------------------- edit


def solve_traj(kin, rp, rq0, psi_full, q_prev, tgt_at, frames, chains_of, psi_prev=None, max_step=0.05):
    """IK on every frame for the heading offsets psi_full, warm-started from q_prev (solved at psi_prev): homotopy in
    steps of at most max_step rad so Gauss-Newton stays on the same branch."""
    q = q_prev.copy()
    psi_prev = np.zeros_like(psi_full) if psi_prev is None else psi_prev
    err = 0.0
    for t in frames:
        k = max(1, int(np.ceil(abs(psi_full[t] - psi_prev[t]) / max_step)))
        for i in range(1, k + 1):
            ps = psi_prev[t] + (psi_full[t] - psi_prev[t]) * i / k
            rq = quat_mul(quat_z(ps)[None], rq0[t][None])[0]
            q[t], e = kin.ik(rp[t], rq, q[t], tgt_at(t, ps), chains=chains_of(t))
        err = max(err, e)
    return q, err


def main(src: Path, dst: Path, frame_offset: int = 149, spin_window: tuple[int, int] = (628, 792), yaw_rate_cap: float = 6.0,
         yaw_margin: float = 0.15, beta: float = 50.0, hip_yaw_cap: float = 10.0, waist_yaw_cap: float = 13.0,
         knee_cap: float = 10.9, knee_floor: float = 0.052, other_cap: float = 10.0, speed_margin: float = 0.2,
         coupling: tuple[float, float] = (1.4, 0.95), soft_factor: float = 0.9,
         land_window: tuple[int, int] = (718, 792), touch: int = 742, lift: int = 776,
         flat_in: tuple[int, int] = (724, 736), flat_out: tuple[int, int] = (778, 790), stance_z: float = 0.08,
         knee_td: float = 0.7, v_td: float = 0.25, v_stance: float = 0.45, vz_td: float = 0.3, land_beta: float = 30.0,
         knee_from: int = 724, c_stance: float = 20.0, cap_from: int = 734, contact_z: tuple[float, float] = (0.18, 0.24), ankle_pitch_min: float = -0.6,
         no_landing: bool = False, no_spin: bool = False, max_outer: int = 12, report: Path | None = None) -> None:
    """spin_window / land_window / touch / lift / flat_*: padded frames; yaw_rate_cap: pelvis body-z [rad/s];
    caps: forward-difference joint speeds [rad/s]; coupling: (k, c) of |roll| + k |pitch| <= c; stance_z: ankle_roll_link
    height of the flat right foot in the stance; knee_td: right knee at touchdown (sets the landing point)."""
    d = dict(np.load(src, allow_pickle=True))
    fps = float(d["fps"][0])
    dt = 1.0 / fps
    off = frame_offset
    q0 = d["joint_pos"].astype(np.float64)
    T = len(q0)
    bn = [str(b) for b in d["body_names"]]
    r = bn.index("pelvis")
    rp = d["body_pos_w"][:, r].astype(np.float64)
    rq0 = d["body_quat_w"][:, r].astype(np.float64)
    rq0 /= np.linalg.norm(rq0, axis=-1, keepdims=True)
    kin = Kin()
    m = kin.m
    rng = m.jnt_range[1:]
    mid, half = rng.mean(1), 0.5 * (rng[:, 1] - rng[:, 0])
    soft_lo, soft_hi = mid - soft_factor * half, mid + soft_factor * half
    tg0 = [kin.targets(rp[t], rq0[t], q0[t]) for t in range(T)]
    rep: dict = {"src": str(src), "dst": str(dst), "frame_offset": off, "params": {
        "spin_window": spin_window, "yaw_rate_cap": yaw_rate_cap, "yaw_margin": yaw_margin, "beta": beta, "hip_yaw_cap": hip_yaw_cap,
        "waist_yaw_cap": waist_yaw_cap, "knee_cap": knee_cap, "knee_floor": knee_floor, "other_cap": other_cap, "coupling": coupling,
        "land_window": land_window, "touch": touch, "lift": lift, "flat_in": flat_in, "flat_out": flat_out, "stance_z": stance_z,
        "knee_td": knee_td, "v_td": v_td, "v_stance": v_stance, "vz_td": vz_td}}

    # joint bounds: soft range, or the original value where the original is already outside; right ankle pitch >=
    # ankle_pitch_min on the landing window; waist coupling <= c or the original; knee floor
    k_c, c_c = coupling
    coup0 = np.abs(q0[:, WAIST_ROLL]) + k_c * np.abs(q0[:, WAIST_PITCH])
    cmax = np.maximum(c_c, coup0)
    lo_b, hi_b = np.minimum(soft_lo[None], q0), np.maximum(soft_hi[None], q0)
    lw = slice(land_window[0] - off, land_window[1] - off + 1)
    lo_b[lw, R_ANKLE_P] = np.maximum(lo_b[lw, R_ANKLE_P], np.minimum(ankle_pitch_min + 0.02, q0[lw, R_ANKLE_P]))
    floor_b = np.minimum(knee_floor, q0[:, list(KNEE)])
    lo_b[:, list(KNEE)] = np.maximum(lo_b[:, list(KNEE)], floor_b)

    # ---------------------------------------------------------------- edit B: right-foot targets
    tg = [dict(x) for x in tg0]
    bf = np.zeros(T, bool)
    q_start = q0.copy()
    if not no_landing:
        cols = LEG_COLS[FEET[1]]
        P_new, R_new, qB, info = landing_targets(kin, rp, rq0, q0, tg0, fps, off, land_window, touch, lift, flat_in, flat_out,
                                                 stance_z, knee_td, v_td, v_stance, vz_td, land_beta, lo_b[:, cols] + 1e-3,
                                                 hi_b[:, cols] - 1e-3, knee_from, c_stance, cap_from)
        rep["landing"] = info
        a, b = info["window_src"]
        for t in range(a, b + 1):
            tg[t][FEET[1]] = (P_new[t], R_new[t])
        bf[a : b + 1] = True
        q_start = qB
        print(f"landing: point {np.round(info['landing_xy'], 3)} (shift {info['landing_shift_m']:.3f} m, frac {info['landing_frac_toward_hip']:.3f}), "
              f"knee at landing (IK) {info['knee_at_landing_ik']:.3f}, QP ok {info['qp_xy_ok']}/{info['qp_z_ok']}, bounded IK pos err "
              f"{info['bounded_ik_max_pos_err_m']:.1e}, rot dev from flat target {info['bounded_ik_rot_dev_from_flat_target_deg']}")

    # ---------------------------------------------------------------- edit A: heading offset
    s0, s1 = spin_window[0] - off, spin_window[1] - off
    n = s1 - s0 + 1
    win = np.arange(s0, s1 + 1)
    frames = sorted(set(win.tolist()) | set(np.nonzero(bf)[0].tolist()))
    in_win = np.zeros(T, bool)
    in_win[win] = True

    # airborne feet follow the heading offset (their leg joints stay as they are), feet on the ground keep their world
    # pose: foot target = original pose rotated about the vertical through the pelvis origin by (1 - contact) * dpsi;
    # contact = smoothstep on the target ankle height (contact_z; edit-B targets for the right foot), time-smoothed
    air = {}
    for f in FEET:
        z = np.array([tg[t][f][0][2] for t in range(T)])
        u = np.clip((contact_z[1] - z) / (contact_z[1] - contact_z[0]), 0, 1)
        c = u * u * (3 - 2 * u)
        c = np.convolve(np.pad(c, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        air[f] = 1.0 - c

    def tgt_at(t, ps):
        out = dict(tg[t])
        for f in FEET:
            a_ = air[f][t] * ps
            if a_ != 0.0:
                Rz = rot_z(a_)
                p_, R_ = tg[t][f]
                out[f] = (rp[t] + Rz @ (p_ - rp[t]), Rz @ R_)
        return out

    def chains_of(t):
        if in_win[t]:
            return (FEET[0], FEET[1], "torso")
        return (FEET[1],)

    def full(psi):
        x = np.zeros(T)
        if not no_spin:
            x[s0 : s1 + 1] = psi
        return x

    def rq_of(psi):
        return quat_mul(quat_z(full(psi)), rq0)

    # constraint frames (forward differences t -> t+1 for t in s0-1 .. s1; joints / rates evaluated on s0-1 .. s1+1)
    ft = np.arange(s0 - 1, s1 + 1)
    vt = np.arange(s0 - 1, s1 + 2)
    caps = np.full(15, other_cap)
    for j in range(15):
        caps[j] = max(other_cap, float(np.abs(np.diff(q0[s0 - 2 : s1 + 3, j])).max() * fps))
    for j in HIP_YAW:
        caps[j] = hip_yaw_cap
    caps[WAIST_YAW] = waist_yaw_cap
    for j in KNEE:
        caps[j] = knee_cap

    def violations(q, psi):
        wc, wf = body_z_rates(rq_of(psi), dt)
        # yaw only where the offset can act (window +- 2 frames); peaks elsewhere are reported by summarize()
        lo_, hi_ = max(0, s0 - 2), min(T, s1 + 3)
        v = {"yaw_c": float(np.abs(wc[lo_:hi_]).max() - yaw_rate_cap), "yaw_f": float(np.abs(wf[lo_ : hi_ - 1]).max() - yaw_rate_cap)}
        sp = np.abs(np.diff(q[:, :15], axis=0)) * fps
        chg = np.any(q[:, :15] != q0[:, :15], axis=1)
        rows = chg[:-1] | chg[1:]
        v["speed"] = float((sp[rows] - caps[None]).max()) if rows.any() else -1.0
        v["soft"] = float(max((lo_b[:, :15] - q[:, :15]).max(), (q[:, :15] - hi_b[:, :15]).max()))
        v["coupling"] = float((np.abs(q[:, WAIST_ROLL]) + k_c * np.abs(q[:, WAIST_PITCH]) - cmax).max())
        v["knee_floor"] = float((floor_b - q[:, list(KNEE)]).max())
        return v

    psi = np.zeros(n)
    q, ik_err = solve_traj(kin, rp, rq0, full(psi), q_start, tgt_at, frames, chains_of)
    print(f"edit B only: IK err {ik_err:.2e}, violations {violations(q, psi)}")
    D = d2_operator(n)
    H = np.eye(n) + beta * D.T @ D
    hstep = 1e-5
    marg = yaw_margin
    sp_marg = speed_margin
    hist = []
    for it in range(max_outer if not no_spin else 0):
        # linearisation at psi: joint sensitivities S[t, j] = dq_j / dpsi_t
        q_lin = q.copy()
        S = np.zeros((T, 15))
        pf = full(psi)
        for t in win:
            rq = quat_mul(quat_z(pf[t] + hstep)[None], rq0[t][None])[0]
            qp, _ = kin.ik(rp[t], rq, q_lin[t], tgt_at(t, pf[t] + hstep), chains=chains_of(t), tol=1e-13)
            S[t] = (qp[:15] - q_lin[t, :15]) / hstep
        S[np.abs(S) < 1e-5] = 0.0  # finite-difference noise (e.g. airborne legs that follow the offset rigidly)
        # yaw-rate Jacobians (finite differences, banded)
        wc0, wf0 = body_z_rates(rq_of(psi), dt)
        Jc = np.zeros((T, n))
        Jf = np.zeros((T - 1, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = hstep
            wc1, wf1 = body_z_rates(rq_of(psi + e), dt)
            Jc[:, i] = (wc1 - wc0) / hstep
            Jf[:, i] = (wf1 - wf0) / hstep
        A, bb = [], []
        cap_y = yaw_rate_cap - marg
        rows_c = np.arange(max(0, s0 - 2), min(T, s1 + 3))
        rows_f = np.arange(max(0, s0 - 2), min(T - 1, s1 + 2))
        for J_, w_, rows in ((Jc, wc0, rows_c), (Jf, wf0, rows_f)):
            A += [J_[rows], -J_[rows]]
            bb += [cap_y - w_[rows] + J_[rows] @ psi, cap_y + w_[rows] - J_[rows] @ psi]
        # joint q(t) ~ q_lin(t) + S(t) (x_t - psi_t) on window frames; outside the window q is fixed
        # selection: E[t] maps x to x_t (zero rows outside the window)
        def qrow(t, j):
            row = np.zeros(n)
            c0 = q_lin[t, j]
            if s0 <= t <= s1:
                row[t - s0] = S[t, j]
                c0 -= S[t, j] * psi[t - s0]
            return row, c0

        # margins only where the current value has room (never demand better than the current / original value)
        def ub(bound, cur, margin):
            return bound - margin if cur <= bound - margin else max(bound, cur)

        tags = ["yaw"] * sum(len(x) for x in A)
        for j in range(15):
            cap = caps[j] / fps
            for t in ft:
                r1, c1 = qrow(t + 1, j)
                r0, c0_ = qrow(t, j)
                g = r1 - r0
                dc = c1 - c0_
                if not g.any():
                    continue
                if abs(dc) + np.abs(g).sum() * 0.6 > 0.6 * cap:
                    lim = ub(cap, abs(dc), sp_marg / fps)
                    A += [g[None], -g[None]]
                    bb += [np.array([lim - dc]), np.array([lim + dc])]
                    tags += [f"speed:{G1_JOINT_NAMES[j]}"] * 2
            for t in vt:
                rr, cc = qrow(t, j)
                if not rr.any():
                    continue
                A += [rr[None], -rr[None]]
                bb += [np.array([ub(hi_b[t, j], cc, 1e-3) - cc]), np.array([ub(-lo_b[t, j], -cc, 1e-3) + cc])]
                tags += [f"range:{G1_JOINT_NAMES[j]}"] * 2
        for t in vt:
            rr_r, cr = qrow(t, WAIST_ROLL)
            rr_p, cp = qrow(t, WAIST_PITCH)
            if not (rr_r.any() or rr_p.any()):
                continue
            for sr in (1, -1):
                for sp_ in (1, -1):
                    cur = sr * cr + sp_ * k_c * cp
                    A.append((sr * rr_r + sp_ * k_c * rr_p)[None])
                    bb.append(np.array([ub(cmax[t], cur, 2e-3) - cur]))
                    tags.append("coupling")
        A_ub, b_ub = np.vstack(A), np.concatenate(bb)
        if (b_ub < -1e-12).any():
            pass
        lp = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n, method="highs")
        if lp.status != 0:
            cats = sorted(set(x.split(":")[0] for x in tags))
            for nm in sorted(set(tags)):
                keep = np.array([x == nm or x == "yaw" for x in tags])
                lp2 = linprog(np.zeros(n), A_ub=A_ub[keep], b_ub=b_ub[keep], bounds=[(None, None)] * n, method="highs")
                keep2 = np.array([x == nm for x in tags])
                lp3 = linprog(np.zeros(n), A_ub=A_ub[keep2], b_ub=b_ub[keep2], bounds=[(None, None)] * n, method="highs")
                if lp2.status != 0 or lp3.status != 0:
                    bad = np.nonzero(keep2 & (b_ub < 0))[0]
                    print(f"  linearised QP infeasible: {nm} alone status {lp3.status}, with yaw {lp2.status}; rows with b<0: {len(bad)}")
        psi_new, res = solve_qp(H, psi, A_ub, b_ub)
        q_new, ik_err = solve_traj(kin, rp, rq0, full(psi_new), q, tgt_at, frames, chains_of, psi_prev=full(psi))
        v = violations(q_new, psi_new)
        lin_viol = float((A_ub @ psi_new - b_ub).max())
        hist.append({"iter": it, "qp_success": bool(res.success), "qp_msg": str(res.message), "lin_viol": lin_viol,
                     "max_abs_dpsi": float(np.abs(psi_new).max()), "ik_err": ik_err, **v})
        print(f"iter {it}: QP {res.success} ({res.message}) lin viol {lin_viol:.2e} max|dpsi| {np.abs(psi_new).max():.4f} "
              f"IK err {ik_err:.1e} exact {v} rows {len(b_ub)}")
        psi, q = psi_new, q_new
        if all(x <= 1e-6 for x in v.values()) and res.success:
            break
        if v["yaw_c"] > 0 or v["yaw_f"] > 0:
            marg += 0.05
        if v["speed"] > 0:
            sp_marg += 0.1
    rep["spin_iterations"] = hist

    # ---------------------------------------------------------------- recompute derived quantities
    rq = rq_of(psi)
    q32 = q.astype(np.float32)
    rq32 = rq.astype(np.float32)
    jp_changed = np.any(q32 != d["joint_pos"], axis=1)
    root_changed = np.abs(full(psi)) > 0
    ch = jp_changed | root_changed
    root_q_store = np.where(root_changed[:, None], rq32, d["body_quat_w"][:, r])
    rq_use = np.where(root_changed[:, None], rq, rq0)
    q_use = np.where(jp_changed[:, None], q32.astype(np.float64), q0)
    jv_new = np.gradient(q_use, dt, axis=0)
    w_new = _so3_derivative(rq_use, dt)
    dil = ch.copy()
    dil[1:] |= ch[:-1]
    dil[:-1] |= ch[1:]
    motion = ResampledMotion(fps=fps, root_pos=rp, root_quat=rq_use, joint_pos=q_use, root_lin_vel_w=np.gradient(rp, dt, axis=0),
                             root_ang_vel_w=w_new, joint_vel=jv_new)
    bodies = forward_kinematics(m, motion)
    assert bodies.body_names == bn
    out = dict(d)
    out["joint_pos"] = np.where(jp_changed[:, None], q32, d["joint_pos"])
    out["joint_vel"] = np.where(dil[:, None], jv_new.astype(np.float32), d["joint_vel"])
    for key, val in (("body_pos_w", bodies.pos_w), ("body_quat_w", bodies.quat_w), ("body_lin_vel_w", bodies.lin_vel_w),
                     ("body_ang_vel_w", bodies.ang_vel_w)):
        out[key] = np.where(dil[:, None, None], val.astype(np.float32), d[key])
    out["body_pos_w"][:, r] = d["body_pos_w"][:, r]  # root position never changes
    out["body_lin_vel_w"][:, r] = d["body_lin_vel_w"][:, r]
    out["body_quat_w"][:, r] = root_q_store
    out["source"] = np.array(f"spin_feasibility({d['source']}; yaw rate cap {yaw_rate_cap} rad/s body z, window {list(spin_window)}; "
                             f"right landing {'off' if no_landing else f'td {touch} lift {lift} stance z {stance_z} knee {knee_td}'})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)

    # ---------------------------------------------------------------- report
    rep.update(summarize(kin, d, out, off, fps, touch))
    rep["max_abs_dpsi_rad"] = float(np.abs(psi).max())
    rep["dpsi_edges"] = [float(psi[0]), float(psi[1]), float(psi[-2]), float(psi[-1])]
    rep["changed_frames_pad"] = [int(np.nonzero(dil)[0].min() + off), int(np.nonzero(dil)[0].max() + off)] if dil.any() else None
    rep["dpsi_pad"] = {int(t + off): round(float(x), 5) for t, x in zip(win, psi)}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("dpsi_pad", "spin_iterations", "params")}, indent=1))
    print(f"wrote {dst}")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(rep, indent=1))


def summarize(kin: Kin, d0: dict, d1: dict, off: int, fps: float, touch: int) -> dict:
    """Before / after numbers on a pair of npz dicts (same frame indexing; off = padded - index)."""
    bn = [str(b) for b in d0["body_names"]]
    r = bn.index("pelvis")
    rf, lf = bn.index(FEET[1]), bn.index(FEET[0])
    out: dict = {}

    def rates(dd):
        rq = dd["body_quat_w"][:, r].astype(np.float64)
        rq /= np.linalg.norm(rq, axis=-1, keepdims=True)
        wst = quat_rotate_inverse(rq, dd["body_ang_vel_w"][:, r].astype(np.float64))[:, 2]
        wc, wf = body_z_rates(rq, 1.0 / fps)
        return wst, wc, wf

    for tag, dd in (("before", d0), ("after", d1)):
        wst, wc, wf = rates(dd)
        q = dd["joint_pos"].astype(np.float64)
        jv = dd["joint_vel"].astype(np.float64)
        fwd = np.abs(np.diff(q, axis=0)) * fps
        coup = np.abs(q[:, WAIST_ROLL]) + 1.4 * np.abs(q[:, WAIST_PITCH])
        V = dd["body_lin_vel_w"].astype(np.float64)
        P = dd["body_pos_w"].astype(np.float64)
        Q = dd["body_quat_w"].astype(np.float64)
        td = touch - off
        sole = []
        for t in range(td - 12, td + 13, 2):
            kin.fk(P[t, r], Q[t, r] / np.linalg.norm(Q[t, r]), q[t])
            sole.append(round(kin.sole_min_z(), 4))
        kin.fk(P[td, r], Q[td, r] / np.linalg.norm(Q[td, r]), q[td])
        Rr = kin.d.xmat[kin.fid[FEET[1]]]
        out[tag] = {
            "pelvis_yaw_rate_peak_stored": [float(np.abs(wst).max()), int(np.abs(wst).argmax() + off)],
            "pelvis_yaw_rate_peak_central": float(np.abs(wc).max()), "pelvis_yaw_rate_peak_fwd": float(np.abs(wf).max()),
            "waist_yaw_speed_peak": [float(fwd[:, WAIST_YAW].max()), float(np.abs(jv[:, WAIST_YAW]).max())],
            "hip_yaw_speed_peak": [float(fwd[:, list(HIP_YAW)].max()), float(np.abs(jv[:, list(HIP_YAW)]).max())],
            "knee_speed_peak": [float(fwd[:, list(KNEE)].max()), float(np.abs(jv[:, list(KNEE)]).max())],
            "knee_min": float(q[:, list(KNEE)].min()),
            "leg_waist_speed_peak_fwd": {G1_JOINT_NAMES[j]: round(float(fwd[:, j].max()), 3) for j in range(15)},
            "all_joint_speed_peak_grad": [float(np.abs(jv).max()), G1_JOINT_NAMES[int(np.abs(jv).max(0).argmax())]],
            "coupling_max": [float(coup.max()), int(coup.argmax() + off)], "coupling_margin_min": float(0.95 - coup.max()),
            "frames_coupling_over_0.95": int((coup > 0.95).sum()),
            "right_at_touch": {"frame": touch, "knee": float(q[td, R_KNEE]), "ankle_pitch": float(q[td, R_ANKLE_P]),
                               "ankle_roll": float(q[td, R_ANKLE_R]), "hip_pitch": float(q[td, R_HIP_PITCH]),
                               "foot_z": float(P[td, rf, 2]), "foot_tilt_deg": float(np.degrees(np.arccos(np.clip(Rr[8], -1, 1))))},
            "right_window_730_746": {"knee": [round(float(x), 3) for x in q[730 - off : 747 - off : 2, R_KNEE]],
                                     "ankle_pitch": [round(float(x), 3) for x in q[730 - off : 747 - off : 2, R_ANKLE_P]],
                                     "foot_xy_speed": [round(float(x), 3) for x in np.linalg.norm(V[730 - off : 747 - off : 2, rf, :2], axis=-1)],
                                     "foot_z": [round(float(x), 3) for x in P[730 - off : 747 - off : 2, rf, 2]]},
            "right_sole_min_z_td-12..td+12_step2": sole,
            "right_foot_xy_speed_peak_td-4..lift": float(np.linalg.norm(V[td - 4 : 776 - off + 1, rf, :2], axis=-1).max()),
            "right_ankle_pitch_min_722_792": float(q[722 - off : 793 - off, R_ANKLE_P].min()),
        }
    # stance-foot drift: feet poses vs the original on the frames where that foot is on the ground (orig z < 0.12)
    drift = {}
    for name, b in (("left", lf), ("right", rf)):
        dp = np.linalg.norm(d1["body_pos_w"][:, b].astype(np.float64) - d0["body_pos_w"][:, b], axis=-1)
        on = d0["body_pos_w"][:, b, 2] < 0.12
        drift[name] = {"max_on_ground_m": float(dp[on].max()), "max_any_m": float(dp.max())}
    tq = bn.index("torso_link")
    qa, qb = d0["body_quat_w"][:, tq].astype(np.float64), d1["body_quat_w"][:, tq].astype(np.float64)
    drift["torso_rot_deg"] = float(np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, -1)), 0, 1))).max())
    out["drift_vs_original"] = drift
    return out


# ----------------------------------------------------------------------------- verify the padded clip


def verify(old: Path, new: Path, clip_start: int = 149, clip_end: int = 898, edit_lo: int = 627, edit_hi: int = 793,
           report: Path | None = None) -> None:
    """FK(stored joints) == stored bodies, velocities == finite differences (converter conventions), padding identical,
    frames outside [edit_lo, edit_hi] bit-identical to the old padded clip."""
    a, b = dict(np.load(old, allow_pickle=True)), dict(np.load(new, allow_pickle=True))
    kin = Kin()
    fps = float(b["fps"][0])
    dt = 1.0 / fps
    bn = [str(x) for x in b["body_names"]]
    r = bn.index("pelvis")
    T = len(b["joint_pos"])
    assert T == len(a["joint_pos"]), (T, len(a["joint_pos"]))
    res: dict = {"frames": T}
    rp = b["body_pos_w"][:, r].astype(np.float64)
    rq = b["body_quat_w"][:, r].astype(np.float64)
    rq /= np.linalg.norm(rq, axis=-1, keepdims=True)
    q = b["joint_pos"].astype(np.float64)
    motion = ResampledMotion(fps=fps, root_pos=rp, root_quat=rq, joint_pos=q, root_lin_vel_w=np.gradient(rp, dt, axis=0),
                             root_ang_vel_w=_so3_derivative(rq, dt), joint_vel=np.gradient(q, dt, axis=0))
    bod = forward_kinematics(kin.m, motion)
    res["fk_err"] = {"body_pos_m": float(np.abs(bod.pos_w - b["body_pos_w"]).max()),
                     "body_quat": float(np.minimum(np.abs(bod.quat_w - b["body_quat_w"]), np.abs(bod.quat_w + b["body_quat_w"])).max()),
                     "body_lin_vel": float(np.abs(bod.lin_vel_w - b["body_lin_vel_w"]).max()),
                     "body_ang_vel": float(np.abs(bod.ang_vel_w - b["body_ang_vel_w"]).max()),
                     "joint_vel_vs_gradient": float(np.abs(motion.joint_vel - b["joint_vel"]).max())}
    res["fk_err_old_clip_same_check"] = None
    same = {}
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        diff = np.any((a[k] != b[k]).reshape(T, -1), axis=1)
        idx = np.nonzero(diff)[0]
        same[k] = [int(idx.min()), int(idx.max())] if len(idx) else None
    res["changed_frame_range"] = same
    outside = np.ones(T, bool)
    outside[edit_lo : edit_hi + 1] = False
    res["outside_bit_identical"] = bool(all(np.array_equal(a[k][outside], b[k][outside]) for k in same))
    pad = np.ones(T, bool)
    pad[clip_start:clip_end] = False
    res["padding_bit_identical"] = bool(all(np.array_equal(a[k][pad], b[k][pad]) for k in same))
    for k in ("segment_start", "segment_end", "segment_kind"):
        if k in a:
            res[f"{k}_same"] = bool(np.array_equal(a[k], b[k]))
    wc, wf = body_z_rates(rq, dt)
    res["pelvis_yaw_rate_peak"] = {"stored_body_z": float(np.abs(quat_rotate_inverse(rq, b["body_ang_vel_w"][:, r].astype(np.float64))[:, 2]).max()),
                                   "central": float(np.abs(wc).max()), "forward": float(np.abs(wf).max())}
    print(json.dumps(res, indent=1))
    if report is not None:
        report.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"edit": main, "verify": verify})
