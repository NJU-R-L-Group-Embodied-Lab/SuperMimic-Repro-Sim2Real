"""Extend a clip past a close-up camera cut: A (existing rows, kept bit-exact) -> B (close-up, legs not visible)
-> C (wide shot again, until the dancer stands still). EXPERIMENTAL: written for two clips of one dance video
(static camera, wide / close-up / wide); check the result with render_full_preview.py.

Inputs
  * --old_csv: the existing CSV of shot A (rows 0..L-1), copied unchanged;
  * --whole_csv: the same pipeline (person boxes from shot_bbx.py so every shot keeps its own box, GVHMR -s,
    gvhmr_to_supermimic.py --floor_mode drift) run on the whole A+B+C clip, which must START AT THE SAME SOURCE
    FRAME as the old CSV (= N below);
  * --gvhmr_pred: that whole-clip run's hmr4d_results.pt (in-camera body orientation and translation).
Construction
  * A tail (rows L..c0-1): N rigidly aligned to the existing rows, residual offset faded out over 12 frames.
  * C (rows c1..): N's whole-body motion. GVHMR's root translation is unreliable through the close-up (the person
    fills the frame), but both wide shots come from the same static camera, so C is placed by the camera:
    heading = in-camera body yaw + the camera->world yaw offset fitted on A, position = A end + camera-frame
    displacement (--place cam, needs --human_height) or so that the planted feet coincide (--place feet).
  * B (rows c0..c1-1): waist, arms, head and pelvis tilt from N (upper body is visible); root xy/yaw/z = smooth
    bridge between the A-end and C-start states plus part of N's own sway; legs by IK on foot targets:
    "fixed": both feet stay planted; "steps": N's stepping rhythm (contact when N's sole is < 3.5 cm), each
    contact locked flat on the floor, swings re-targeted between locks (average swing speed <= 2.5 m/s).
  * finally joint spikes > 12 rad/s are re-interpolated, the clip ends 15 frames after the dancer reaches the
    still pose (held exactly), joints clamped to the robot model's limits.

Run inside the GMR environment (needs mujoco, numpy, scipy, torch):
  python tools/video2motion/extend_full.py --old_csv A.csv --whole_csv N.csv --gvhmr_pred N/hmr4d_results.pt \
      --cuts 271 334 --feet steps --place cam --human_height 1.70 --out clip_full.csv
"""

import argparse
import json
import os

import mujoco
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import g1_model

FPS = g1_model.FPS
FADE = 12

m = d = LO = HI = FB = SPH = RAD = PELVIS = None


def init_model(robot_xml=None):
    global m, d, LO, HI, FB, SPH, RAD, PELVIS
    m = g1_model.load_g1(robot_xml)
    d = mujoco.MjData(m)
    LO, HI = g1_model.joint_limits(m)
    feet_pts = g1_model.foot_contact_points(m)
    FB = [b for b, _, _ in feet_pts]
    SPH = [c for _, c, _ in feet_pts]  # contact sphere centres in the foot frame
    RAD = [r for _, _, r in feet_pts]
    PELVIS = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")


def setq(row):
    g1_model.set_row(m, d, row)


def feet(row):
    """[(pos, rotmat, sole_z)] for left/right ankle_roll_link."""
    setq(row)
    return [
        (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), float((d.xpos[b][2] + SPH[s] @ d.xmat[b].reshape(3, 3)[2] - RAD[s]).min()))
        for s, b in enumerate(FB)
    ]


def sole_offset(Rm, s):
    """z of the lowest sphere bottom relative to the foot origin, for foot rotation Rm."""
    return ((SPH[s] @ Rm.T)[:, 2] - RAD[s]).min()


def yaw(q):
    return R.from_quat(q).as_euler("ZYX")[..., 0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circmean(a):
    return np.arctan2(np.sin(a).mean(), np.cos(a).mean())


def rotz(a):
    return R.from_euler("z", a)


def hermite(p0, v0, p1, v1, n):
    """n in-between samples from p0 (frame -1) to p1 (frame n), velocities per frame."""
    T = n + 1
    u = (np.arange(1, n + 1) / T)[:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1


def bridge(p0, v0, p1, v1, n, tau=4.0):
    """Like hermite() but without the long-gap overshoot: smoothstep from p0 to p1 plus the end velocities
    decaying over ~tau frames (a 2 s cubic Hermite with the end velocities bulges by up to 0.15*T*v)."""
    k = np.arange(1, n + 1)[:, None].astype(float)
    u = k / (n + 1)
    base = p0 + (p1 - p0) * u * u * (3 - 2 * u)
    kb = (n + 1) - k
    return base + v0 * k * np.exp(-k / tau) - v1 * kb * np.exp(-kb / tau)


def fade(k):
    """1 at k=0 -> 0 at k=FADE (cosine)."""
    k = np.clip(k, 0, FADE)
    return 0.5 * (1 + np.cos(np.pi * k / FADE))


def ik_legs(row, targets, iters=60):
    """Solve both 6-dof legs of `row` (in place) so the ankle_roll_links reach targets [(pos, rotmat)]."""
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for s, (tp, tR) in enumerate(targets):
        js = slice(6 * s, 6 * s + 6)
        for _ in range(iters):
            setq(row)
            mujoco.mj_comPos(m, d)
            b = FB[s]
            ep = tp - d.xpos[b]
            er = R.from_matrix(tR @ d.xmat[b].reshape(3, 3).T).as_rotvec()
            e = np.concatenate([ep, 0.3 * er])
            if np.linalg.norm(e) < 2e-4:
                break
            mujoco.mj_jacBody(m, d, jacp, jacr, b)
            cols = 6 + 6 * s + np.arange(6)
            J = np.vstack([jacp[:, cols], 0.3 * jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            dq *= min(1.0, 0.15 / np.abs(dq).max())
            row[7:][js] = np.clip(row[7:][js] + dq, LO[js], HI[js])
    f = feet(row)
    return max(np.linalg.norm(f[s][0] - targets[s][0]) for s in range(2))


def runs(mask):
    out, s = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def build(cfg):
    """cfg: old_csv, whole_csv, gvhmr_pred, cuts (c0, c1), feet, place, height, start, out. Returns the log dict."""
    c0, c1 = cfg["cuts"]
    old = np.loadtxt(cfg["old_csv"], delimiter=",", ndmin=2)
    N = np.loadtxt(cfg["whole_csv"], delimiter=",", ndmin=2)
    res = torch.load(cfg["gvhmr_pred"], map_location="cpu")
    L, T = len(old), len(N)
    assert old.shape[1] == N.shape[1] == g1_model.CSV_COLS, (old.shape, N.shape)
    if not (30 <= L < c0 < c1 < T - 16):
        raise SystemExit(f"need 30 <= kept rows ({L}) < c0 ({c0}) < c1 ({c1}) < whole-clip rows ({T}) - 16")
    if cfg["place"] == "cam" and not cfg["height"]:
        raise SystemExit("--place cam needs --human_height (printed by gvhmr_to_supermimic.py)")
    log = {"L": L, "T": T}

    # ---- align N to the existing rows (yaw + translation fitted on the last 30 A rows)
    w = slice(L - 30, L)
    dyaw = circmean(yaw(old[w, 3:7]) - yaw(N[w, 3:7]))
    Nw = N.copy()
    Nw[:, :2] = rotz(dyaw).apply(np.c_[N[:, :2], np.zeros(T)])[:, :2]
    Nw[:, :2] += (old[w, :2] - Nw[w, :2]).mean(0)
    Nw[:, 2] += (old[w, 2] - Nw[w, 2]).mean()
    Nw[:, 3:7] = (rotz(dyaw) * R.from_quat(N[:, 3:7])).as_quat()

    out = Nw.copy()
    out[:L] = old
    # residual offsets at the seam, faded out over FADE rows
    o_pos = old[L - 1, :3] - Nw[L - 1, :3]
    o_rot = R.from_quat(old[L - 1, 3:7]) * R.from_quat(Nw[L - 1, 3:7]).inv()
    o_j = old[L - 1, 7:] - N[L - 1, 7:]
    k = np.arange(L, T) - (L - 1)
    fw = fade(k)
    out[L:, :3] = Nw[L:, :3] + fw[:, None] * o_pos
    out[L:, 3:7] = (R.from_rotvec(fw[:, None] * o_rot.as_rotvec()) * R.from_quat(Nw[L:, 3:7])).as_quat()
    out[L:, 7:] = N[L:, 7:] + fw[:, None] * o_j
    log["seam_fit"] = dict(dyaw_deg=np.degrees(dyaw), o_pos=o_pos.round(3).tolist(), o_rot_deg=float(np.degrees(o_rot.magnitude())))

    # ---- place C with the static wide-shot camera
    go = R.from_rotvec(res["smpl_params_incam"]["global_orient"].numpy())
    fwd = go.apply([0, 0, 1])
    cyaw = np.arctan2(fwd[:, 0], -fwd[:, 2])  # body heading seen from the camera, CCW about up
    cam_off = circmean(yaw(old[:, 3:7]) - cyaw[:L])
    log["cam_yaw_offset_std_deg"] = float(np.degrees(np.std(wrap(yaw(old[:, 3:7]) - cyaw[:L] - cam_off))))
    C = slice(c1, T)
    dC = circmean(cyaw[C] + cam_off - yaw(Nw[C, 3:7]))
    piv = Nw[c1, :2].copy()
    Cw = Nw[C].copy()
    Cw[:, :2] = rotz(dC).apply(np.c_[Cw[:, :2] - piv, np.zeros(T - c1)])[:, :2] + piv
    Cw[:, 3:7] = (rotz(dC) * R.from_quat(Cw[:, 3:7])).as_quat()
    a_end = out[c0 - 1]
    if cfg["place"] == "cam":
        tr = res["smpl_params_incam"]["transl"].numpy()
        dc = tr[c1 : c1 + 3].mean(0) - tr[c0 - 3 : c0].mean(0)
        scale = 0.9 * cfg["height"] / 1.8  # GMR smplx_to_g1 root scale
        dw = scale * rotz(cam_off).apply([-dc[2], dc[0], 0])[:2]
        shift = a_end[:2] + dw - Cw[0, :2]
        log["C_place"] = dict(cam_disp_m=dc[[0, 2]].round(3).tolist(), world_disp_m=dw.round(3).tolist())
    else:
        fa = feet(a_end)
        fc = feet(Cw[0])
        shift = np.mean([fa[s][0][:2] - fc[s][0][:2] for s in range(2)], 0)
        log["C_place"] = dict(per_foot_residual_m=[float(np.linalg.norm(fa[s][0][:2] - fc[s][0][:2] - shift)) for s in range(2)])
    Cw[:, :2] += shift
    # floorfix (same 1 s rolling lower envelope as gvhmr_to_supermimic --floor_mode drift) on C's own frames, so
    # the hallucinated B legs of the whole-clip run do not leak into C's floor
    minz = np.array([min(x[2] for x in feet(r)) for r in Cw])
    pad = np.pad(minz, 15, mode="edge")
    env = np.array([pad[i : i + 30].min() for i in range(len(minz))])
    pad = np.pad(env, 15, mode="edge")
    Cw[:, 2] -= np.convolve(pad, np.ones(30) / 30, mode="valid")[: len(minz)]
    out[C] = Cw
    # GVHMR sometimes leaves a planted foot hovering (yaotou C start: left foot ~10 cm up in a horse stance): a foot
    # that stays put (< 0.15 m/s) for >= 8 frames while 2-12 cm up is put on the floor by leg IK, blended in/out
    FC = [feet(r) for r in out[C]]
    grounded = []
    for s_ in range(2):
        pz = np.array([f[s_][2] for f in FC])
        pxy = np.array([f[s_][0][:2] for f in FC])
        spd = np.r_[0, np.linalg.norm(np.diff(pxy, axis=0), axis=1) * 30]
        for a, b in runs((spd < 0.15) & (pz > 0.02) & (pz < 0.12)):
            if b - a < 7:
                continue
            a0 = 0 if a <= 6 else a - 4  # a run starting right after the cut is grounded from C's first frame
            for j in range(a0, min(b + 5, len(FC))):
                g = min(1.0, (j - (a0 - 1)) / 5 if a0 else 1.0, ((b + 5) - j) / 5)
                row = out[c1 + j]
                tgt = [(f[0], f[1]) for f in feet(row)]
                p = tgt[s_][0].copy()
                p[2] -= g * FC[j][s_][2]
                tgt[s_] = (p, tgt[s_][1])
                ik_legs(row, tgt)
            grounded.append((s_, a + c1 + cfg["start"], b + c1 + cfg["start"], round(float(pz[a : b + 1].max()), 3)))
    log["C_hovering_feet_grounded(foot,src from,to,max z)"] = grounded
    log["C_yaw_rot_deg"] = float(np.degrees(dC))
    log["C_start_vs_gvhmr_root_m"] = float(np.linalg.norm(Cw[0, :2] - Nw[c1, :2]))  # = GVHMR translation drift through B

    # ---- B: root
    nB = c1 - c0
    B = slice(c0, c1)

    def ends(src_a, src_c, f):
        xa, xc = f(src_a), f(src_c)
        return xa[-1], xa[-1] - xa[-2], xc[0], xc[1] - xc[0]

    def interp_with_residual(fa, fn):
        """Hermite between out's A end / C start states + N's own deviation from its Hermite."""
        p0, v0, p1, v1 = ends(out[c0 - 3 : c0], out[c1 : c1 + 2], fa)
        q0, u0, q1, u1 = ends(Nw[c0 - 3 : c0], Nw[c1 : c1 + 2], fn)
        return bridge(p0, v0, p1, v1, nB) + (fn(Nw[B]) - bridge(q0, u0, q1, u1, nB))

    out[B, :2] = interp_with_residual(lambda x: x[:, :2], lambda x: x[:, :2])
    # GVHMR's B translation is unreliable (person fills the frame): keep only part of its pelvis sway
    gain, lim = (0.3, 0.05) if cfg["feet"] == "fixed" else (0.5, 0.2)
    herm = bridge(out[c0 - 1, :2], out[c0 - 1, :2] - out[c0 - 2, :2], out[c1, :2], out[c1 + 1, :2] - out[c1, :2], nB)
    out[B, :2] = herm + np.clip(gain * (out[B, :2] - herm), -lim, lim)
    out[B, 2] = interp_with_residual(lambda x: x[:, 2:3], lambda x: x[:, 2:3])[:, 0]
    ya = np.unwrap(yaw(out[c0 - 3 : c1 + 2, 3:7]))
    yn = np.unwrap(yaw(Nw[c0 - 3 : c1 + 2, 3:7]))
    ya_end, ya_v = ya[2], ya[2] - ya[1]
    yc0 = ya[2] + wrap(yaw(out[c1, 3:7]) - ya[2])
    yc_v = wrap(yaw(out[c1 + 1, 3:7]) - yaw(out[c1, 3:7]))
    yn_end, yn_v = yn[2], yn[2] - yn[1]
    ync0 = yn[2] + wrap(yaw(Nw[c1, 3:7]) - yn[2])
    ync_v = wrap(yaw(Nw[c1 + 1, 3:7]) - yaw(Nw[c1, 3:7]))
    yB = bridge(ya_end, ya_v, yc0, yc_v, nB)[:, 0] + (np.unwrap(yaw(Nw[c0 - 1 : c1, 3:7]))[1:] - bridge(yn_end, yn_v, ync0, ync_v, nB)[:, 0])
    base = R.from_quat(out[B, 3:7])  # N tilt (+ faded seam offset)
    out[B, 3:7] = (rotz(yB - yaw(base.as_quat())) * base).as_quat()

    # ---- B: legs by IK
    fa_end = feet(out[c0 - 1])
    fc_beg = feet(out[c1])
    targets = [[None] * nB for _ in range(2)]
    if cfg["feet"] == "fixed":
        for s in range(2):
            sl = Slerp([0, 1], R.from_matrix([fa_end[s][1], fc_beg[s][1]]))
            for i in range(nB):
                u = (i + 1) / (nB + 1)
                u = u * u * (3 - 2 * u)
                p, Rm = (1 - u) * fa_end[s][0] + u * fc_beg[s][0], sl(u).as_matrix()
                g = min(1.0, (i + 1) / 8, (nB - i) / 8)  # sole onto the floor away from the two boundary frames
                p[2] += g * (-sole_offset(Rm, s) - p[2])
                targets[s][i] = (p, Rm)
        log["B_feet"] = "both planted; A-end -> C-start drift " + ", ".join(f"{np.linalg.norm(fa_end[s][0][:2] - fc_beg[s][0][:2]) * 100:.1f} cm" for s in range(2))
    else:
        idx = np.arange(c0 - 1, c1 + 1)  # include the fixed boundary rows
        raw = [[None] * len(idx) for _ in range(2)]
        nz = np.zeros((2, len(idx)))
        for j, t in enumerate(idx):
            fN = feet(N[t])
            setq(N[t])
            rootN_p, rootN_R = d.xpos[PELVIS].copy(), d.xmat[PELVIS].reshape(3, 3).copy()
            setq(out[t])
            ro_p, ro_R = d.xpos[PELVIS].copy(), d.xmat[PELVIS].reshape(3, 3).copy()
            for s in range(2):
                lp = rootN_R.T @ (fN[s][0] - rootN_p)
                lR = rootN_R.T @ fN[s][1]
                raw[s][j] = (ro_p + ro_R @ lp, ro_R @ lR)
                nz[s, j] = fN[s][2]
        # boundary offsets (raw at the boundary rows uses N's legs, the real rows use out's legs)
        for s in range(2):
            act = [fa_end[s], fc_beg[s]]
            op = [act[0][0] - raw[s][0][0], act[1][0] - raw[s][-1][0]]
            orr = [R.from_matrix(act[0][1] @ raw[s][0][1].T).as_rotvec(), R.from_matrix(act[1][1] @ raw[s][-1][1].T).as_rotvec()]
            for j in range(len(idx)):
                u = j / (len(idx) - 1)
                raw[s][j] = (raw[s][j][0] + (1 - u) * op[0] + u * op[1], R.from_rotvec((1 - u) * orr[0] + u * orr[1]).as_matrix() @ raw[s][j][1])
            raw[s][0], raw[s][-1] = (act[0][0], act[0][1]), (act[1][0], act[1][1])
        contact_log = []
        for s in range(2):
            rp = np.array([r[0] for r in raw[s]])
            rsole = rp[:, 2] + np.array([sole_offset(r[1], s) for r in raw[s]])
            rspd = np.r_[0, np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * 30]
            con = (nz[s] < 0.035) | ((rsole < 0.04) & (rspd < 0.8))
            con[0], con[-1] = fa_end[s][2] < 0.035, fc_beg[s][2] < 0.035
            for a, b in zip(runs(con)[:-1], runs(con)[1:]):  # a foot does not leave the floor for <= 3 frames
                if b[0] - a[1] <= 4:
                    con[a[1] : b[0]] = True
            rr = [r for r in runs(con) if r[1] - r[0] >= 2 or r[0] == 0 or r[1] == len(idx) - 1]
            last = len(idx) - 1
            locks = []
            for a, b in rr:
                if a == 0:
                    p, Rm = raw[s][0]
                elif b == last:
                    p, Rm = raw[s][-1]
                else:
                    Rm = rotz(yaw(R.from_matrix(raw[s][(a + b) // 2][1]).as_quat())).as_matrix()  # flat foot
                    p = np.mean([raw[s][j][0] for j in range(a, b + 1)], 0)
                    p[2] = -sole_offset(Rm, s)
                locks.append([a, b, p, Rm])
            # a swing may not need more than 2.5 m/s on average: lengthen it by shortening the neighbouring contacts
            for k in range(len(locks) - 1):
                l0, l1 = locks[k], locks[k + 1]
                need = int(np.ceil(np.linalg.norm(l1[2][:2] - l0[2][:2]) / (2.5 / 30)))
                while l1[0] - l0[1] - 1 < need:
                    can0 = l0[1] - l0[0] >= 2 and l0[1] > 0
                    can1 = l1[1] - l1[0] >= 2 and l1[0] < last
                    if can0 and (l0[1] - l0[0] >= l1[1] - l1[0] or not can1):
                        l0[1] -= 1
                    elif can1:
                        l1[0] += 1
                    else:
                        break
            lock = {}
            for a, b, p, Rm in locks:
                for j in range(a, b + 1):
                    lock[j] = (p, Rm)
                contact_log.append((s, int(idx[a] + cfg["start"]), int(idx[b] + cfg["start"])))
            # swings: raw shifted by an offset that blends from the previous lock to the next lock
            tgt = [None] * len(idx)
            j = 0
            while j < len(idx):
                if j in lock:
                    tgt[j] = lock[j]
                    j += 1
                    continue
                a = j
                while j < len(idx) and j not in lock:
                    j += 1
                b = j - 1
                pa = lock[a - 1][0] - raw[s][a - 1][0] if a - 1 in lock else np.zeros(3)
                pb = lock[b + 1][0] - raw[s][b + 1][0] if b + 1 in lock else pa
                ra = R.from_matrix(lock[a - 1][1] @ raw[s][a - 1][1].T).as_rotvec() if a - 1 in lock else np.zeros(3)
                rb = R.from_matrix(lock[b + 1][1] @ raw[s][b + 1][1].T).as_rotvec() if b + 1 in lock else ra
                for jj in range(a, b + 1):
                    u = (jj - a + 1) / (b - a + 2)
                    u = u * u * (3 - 2 * u)
                    p = raw[s][jj][0] + (1 - u) * pa + u * pb
                    Rm = R.from_rotvec((1 - u) * ra + u * rb).as_matrix() @ raw[s][jj][1]
                    so = p[2] + sole_offset(Rm, s)
                    p = p.copy()
                    clear = 0.05 * np.sin(np.pi * (jj - a + 1) / (b - a + 2))  # lift-off / touch-down clearance
                    p[2] -= min(so - clear, 0.0) + max(so - 0.20, 0.0)  # sole above the floor, lift capped at 20 cm
                    tgt[jj] = (p, Rm)
            for i in range(nB):
                targets[s][i] = tgt[i + 1]
        log["B_contacts(src frames)"] = contact_log

    def solve_B():
        errs = np.zeros((nB, 2, 3))  # achieved - target, per foot
        for i, t in enumerate(range(c0, c1)):
            # two starts: the previous solved frame and N's (hallucinated) legs; keep the better one
            best = None
            for init in (out[t - 1, 7:19], N[t, 7:19]):
                row = out[t].copy()
                row[7:19] = init
                e_ = ik_legs(row, [targets[0][i], targets[1][i]], iters=150)
                if best is None or e_ < best[0] - 1e-4:
                    best = (e_, row)
            out[t] = best[1]
            f = feet(out[t])
            errs[i] = [f[s][0] - targets[s][i][0] for s in range(2)]
        return errs

    # where a leg cannot reach, move the pelvis toward its foot target (smoothed in time, zero at both ends)
    taper = np.minimum(1.0, np.minimum(np.arange(1, nB + 1), np.arange(nB, 0, -1)) / 6.0)[:, None]
    for _ in range(4):
        errs = solve_B()
        n_err = np.linalg.norm(errs, axis=2)
        if n_err.max() < 0.01:
            break
        corr = -(errs * (n_err > 0.005)[..., None]).sum(1) / np.maximum(1, (n_err > 0.005).sum(1))[:, None]
        out[B, :3] += 0.8 * gaussian_filter1d(corr, 2.0, axis=0, mode="nearest") * taper
    errs = solve_B()
    worst = np.linalg.norm(errs, axis=2).max()
    log["B_ik_worst_foot_err_m"] = round(float(worst), 4)

    # ---- de-spike new rows (IK branch flips / GVHMR arm flips) and smooth B legs/root lightly
    for _ in range(20):
        v = np.abs(np.diff(out[:, 7:], axis=0)) * 30
        v[: L - 1] = 0
        bad = np.argwhere(v > 12.0)
        if not len(bad):
            break
        for i, j in bad:
            a, b = max(i - 4, L - 1), min(i + 5, T - 1)
            if b - a < 2:
                continue
            p0, p1 = out[a, 7 + j], out[b, 7 + j]
            v0 = out[a, 7 + j] - out[a - 1, 7 + j]
            v1 = out[min(b + 1, T - 1), 7 + j] - out[b, 7 + j]
            out[a + 1 : b, 7 + j] = hermite(np.array([p0]), np.array([v0]), np.array([p1]), np.array([v1]), b - a - 1)[:, 0]
    log["despike_left_rad_s"] = float((np.abs(np.diff(out[L - 1 :, 7:], axis=0)) * 30).max())

    # ---- stance: first row from which the pose stays within 0.12 rad (wrists ignored, GVHMR jitters them) / 3 cm
    # of the final pose; then 15 held rows (the source ends at T-1, so the stance is pulled to <= T-16 if needed)
    nw = np.r_[0:19, 22:26]
    fin_q, fin_p = out[-8:, 7:].mean(0), out[-8:, :2].mean(0)
    near = (np.abs(out[:, 7:][:, nw] - fin_q[nw]).max(1) < 0.12) & (np.linalg.norm(out[:, :2] - fin_p, axis=1) < 0.03)
    st = T - 1
    while st > c1 and near[st - 1]:
        st -= 1
    log["stance_detected_src_frame"] = st + cfg["start"]
    st = min(st, T - 16)
    end = st + 15
    hold = out[st : end + 1].copy()
    H = hold.mean(0)
    H[3:7] = R.from_quat(hold[:, 3:7]).mean().as_quat()
    f = feet(H)
    H[2] -= min(f[0][2], f[1][2])
    ramp = 10
    for t in range(st - ramp, end + 1):
        u = np.clip((t - (st - ramp)) / ramp, 0, 1)
        u = u * u * (3 - 2 * u)
        out[t, :3] = (1 - u) * out[t, :3] + u * H[:3]
        out[t, 7:] = (1 - u) * out[t, 7:] + u * H[7:]
        out[t, 3:7] = Slerp([0, 1], R.from_quat([out[t, 3:7], H[3:7]]))(u).as_quat()
    out = out[: end + 1]
    log["stance_src_frame"] = st + cfg["start"]
    log["end_src_frame"] = end + cfg["start"]
    log["final_feet_sole_z"] = [round(float(x[2]), 4) for x in feet(out[-1])]

    # ---- limits / checks / save
    out[L:, 7:] = np.clip(out[L:, 7:], LO, HI)
    assert np.abs(out[:L] - old).max() <= 1e-6
    qd = np.abs(np.diff(out[:, 7:], axis=0)) * 30
    rv = np.linalg.norm(np.diff(out[:, :3], axis=0), axis=1) * 30
    S = cfg["start"]
    log["checks"] = dict(
        rows=len(out),
        new_rows=len(out) - L,
        new_src=(L + S, len(out) - 1 + S),
        A_tail_src=(L + S, c0 - 1 + S),
        B_src=(c0 + S, c1 - 1 + S),
        C_src=(c1 + S, len(out) - 1 + S),
        max_qd_new=float(qd[L - 1 :].max()),
        max_qd_A_old=float(qd[: L - 1].max()),
        qd_at_seams={f"{nm}({r + S - 1}->{r + S})": float(qd[r - 3 : r + 2].max()) for nm, r in (("A_old->new", L), ("A->B", c0), ("B->C", c1))},
        root_step_at_seams_m={f"{nm}": float(np.linalg.norm(out[r, :3] - out[r - 1, :3])) for nm, r in (("A_old->new", L), ("A->B", c0), ("B->C", c1))},
        max_root_speed_new=float(rv[L - 1 :].max()),
        last15_max_qd=float(qd[-15:].max()),
        joint_limit_violations=int(((out[:, 7:] < LO - 1e-6) | (out[:, 7:] > HI + 1e-6)).sum()),
    )
    # foot slip of planted feet in B (sole < 1 cm both consecutive frames)
    slip = []
    prev = feet(out[c0 - 1])
    for t in range(c0, c1 + 1):
        cur = feet(out[t])
        for s in range(2):
            if cur[s][2] < 0.01 and prev[s][2] < 0.01:
                slip.append(np.linalg.norm(cur[s][0][:2] - prev[s][0][:2]) * 30)
        prev = cur
    sole = np.array([[x[2] for x in feet(r)] for r in out[c0 : c1 + 1]])
    log["B_feet"] = log.get("B_feet", "") + f" | planted-foot slip max {max(slip) if slip else 0:.3f} m/s, sole z min {sole.min():+.3f} max {sole.max():.3f}"
    g1_model.save_csv(cfg["out"], out)
    # fmt rounding must not touch the kept rows (they were written with the same %.6f)
    chk = np.loadtxt(cfg["out"], delimiter=",", ndmin=2)
    assert chk.shape[1] == g1_model.CSV_COLS and np.abs(chk[:L] - old).max() <= 1e-6
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old_csv", required=True, help="existing CSV of shot A (rows kept bit-exact)")
    ap.add_argument("--whole_csv", required=True, help="drift-floorfix CSV of the whole A+B+C clip (same start frame)")
    ap.add_argument("--gvhmr_pred", required=True, help="hmr4d_results.pt of the whole-clip GVHMR run")
    ap.add_argument("--cuts", type=int, nargs=2, required=True, metavar=("C0", "C1"), help="first row of shot B and of shot C")
    ap.add_argument("--feet", choices=("fixed", "steps"), required=True, help="B legs: both feet planted, or N's stepping rhythm")
    ap.add_argument("--place", choices=("cam", "feet"), default="feet", help="place C by camera displacement or by matching planted feet")
    ap.add_argument("--human_height", type=float, default=None, help="m, as printed by gvhmr_to_supermimic.py (needed for --place cam)")
    ap.add_argument("--src_offset", type=int, default=0, help="source-video frame of row 0 (only used in the log)")
    ap.add_argument("--robot_xml", default=None, help="default: $G1_XML, else mjlab's g1.xml")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--log", default=None, help="JSON log (default: <out stem>_log.json)")
    a = ap.parse_args()
    init_model(a.robot_xml)
    cfg = dict(
        old_csv=a.old_csv, whole_csv=a.whole_csv, gvhmr_pred=a.gvhmr_pred, cuts=tuple(a.cuts), feet=a.feet,
        place=a.place, height=a.human_height, start=a.src_offset, out=a.out,
    )
    lg = build(cfg)
    js = json.dumps(lg, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(js)
    log_path = a.log or os.path.splitext(a.out)[0] + "_log.json"
    with open(log_path, "w") as f:
        f.write(js)
    print("->", a.out, log_path)


if __name__ == "__main__":
    main()
