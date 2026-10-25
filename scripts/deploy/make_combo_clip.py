"""Concatenate BONES-SEED clips into ONE continuous reference (same 50 Hz npz format) for a single policy run.

  PYTHONPATH= .venv/bin/python scripts/deploy/make_combo_clip.py \
      --clips itching_left_arm_R_003__A121 shoulder_clap_004__A041_M walk_forward_stop_001__A026 walk_sideway_090_stop_003__A037 \
      --name combo_itch_clap_fwd_side

Construction (all in float64, stored as float32 like the converter):
  * clip k+1 is moved rigidly (rotation about world z + xy translation) so that its frame-0 pelvis xy and heading yaw
    coincide with the previous clip's last frame (pelvis height untouched);
  * between two clips a transition of --blend-s seconds is inserted (blend_s * fps - 1 new frames, so the last frame of
    A and frame 0 of B are blend_s apart): with s = (1 - cos(pi u)) / 2, u in (0, 1), joints q_A + s (q_B - q_A),
    pelvis position lerp(p_A, p_B, s), pelvis orientation slerp(q_A, q_B, s) (shortest path);
  * --end-settle-s > 0 appends a "settle" segment after the last clip: every joint, the pelvis position and the pelvis
    orientation keep moving with the last frame's velocity v, decaying as v (1 - u)^2 (u = 0 -> 1 over end_settle_s),
    so the reference comes to rest (zero velocity at the end) instead of stopping on a moving frame; joints are
    clipped to the soft range;
  * --fade-out-s > 0 with --fade-clips: those clips (mocap cut mid-motion, e.g. still shaking) get a "fade" segment
    appended before the next transition: the upper body keeps its last oscillation period playing with an amplitude
    fading to zero (position and velocity continuous at the seam), legs and pelvis decelerate within 0.2 s and hold
    (see fade_extend);
  * every velocity is recomputed on the whole sequence with the converter's conventions
    (supermimic.data.motion_convert: np.gradient for joint and pelvis linear velocity, central SO(3) difference in the
    world frame for the pelvis angular velocity) and every body state comes from forward_kinematics (CPU MuJoCo, G1).

Writes <out-dir>/<name>.npz (keys of the BONES-SEED npz files + segment_* / combo_* metadata) and <out-dir>/<name>.json
(construction parameters and the verification numbers). The source npz files are only read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    ResampledMotion,
    _so3_derivative,
    forward_kinematics,
    load_g1_model,
    quat_conjugate,
    quat_mul,
    rotvec_from_quat,
)

KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


# ----------------------------------------------------------------------------- small helpers (wxyz quaternions)


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_z(a: float) -> np.ndarray:
    return np.array([np.cos(0.5 * a), 0.0, 0.0, np.sin(0.5 * a)])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def slerp(q0: np.ndarray, q1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Shortest-path slerp from q0 to q1 (single quaternions) at the fractions s (N,) -> (N, 4)."""
    q1 = q1 if np.dot(q0, q1) >= 0 else -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    th = np.arccos(dot)
    s = s[:, None]
    if th < 1e-8:
        out = (1 - s) * q0 + s * q1
    else:
        out = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two quaternion arrays (sign-invariant)."""
    d = np.abs(np.sum(a * b, axis=-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def frame_jumps(x: np.ndarray, key: str) -> np.ndarray:
    """(T-1,) per-frame jump: joints max |dx|; 3-vectors max over bodies of |dx| (norm); quaternions max relative angle."""
    if key == "body_quat_w":
        return quat_angle(x[1:], x[:-1]).max(-1)
    d = np.diff(x, axis=0)
    if key.startswith("joint"):
        return np.abs(d).max(-1)
    return np.linalg.norm(d, axis=-1).max(-1)


def _quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-12, rv / np.maximum(ang, 1e-12), 0.0)
    return np.concatenate([np.cos(0.5 * ang), np.sin(0.5 * ang) * axis], -1)


def fade_extend(p: np.ndarray, q: np.ndarray, jp: np.ndarray, n: int, fps: float, lo: np.ndarray, hi: np.ndarray,
                decay_s: float = 0.2, upper_from: int = 12):
    """n frames continuing a clip that ends mid-motion.

    Upper body (joints >= upper_from: waist + arms): the last oscillation period (autocorrelation of the joint velocity
    over the last 1.5 s, lags 0.3-0.9 s), made into a seamless loop (linear ramp removing its endpoint mismatch), replays
    around its mean with the amplitude fading 1 -> 0 (half cosine); a cubic Hermite correction over ~1/3 period absorbs
    what is left of the position / velocity mismatch at the seam. If no period is found the
    upper body uses the decay below. Legs, pelvis position and pelvis orientation: the last velocity decays to zero as
    v (1 - u)^2 over decay_s, then hold (no long extrapolation: a leg in the air must not swing further).
    """
    qe = q[-1]
    qa = np.where((np.sum(q * qe, -1) < 0)[:, None], -q, q)
    rv = rotvec_from_quat(quat_mul(qa, quat_conjugate(np.broadcast_to(qe, qa.shape))))  # world-left, 0 at the end
    k = np.arange(1, n + 1)
    nd = max(2, int(round(decay_s * fps)))
    ud = np.minimum(k / nd, 1.0)[:, None]

    def decay(x):
        return x[-1] + (x[-1] - x[-2]) * nd * (1 - (1 - ud) ** 3) / 3

    W = min(len(jp), int(round(1.5 * fps)))
    v = np.diff(jp[-W:, upper_from:], axis=0)
    best, P = -1.0, 0
    for L in range(int(0.3 * fps), min(int(0.9 * fps), len(v) - 10) + 1):
        a, b = v[:-L], v[L:]
        c = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-12))
        if c > best:
            best, P = c, L
    ej = decay(jp)
    if best >= 0.3:
        x = jp[:, upper_from:]
        # seamless loop: the last period x[T-P..T-1] plus a linear ramp that removes its endpoint mismatch, so the
        # loop starts exactly where the clip's next frame would be and every wrap steps like an ordinary frame
        D = (2 * x[-1] - x[-2]) - x[-P]
        loop = x[-P:] + D * (1 - np.arange(P) / P)[:, None]
        m = loop.mean(0)
        d = (0.5 + 0.5 * np.cos(np.pi * k / n))[:, None]  # amplitude 1 -> 0, zero slope at both ends
        raw = m + d * (loop[(k - 1) % P] - m)
        v_end = x[-1] - x[-2]
        e_p = (x[-1] + v_end) - raw[0]
        e_v = v_end - (raw[1] - raw[0])
        nc = max(4, P // 3)
        sc = np.clip((k - 1) / nc, 0, 1)[:, None]
        h00, h10 = 2 * sc**3 - 3 * sc**2 + 1, sc**3 - 2 * sc**2 + sc
        ej[:, upper_from:] = raw + e_p * h00 + e_v * nc * h10
    ej = np.clip(ej, np.minimum(lo, jp[-1]), np.maximum(hi, jp[-1]))
    ep, erv = decay(p), decay(rv)
    eq = quat_mul(_quat_from_rotvec(erv), np.broadcast_to(qe, (n, 4)))
    return ep, eq, ej, {"period_frames": P if best >= 0.3 else None, "autocorr": best}


def load_clip(path: Path) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    assert tuple(str(n) for n in out["joint_names"]) == G1_JOINT_NAMES, f"{path}: joint order"
    assert abs(float(out["fps"][0]) - 50.0) < 1e-6, f"{path}: fps {out['fps']}"
    return out


# ----------------------------------------------------------------------------- main


def main(
    clips: tuple[str, ...],
    name: str,
    npz_dir: Path = Path("data/npz/bones_seed"),
    out_dir: Path = Path("data/npz/bones_seed_combo"),
    blend_s: float = 1.0,
    soft_factor: float = 0.9,
    overwrite: bool = False,
    end_settle_s: float = 0.0,
    last_blend_s: float | None = None,
    fade_out_s: float = 0.0,
    fade_clips: tuple[str, ...] = (),
) -> None:
    """clips: ordered clip names in npz_dir; name: output stem; blend_s: transition duration between two clips [s];
    last_blend_s: duration of the last transition only (default blend_s)."""
    assert len(clips) >= 2, "need at least two clips"
    out_npz, out_json = out_dir / f"{name}.npz", out_dir / f"{name}.json"
    if out_npz.exists() and not overwrite:
        raise SystemExit(f"{out_npz} exists (--overwrite to replace)")
    if out_npz.resolve().parent == npz_dir.resolve():
        raise SystemExit("out-dir must differ from npz-dir (never write next to the source clips)")
    model = load_g1_model()
    src = [load_clip(npz_dir / f"{c}.npz") for c in clips]
    body_names = [str(n) for n in src[0]["body_names"]]
    names_fk = [model.body(i).name for i in range(1, model.nbody)]
    assert body_names == names_fk, ("npz body order != MJCF body order", body_names, names_fk)
    for s in src:
        assert [str(n) for n in s["body_names"]] == body_names
    r = body_names.index("pelvis")
    fps = 50.0
    dt = 1.0 / fps
    n_blend = int(round(blend_s * fps))
    assert n_blend >= 2, "blend too short"

    rng = model.jnt_range[1:]
    mid, half = rng.mean(1), 0.5 * (rng[:, 1] - rng[:, 0])
    lo, hi = mid - soft_factor * half, mid + soft_factor * half
    n_fade = int(round(fade_out_s * fps))
    fades = {}

    # ---------------------------------------------------------------- assemble pelvis pose + joints
    root_pos, root_quat, joint_pos = [], [], []
    segments: list[tuple[str, str, int, int]] = []  # (name, kind, start, end exclusive)
    xforms = []  # per clip: (yaw offset, source frame-0 xy, target xy)
    t = 0
    for k, (cname, s) in enumerate(zip(clips, src)):
        p = s["body_pos_w"][:, r].astype(np.float64)
        q = s["body_quat_w"][:, r].astype(np.float64)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        jp = s["joint_pos"].astype(np.float64)
        if k == 0:
            dpsi, src_xy, dst_xy = 0.0, np.zeros(2), np.zeros(2)
        else:
            pa, qa, ja = root_pos[-1][-1], root_quat[-1][-1], joint_pos[-1][-1]
            dpsi = float(yaw_of(qa) - yaw_of(q[0]))
            src_xy, dst_xy = p[0, :2].copy(), pa[:2].copy()
            R = rot_z(dpsi)
            off = np.array([src_xy[0], src_xy[1], 0.0])
            p = (p - off) @ R.T + np.array([dst_xy[0], dst_xy[1], 0.0])
            q = quat_mul(quat_z(dpsi)[None], q)
            if np.dot(q[0], qa) < 0:
                q = -q
            # transition A(last) -> B(0)
            nb = int(round(last_blend_s * fps)) if (last_blend_s is not None and k == len(clips) - 1) else n_blend
            u = np.arange(1, nb) / nb
            sw = 0.5 - 0.5 * np.cos(np.pi * u)
            tj = ja + sw[:, None] * (jp[0] - ja)
            tp = pa + sw[:, None] * (p[0] - pa)
            tq = slerp(qa, q[0], sw)
            segments.append((f"T{k}: {clips[k - 1]} -> {cname}", "transition", t, t + len(u)))
            root_pos.append(tp), root_quat.append(tq), joint_pos.append(tj)  # noqa: E702
            t += len(u)
        xforms.append({"clip": cname, "yaw_offset_deg": float(np.degrees(dpsi)), "src_frame0_xy": src_xy.tolist(), "dst_xy": dst_xy.tolist()})
        segments.append((cname, "clip", t, t + len(p)))
        root_pos.append(p), root_quat.append(q), joint_pos.append(jp)  # noqa: E702
        t += len(p)
        if cname in fade_clips and n_fade > 0:
            fp_, fq_, fj_, info = fade_extend(p, q, jp, n_fade, fps, lo, hi)
            fades[cname] = info
            segments.append((f"{cname} fade-out", "fade", t, t + n_fade))
            root_pos.append(fp_), root_quat.append(fq_), joint_pos.append(fj_)  # noqa: E702
            t += n_fade
    root_pos, root_quat, joint_pos = np.concatenate(root_pos), np.concatenate(root_quat), np.concatenate(joint_pos)
    n_settle = int(round(end_settle_s * fps))
    if n_settle > 0:
        # displacement integral of v (1 - u)^2 over [0, u] * end_settle_s; last-frame velocity by backward difference
        u = np.arange(1, n_settle + 1) / n_settle
        f = (end_settle_s * (1.0 - (1.0 - u) ** 3) / 3.0)[:, None]
        vj = (joint_pos[-1] - joint_pos[-2]) / dt
        vp = (root_pos[-1] - root_pos[-2]) / dt
        w = rotvec_from_quat(quat_mul(root_quat[-1], quat_conjugate(root_quat[-2]))) / dt
        sj = np.clip(joint_pos[-1] + f * vj, np.minimum(lo, joint_pos[-1]), np.maximum(hi, joint_pos[-1]))
        sp = root_pos[-1] + f * vp
        rv = f * w  # world-frame rotation vector, applied on the left
        ang = np.linalg.norm(rv, axis=-1, keepdims=True)
        axis = np.where(ang > 1e-12, rv / np.maximum(ang, 1e-12), 0.0)
        dq = np.concatenate([np.cos(0.5 * ang), np.sin(0.5 * ang) * axis], -1)
        sq = quat_mul(dq, np.broadcast_to(root_quat[-1], dq.shape))
        segments.append(("settle", "settle", len(root_pos), len(root_pos) + n_settle))
        root_pos, root_quat, joint_pos = np.concatenate([root_pos, sp]), np.concatenate([root_quat, sq]), np.concatenate([joint_pos, sj])
    T = len(root_pos)

    # ---------------------------------------------------------------- velocities (converter conventions) + FK
    motion = ResampledMotion(
        fps=fps, root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos,
        root_lin_vel_w=np.gradient(root_pos, dt, axis=0), root_ang_vel_w=_so3_derivative(root_quat, dt),
        joint_vel=np.gradient(joint_pos, dt, axis=0),
    )
    bodies = forward_kinematics(model, motion)
    assert bodies.body_names == body_names
    combo = {
        "joint_pos": motion.joint_pos, "joint_vel": motion.joint_vel, "body_pos_w": bodies.pos_w, "body_quat_w": bodies.quat_w,
        "body_lin_vel_w": bodies.lin_vel_w, "body_ang_vel_w": bodies.ang_vel_w,
    }
    combo32 = {k: v.astype(np.float32) for k, v in combo.items()}

    # ---------------------------------------------------------------- verification
    rep: dict = {"name": name, "clips": list(clips), "blend_s": blend_s, "last_blend_s": last_blend_s, "fade_out_s": fade_out_s, "fade_clips": list(fade_clips), "end_settle_s": end_settle_s, "fps": fps, "frames": T, "duration_s": (T - 1) / fps,
                 "transforms": xforms, "segments": [{"name": n, "kind": kd, "start": a, "end": b} for n, kd, a, b in segments]}

    # (a) FK / rigid-transform consistency on the original clip frames (interior: velocities unchanged there)
    fk_err = []
    for (cname, kind, a, b), s, xf in zip([sg for sg in segments if sg[1] == "clip"], src, xforms):
        R = rot_z(np.radians(xf["yaw_offset_deg"]))
        off, dst = np.array([*xf["src_frame0_xy"], 0.0]), np.array([*xf["dst_xy"], 0.0])
        qz = quat_z(np.radians(xf["yaw_offset_deg"]))
        exp = {
            "joint_pos": s["joint_pos"], "joint_vel": s["joint_vel"],
            "body_pos_w": (s["body_pos_w"].astype(np.float64) - off) @ R.T + dst,
            "body_lin_vel_w": s["body_lin_vel_w"].astype(np.float64) @ R.T, "body_ang_vel_w": s["body_ang_vel_w"].astype(np.float64) @ R.T,
        }
        e = {}
        for key, v in exp.items():
            d = np.abs(combo[key][a:b] - v)
            e[key + "_interior"] = float(d[1:-1].max())
            e[key + "_endframes"] = float(max(d[0].max(), d[-1].max()))
        qe = quat_angle(combo["body_quat_w"][a:b], quat_mul(qz[None, None], s["body_quat_w"].astype(np.float64)))
        e["body_quat_w_angle_rad"] = float(qe.max())
        fk_err.append({"clip": cname, **e})
    rep["fk_vs_original"] = fk_err

    # (b) soft joint range (training: mid +- soft_factor * half range)
    exc = np.maximum(lo - joint_pos, joint_pos - hi)  # > 0: outside
    soft = []
    for n, kind, a, b in segments:
        e = exc[a:b]
        bad = np.nonzero((e > 0).any(0))[0]
        soft.append({"segment": n, "frames_outside": int((e > 0).any(1).sum()), "max_excess_rad": float(max(e.max(), 0.0)),
                     "joints": [G1_JOINT_NAMES[j] for j in bad]})
    rep["soft_range"] = soft

    # (c) continuity: per-frame jumps inside the original clips vs at the seams / inside the transitions
    kind_of = np.empty(T, object)
    seg_of = np.empty(T, int)
    for i, (n, kind, a, b) in enumerate(segments):
        kind_of[a:b], seg_of[a:b] = kind, i
    cont = {}
    for key in KEYS:
        orig = max(float(frame_jumps(s[key].astype(np.float64), key).max()) for s in src)
        j = frame_jumps(combo[key], key)
        inside = (kind_of[:-1] == "clip") & (seg_of[:-1] == seg_of[1:])
        # transition windows incl. both seams (last clip frame -> first inserted ... last inserted -> next clip frame 0)
        trans = {}
        for i, (n, kind, a, b) in enumerate(segments):
            if kind != "transition":
                continue
            w = j[a - 1 : b]
            trans[n.split(":")[0]] = {"max": float(w.max()), "seam_in": float(j[a - 1]), "seam_out": float(j[b - 1]), "ratio_vs_orig_max": float(w.max() / orig)}
        cont[key] = {"orig_clips_max": orig, "combo_inside_clips_max": float(j[inside].max()), "transitions": trans}
    rep["continuity"] = cont

    # (d) transition kinematics: joint / pelvis speeds, feet
    feet = [body_names.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    tk = []
    for n, kind, a, b in segments:
        if kind != "transition":
            continue
        w = slice(a - 1, b + 1)  # incl. both end frames
        jv = np.abs(motion.joint_vel[w])
        ja = np.abs(np.gradient(motion.joint_vel, dt, axis=0)[w])
        fp = bodies.pos_w[w][:, feet]
        tk.append({
            "segment": n, "frames": b - a, "max_joint_speed_rad_s": float(jv.max()), "joint": G1_JOINT_NAMES[int(np.argmax(jv.max(0)))],
            "max_joint_acc_rad_s2": float(ja.max()),
            "max_pelvis_lin_speed_m_s": float(np.linalg.norm(motion.root_lin_vel_w[w], axis=-1).max()),
            "max_pelvis_ang_speed_rad_s": float(np.linalg.norm(motion.root_ang_vel_w[w], axis=-1).max()),
            "pelvis_dz_m": float(root_pos[b, 2] - root_pos[a - 1, 2]),
            "pelvis_tilt_change_deg": float(np.degrees(quat_angle(root_quat[b], root_quat[a - 1]))),
            "max_abs_joint_change_rad": float(np.abs(joint_pos[b] - joint_pos[a - 1]).max()),
            "joint_change_joint": G1_JOINT_NAMES[int(np.argmax(np.abs(joint_pos[b] - joint_pos[a - 1])))],
            "foot_xy_slide_m": [float(np.linalg.norm(fp[:, i, :2] - fp[0, i, :2], axis=-1).max()) for i in range(2)],
            "foot_z_range_m": [[float(fp[:, i, 2].min()), float(fp[:, i, 2].max())] for i in range(2)],
            "orig_end_speeds": {"A_last_max_joint_speed": float(np.abs(motion.joint_vel[a - 1]).max()), "B_first_max_joint_speed": float(np.abs(motion.joint_vel[b]).max())},
        })
    rep["transitions"] = tk
    fk = []
    for n, kind, a, b in segments:
        if kind != "fade":
            continue
        w = slice(a - 1, b + 1)
        fp = bodies.pos_w[w][:, feet]
        fk.append({"segment": n, **fades[n.split(" fade-out")[0]],
                   "max_joint_speed_start_rad_s": float(np.abs(motion.joint_vel[a - 1]).max()),
                   "max_joint_speed_end_rad_s": float(np.abs(motion.joint_vel[b - 1]).max()),
                   "seam_joint_vel_jump": float(np.abs(motion.joint_vel[a] - motion.joint_vel[a - 1]).max()),
                   "max_joint_acc_rad_s2": float(np.abs(np.gradient(motion.joint_vel, dt, axis=0)[w]).max()),
                   "pelvis_move_m": float(np.linalg.norm(root_pos[b - 1] - root_pos[a - 1])),
                   "foot_xy_slide_m": [float(np.linalg.norm(fp[:, i, :2] - fp[0, i, :2], axis=-1).max()) for i in range(2)],
                   "foot_z_range_m": [[float(fp[:, i, 2].min()), float(fp[:, i, 2].max())] for i in range(2)]})
    rep["fades"] = fk
    if n_settle > 0:
        a = T - n_settle
        fp = bodies.pos_w[a - 1:][:, feet]
        rep["settle"] = {
            "frames": n_settle, "max_joint_speed_start_rad_s": float(np.abs(motion.joint_vel[a - 1]).max()),
            "max_joint_speed_end_rad_s": float(np.abs(motion.joint_vel[-1]).max()),
            "max_joint_acc_rad_s2": float(np.abs(np.gradient(motion.joint_vel, dt, axis=0)[a - 1:]).max()),
            "pelvis_move_m": float(np.linalg.norm(root_pos[-1] - root_pos[a - 1])),
            "pelvis_rot_deg": float(np.degrees(quat_angle(root_quat[-1], root_quat[a - 1]))),
            "foot_xy_slide_m": [float(np.linalg.norm(fp[:, i, :2] - fp[0, i, :2], axis=-1).max()) for i in range(2)],
            "foot_z_end_m": [float(z) for z in fp[-1, :, 2]],
        }
    fz = bodies.pos_w[:, feet, 2]
    rep["foot_z_min_in_clips_m"] = float(fz[kind_of == "clip"].min())
    rep["max_joint_speed_in_clips_rad_s"] = float(np.abs(motion.joint_vel[kind_of == "clip"]).max())
    rep["max_pelvis_speed_in_clips_m_s"] = float(np.linalg.norm(motion.root_lin_vel_w[kind_of == "clip"], axis=-1).max())

    # ---------------------------------------------------------------- write
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        fps=np.array([fps], dtype=np.float32),
        **combo32,
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(body_names),
        source=np.array("combo(" + " + ".join(str(s["source"]) for s in src) + f"; blend {blend_s} s)"),
        input_fps=np.array([fps], dtype=np.float32),  # built on the 50 Hz grid of the source npz files
        segment_names=np.array([n for n, *_ in segments]),
        segment_kind=np.array([kd for _, kd, *_ in segments]),
        segment_start=np.array([a for _, _, a, _ in segments], dtype=np.int64),
        segment_end=np.array([b for *_, b in segments], dtype=np.int64),
        combo_clips=np.array(clips),
        combo_blend_s=np.array([blend_s], dtype=np.float32),
        combo_yaw_offset_deg=np.array([x["yaw_offset_deg"] for x in xforms], dtype=np.float64),
    )
    out_json.write_text(json.dumps(rep, indent=1))
    # ---------------------------------------------------------------- print
    print(f"wrote {out_npz}: {T} frames ({(T - 1) / fps:.2f} s); {out_json}")
    for n, kind, a, b in segments:
        print(f"  [{a:5d},{b:5d}) {kind:10s} {n}")
    for x in xforms:
        print(f"  transform {x['clip'][:36]:36s} yaw {x['yaw_offset_deg']:+8.2f} deg, frame-0 xy {np.round(x['src_frame0_xy'], 3)} -> {np.round(x['dst_xy'], 3)}")
    print("  FK / rigid transform vs original (max abs; interior | first/last frame of each clip):")
    for e in fk_err:
        print(f"    {e['clip'][:36]:36s} " + " ".join(f"{k.replace('_interior', '')} {e[k]:.1e}|{e[k.replace('interior', 'endframes')]:.1e}" for k in e if k.endswith("_interior"))
              + f" quat {e['body_quat_w_angle_rad']:.1e}")
    print(f"  soft range ({soft_factor}): " + "; ".join(f"{x['segment'][:28]} {x['frames_outside']} fr {x['max_excess_rad']:.3f} rad {x['joints']}" for x in soft))
    print("  per-frame jump max: original clips | combo inside clips | each transition window incl. seams (ratio vs original)")
    for key, c in cont.items():
        print(f"    {key:15s} {c['orig_clips_max']:.4f} | {c['combo_inside_clips_max']:.4f} | "
              + " ".join(f"{k} {v['max']:.4f} ({v['ratio_vs_orig_max']:.2f}x; seams {v['seam_in']:.4f}/{v['seam_out']:.4f})" for k, v in c["transitions"].items()))
    for x in tk:
        print(f"  {x['segment'][:70]}: max joint speed {x['max_joint_speed_rad_s']:.3f} rad/s ({x['joint']}), acc {x['max_joint_acc_rad_s2']:.2f}, pelvis {x['max_pelvis_lin_speed_m_s']:.3f} m/s "
              f"{x['max_pelvis_ang_speed_rad_s']:.3f} rad/s, dz {x['pelvis_dz_m']:+.3f} m, tilt change {x['pelvis_tilt_change_deg']:.2f} deg, max joint change {x['max_abs_joint_change_rad']:.3f} ({x['joint_change_joint']}), "
              f"foot slide {np.round(x['foot_xy_slide_m'], 3)} m, foot z {np.round(x['foot_z_range_m'], 3)}")
    for x in rep["fades"]:
        print(f"  fade: {x}")
    if "settle" in rep:
        print(f"  settle: {rep['settle']}")
    print(f"  in clips: max joint speed {rep['max_joint_speed_in_clips_rad_s']:.3f} rad/s, max pelvis speed {rep['max_pelvis_speed_in_clips_m_s']:.3f} m/s, foot z min {rep['foot_z_min_in_clips_m']:.3f} m")


if __name__ == "__main__":
    tyro.cli(main)
