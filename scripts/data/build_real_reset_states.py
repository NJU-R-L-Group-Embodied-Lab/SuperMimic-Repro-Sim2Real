"""Build a real-log reset-state file for --env.real-reset-file (DEVIATION from the paper, opt-in).

Each selected policy row of a real deploy log becomes one simulator start state on a library clip:

  * joint q / dq: as measured (policy joint order);
  * pelvis orientation: the logged IMU quaternion (wxyz) with its yaw replaced by the reference pelvis yaw of the
    clip frame (the IMU yaw is arbitrary; roll / pitch / relative heading of the body are kept);
  * pelvis angular velocity: the logged gyro (body frame);
  * root position: the clip's reference xy, height such that the lowest point of the lowest foot collision capsule
    touches the floor (+0.5 mm) -- that foot is the stance foot;
  * root linear velocity, two variants: ``root_lin_vel_clip`` = the clip's reference pelvis velocity (world, the clip
    and the re-aligned state share the heading), and ``root_lin_vel_foot`` = the velocity that makes the stance contact
    point stationary given the measured joint rates and gyro (stance-foot-zero-velocity estimate).

Frame mapping. A log row with ``frame == f`` is the step at which DeployCore advanced from frame f-1 to f: the robot
state was measured "at" reference frame f-1 and the policy saw goals f.. (DeployCore.goal). In the training env a
reset to command frame F is followed by one command update (F -> F+1) before the first observation, so the same
situation is reset frame F = f-1. The file stores ``log_frame`` (f) and ``ref_frame`` (f-1, local to the clip).

Usage (training venv, repo root):
  PYTHONPATH= .venv/bin/python scripts/data/build_real_reset_states.py \
      --library data/library/lean35kfs_only.pt --clip shanshen_duosuomao_lean35kfs_pad2 --frames 696 724 \
      --out data/real_reset/lean35kfs_landing_real.npz deploy_logs/real/<session>*/run_*.npz ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

POLICY_STATE = 3


def _qnorm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def set_yaw(q: np.ndarray, yaw: float) -> np.ndarray:
    """q with its heading replaced by `yaw` (left-multiplied world-z rotation; roll/pitch unchanged)."""
    d = yaw - yaw_of(q)
    return _qnorm(_qmul(np.array([np.cos(d / 2), 0.0, 0.0, np.sin(d / 2)]), q))


def foot_capsules(m: mujoco.MjModel) -> dict[str, list[int]]:
    out = {}
    for side in ("left", "right"):
        out[side] = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{side}_foot")
                     and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_collision")]
        assert out[side] and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in out[side]), side
    return out


def lowest_point(m: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> tuple[float, np.ndarray, int]:
    """(z, world point, geom) of the lowest surface point over capsule geoms (endpoint - radius)."""
    best = (np.inf, None, -1)
    for g in geoms:
        c, ax, hl, r = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)[:, 2], m.geom_size[g, 1], m.geom_size[g, 0]
        for e in (c + hl * ax, c - hl * ax):
            z = e[2] - r
            if z < best[0]:
                best = (z, e - np.array([0.0, 0.0, r]), g)
    return best


def build_state(m, d, feet, q, dq, quat, gyro, ref_root_pos, ref_root_quat, ref_lin_vel, clearance=5e-4):
    """One reset state (dict of arrays) from measured q/dq/quat/gyro and the clip frame's reference pelvis."""
    quat = set_yaw(_qnorm(quat), yaw_of(ref_root_quat))
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = [ref_root_pos[0], ref_root_pos[1], 1.0]
    d.qpos[3:7] = quat
    d.qpos[7:] = q
    d.qvel[3:6] = gyro
    d.qvel[6:] = dq
    mujoco.mj_forward(m, d)
    lows = {s: lowest_point(m, d, feet[s]) for s in feet}
    stance = min(lows, key=lambda s: lows[s][0])
    z, pt, g = lows[stance]
    d.qpos[2] -= z - clearance
    mujoco.mj_forward(m, d)
    pt = pt.copy()
    pt[2] -= z - clearance
    jac = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jac, None, pt, m.geom_bodyid[g])
    v_pt = jac[:, 3:] @ d.qvel[3:]  # contact-point velocity with zero root linear velocity
    other = "right" if stance == "left" else "left"
    return dict(
        joint_pos=np.asarray(q, np.float32), joint_vel=np.asarray(dq, np.float32),
        root_pos=d.qpos[0:3].astype(np.float32), root_quat=quat.astype(np.float32),
        root_ang_vel_b=np.asarray(gyro, np.float32),
        root_lin_vel_clip=np.asarray(ref_lin_vel, np.float32), root_lin_vel_foot=(-v_pt).astype(np.float32),
        stance=np.int8(0 if stance == "left" else 1),
        other_foot_height=np.float32(lows[other][0] - z + clearance),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="real deploy log npz files")
    ap.add_argument("--library", type=Path, required=True, help="packed compact library (.pt) holding --clip")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=(696, 724), help="inclusive log-frame range")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch

    from supermimic.data.motion_convert import load_g1_model

    pack = torch.load(a.library, map_location="cpu", weights_only=False)
    ci = list(pack["names"]).index(a.clip)
    start, length = int(pack["starts"][ci]), int(pack["lengths"][ci])
    ref = pack["ref"].float().numpy()
    m = load_g1_model()
    dd = mujoco.MjData(m)
    feet = foot_capsules(m)
    rows, src, logf = [], [], []
    for li, p in enumerate(a.logs):
        z = np.load(p, allow_pickle=True)
        pol = np.nonzero(z["state"] == POLICY_STATE)[0]
        for f in range(a.frames[0], a.frames[1] + 1):
            hit = pol[z["frame"][pol] == f]
            if hit.size != 1:
                print(f"[WARN] {p.name}: frame {f} has {hit.size} policy rows, skipped")
                continue
            i = int(hit[0])
            rf = f - 1
            assert 0 <= rf < length, (f, length)
            r = ref[start + rf]
            st = build_state(m, dd, feet, z["q"][i], z["dq"][i], z["quat"][i], z["gyro"][i], r[0:3], r[3:7], r[36:39])
            rows.append(st)
            src.append(li)
            logf.append(f)
    keys = rows[0].keys()
    out = {k: np.stack([r[k] for r in rows]) for k in keys}
    out.update(
        clip=np.array(a.clip), log_frame=np.array(logf, np.int32), ref_frame=np.array(logf, np.int32) - 1,
        source=np.array(src, np.int16), source_names=np.array([str(p) for p in a.logs]),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    n = len(rows)
    print(f"[INFO] wrote {a.out}: {n} states, clip {a.clip}, log frames {a.frames[0]}..{a.frames[1]}")
    for li, p in enumerate(a.logs):
        print(f"  {np.sum(out['source'] == li):3d} from {p}")
    print(f"  stance left/right: {np.sum(out['stance'] == 0)}/{np.sum(out['stance'] == 1)}; root z "
          f"{out['root_pos'][:, 2].min():.3f}..{out['root_pos'][:, 2].max():.3f} m; |v_foot - v_clip| mean "
          f"{np.linalg.norm(out['root_lin_vel_foot'] - out['root_lin_vel_clip'], axis=1).mean():.2f} m/s")


if __name__ == "__main__":
    main()
