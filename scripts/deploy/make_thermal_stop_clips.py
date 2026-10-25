"""Sim2sim test clips for the runner's thermal stop (SIM ONLY).

For each stop frame k of a deploy clip: reference = clip frames 0..k, then the runner's own thermal_stop_clip blend
(bridge/g1_runner_bridge.py, --thermal-stop-s) to --pose (current: frame k held still, the bridge runner's default
--thermal-stop-pose; upright: the same with the waist at its frame-0 values; frame0: blend back to the clip's frame-0 stand
pose). This script's own --pose default is upright: pass --pose current to test the pose the runner uses by default), then --hold-s of that static row (what STANDBY_POST holds with advance=False). Raw state arrays (used by sim2sim for the initial state, the info-only
training-termination check and the displacement report) hold the frame-k root/joint state through the stop.

  source $G1_DEPLOY_HOME/env.sh   # default ~/workspace/g1_deploy
  python scripts/deploy/make_thermal_stop_clips.py --clip combo_c4_full_vc_b15f20 --frames 380 1289 1464
  python scripts/deploy/sim2sim.py --clips <printed names> --tag thermal --ckpt e13i_0291000 --cmd g2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "deploy_artifacts"
sys.path.insert(0, str(REPO / "src"))
from supermimic.deploy.core import Clip, static_clip  # noqa: E402


def load_runner(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("g1_runner_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--stop-s", type=float, default=1.0)
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--pose", choices=("upright", "current", "frame0"), default="current")
    ap.add_argument("--runner", type=Path, default=REPO / "scripts/deploy/bridge/g1_runner_bridge.py",
                    help="runner whose thermal_stop_clip / WAIST_IDX are used (default: the bridge runner)")
    a = ap.parse_args()
    R = load_runner(a.runner)
    src = ART / "clips" / f"{a.clip}.npz"
    clip = Clip.load(src)
    raw = dict(np.load(src))
    dt = 1.0 / float(np.asarray(raw["fps"]).reshape(-1)[0])
    n = max(2, int(round(a.stop_s / dt)))
    nh = int(round(a.hold_s / dt))
    stand = static_clip(clip, 0)
    names = []
    for k in a.frames:
        target = stand if a.pose == "frame0" else static_clip(clip, k)
        if a.pose == "upright":
            target.goal[:, R.WAIST_IDX] = stand.goal[0, R.WAIST_IDX]
            target.joint_pos[:, R.WAIST_IDX] = stand.joint_pos[0, R.WAIST_IDX]
        stop = R.thermal_stop_clip(clip, k, target, n, dt)
        hold = static_clip(stop, stop.num_frames - 1)
        goal = np.concatenate([clip.goal[: k + 1], stop.goal[1:], np.repeat(hold.goal[:1], nh, 0)])
        jp = np.concatenate([clip.joint_pos[: k + 1], stop.joint_pos[1:], np.repeat(hold.joint_pos[:1], nh, 0)])
        pg = np.concatenate([clip.proj_grav[: k + 1], stop.proj_grav[1:], np.repeat(hold.proj_grav[:1], nh, 0)])
        T, extra = goal.shape[0], goal.shape[0] - (k + 1)
        name = f"{a.clip}_thstop{dict(frame0='', current='cur', upright='up')[a.pose]}{k}"
        out = {key: v for key, v in raw.items()}
        out.update(name=np.array(name), goal=goal.astype(np.float32), joint_pos=jp, proj_grav=pg)
        for key in ("root_pos", "root_quat_wxyz", "root_lin_vel_w", "root_ang_vel_w", "joint_vel"):
            v = raw[key][: k + 1]
            tail = np.repeat(v[-1:], extra, 0)
            if key in ("root_lin_vel_w", "root_ang_vel_w", "joint_vel"):
                tail = np.zeros_like(tail)
            out[key] = np.concatenate([v, tail])
        if "segment_names" in raw:
            keep = raw["segment_start"] <= k
            out["segment_names"] = np.concatenate([raw["segment_names"][keep], np.array(["thermal_stop", "thermal_hold"])])
            out["segment_kind"] = np.concatenate([raw["segment_kind"][keep], np.array(["transition", "settle"])])
            out["segment_start"] = np.concatenate([raw["segment_start"][keep], [k + 1, k + n + 1]])
            out["segment_end"] = np.concatenate([np.minimum(raw["segment_end"][keep], k + 1), [k + n + 1, T]])
        np.savez_compressed(ART / "clips" / f"{name}.npz", **out)
        names.append(name)
        print(f"{name}: T={T} (motion 0..{k}, stop {n} frames, hold {nh})")
    print(" ".join(names))


if __name__ == "__main__":
    main()
