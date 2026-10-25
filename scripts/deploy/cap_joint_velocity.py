"""Locally smooth reference joints whose speed exceeds a cap, so a faithful tracker stays under the deploy velocity abort.

  uv run python scripts/deploy/cap_joint_velocity.py --src data/npz/custom_dance/gouxiong_duosuomao.npz \
      --dst data/npz/custom_dance/gouxiong_duosuomao_vc.npz --deploy-config deploy_artifacts/e13i_0291000/deploy_config.json

For every joint and every run of frames with |joint_vel| > cap_frac * max_vel_frac * velocity_limit, the joint position is
replaced by q + w (g_sigma * q - q) in a window of +-half_window_s around the run, where g_sigma is a Gaussian filter
and w a cos^2 bump (1 at the run, 0 at the window edges); sigma grows until the window is under the cap. Only joint_pos
and joint_vel (np.gradient, as the converter) change; body states are recomputed later by make_combo_clip (FK).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro
from scipy.ndimage import gaussian_filter1d


def main(src: Path, dst: Path, deploy_config: Path, cap_frac: float = 0.75, max_vel_frac: float = 0.8,
         half_window_s: float = 0.4) -> None:
    cfg = json.loads(deploy_config.read_text())
    d = dict(np.load(src, allow_pickle=True))
    fps = float(np.ravel(d["fps"])[0]) if "fps" in d else 50.0
    q = d["joint_pos"].astype(np.float64)
    cap = cap_frac * max_vel_frac * np.asarray(cfg["velocity_limit"], np.float64)
    hw = int(round(half_window_s * fps))
    T = len(q)
    for j in range(q.shape[1]):
        for _ in range(10):  # runs can reappear next to a smoothed one
            v = np.gradient(q[:, j], 1.0 / fps)
            over = np.where(np.abs(v) > cap[j])[0]
            if not len(over):
                break
            a, b = over[0], over[0]
            while b + 1 < T and (b + 1 in over or b + 2 in over):
                b += 1
            lo, hi = max(0, a - hw), min(T, b + hw + 1)
            k = np.arange(lo, hi)
            w = np.ones(len(k))
            left, right = k < a, k > b
            w[left] = np.cos(0.5 * np.pi * (a - k[left]) / max(1, a - lo)) ** 2
            w[right] = np.cos(0.5 * np.pi * (k[right] - b) / max(1, hi - 1 - b)) ** 2
            for sigma in np.arange(0.5, 12.0, 0.25):
                cand = q[:, j].copy()
                sm = gaussian_filter1d(q[:, j], sigma, mode="nearest")
                cand[k] = q[k, j] + w * (sm[k] - q[k, j])
                if np.abs(np.gradient(cand, 1.0 / fps)[lo:hi]).max() <= cap[j]:
                    break
            dev = np.abs(cand - q[:, j]).max()
            print(f"{cfg['joint_names'][j]}: frames {a}-{b} peak {np.abs(v[a:b + 1]).max():.2f} rad/s > cap {cap[j]:.2f}; "
                  f"sigma {sigma:.2f}, window {lo}-{hi - 1}, max |dq_pos| {dev:.3f} rad")
            q[:, j] = cand
    d["joint_pos"] = q.astype(np.float32)
    d["joint_vel"] = np.gradient(q, 1.0 / fps, axis=0).astype(np.float32)
    np.savez(dst, **d)
    print(f"wrote {dst}: max |joint_vel|/cap {np.max(np.abs(d['joint_vel']) / cap):.3f}")


if __name__ == "__main__":
    tyro.cli(main)
