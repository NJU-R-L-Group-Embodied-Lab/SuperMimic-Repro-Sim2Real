"""Appendix E filters for BONES-SEED: name filter + feasibility filter.

Reads converted npz clips (50 Hz, with 30 body link states) and writes a manifest csv with one row
per clip: name, frames, keep, reason.

Two name-filter scopes (plan D16 correction):
  --scope paper (default): SONIC's full keyword list (incl. the 8 acrobatics terms) + our paper-example
      additions -> <npz_dir>/manifest_paper.csv; ~129,647 after name filter, 129,491 final (paper 129,667 / 129,514).
  --scope d16: the original D16 list (SONIC minus the 8 acrobatics terms) -> <npz_dir>/manifest_filtered.csv
      (129,943 final; kept for comparability with e2-e11).

The feasibility filter does not depend on the scope, so --reuse-feasibility <manifest.csv> takes the
feasibility verdicts of clips that pass the name filter from an existing manifest instead of recomputing
them (clips not in that manifest or name-filtered there are recomputed).

  uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 24
  uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --reuse-feasibility data/npz/bones_seed/manifest_filtered.csv
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

# SONIC's keyword list, from https://github.com/NVlabs/GR00T-WholeBodyControl
# (gear_sonic/data_process/filter_and_copy_bones_data.py, Apache-2.0; see LICENSES/THIRD_PARTY_NOTICES.md), minus pure-acrobatics terms,
# plus the paper's examples, underscore-delimited to avoid substring false positives (plan D16).
# The 'paper' scope restores the 8 acrobatics terms (D16 correction).
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu", "50cm_box_", "on_box",
    "fall_from", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_",
    "stair", "box_jump", "box_dips", "walking_on_edge", "push_obstacle",
]
REMOVED_ACROBATICS = ["acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll", "kozak_dance", "jump_to_shoulder_roll", "monkey_jump"]
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
SCOPE_KEYWORDS = {
    "paper": sorted(set(SONIC_KEYWORDS + REMOVED_ACROBATICS + PAPER_EXTRA)),
    "d16": sorted(set(SONIC_KEYWORDS + PAPER_EXTRA)),
}
SCOPE_MANIFEST = {"paper": "manifest_paper.csv", "d16": "manifest_filtered.csv"}
KEYWORDS = SCOPE_KEYWORDS["paper"]

# Feasibility thresholds (paper Appendix E + plan D16 choices).
ELEVATED_Z = 0.20
ELEVATED_S = 0.6
PENETRATE_Z_HARD = -0.12
PENETRATE_Z_SOFT = -0.08
PENETRATE_FRAC = 0.80
FLOAT_Z = 0.02
FLOAT_S = 0.5
UPRIGHT_GZ = 0.9
STAND_Z = (0.60, 0.85)
STILL_V = 0.1
STILL_QD = 0.5
FPS = 50


def name_filter(name: str, keywords: list[str] | None = None) -> str | None:
    n = name.lower()
    for k in KEYWORDS if keywords is None else keywords:
        if k in n:
            return f"name:{k}"
    return None


def feasibility_filter(npz_path: str) -> str | None:
    with np.load(npz_path) as d:
        names = [str(x) for x in d["body_names"]]
        pos = d["body_pos_w"]  # (T, 30, 3)
        quat = d["body_quat_w"]  # (T, 30, 4) wxyz
        jv = d["joint_vel"]
        lin = d["body_lin_vel_w"]
    T = pos.shape[0]
    zmin = pos[:, :, 2].min(1)
    lf, rf, pel = names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link"), names.index("pelvis")
    # elevated support: zmin > 0.20 for >= 0.6 s
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    q = quat[:, pel]
    # gravity z-component in pelvis frame: -(R^T e_z)_z = -(1 - 2(x^2+y^2)) for wxyz quaternion
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    upright = gz < -UPRIGHT_GZ
    standing = (pos[:, pel, 2] > STAND_Z[0]) & (pos[:, pel, 2] < STAND_Z[1])
    still = (np.linalg.norm(lin[:, pel], axis=1) < STILL_V) & (np.sqrt((jv**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        # Exact lowest point of the foot collision capsules (paper: mj_geomDistance to the plane).
        foot_z = _foot_sole_height(npz_path, gate)
        floating = gate.copy()
        floating[gate] = foot_z > FLOAT_Z
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


_MODEL = None
_FOOT_GEOMS = None


def _foot_sole_height(npz_path: str, gate: np.ndarray) -> np.ndarray:
    """Lowest point over both feet's collision capsules, for the gated frames (returns len(gate.sum()),)."""
    global _MODEL, _FOOT_GEOMS
    import mujoco
    import re

    if _MODEL is None:
        from supermimic.data.motion_convert import load_g1_model

        _MODEL = load_g1_model()
        _FOOT_GEOMS = [g for g in range(_MODEL.ngeom) if re.match(r"^(left|right)_foot[1-7]_collision$", mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(_FOOT_GEOMS) == 14 and all(_MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in _FOOT_GEOMS)
    m = _MODEL
    d = mujoco.MjData(m)
    with np.load(npz_path) as z:
        names = [str(x) for x in z["body_names"]]
        pel = names.index("pelvis")
        root_pos = z["body_pos_w"][:, pel]
        root_quat = z["body_quat_w"][:, pel]
        jp = z["joint_pos"]
    idx = np.nonzero(gate)[0]
    out = np.empty(len(idx))
    radius = m.geom_size[_FOOT_GEOMS, 0]
    half = m.geom_size[_FOOT_GEOMS, 1]
    for k, t in enumerate(idx):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t]
        d.qpos[7:] = jp[t]
        mujoco.mj_kinematics(m, d)
        cz = d.geom_xpos[_FOOT_GEOMS, 2]
        axis_z = d.geom_xmat[_FOOT_GEOMS].reshape(-1, 3, 3)[:, 2, 2]  # world z of the capsule axis
        out[k] = (cz - half * np.abs(axis_z) - radius).min()
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


_KEYWORDS: list[str] = KEYWORDS
_REUSE: dict[str, tuple[int, str]] = {}


def _init(keywords: list[str], reuse: dict[str, tuple[int, str]]) -> None:
    global _KEYWORDS, _REUSE
    _KEYWORDS, _REUSE = keywords, reuse


def _one(path: str) -> tuple[str, int, str | None]:
    name = Path(path).stem
    r = name_filter(name, _KEYWORDS)
    if r is None and name in _REUSE:
        frames, reason = _REUSE[name]
        return name, frames, reason or None
    if r is None:
        r = feasibility_filter(path)
    with np.load(path) as d:
        frames = int(d["joint_pos"].shape[0])
    return name, frames, r


def _load_feasibility(manifest: Path) -> dict[str, tuple[int, str]]:
    """Feasibility verdicts (frames, reason or '') of the clips that passed the name filter in ``manifest``."""
    with open(manifest) as f:
        return {r["name"]: (int(r["frames"]), r["reason"]) for r in csv.DictReader(f) if not r["reason"].startswith("name:")}


def main(npz_dir: Path, scope: Literal["paper", "d16"] = "paper", workers: int = 24, out: Path | None = None, reuse_feasibility: Path | None = None) -> None:
    files = sorted(str(p) for p in npz_dir.glob("*.npz"))
    keywords = SCOPE_KEYWORDS[scope]
    reuse = _load_feasibility(reuse_feasibility) if reuse_feasibility is not None else {}
    print(f"{len(files)} clips, scope {scope} ({len(keywords)} keywords), reusing {len(reuse)} feasibility verdicts")
    with mp.Pool(workers, initializer=_init, initargs=(keywords, reuse)) as pool:
        rows = pool.map(_one, files, chunksize=64)
    out = out or npz_dir / SCOPE_MANIFEST[scope]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        for name, frames, r in rows:
            w.writerow([name, frames, int(r is None), r or ""])
    reasons = Counter(r for _, _, r in rows)
    kept = sum(1 for _, _, r in rows if r is None)
    name_removed = sum(1 for _, _, r in rows if r and r.startswith("name:"))
    feas_removed = sum(1 for _, _, r in rows if r and not r.startswith("name:"))
    hours = sum(fr for _, fr, r in rows if r is None) / FPS / 3600
    print(f"raw {len(rows)} -> after name filter {len(rows) - name_removed} (paper 129,667) -> after feasibility {kept} (paper 129,514); {hours:.2f} h kept (paper 264.56)")
    print("by reason:", dict(reasons.most_common(40)))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
