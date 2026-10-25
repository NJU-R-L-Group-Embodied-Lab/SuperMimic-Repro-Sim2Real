"""Batch-convert G1 retarget CSVs to mjlab-compatible 50 Hz motion npz files.

Example:
  uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 \
      --output-dir data/npz/lafan1 --input-fps 30 --workers 16

Writes <output_dir>/<clip>.npz for every CSV and <output_dir>/manifest.csv.
A per-clip fps override can be given with --fps-map (CSV with columns `name,fps`).
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import tyro

_MODEL = None


def _init_worker() -> None:
    global _MODEL
    # Keep BLAS single-threaded inside workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from supermimic.data.motion_convert import load_g1_model

    _MODEL = load_g1_model()


def _convert_one(args: tuple[str, str, float, float]) -> dict:
    from supermimic.data.motion_convert import convert_csv_to_npz

    src, dst, in_fps, out_fps = args
    try:
        info = convert_csv_to_npz(src, dst, in_fps, out_fps, model=_MODEL)
        info["ok"] = True
        info["error"] = ""
    except Exception as e:  # noqa: BLE001
        info = {"source": src, "out": dst, "ok": False, "error": repr(e), "input_frames": 0, "output_frames": 0, "duration_s": 0.0}
    return info


def main(
    input_dir: Path,
    output_dir: Path,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    workers: int = 16,
    pattern: str = "**/*.csv",
    fps_map: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    files = sorted(input_dir.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"no files matching {pattern} under {input_dir}", file=sys.stderr)
        sys.exit(1)
    fps_by_name: dict[str, float] = {}
    if fps_map is not None:
        with open(fps_map) as f:
            for row in csv.DictReader(f):
                fps_by_name[row["name"]] = float(row["fps"])

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src in files:
        name = src.stem
        dst = output_dir / f"{name}.npz"
        if dst.exists() and not overwrite:
            continue
        jobs.append((str(src), str(dst), fps_by_name.get(name, input_fps), output_fps))
    print(f"{len(files)} csv files, {len(jobs)} to convert, {workers} workers")

    t0 = time.time()
    results: list[dict] = []
    if jobs:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            for i, info in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=4), 1):
                results.append(info)
                if i % 200 == 0 or i == len(jobs):
                    dt = time.time() - t0
                    print(f"  {i}/{len(jobs)} done, {dt:.0f}s elapsed, {i / dt:.1f} clips/s")
    failures = [r for r in results if not r["ok"]]
    for r in failures[:20]:
        print("FAILED", r["source"], r["error"])

    # Manifest over everything present in output_dir (so re-runs stay complete).
    import numpy as np

    rows = []
    for npz in sorted(output_dir.glob("*.npz")):
        with np.load(npz) as d:
            n = int(d["joint_pos"].shape[0])
            fps = float(d["fps"][0])
            in_fps = float(d["input_fps"][0]) if "input_fps" in d else float("nan")
        rows.append({"name": npz.stem, "frames": n, "fps": fps, "duration_s": (n - 1) / fps, "input_fps": in_fps})
    with open(output_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "frames", "fps", "duration_s", "input_fps"])
        w.writeheader()
        w.writerows(rows)
    total_h = sum(r["duration_s"] for r in rows) / 3600.0
    print(f"manifest: {len(rows)} clips, {total_h:.3f} h total, {len(failures)} failures, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    tyro.cli(main)
