"""Print the latest TensorBoard scalars of a run (no UI needed).

  uv run python scripts/tb_tail.py logs/e1_walk1 [--last 3] [--filter policy/]
"""

from __future__ import annotations

from pathlib import Path

import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(run_dir: Path, last: int = 3, filter: str = "") -> None:  # noqa: A002
    tb = run_dir / "tb" if (run_dir / "tb").exists() else run_dir
    acc = EventAccumulator(str(tb), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(t for t in acc.Tags()["scalars"] if filter in t)
    for t in tags:
        ev = acc.Scalars(t)[-last:]
        vals = ", ".join(f"{e.step}:{e.value:.4g}" for e in ev)
        print(f"{t:40s} {vals}")


if __name__ == "__main__":
    tyro.cli(main)
