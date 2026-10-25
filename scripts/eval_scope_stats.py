"""Report a full BONES-SEED benchmark under both filter scopes (see scripts/data/filter_bones_seed.py).

  paper: manifest_paper.csv (SONIC keywords + paper examples, 129,491 clips; primary scope, compare to Table A.7/A.8)
  d16:   manifest_filtered.csv (D16 list without the 8 acrobatics terms, 129,943 clips; skipped if the manifest is missing)

Input: a benchmark json ({"summary", "clips"}) from scripts/eval_benchmark.py, or a
chunked output directory (uses its benchmark_full.json, else merges chunk_*.json). Pure post-processing: no eval.

  python scripts/eval_scope_stats.py logs/<run>/bench.json
  python scripts/eval_scope_stats.py logs/<run>/full_<step> --json-out logs/<run>/full_<step>/scope_stats.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

# Table A.8 groups, matched in this order on the lowercased clip name (first match wins). The paper does not
# publish its grouping rules; these are our name rules. Side rolls (22), fire-in-the-hole (20) and crossed legs (10)
# reproduce the paper's group sizes exactly in the paper scope; flips are 54 here vs 44 in the paper, and the
# lying/big-jump/dance sizes are not comparable (paper 99/24/8), so read only the failure counts against the paper.
GROUPS: list[tuple[str, str, int]] = [  # (label, regex, paper failures SM 1 GPU x 3 h)
    ("side rolls", r"side_roll", 2),
    ("flips 360/180/90", r"(^|_)flip_", 1),
    ("fire-in-the-hole", r"fire_in_the_hole", 3),
    ("crossed legs", r"crossed_legs", 0),
    ("lying/rest/get-up", r"lying|^rest_|_rest_|fetal|spread_out|cross_legged|meditate|death|postmortem|faint|stand_up|get_up", 8),
    ("big jumps", r"(hook|reach|turn|high|broad|long|tuck|star)_jump|jump_over|jump_off", 0),
    ("dance", r"dance", 0),
]
OTHER = ("other", 11)
PAPER_TOTAL = 25


def group_of(name: str) -> str:
    n = name.lower()
    for label, pat, _ in GROUPS:
        if re.search(pat, n):
            return label
    return OTHER[0]


def prefix_of(name: str) -> str:
    """flip_180_R_001__A304_M -> flip_180; mohak_forward_stop_002__A123 -> mohak_forward_stop."""
    base = re.sub(r"__A\d+(_M)?$", "", name)
    return re.sub(r"(_[RL])?(_(opt_\d+|fast|slow))?(_\d+)?$", "", base)


def load_clips(path: Path) -> list[dict]:
    if path.is_dir():
        full = path / "benchmark_full.json"
        if full.exists():
            path = full
        else:
            return [c for f in sorted(glob.glob(str(path / "chunk_*.json"))) for c in json.load(open(f))["clips"]]
    return json.load(open(path))["clips"]


def load_scope(manifest: Path) -> set[str]:
    with open(manifest) as f:
        return {r["name"] for r in csv.DictReader(f) if r["keep"] in ("1", "True", "true")}


def stats(clips: list[dict]) -> dict:
    n = len(clips)
    steps = sum(c["steps"] for c in clips)
    return {
        "clips": n,
        "failed": sum(bool(c["failed"]) for c in clips),
        "mean_norm_return": sum(c["norm_return"] for c in clips) / max(n, 1),
        "step_weighted_norm_return": sum(c["norm_return"] * c["steps"] for c in clips) / max(steps, 1),
        "low_return_lt_0.6_not_failed": sum((not c["failed"]) and c["norm_return"] < 0.6 for c in clips),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", type=Path, help="benchmark json or chunked output dir")
    ap.add_argument("--paper-manifest", type=Path, default=Path("data/npz/bones_seed/manifest_paper.csv"))
    ap.add_argument("--d16-manifest", type=Path, default=Path("data/npz/bones_seed/manifest_filtered.csv"))
    ap.add_argument("--top", type=int, default=20, help="number of failing name prefixes to list")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    clips = load_clips(args.bench)
    by_name = {c["name"]: c for c in clips}
    print(f"{args.bench}: {len(clips)} clips in file")
    out: dict = {"source": str(args.bench), "scopes": {}}
    for scope, manifest in (("paper", args.paper_manifest), ("d16", args.d16_manifest)):
        if not manifest.exists():
            print(f"\nskip scope {scope}: {manifest} missing")
            continue
        names = load_scope(manifest)
        sel = [by_name[n] for n in sorted(names) if n in by_name]
        missing = len(names) - len(sel)
        s = stats(sel)
        s["expected_clips"], s["missing_clips"] = len(names), missing
        fails = [c for c in sel if c["failed"]]
        groups = Counter(group_of(c["name"]) for c in fails)
        sizes = Counter(group_of(n) for n in names)
        prefixes = Counter(prefix_of(c["name"]) for c in fails)
        s["groups"] = {g: {"failed": groups.get(g, 0), "clips": sizes.get(g, 0)} for g in [g for g, _, _ in GROUPS] + [OTHER[0]]}
        s["top_prefixes"] = prefixes.most_common(args.top)
        out["scopes"][scope] = s
        flag = "" if missing == 0 else f"  [WARNING: {missing} of {len(names)} scope clips missing from file]"
        print(f"\n== scope {scope} ({manifest}){flag}")
        print(f"clips {s['clips']} | failed {s['failed']} ({100 * s['failed'] / max(s['clips'], 1):.3f}%) | mean {s['mean_norm_return']:.4f} | "
              f"step-weighted {s['step_weighted_norm_return']:.4f} | not-failed norm_return<0.6: {s['low_return_lt_0.6_not_failed']}")
        print(f"failures by Table A.8 group (our name rules; [paper SM 1 GPU x 3 h]; clips = group size in this scope):")
        for g, _, paper in GROUPS + [(OTHER[0], "", OTHER[1])]:
            print(f"  {g:<20} {groups.get(g, 0):>5}  [{paper:>2}]  clips {sizes.get(g, 0)}")
        print(f"  {'total':<20} {len(fails):>5}  [{PAPER_TOTAL:>2}]")
        print(f"top {args.top} failing name prefixes:")
        print("  " + ", ".join(f"{p} {k}" for p, k in s["top_prefixes"]))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(out, indent=1))
        print("saved", args.json_out)


if __name__ == "__main__":
    main()
