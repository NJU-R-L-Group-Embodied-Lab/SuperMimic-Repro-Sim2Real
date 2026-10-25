#!/bin/bash
# Pull new checkpoints of a remote run and evaluate them locally on the BONES-SEED 2000-clip subset.
#   scripts/pull_and_eval.sh <remote_host> <run_name> [num_envs]
# <remote_host> is any ssh destination (user@host or an ~/.ssh/config alias). The repo location on the remote
# defaults to ~/SuperMimic; override with SM_REMOTE_REPO.
host=$1; run=$2; nenv=${3:-1024}
remote_repo=${SM_REMOTE_REPO:-'~/SuperMimic'}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/$run
while true; do
  rsync -a --include='ckpt_0*.pt' --include='config.json' --include='eval_summary.txt' --exclude='*' $host:$remote_repo/logs/$run/ logs/$run/ 2>/dev/null
  for ck in $(ls logs/$run/ckpt_0*.pt 2>/dev/null | sort); do
    out=logs/$run/benchmark_$(basename $ck .pt).json
    [ -f "$out" ] && continue
    env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt $ck --npz-dir data/npz/bones_seed --clip-list data/npz/bones_seed/subset2000.txt --num-envs $nenv > logs/$run/eval_$(basename $ck .pt).log 2>&1
    echo "$(date +%H:%M) $(basename $ck): $(grep -E 'clips \|' logs/$run/eval_$(basename $ck .pt).log)" >> logs/$run/eval_summary_local.txt
  done
  sleep 180
done
