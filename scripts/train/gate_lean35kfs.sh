#!/bin/bash
# sim2sim gate used to pick e24_0355000 for the real robot (clip shanshen_duosuomao_lean35kfs_pad2).
# Background and how to read the numbers: docs/REAL_ROBOT_LEAN35KFS.md#which-sim-gates-predicted-the-real-result
#
#   bash scripts/train/gate_lean35kfs.sh <ckpt.pt | deploy name> [name]
#
#   * <ckpt.pt> (training checkpoint): exported first in the training environment (uv) to deploy_artifacts/<name>/
#     ({policy.onnx, deploy_config.json}; `export --cmd config/onnx` need data/npz/bones_seed/neutral_idle_loop_001__A081.npz,
#     see docs/SIM2SIM.md#exporting-the-deploy-bundle). <name> defaults to the file stem (ckpt_0355000).
#   * <deploy name> (an existing deploy_artifacts/<name>/, e.g. e24_0355000 from the v1.1 Release asset): export skipped.
#
# Stages (each one is skipped when its output already exists):
#   1. export   (training env)  deploy bundle + deploy_artifacts/clips/<clip>.npz from data/npz/lean35kfs_only/
#   2. G2 / G3  (deploy env)    scripts/deploy/sim2sim.py --cmd g2,g3,report on the Unitree MJCF with the real-G1 models
#                               used for e24: --imu-model (gyro saturation at 500 deg/s, gyro-only attitude),
#                               --waist-coupling (|roll| + 1.4 |pitch| <= 0.95), --waist-roll-range 0.72,
#                               --waist-roll-damping 1.8. 36 G2 runs per profile (latency 0/10/20 ms x 6 kp/mass x mu 0.6/1.0).
#   3. standby  (deploy env)    sim2sim.py --cmd standby (STANDBY_PRE -> motion -> STANDBY_POST without support)
#   4. summary                  the numbers we used to choose (ground profile): aborts, falls, worst tilt error, worst
#                               IMU gravity-estimate error, 1 kHz knee peak. Rule of thumb that matched the real result
#                               (chosen after the fact, one successful checkpoint): 0/36 aborts, 0 falls, worst tilt
#                               error <= 14 deg, IMU gravity error <= ~1 deg.
#   5. real-state gate (optional, REAL_LOGS="deploy_logs/real/<run_a> ...")  scripts/deploy/realstate_gate.py: sim started
#                               from YOUR real logged states of this clip. Only its knee-abort count was informative
#                               for us; do not reject a candidate on its clean rate, tilt or fall counts.
#
# Env: G1_DEPLOY_HOME (deploy environment, default ~/workspace/g1_deploy), CLIP (default
# shanshen_duosuomao_lean35kfs_pad2), NPZ_DIR (default data/npz/lean35kfs_only), TAG (report tag, default lean35kfs),
# WORKERS (default 16), SKIP_STANDBY=1, REAL_LOGS, RS_ARGS (extra realstate_gate.py arguments).
# Reports: deploy_artifacts/reports/sim2sim_<TAG>_wc_wr_imu_<name>.json (+ SIM2SIM_*.md, sim2sim_standby_*).

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

SRC=${1:?usage: gate_lean35kfs.sh <ckpt.pt | deploy name> [name]}
CLIP=${CLIP:-shanshen_duosuomao_lean35kfs_pad2}
NPZ_DIR=${NPZ_DIR:-data/npz/lean35kfs_only}
TAG=${TAG:-lean35kfs}
WORKERS=${WORKERS:-16}
G1_DEPLOY_HOME=${G1_DEPLOY_HOME:-~/workspace/g1_deploy}

if [[ "$SRC" == *.pt ]]; then
  NAME=${2:-$(basename "$SRC" .pt)}
else
  NAME=${2:-$SRC}
fi
ART=deploy_artifacts/$NAME

# ---- 1. export (training environment) -------------------------------------------------------------------------------
if [ ! -f "$ART/policy.onnx" ] || [ ! -f "$ART/deploy_config.json" ]; then
  [[ "$SRC" == *.pt ]] || { echo "no $ART/{policy.onnx,deploy_config.json} and no checkpoint given" >&2; exit 1; }
  mkdir -p deploy_artifacts/ckpts
  CK=deploy_artifacts/ckpts/$NAME.pt   # export names the bundle after the checkpoint file stem
  [ -f "$CK" ] || cp "$SRC" "$CK"
  EXP=(env -u PYTHONPATH uv run --with onnxruntime python -m supermimic.deploy.export)
  "${EXP[@]}" --cmd config --ckpts "$CK" || exit 1
  "${EXP[@]}" --cmd onnx --ckpts "$CK" || exit 1
fi
if [ ! -f "deploy_artifacts/clips/$CLIP.npz" ]; then
  env -u PYTHONPATH uv run --with onnxruntime python -m supermimic.deploy.export --cmd clips \
    --npz-dir "$NPZ_DIR" --clip-names "$CLIP" || exit 1
fi

# ---- 2.-3. sim2sim with the real-G1 models (deploy environment) -------------------------------------------------------
# shellcheck disable=SC1091
source "$G1_DEPLOY_HOME/env.sh" || { echo "cannot source $G1_DEPLOY_HOME/env.sh (see docs/DEPLOY_G1.md)" >&2; exit 1; }
REAL_G1=(--imu-model --waist-coupling --waist-roll-range 0.72 --waist-roll-damping 1.8)
SUF=wc_wr_imu   # report-name suffix sim2sim.py adds for these models
JSON=deploy_artifacts/reports/sim2sim_${TAG}_${SUF}_${NAME}.json
if [ ! -f "$JSON" ]; then
  python scripts/deploy/sim2sim.py --cmd g2,g3,report --clips "$CLIP" --tag "$TAG" --ckpt "$NAME" \
    "${REAL_G1[@]}" --workers "$WORKERS" || exit 1
fi
if [ "${SKIP_STANDBY:-0}" != 1 ]; then
  python scripts/deploy/sim2sim.py --cmd standby --clips "$CLIP" --tag "$TAG" --ckpt "$NAME" \
    "${REAL_G1[@]}" --workers "$WORKERS" --standby-pre-s 6 --standby-post-s 10
fi

# ---- 4. summary (ground profile = the stage the real runs used) -------------------------------------------------------
python - "$JSON" "$CLIP" <<'EOF'
import collections, json, sys
d = json.load(open(sys.argv[1])); clip = sys.argv[2]
g2 = [r for r in d["g2"] if r["clip"] == clip]
rs = [r for r in g2 if r["key"] == "ground"]
ab = [r for r in rs if r["first_abort"]]
where = collections.Counter(str(r["first_abort"][1])[:24] + "@" + str(r["first_abort"][0] // 25 * 25) for r in ab)
tilt = max(r["max_tilt_err_deg"] for r in rs)
print(f"{d['ckpt']} G2 ground: aborts {len(ab)}/{len(rs)}, falls {sum(r['fall'] for r in rs)}, worst tilt err {tilt:.1f} deg {dict(where)}")
im = [r for r in rs if "imu_grav_err_max_deg" in r]
if im:
    w = max(im, key=lambda r: r["imu_grav_err_max_deg"])
    sat = sorted(r["imu_sat_z_frames"] for r in im)
    print(f"{d['ckpt']} IMU: worst gravity-estimate err {w['imu_grav_err_max_deg']:.1f} deg (frame {w['imu_grav_err_max_frame']}), "
          f"gyro-z saturated frames min/median/max {sat[0]}/{sat[len(sat) // 2]}/{sat[-1]}")
kp = [r for r in g2 if "knee_peak_1khz" in r]
if kp:
    w = max(kp, key=lambda r: r["knee_peak_1khz"])
    print(f"{d['ckpt']} 1 kHz knee peak max {w['knee_peak_1khz']:.1f} rad/s ({w['knee_peak_1khz_joint']} frame {w['knee_peak_1khz_frame']}, "
          f"profile {w['key']}); informational only, it did not predict the real result")
ok = not ab and not sum(r["fall"] for r in rs) and tilt <= 14.0 and (not im or max(r["imu_grav_err_max_deg"] for r in im) <= 1.0)
print(f"{d['ckpt']} rule of thumb (0 aborts, 0 falls, tilt <= 14 deg, IMU grav err <= 1 deg): {'MET' if ok else 'NOT MET'}")
EOF

# ---- 5. optional real-state gate ----------------------------------------------------------------------------------------
if [ -n "${REAL_LOGS:-}" ]; then
  # shellcheck disable=SC2086
  python scripts/deploy/realstate_gate.py --artifacts "$ART" --clip "deploy_artifacts/clips/$CLIP.npz" \
    --real-logs $REAL_LOGS --waist-roll-range 0.72 --workers "$WORKERS" ${RS_ARGS:-} | tail -n 5
fi
