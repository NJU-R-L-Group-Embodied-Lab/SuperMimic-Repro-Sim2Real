#!/bin/bash
# Training recipe of the released checkpoint e24_0355000 (clip shanshen_duosuomao_lean35kfs_pad2), the first policy that
# ran this spin / landing dance from frame 0 to the end on a real G1 (twice, rope attached; see
# docs/REAL_ROBOT_LEAN35KFS.md). Run from the repository root in the training environment:
#
#   PARENT_CKPT=<parent checkpoint .pt> bash scripts/train/train_lean35kfs_e24.sh
#
# Lineage (honest version). e24 is NOT trained from scratch. It is a short fine-tune (342k -> 356k iterations, constant lr
# 1e-5, this single clip) of an earlier lean35kfs checkpoint (internal name e23_0342000), which itself descends from a
# chain of earlier fine-tunes of a BONES-SEED-based model on successive versions of this dance (lean35 -> lean35kf ->
# lean35kfs), each adding some of the flags below. The released checkpoint (e24_0355000.pt in the v1.1 Release asset) is the RESULT of
# this command; the parent checkpoint e23_0342000 is NOT part of the release. So this script documents exactly what was
# run, but cannot reproduce e24 bit for bit. Options:
#   * PARENT_CKPT=<your own checkpoint fine-tuned on this clip>  (closest to what we did; set TOTAL_ITERS = its iteration
#     + about 14000, since --schedule.total-iterations counts from the resumed step);
#   * PARENT_CKPT=<path>/e24_0355000.pt to continue fine-tuning the released policy (TOTAL_ITERS > 355000).
# There is no ablation behind any flag below: we only know that this combination produced one checkpoint that worked on
# the real robot (n = 1 checkpoint, 2 real runs).
#
# Inputs (see docs/REAL_ROBOT_LEAN35KFS.md#files-and-where-they-go):
#   data/library/lean35kfs_only.pt                   packed library with the one clip (v1.1 asset, or build_library.py)
#   motions/shanshen_duosuomao/reset_points_lean35kfs.csv   failure-point reset list (in the repo)
#   data/real_reset/lean35kfs_landing_real_e22e23.npz       87 real-robot states (v1.1 asset)
#
# After (or while) training, gate every checkpoint with scripts/train/gate_lean35kfs.sh (sim2sim with the real-G1 IMU,
# waist coupling and waist roll models), e.g.
#   bash scripts/train/gate_lean35kfs.sh logs/lean35kfs_e24/ckpt_0355000.pt e24_0355000

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

: "${PARENT_CKPT:?set PARENT_CKPT to the checkpoint to fine-tune from (see the header of this script)}"
RUN_NAME=${RUN_NAME:-lean35kfs_e24}
TOTAL_ITERS=${TOTAL_ITERS:-356000}   # e24: resumed at 342000, so 14000 fine-tune iterations
MAX_HOURS=${MAX_HOURS:-3.5}          # e24 ran on one RTX 4090; 355000 was picked by the gate, not the last checkpoint

args=(
  --run-name "$RUN_NAME"
  # one-clip library (shanshen_duosuomao_lean35kfs_pad2), 4096 envs, fine-tune from the parent at a constant lr
  --data.library data/library/lean35kfs_only.pt
  --env.num-envs 4096 --policy-trainer.batch-size 4096
  --resume "$PARENT_CKPT"
  --schedule.lr-constant 1e-5 --schedule.total-iterations "$TOTAL_ITERS" --schedule.max-hours "$MAX_HOURS"
  --schedule.checkpoint-interval 1000
  # reset sampler: the defaults, written out (horizon G_max, softmax temperature 25, visit-count warm start)
  --env.sampler-gmax-mode horizon --env.sampler-softmax-temp 25 --env.sampler-count-warmstart
  # [deviation] half of the resets start in the 3 s before a known failure frame of this clip (clip_name,fail_step list:
  # frames where earlier checkpoints failed in sim or on the real robot)
  --env.fail-reset-frac 0.5
  --env.fail-reset-list motions/shanshen_duosuomao/reset_points_lean35kfs.csv
  --env.fail-reset-window-s 3.0
  # [deviation] tilt hinge: -40 * max(0, tilt_err - 0.12 rad); the real runner aborts at a 20 deg tilt error, training
  # terminates only at ~78 deg
  --env.tilt-hinge-weight -40.0 --env.tilt-hinge-knee 0.12
  # [deviation] per-episode PD stiffness scale and extra torso mass (the sim2sim G2 axes kp 0.85/1.15, +1.8 kg)
  --env.dr-kp-range 0.8 1.2 --env.dr-torso-mass-add 0.0 2.5
  # [deviation] command-latency DR: 0..4 physics substeps (0-20 ms) of delay per episode (runner -> bridge -> motor)
  --env.dr-action-delay-substeps 4
  # [deviation] real-G1 waist roll/pitch linkage as a physics constraint |roll| + 1.4 |pitch| <= 0.95
  # (supermimic.deploy.waist_coupling; fitted on real logs)
  --env.waist-coupling
  # [deviation] imagined-reward torque penalty (waist / knee heating on the real robot)
  --env.joint-torque-weight -0.2
  # [deviation] imagined-reward knee speed hinge: -0.1 * sum max(0, |dq_knee| - 12 rad/s)^2 (real abort at 16 rad/s)
  --env.joint-vel-hinge-weight -0.1 --env.joint-vel-hinge-thr 12.0
  # [deviation] saturating IMU observation model in 75% of episodes: gyro clipped at +-500 deg/s, gravity integrated from
  # the clipped gyro (the real pelvis gyro saturates in the spins); plus a pelvis yaw-rate hinge above 7.5 rad/s
  --env.imu-sat-prob 0.75 --env.yaw-rate-hinge-weight 0.1 --env.yaw-rate-hinge-thr 7.5
  # ---- new in e24 relative to its parent (the three changes discussed in docs/REAL_ROBOT_LEAN35KFS.md) ----
  # [deviation] 20% of resets start from one of 87 REAL robot states logged at clip frames 696-724 in three failed real
  # runs of earlier checkpoints (scripts/data/build_real_reset_states.py); root velocity taken from the clip
  --env.real-reset-file data/real_reset/lean35kfs_landing_real_e22e23.npz --env.real-reset-prob 0.2 --env.real-reset-vel clip
  # [deviation] real-G1 waist roll: range +-0.72 instead of the MJCF +-0.52, plus per-episode waist-roll damping
  # U[0, 3.6] Nm s/rad and frictionloss U[0, 2] Nm (supermimic.deploy.waist_roll)
  --env.waist-roll-range 0.72 --env.dr-waist-roll-damping 0.0 3.6 --env.dr-waist-roll-friction 0.0 2.0
  # [deviation] pelvis push DR: with p 0.2 per episode, a short random pelvis force/torque starting at a clip frame in
  # 690-722 (between the first spin and the second jump), only on clips whose name contains "lean35"
  --env.dr-push-prob 0.2 --env.dr-push-frames 690 722 --env.dr-push-clips lean35
)

mkdir -p logs
exec uv run sm-train "${args[@]}" "$@"
