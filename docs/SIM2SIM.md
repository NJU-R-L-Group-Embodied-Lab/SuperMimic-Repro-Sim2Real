# Export and sim2sim gates

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

## Exporting the deploy bundle

Run this in the training environment. `onnxruntime` is not among the training dependencies; use `uv run --with onnxruntime`, or set `SM_ORT_OVERLAY=<a directory with onnxruntime installed>`:

```bash
EXP="uv run --with onnxruntime python -m supermimic.deploy.export"
$EXP --cmd clips                                            # by default exports the BONES-SEED SAFE/HARDER test clips to deploy_artifacts/clips/ (the onnx parity check in the next step needs at least one of them)
$EXP --cmd freeze --ckpts logs/custom_ft/ckpt_0291000.pt   # copies into deploy_artifacts/ckpts/
$EXP --cmd config --ckpts logs/custom_ft/ckpt_0291000.pt   # deploy_artifacts/ckpt_0291000/deploy_config.json (builds a small environment from BONES-SEED neutral_idle_loop_001__A081 for a cross-check)
$EXP --cmd onnx   --ckpts logs/custom_ft/ckpt_0291000.pt   # deploy_artifacts/ckpt_0291000/policy.onnx, checked for parity against torch
$EXP --cmd clips  --npz-dir data/npz/custom_combo --clip-names combo_c4_full_vc_b15f20   # deploy_artifacts/clips/combo_c4_full_vc_b15f20.npz (goal table)
$EXP --cmd clips  --clip-names walk_forward_stop_001__A026  # used by the standby / standby_exit / standby_nan tests of run_sim_e2e_bridge.sh
$EXP --cmd mjcf                                             # training-scene MJCF, used by the sim2sim g1 gate
```

`config`, `onnx` and `mjcf` all need `data/npz/bones_seed/neutral_idle_loop_001__A081.npz` (the BONES-SEED conversion output).

The deploy bundle is just `deploy_artifacts/<name>/{policy.onnx, deploy_config.json}` plus `deploy_artifacts/clips/<clip>.npz`. For the numerical parity test, see `tests/test_deploy_parity.py` under "[How to run](#how-to-run)" below.

## sim2sim gates overview

Before going on the robot, every policy must first pass a set of gates in simulation. The script is `scripts/deploy/sim2sim.py`, and it runs in the deploy environment (CPU MuJoCo + onnxruntime, no torch needed). Observations, the goal table, ONNX inference, clamping and the abort criteria all go through the same `DeployCore` (`src/supermimic/deploy/core.py`) that is used on the real robot, so what the simulation tests is exactly the code that runs on the robot. **Everything below is simulation results.** The representative numbers come from early e11 BONES-SEED checkpoints (`ckpt_0100000` / `ckpt_0116000` / `ckpt_0140000` / `ckpt_0144000`), which are **not** in the Release. Only the subsection "[sim2sim results for the Release model](#sim2sim-results-for-the-release-model)" covers `e13i_0291000`.

## What each gate checks

| Gate (`--cmd`) | Simulation model and procedure | Quantities checked |
|---|---|---|
| `g1` same model | The exported training-scene MJCF (position actuators = training PD), dt 5 ms, the policy is called once every 4 substeps. Starts from frame 0 of the clip (including reference velocities) and is compared step by step against a rollout in the training environment | Whether failure matches the training environment (failure = training termination condition: pelvis height deviation > 0.25 m or gravity-direction deviation > 0.8); ratio of joint tracking RMSE; first-step obs/goal/action differences; time at which divergence starts (joint RMS > 0.05 rad or horizontal position > 0.10 m); number of clamps |
| `g2` real model | Unitree's official G1 29-DoF MJCF (torque motors). Here PD is computed at 1 kHz with the training kp/kd from the latest (delayed) target and clipped to effort. The robot starts standing still on the ground in the frame-0 pose | Falls (pelvis height < 0.5 m or tilt > 60°); false aborts (an abort triggered without a fall); maximum tilt; \|tau\|/limit, and the fraction of time with \|tau\| ≤ 0.9 limit; first-step torque; RMSE |
| `g3` transitions | The runner flow: q0 standing → 4 s cosine interpolation to frame 0 (standing gains) → 1 s HOLD → POLICY (the first target and the training gains take effect at the same time) → END transition (over 1 s, kp/kd ramp linearly to the standing gains and the target ramps to the last-frame pose clipped to the soft limits, with effort clipping every 2 ms) → 2 s hold. The gantry is modeled as a spring-damper on the pelvis (2000 N/m, 200 N·s/m). There are also checks of the per-stage abort thresholds: air (lifted from the torso CoM, 20 kN/m, feet 5 cm off the ground), feet (gantry carries the load throughout), slack (gantry released when the policy starts) | Whether it stays upright, number of aborts, torque rate at each switch (the largest \|τ(t)−τ(t−1 ms)\| over all joints, in Nm/ms), margin to each stage's thresholds |
| `standby` policy standby | The runner's current flow: HOLD (gantry) → R1+A → STANDBY_PRE (`static_clip(clip, 0)`, `step(advance=False)`, training gains, **no support of any kind from here on**) → R1+X → `switch_clip(clip, 0)`, history kept, motion starts → after playback `switch_clip(static_clip(clip, last frame))` → STANDBY_POST | Falls, aborts, tilt error during standby, drift and yaw, switch torque rates, whether the R1+X check lets the motion through |
| `g5` | Timing of `DeployCore.step`, single-threaded | p50 / p99 |
| `margins` | Statistics along the reference rollout of the fp32 training environment | \|a\| and dq/velocity limit |
| `compare` | Training MJCF vs. Unitree MJCF | Mass, inertia, armature, damping, frictionloss, joint range, effort |
| `lw` / `lw_robust` | Paired comparison of the waist-locked and non-locked robot variants, see "[sim2sim evaluation of the waist-locked G1](#sim2sim-evaluation-of-the-waist-locked-g1)" | — |
| End-to-end (`run_sim_e2e*.sh`) | The real runner connected over DDS to a headless unitree_mujoco G1 simulation (domain 1, `lo`, dt 0.002) | State machine, buttons, watchdog and various fault injections, see "Simulated end-to-end tests (DDS)" in [Real-robot deployment](DEPLOY_G1.md) |

`DeployCore` defaults: soft-limit clipping **off**; torque clipping **on**, capped at 1.5 × effort; abort when the tilt error > 20° (0.35 rad), joint velocity > 0.8 × velocity limit, \|a\| > 50, or any non-finite value appears. The runner's `--stage` tightens these (air/feet: 0.3 / 10 / 0.5×; slack: 0.5 / 15 / 1.0×; ground: 0.8 / 50 / 1.5×); for the full table see "Staged safety profiles `--stage` (required)" in [Real-robot deployment](DEPLOY_G1.md).

## Perturbation grid

| Mode | Latency [ms] | kp scale | Added torso mass [kg] | Friction μ | Abort profile |
|---|---|---|---|---|---|
| `g2` on the built-in clip set (nominal) | 0 / 5 / 10 / 20 | 1.0 | 0 | 0.6 / 1.0 | core defaults |
| `g2` on the built-in clip set (full sweep, only for the recommended `ckpt_0116000`) | 0 / 5 / 10 / 20 | 0.85 / 1.0 / 1.15 | 0 / 1.8 | 0.6 / 1.0 | core defaults |
| `g2` in `--clips` mode | 0 / 10 / 20 | 6 (kp, kg) combinations: (1.0,0) (nominal), (0.85,0), (1.15,0), (1.0,1.8), (0.85,1.8), (1.15,1.8) | as in the previous column | 0.6 / 1.0 | `slack`, `ground` |
| `g3` in `--clips` mode | 0 / 10 / 20 | 1.0 | 0 | — | `slack`, `ground`; the END transition command has 2 ms latency |
| `standby` | 0 / 10 / 20 | 0.85 / 1.0 / 1.15 (all commanded gains scaled together) | 0 / 1.8 | 0.6 / 1.0 | `slack`, `ground`. 36 runs per clip per profile |

The built-in clip set has two groups: SAFE is 5 slow motions (`neutral_idle_loop_001__A081`, `idle_loop_002__A060`, `shoulder_clap_004__A041_M`, `itching_left_arm_R_003__A121`, `walk_sideway_090_stop_003__A037`), and HARDER is `wave_R_001__A431`, `jog_sideway_045_loop_002__A025`, `dance_hiphop_alt_stump_R_fast_001__A318_M`.

## Pass criteria

| Gate | Pass criterion |
|---|---|
| G1 (built-in) | For every SAFE clip the failure outcome is the same as in the fp32 training environment, and s2s joint RMSE ≤ 1.2 × environment RMSE |
| G1 (`--clips`) | Under both the `ground` and `slack` profiles: failure outcome matches, s2s does not fail, RMSE ratio ≤ 1.2 |
| G2 (built-in) | 0 falls and 0 false aborts in the 40 SAFE nominal runs (5 clips × 4 latencies × 2 frictions), and in every run the fraction of 1 kHz steps with \|tau\| ≤ 0.9 limit is ≥ 99% |
| G2 / G3 / margins (`--clips`, per clip) | `slack` nominal G2: 0 falls, 0 aborts. `slack` G3: all upright, all run to completion, 0 aborts. `slack` margins (threshold ÷ the maximum seen in G2 nominal and G3, computed from the 50 Hz sampled state that the core actually checks) must be ≥ 1.2 for \|a\|, dq/velocity limit and the 20° tilt abort |
| standby | The motion runs to completion and STANDBY_POST is reached; 0 falls, 0 aborts; standby tilt error < 8°; torque rates for STANDBY_PRE→POLICY and POLICY→STANDBY_POST ≤ 0.5 × that of HOLD→STANDBY_PRE in the same run. R1+X is pressed at `--standby-pre-s` (default 60 s), and the runner only starts the motion once its check passes: standby ≥ 3 s, and over the last 3 s tilt error < 8° and max\|dq\| < 1 rad/s. If it does not pass, the press is repeated every step for up to another `--standby-r1x-cap-s` (default 30 s); if it still does not pass, the run is recorded as never, the motion is not started, and STANDBY_POST is not entered |
| Requirements for going on the robot (our own rules) | G2 perturbation grid with 0 falls and 0 aborts; maximum tilt error leaves margin to the 20° abort threshold; all standby runs pass; all end-to-end tests pass |

## How to run

```bash
export G1_DEPLOY_HOME=${G1_DEPLOY_HOME:-~/workspace/g1_deploy}
source $G1_DEPLOY_HOME/env.sh   # deploy environment; the Unitree MJCF defaults to $G1_DEPLOY_HOME/unitree_mujoco/unitree_robots/g1/scene_29dof.xml and can also be set with G1_UNITREE_XML

# Arbitrary clip set (use this for the Release model): results go to sim2sim_<tag>_<ckpt>.json; when --cmd includes report, SIM2SIM_<tag>_<ckpt>.md is written as well
python scripts/deploy/sim2sim.py --cmd g2,g3,report --clips combo_c4_full_vc_b15f20 --tag combo --ckpt e13i_0291000 --workers 16
# Policy standby: sim2sim_standby_<tag>_<ckpt>.json and .md are always written together
python scripts/deploy/sim2sim.py --cmd standby --clips combo_c4_full_vc_b15f20 --tag combo --ckpt e13i_0291000 --workers 16
# Full gate suite on the built-in SAFE/HARDER set (g1,g2,g3,g5,margins,compare,report)
python scripts/deploy/sim2sim.py --cmd all --ckpt <name> --workers 10
```

- For setting up the deploy environment (`G1_DEPLOY_HOME`, `env.sh`), see "Deploy environment: `G1_DEPLOY_HOME` and `env.sh`" in [Real-robot deployment](DEPLOY_G1.md).
- `--ckpt` is a directory name under `deploy_artifacts/`. The default `ckpt_0100000 ckpt_0116000` are early policies that are not in the Release; when using the Release model you must pass it explicitly.
- All results are written to `deploy_artifacts/reports/`. The built-in set's `report` only summarizes `ckpt_0100000` / `ckpt_0116000` and writes `SIM2SIM.md`; for other models use the `--clips … --tag` mode.
- Other switches: `--g1` / `--g2` select variants, and `--lock-waist` is for the waist-locked robot. The standby grid can be overridden with `--standby-profiles/-lat/-kps/-mass/-fric`; when overriding, name the report with `--tag`. `--standby-pre-s` / `--standby-post-s` default to 60 / 30 s. `--standby-r1x-window-s` / `-min-s` / `-cap-s` correspond to the runner's `--standby-window-s` / `--standby-min-s`. `--cmd standby_report --standby-json <json>` regenerates the md from a json.
- **The `g1` gate needs a training-side reference rollout.** Generate it in the training environment (GPU); the file name must be `reports/sim2sim/ref_<tag>_<ckpt>_<tf32|no-tf32>.npz` (the built-in set has no `<tag>_` prefix):

  ```bash
  PYTHONPATH= uv run python scripts/deploy/sim2sim_ref.py --ckpt deploy_artifacts/e13i_0291000/ckpt_e13i_0291000.pt \
      --clips combo_c4_full_vc_b15f20 --npz-dir data/npz/custom_combo --no-tf32 \
      --out deploy_artifacts/reports/sim2sim/ref_combo_e13i_0291000_no-tf32.npz
  ```

- **Numerical parity**: `tests/test_deploy_parity.py` (needs CUDA, the BONES-SEED npz, onnxruntime, and the exported `deploy_artifacts/<ckpt>/policy.onnx` and `deploy_artifacts/ckpts/<ckpt>.pt`); select the model with `SM_PARITY_CKPT=<name>` (default `ckpt_0116000`; skipped if it does not exist). It uses an explicit no-safety configuration and covers 4 clips × 200 steps, with tolerances of 1e-4 for obs/goal, 1e-3 for action (core ONNX vs. torch fp32), 1e-5 for q_target, and goal-row misalignment must be 0. This test does **not** cover the default clamping; clamping is checked by the G1 gate.
- **End-to-end DDS tests**: `bash scripts/deploy/bridge/run_sim_e2e_bridge.sh <test>` tests the bridge runner (run `bridge/build.sh` first), and `bash scripts/deploy/run_sim_e2e.sh <test>` tests the pure-Python runner. Environment variables are `CKPT`, `CLIPS`, `STAGE` (default ground), `LOCK=1`, `LW_DIRECT`, `SIM_DT`, `E2E_NAME`, `G1_DEPLOY_HOME`. For the test list and results, see "Simulated end-to-end tests (DDS)" in [Real-robot deployment](DEPLOY_G1.md).

## Real-G1 models and extra checks (opt-in)

These options were added while bringing the spin / landing dance `shanshen_duosuomao_lean35kfs_pad2` to the real robot (see [Real-robot result: lean35kfs](REAL_ROBOT_LEAN35KFS.md)). All of them are off by default, so the gates above are unchanged. The settings are inherited by the worker processes, and each one adds a suffix to the report names (`_wc`, `_wr`, `_imu`, in this order: `sim2sim_<tag>_wc_wr_imu_<ckpt>.json`), so these results never overwrite the plain ones.

| Option (`sim2sim.py`) | Environment variable | What it models |
|---|---|---|
| `--imu-model` | `SM_IMU_MODEL=1` | Real pelvis IMU: gyro clipped at ±500°/s per axis, orientation integrated from the clipped gyro with no accelerometer correction (`src/supermimic/deploy/imu_model.py`). G2 rows then carry `imu_sat_z_frames`, `imu_grav_err_max_deg` (+ `_frame`) and `imu_rot_lost_z_rad` |
| `--waist-coupling` (`--waist-coupling-k 1.4 --waist-coupling-c 0.95`) | `SM_WAIST_COUPLING=k,c` (or `1` for the defaults) | Waist roll/pitch linkage \|roll\| + k \|pitch\| ≤ c as two limited tendons in every gate model (`src/supermimic/deploy/waist_coupling.py`; k, c fitted on real logs) |
| `--waist-roll-range R --waist-roll-damping D --waist-roll-friction F` | `SM_WAIST_ROLL="range=0.72,damping=1.8,friction=2"` | Waist roll range ±R instead of ±0.52 in every model; extra waist-roll damping D [Nm·s/rad] and frictionloss F [Nm] in the Unitree models only (`src/supermimic/deploy/waist_roll.py`) |
| `--vel-check-1khz` | `SM_VEL_CHECK_1KHZ=1` | G2 also aborts when any 1 kHz physics sample exceeds the joint-speed threshold (the real runner checks one sample per 50 Hz step). The 1 kHz peaks (`knee_peak_1khz`, `peak_joint_1khz`, `dq_peak_1khz`, `vel_over_1khz_ms`) are recorded in either case |

- The G1 gate compares against the training environment, so its reference rollout must use the same physics. `sim2sim_ref.py` takes the waist coupling and the waist roll range from the checkpoint (what it was trained with), or from `SM_WAIST_COUPLING` / `SM_WAIST_ROLL` when they are set.
- `--waist-coupling`, `--waist-roll-*` and `--imu-model` cannot be combined with `--lock-waist` or `--cmd lw`.
- Example (the setting used to choose `e24_0355000`):

  ```bash
  python scripts/deploy/sim2sim.py --cmd g2,g3,report --clips shanshen_duosuomao_lean35kfs_pad2 --tag lean35kfs --ckpt e24_0355000 \
      --imu-model --waist-coupling --waist-roll-range 0.72 --waist-roll-damping 1.8 --workers 16
  ```

  `scripts/train/gate_lean35kfs.sh` runs this, plus export, standby and a summary.

**Real-state gate** (`scripts/deploy/realstate_gate.py`). It starts the Unitree-model simulation from *real* logged robot states (joint q/dq, IMU orientation, gyro, the policy's observation history) at given clip frames of your own real runs. From there it runs the deploy core over a latency × kp × mass × friction × velocity-estimate grid. The IMU model and the waist coupling are on by default; `--waist-roll-range R` (or `SM_WAIST_ROLL_RANGE=R`) widens the waist roll for this gate. It prints one `RS` line, with PASS = 0 knee joint-speed aborts and p95 of the 1 kHz knee peak ≤ 13 rad/s, and writes `deploy_artifacts/reports/realstate_<tag>.json`.

```bash
python scripts/deploy/realstate_gate.py --artifacts deploy_artifacts/<ckpt> \
    --clip deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz \
    --real-logs deploy_logs/real/<run_a> deploy_logs/real/<run_b> --waist-roll-range 0.72 --workers 16
```

On this clip, only the knee-abort count of this gate ranked checkpoints in the same order as the real robot did. Its other numbers start from the failure states of *other* policies and were misleading; see [Which sim gates predicted the real result](REAL_ROBOT_LEAN35KFS.md#which-sim-gates-predicted-the-real-result).

## Representative results (early e11 policies, round 2 core defaults)

**Summary**:

| Gate / variant | ckpt_0100000 | ckpt_0116000 |
|---|---|---|
| G1, core defaults (soft-limit clipping off, torque clipping 1.5×) | PASS | PASS |
| G1, torque clipping 1.0× / 0.5× | — | PASS / **FAIL** |
| G1, no clipping | PASS | PASS |
| G1, soft-limit clipping on + 1.5× (round-1 defaults, for reference) | — | **FAIL** (walk_sideway RMSE 2.35×) |
| G2, core defaults, standing start | PASS | PASS |
| G2, hold frame 0 with training gains for 0.5 s before handing over to the policy | **FAIL** | **FAIL** |
| G2, targets clipped to the hard joint range (assuming the firmware clips) | — | PASS (but tracking clearly worse) |
| G2, hip roll effort 88 Nm | — | PASS (bit-identical to baseline) |
| G2, torque clipping 1.0 / 0.5 / no clipping | — | PASS / **FAIL** / PASS |
| G3 transitions (multipliers 0.5/1.0/1.5, latency 0/20 ms; 60 s looping idle) | — | PASS |
| G5 `DeployCore.step` p50 / p99 (single-threaded) | 1.28 / 1.35 ms | 1.28 / 1.34 ms |

**G2 (ground, standing start)**: both ckpts pass. 0 falls and 0 false aborts in the 40 SAFE nominal runs:

| | ckpt_0116000 | ckpt_0100000 |
|---|---|---|
| SAFE RMSE [rad] | 0.047 | 0.061 |
| Fraction of 1 kHz steps with \|tau\| ≤ 0.9 limit | 100% | 99.56% |

The `ckpt_0116000` full sweep is 384 runs in total (kp ×0.85–1.15, +0/+1.8 kg, latency 0–20 ms, μ 0.6/1.0), with 0 falls. All 29 false aborts are in dance_hiphop (HARDER) at around step 190, caused by knee velocity > 0.8 × velocity limit; there are no false aborts on SAFE.

**Torque-clipping multiplier** (G2 nominal, SAFE, 40 runs per row):

| Multiplier | Falls | False aborts | RMSE | Max first-step tau | Steps with clipping | Wrist clipping frequency |
|---|---|---|---|---|---|---|
| 1.5 | 0 | 0 | 0.0472 | 23.6 Nm (knee) | 0 | 0 |
| 1.0 | 0 | 0 | 0.0472 | 23.6 Nm | 0 | 0 |
| 0.5 | **2** | 1 | 0.0598 | 23.6 Nm | up to 66% | worst wrist joint 0.6% |

At multiplier 0.5, both falls are walk_sideway (μ 1.0, latency 10 / 20 ms), and jog_sideway in HARDER falls in all 8 runs. The first-step torque does not depend on the multiplier: clipping has no effect on the first step, and the first target asks for at most 0.30–0.50 × limit.

**G3 transitions** (`ckpt_0116000`, idle clip). The table below gives the maximum torque rate at the switch:

| END variant | Max torque rate |
|---|---|
| New END transition (1 s linear ramp, effort clipping every 2 ms) | **0.1 Nm/ms**, both during the transition and during the 2 s hold after it |
| Cosine target curve; transition command latency 2 ms | 0.1 Nm/ms |
| round-1 sim2sim: switch to standing gains + last-frame pose in one step | 14.8–14.9 Nm/ms (ankle pitch) |
| round-1 runner: standing gains + last policy target | 31.3–31.7 Nm/ms (waist pitch) |

- The new END transition holds at multipliers 0.5/1.0/1.5, latency 0/20 ms and 60 s looping idle (4 loops). Max tau/limit during the transition is 0.31; with the gantry it stays upright with 0 aborts.
- **Without the gantry at END, the robot falls during the standing hold after the transition** (tilt 56°).
- The rate at the start of POLICY is 21.7–21.9 Nm/ms (knee; 13–23.8 across the SAFE clips). It comes from the policy's first target, not from the gain switch: ramping the gains from standing to training gains over the last 1 s of HOLD only brings it down to 20.9. The first-step torque is 17.9–18.6 Nm, about 0.30 × limit. The only way to reduce it would be a target ramp, which is out of distribution and was not simulated.

**Margins to the per-stage abort thresholds**. Margin = threshold ÷ maximum seen; dq is the 50 Hz state checked by the core:

| Stage | clip | Max \|a\| (margin) | Max dq/velocity limit (margin) | Aborts |
|---|---|---|---|---|
| air (feet 5 cm off the ground, lifted from the torso) | idle clip | ≤ 3.55 (≥ 2.8×) | ≤ 0.169 (≥ 1.8×) | 0 |
| feet (gantry carries the load) | idle clip | ≤ 2.14 | ≤ 0.106 (2.8×) | 0 |
| air | walk_sideway | 9.10 (1.10×) | 0.28 (1.07×) | 0, but close |
| air | shoulder_clap | 6.66 | — | 0 |
| slack (multiplier 1.5) | SAFE G2 nominal, 40 runs | 5.45 (2.75× relative to 15) | 0.188 (2.66× relative to 0.5) | 0 |
| slack full sweep | worst: walk_sideway, kp 0.85, +1.8 kg | 9.33 | 0.43 (1.17×) | 0 |
| Training environment fp32 (finite-difference dq) | SAFE | ≤ 5.43 | ≤ 0.187 | — |

**Policy standby** (`ckpt_0116000`, 4 clips: idle, walk_forward_stop, combo_itch_clap_fwd_side, ib_dodge_back_L; each run has 60 s standby + 30 s standby). The `slack` and `ground` profiles give identical results, because torque clipping never triggered:

| kp scale | Passed | Falls | With aborts | Max standby tilt error | Standby horizontal drift | Switch PRE→POLICY / POLICY→POST [Nm/ms] |
|---|---|---|---|---|---|---|
| 1.0 / 1.15 | **192/192** | 0 | 0 | ≤ 5.15° | ≤ 4.8 cm | ≤ 8.61–9.90 (only ib_dodge, whose frame 0 carries velocity); all others ≤ 2.49 |
| **0.85** | **24/96** | 0 | 11 | 15.25° | **5.0 m** (stepping in place, walking around, yaw up to 180°) | 25.2 / 20.8 |

HOLD→STANDBY_PRE is 10–25 Nm/ms, which is the same jump that the old HOLD→POLICY had. Re-running 64 times with the R1+X check actually enforced (latency 0/20 ms): kp 1.0 gives 32/32, with the condition first satisfied at 3.04–3.14 s of standby. kp 0.85 gives 9/32: the check blocked 3 cases that kept walking (never), but 4 runs still aborted during the motion after the check had passed. So **at low stiffness the motion itself fails, and the 3 s check cannot catch it**. For a 300 s long standby, kp ≥ 1.0 gives 32/32, but yaw keeps accumulating (idle ≤ 34.6°, ib_dodge up to 164°). The other 4 SAFE clips give 96/96 at kp ≥ 1.0 and 19/48 at kp 0.85.

**Other**:

- **G5**: `DeployCore.step` p50 1.28 ms, p99 1.34–1.35 ms, of which ONNX itself is p50 1.14 ms. Single-threaded, intra-op spinning disabled, Intel Core Ultra 7 265K, onnxruntime 1.23.2, measured while the same machine was training.
- **Parity**: both ckpts pass. Max differences: obs 4.8e-6 / 5.7e-6, goal 1.8e-7, action 4.8e-6 / 5.7e-6, q_target 1.3e-7, goal-row misalignment 0. TF32 differs from fp32 by 2.8e-3 / 2.5e-3, for reference only.
- **Comparison of the two MJCFs**:

  | Item | Training MJCF | Unitree MJCF |
  |---|---|---|
  | Total mass | 33.341 kg | 35.112 kg |
  | torso_link mass | 7.818 kg | 9.598 kg |
  | Standing pelvis height | 0.757 m | 0.757 m |
  | Joint range | identical | identical |
  | Joint damping / frictionloss | 0 / 0 | 0.05 / 0.2 (frictionloss 0.1 for wrist pitch/yaw) |
  | Joint armature | differs per joint | uniform 0.01 |
  | hip roll effort | 139 Nm (deploy_config is also 139) | **88 Nm** |

  The Unitree MJCF's IMU sensors agree with the state extraction (difference < 1e-15).

## Deployment rules derived from the gates

1. **Soft-limit clipping is off by default.** During training, 61–74% of steps point the target beyond the 0.9 soft limit, so clipping them is out of distribution: walk_sideway's RMSE becomes 2.35×, and the gate FAILs.
2. **Keep the torque-clipping multiplier at the default 1.5**; 1.0 and 1.5 give identical results. 0.5 is only suitable for the idle clip in the air/feet stages, where it never triggers; walk_sideway and faster motions need 1.0 or 1.5.
3. **The policy must start from a HOLD supported by the gantry; do not first hold frame 0 on the ground and then hand over to the policy** (doing that gives 28 false aborts out of 40 runs). Neither HOLD nor END_HOLD can balance on its own on the ground; the robot tips over after about 1–1.3 s, so both R1+A and END must happen while the gantry carries the load. Standby after the policy has taken over needs no support.
4. **The air/feet stages (0.3 / 10) run only the idle clip.** The slack stage cannot use the 0.3 threshold: in the full sweep it triggers in 7 of 240 SAFE runs. HARDER clips need wider thresholds: jog_sideway's \|a\| reaches 10.9, and dance_hiphop's dq/velocity limit reaches 0.90.
5. **Whether the firmware clips out-of-range targets must be confirmed on the gantry first.** If it does, walk_sideway's RMSE becomes 2.2× (0.12–0.14 vs 0.063), tilt is 14.7–17.6°, leaving only about 2.4° to the 20° abort, and \|a\| reaches 12.4.
6. **hip roll**: the Unitree MJCF already limits hip roll to 88 Nm, so all G2/G3 runs were in fact simulated with an 88 Nm motor. In the full sweep, the right hip roll saturated for a cumulative 749 ms over 15 runs, with no falls. Setting `effort_limit` to 88 in the deploy config costs nothing in simulation. The actual motor rating is **UNVERIFIED**.
7. **Real-robot motor stiffness is UNVERIFIED.** At kp 0.85, standby wanders and motions abort. Observe the first real-robot standby under conditions where the gantry can catch the robot, and do not prolong standby needlessly; if you see stepping, sliding or noticeable turning, first let the gantry take the load, then press L2+B.
8. **Limitations**: all gantries are ideal springs (a point force on the pelvis or torso CoM, with no rope-slack dynamics). Real motor friction, firmware handling of targets, the hip roll rating and the IMU frame are all **UNVERIFIED**.

## sim2sim results for the Release model

The deploy clip for `e13i_0291000` is `combo_c4_full_vc_b15f20` (about 46 s). On the Unitree MJCF, the G2 perturbation grid has 0/36 aborts, standby passes 36/36, and the maximum tilt error is 16.1°. Before going on the robot, re-run the two commands from "[How to run](#how-to-run)" above in your own environment, then follow [Real-robot deployment (sim2real)](DEPLOY_G1.md) and open up stage by stage starting from `--stage air`.

## sim2sim evaluation of the waist-locked G1

Some G1 units have the waist roll/pitch (motors 13/14) locked: in lowstate, q, dq and temperature of these two motors are always 0. `--lock-waist` uses MjSpec to remove `waist_roll_joint`, `waist_pitch_joint` and their actuators, sensors and keyframes from **both** MJCFs, welding the torso to waist_yaw_link at q = 0. This matches Unitree's `g1_29dof_lock_waist.xml`: the differences in mass, position and inertia of every body are 0. The robot variant reports q = dq = 0 for 13/14 to DeployCore; the policy outputs the 29-dimensional action as usual, and the targets for 13/14 are computed but not applied. For comparison, waist-locked and non-locked runs are paired under exactly the same conditions; RMSE27 is the RMSE over the 27 joints other than 13/14.

```bash
python scripts/deploy/sim2sim.py --cmd lw        --ckpt ckpt_0116000 ckpt_0100000 ckpt_0140000 ckpt_0144000 --workers 18  # nominal, G1-style + G2
python scripts/deploy/sim2sim.py --cmd lw_robust --ckpt ckpt_0144000 ckpt_0116000 --workers 18                          # kp 0.85/1.15 × +0/+1.8 kg
python scripts/deploy/sim2sim.py --cmd g3 --lock-waist --ckpt ckpt_0144000 ckpt_0116000 --workers 18                    # -> sim2sim_lw_<ckpt>.json
```

Results of `lw` / `lw_robust` are written to `sim2sim_lockwaist.json`. Any g1/g2/g3 gate can take `--lock-waist`; its results are written separately and do not overwrite the non-locked results.

**Gate**: the 5 SAFE clips, G2 kp 1.0 / +0 kg, latency 0/10/20 ms × μ 0.6/1.0. Requires 0 falls, 0 false aborts, and for every clip and every run, RMSE27 ≤ 1.5 × the non-locked result under the same conditions.

| ckpt | G2 falls / false aborts (30 runs) | RMSE27 non-locked → locked | Mean ratio | Worst clip / worst single run | Max tilt error non-locked / locked |
|---|---|---|---|---|---|
| **ckpt_0144000** | 0 / 0 | 0.0561 → 0.0676 | **1.20** | 1.40 / 1.41 → **PASS** | 8.5° / 17.1° |
| ckpt_0116000 | 0 / 0 | 0.0479 → 0.0673 | 1.40 | **1.88 / 2.35** (idle_loop_002) | 9.9° / 15.8° |
| ckpt_0100000 | 0 / 0 | 0.0616 → 0.0761 | 1.23 | 1.26 / 1.38 (largest absolute RMSE) | 7.6° / 16.6° |
| ckpt_0140000 | 0 / 0 | 0.0544 → 0.0750 | 1.38 | 2.15 / 2.31 | 13.0° / 18.2° |

- **How the policy compensates**: the reference motion leans the waist pitch forward by 0.10–0.18 rad. With the waist locked, the torso can only move together with the pelvis, so the policy tilts the whole pelvis forward by about 7–10° (peak 15–17°) to bring the torso to the reference pose. The abort criterion looks at the pelvis, so on SAFE the tilt margin shrinks from about 10° to about 3–4°.
- **Torque**: max \|tau\|/limit rises to 1.18–1.26, mainly at ankle pitch. The original G2 condition "≥ 99% of time ≤ 0.9 limit" is not met by any of the 4 ckpts when the waist is locked (lowest 0.961–0.977).
- **a13/a14**: grow larger with the waist locked (mean \|a\| of `ckpt_0144000` goes from 0.17 to 0.31, max 3.04), but do not saturate: assuming the motors existed, the PD torque demand on SAFE would be at most 0.76 × effort.
- **Robustness** (`ckpt_0144000`, 150 SAFE runs): 0 falls, but with kp 0.85 or +1.8 kg, walk_sideway has 7 false aborts from 20–22° tilt, versus 0 without the waist lock. On HARDER, jog_sideway falls with the waist locked; **HARDER clips are not run on a waist-locked robot**.
- **G3**: the waist lock does not affect the HOLD → POLICY → END transitions (start 21.5–22.0 Nm/ms, END ≤ 0.07 Nm/ms). Without the gantry at END it falls just the same. walk_sideway exceeds 0.3 × velocity limit (0.38–0.69) in the air/feet stages.
- **Corresponding measures in the runner** (see "Waist-locked G1 (`mode_machine = 6`)" in [Real-robot deployment](DEPLOY_G1.md)):
  - 13/14 are excluded from the R1+A `max|q−f0|` check. Do not work around it by raising `--hold-tol`; that loosens the check for all joints.
  - The clip whitelist by default contains only `neutral_idle_loop_001__A081` and `shoulder_clap_004__A041_M`.
  - **Policy standby with the waist locked did not pass the simulation gate**: STANDBY_POST after shoulder_clap passes 0/6, with the robot stepping. Therefore `--lock-waist` implies `--end-hold-legacy`, and the waist-locked standby whitelist is empty by default. Only after `sim2sim.py --cmd standby --lock-waist --ckpt ckpt_0144000` passes in full may clips be added with `--lock-waist-standby-allow`.
- **UNVERIFIED items for the waist-locked variant**: the real waist is locked at 0 rad; the locking structure adds no extra mass; the meaning of `mode_machine = 6`; lowcmd sent to 13/14 is ignored by the firmware; whether the real pelvis also tilts forward by 7–10°.
