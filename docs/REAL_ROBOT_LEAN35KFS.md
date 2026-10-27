# Real-robot result: spin and landing dance `shanshen_duosuomao_lean35kfs_pad2` (checkpoint `e24_0355000`)

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

> **Safety warning**: everything in [Real-robot deployment](DEPLOY_G1.md) applies. This clip contains two fast spins and a jump with a one-foot landing. Both successful runs were done with the rope attached. Use at your own risk.

**Result.** Checkpoint `e24_0355000` with the clip `shanshen_duosuomao_lean35kfs_pad2` (1097 frames at 50 Hz, 21.9 s) is the first policy that ran this dance on our real G1 from frame 0 to the end of the clip (frame 1096). It did so in two runs in a row, both at `--stage ground` with the rope attached but slack. Before it, every real run of 9 earlier checkpoints of this dance aborted between frames 550 and 744, or never started (one run stopped on motor temperature during standby).

**How much this shows.** This is one successful checkpoint, run twice. The two runs show that the result repeats; they are not two independent samples. We did no ablation, so we cannot say which training change made the difference. The explanations below come from the real logs, but they are our reading of them, not tested causes. We have not run it without the rope.

The checkpoint, the deploy bundle, the clip and the training inputs are in the [v1.1 Release](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases/tag/v1.1). The training recipe is [`scripts/train/train_lean35kfs_e24.sh`](../scripts/train/train_lean35kfs_e24.sh), and the sim2sim gate is [`scripts/train/gate_lean35kfs.sh`](../scripts/train/gate_lean35kfs.sh).

## Files and where they go

Release asset `e24_0355000_lean35kfs.tar.gz` (v1.1). The archive is laid out relative to the repository root, so extract it there:

```bash
tar -xzf /path/to/e24_0355000_lean35kfs.tar.gz -C /path/to/SuperMimic-Repro-Sim2Real
```

| Path after extraction | What it is | Used by |
|---|---|---|
| `deploy_artifacts/e24_0355000/policy.onnx` | The policy, **ONNX IR version 9**: it loads on PC2's older onnxruntime as it is (unlike v1.0, no renaming is needed) | runner, sim2sim |
| `deploy_artifacts/e24_0355000/policy_ir10.onnx` | The same graph as exported (IR 10), for newer onnxruntime versions | — |
| `deploy_artifacts/e24_0355000/deploy_config.json` | Joint order, gains, limits, action scale | runner, sim2sim |
| `deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz` | Goal table of the clip (`export --cmd clips`) | runner, sim2sim |
| `deploy_artifacts/ckpts/e24_0355000.pt` | Training checkpoint | `sim2sim_ref.py`, `play.py`, further fine-tuning |
| `data/npz/lean35kfs_only/shanshen_duosuomao_lean35kfs_pad2.npz` | The 50 Hz reference clip (training format) | `export --cmd clips`, `build_library.py`, `play.py` |
| `data/library/lean35kfs_only.pt` | Packed one-clip library (`--data.library`) | training |
| `data/real_reset/lean35kfs_landing_real_e22e23.npz` | 87 real-robot start states (`--env.real-reset-file`) | training |
| `deploy_logs/real/e24_0355000_lean35kfs_run1/` | Runner log (`.npz` + console) of the first successful real run | `realstate_gate.py`, `build_real_reset_states.py`, your own analysis |
| `deploy_artifacts/e24_0355000/SHA256SUMS` | SHA-256 of every file in the archive | `cd` to the repo root, then `shasum -a 256 -c deploy_artifacts/e24_0355000/SHA256SUMS` |

In the repository itself: `motions/shanshen_duosuomao/reset_points_lean35kfs.csv`, the failure-point reset list (`clip_name,fail_step`, 37 rows) used with `--env.fail-reset-list`.

To rebuild the library from the clip npz:

```bash
uv run python scripts/data/build_library.py --npz-dir data/npz/lean35kfs_only --out data/library/lean35kfs_only.pt
```

The real-reset file was made with `scripts/data/build_real_reset_states.py` from three failed real runs of two earlier checkpoints (29 policy rows each, log frames 696–724, the stretch before the second jump). Its source real logs are not released. With your own logs:

```bash
uv run python scripts/data/build_real_reset_states.py \
    --library data/library/lean35kfs_only.pt --clip shanshen_duosuomao_lean35kfs_pad2 --frames 696 724 \
    --out data/real_reset/lean35kfs_landing_real.npz deploy_logs/real/<run_a>/run_*.npz deploy_logs/real/<run_b>/run_*.npz
```

## How the clip was made

The source motion `shanshen_duosuomao` is a G1 retarget of a dance video (30 fps, 450 rows), made in the way described in [Video to motion](VIDEO_TO_MOTION.md). Its source CSV is not part of the repository; the released `_pad2` npz is the end product. It went through three feasibility edits. Each one fixed a problem that we saw on the real robot or in sim2sim:

1. **lean35: pelvis tilt limit.** In one fast spin the retargeted pelvis tilts up to 59.3°. Pose estimation from the video gives at most 51° for the human, so the extra tilt comes from retargeting. A one-off script (not in this repository) shrank the swing part of the pelvis orientation (swing-twist decomposition: twist about world z unchanged, swing soft-capped at 31°, cosine blends of 6 rows at the window edges). It then re-solved the waist joints to keep the torso orientation close to the original, and re-solved both legs by IK to keep the original foot positions. The maximum pelvis tilt went from 59.3° to 31.0°. The arms are unchanged. Converted to 50 Hz with `scripts/data/csv_to_npz_batch.py`, this is `data/npz/shanshen_lean35_src/shanshen_duosuomao_lean35.npz` (unpadded; steps 2 and 3 work on unpadded clips).
2. **kf: knee feasibility.** The knee speed of the reference is capped at 10.9 rad/s (the real runner aborts at 16 rad/s). Around frames 535–555 the right knee is kept at or above 0.15 rad, because a real run aborted at frame 550 when the right knee flexed again from almost straight (16.5 rad/s). The edit goes into the swing phase, and the stance hip and ankle are re-solved to keep the foot in place:

   ```bash
   uv run python scripts/deploy/knee_feasibility.py \
       --src data/npz/shanshen_lean35_src/shanshen_duosuomao_lean35.npz \
       --dst data/npz/shanshen_lean35kf_src/shanshen_duosuomao_lean35kf.npz \
       --frame-offset 149 --floor-windows 535 555 0.15
   ```

3. **s: spins and landing.** The pelvis yaw rate in the spins (frames 628–792) is capped at 6.0 rad/s, below the 8.73 rad/s (500°/s) range of the real pelvis gyro. The right-foot landing of the second jump is changed to a flat foot with the knee pre-bent to 0.7 rad at touchdown (frame 742), and the landing point is moved under the hip:

   ```bash
   uv run python scripts/deploy/spin_feasibility.py edit \
       --src data/npz/shanshen_lean35kf_src/shanshen_duosuomao_lean35kf.npz \
       --dst data/npz/shanshen_lean35kfs_src/shanshen_duosuomao_lean35kfs.npz --frame-offset 149 \
       --report data/npz/shanshen_lean35kfs_src/spin_feasibility.json
   ```

4. **Padding.** A 1 s standing pose `stand_idle` is added at both ends with 2 s cosine transitions, and a 1 s settle at the end. The frame offset 149 above = 50 `stand_idle` frames + 99 transition frames. To make a `stand_idle`, see [Custom motions](CUSTOM_MOTIONS.md#csv--training-and-deployment).

   ```bash
   uv run python scripts/deploy/make_combo_clip.py --npz-dir data/npz/shanshen_lean35kfs_src --out-dir data/npz/shanshen_lean35kfs_pad \
       --clips stand_idle shanshen_duosuomao_lean35kfs stand_idle \
       --name shanshen_duosuomao_lean35kfs_pad2 --blend-s 2.0 --end-settle-s 1.0
   uv run python scripts/deploy/spin_feasibility.py verify \
       --old data/npz/shanshen_lean35kf_pad/shanshen_duosuomao_lean35kf_pad2.npz \
       --new data/npz/shanshen_lean35kfs_pad/shanshen_duosuomao_lean35kfs_pad2.npz
   ```

All frame numbers in this document are frames of the padded clip.

## What makes frames 640–800 hard

This stretch is two spins, not one landing. Reference (forward kinematics of the clip):

- 640: crouch. 664–686: spin on the right foot (first spin); the left foot rises to 0.36 m and the pelvis yaw rate peaks at about 6.8 rad/s in the original clip.
- 690–705: both feet down again, crouching, reference knee angle 1.0–1.18 rad.
- Second jump: the left foot leaves the ground at about 723, flight about 723–734, the right foot lands flat at about 734–742 while turning at about 5.8 rad/s.

On the real robot the landing comes 5–10 frames earlier than in the reference (about 730 for the successful policy, 726–729 for the failed ones). Every failure after the first spin happened at the take-off or around the right-foot landing of the second jump.

## Failures on the real robot and their causes

Real runs of this dance before the successful one, in order. Each checkpoint label gives the clip version it was fine-tuned on and its training iteration.

| Checkpoint | Clip | Real outcome | What we learned |
|---|---|---|---|
| lean35 fine-tune, 310k (2 runs) | lean35 | Abort at 742 / 743, right knee over speed | Landing knee speed (below) |
| lean35 fine-tune, 319k | lean35 | Abort at 716, tilt error 20° | Waist roll could not reach the target (below); command latency |
| + waist coupling, 325k / 327k | lean35 | Abort at 550 (right knee 16.5 rad/s), 739 (right knee) | Reference knee too fast: made the kf clip |
| lean35 fine-tune, 321k | lean35 | Stopped on motor temperature in standby; no motion | — |
| lean35kf, 335k | lean35kf | Abort at 737, right knee −16.8 rad/s | Landing reflex |
| lean35kf, 339k | lean35kf | Abort at 744, tilt error 22° | Landing reflex, knee collapse |
| lean35kfs, 342k (parent of e24) | lean35kfs | Abort at 734, right knee 17.3 rad/s | Landing reflex |
| lean35kfs, 336k | lean35kfs | Abort at 727, left knee −16.7 rad/s | Collapse already in the first spin |
| **`e24_0355000`** (2 runs) | **lean35kfs** | **Full clip, 0 → 1096** (standby before the motion 32.8 s and 44.8 s) | — |

Abort thresholds are those of `--stage ground`: joint speed > 0.8 × velocity limit (16 rad/s for the knees on the 50 Hz sample), tilt error > 20°.

### 1. Pelvis gyro saturation in the spins

The real G1 pelvis gyro saturates at ±500°/s (8.73 rad/s) per axis. During the spins the z axis sat at that limit for 3–10 frames. Over these short times the onboard attitude behaves as a pure gyro integrator. Integrating the clipped gyro reproduces the logged orientation, so the rotation the gyro did not report is missing from the attitude, and the projected-gravity observation is 10–20° off after a spin. The policy never saw this in training.

What we changed: the spin feasibility edit (yaw rate ≤ 6 rad/s); a saturating IMU observation model in training (`--env.imu-sat-prob`, `src/supermimic/envs/imu_model.py`) and in sim2sim (`--imu-model`, `src/supermimic/deploy/imu_model.py`); and a pelvis yaw-rate hinge (`--env.yaw-rate-hinge-*`). Saturation was not removed: the successful run still had 4 saturated frames at 724–727 (take-off to touchdown of the second jump). The difference is the first spin. The successful policy's yaw rate peaked at 7.96 rad/s in frames 650–700, with no saturation, while every other run that reached this stretch saturated for 3–11 frames there.

### 2. Waist roll: coupling and range

Both MJCFs model waist roll and pitch as independent joints with ±0.52 rad ranges. On the real G1 they are driven through a linkage. From all real logs, the reachable roll shrinks with pitch: |roll| + 1.4 |pitch| ≤ 0.95. On the real robot, at frames 702–711 the waist was at pitch 0.53 and roll stalled at −0.17 to −0.22 against a −0.95 to −1.65 target, which led to a tilt abort. On the other hand, the real roll goes beyond the MJCF limit when pitch is small: up to |roll| 0.70. In sim it stops at 0.52. A fit of the waist-roll tracking also shows that the real joint is about 2× more damped than in sim.

What we changed: the coupling as a physics constraint (`--env.waist-coupling`; sim2sim `--waist-coupling`); waist roll range ±0.72 with damping and friction randomisation (`--env.waist-roll-range 0.72 --env.dr-waist-roll-damping 0 3.6 --env.dr-waist-roll-friction 0 2`; sim2sim `--waist-roll-range 0.72 --waist-roll-damping 1.8`). On the real robot, the successful policy's maximum waist roll over frames 640–780 was 0.53. The four failed lean35kf/kfs runs reached 0.68–0.71.

### 3. Landing "tuck" reflex and knee over speed

This was the failure that remained after the clip edits. In the failed lean35kf/kfs runs (335k, 339k, 342k), the policy followed the reference into a toe-down foot before touchdown (foot pitch −58° to −71°). Then, 2–3 frames after touchdown, it commanded the leg to tuck:

- knee target jumps to about 2.0 rad (2.03 / 1.97 / 2.01 in the three runs);
- hip target −2.6 rad (hip flexion);
- the ankle pushes down.

As a result, the right knee speed went 0 → 17.3 rad/s and the run aborted at 734. In the 339k run, the knee sank to 1.45–1.49 rad and could not be pulled back even at the 139 Nm torque limit; the waist roll reached +0.68 and the tilt reached 22°, so it aborted at 744. The 336k checkpoint failed earlier. In the first spin (670–681) the stance right knee sank to 1.47 (reference 0.28) with the ankle at its 50 Nm limit, and the left knee target later swung between −0.64 and 2.24 (bang-bang) before a left-knee abort at 727.

### Things we ruled out

- **The state before the spins (500–640).** The four runs we compared had a similar state there: mean tilt 2.4–8.1°, about half the reference yaw rate, and the successful policy's waist roll was not smaller. The difference comes from what the policy does in 670–740.
- **Better joint tracking.** No. In 700–722 the successful policy has the largest leg RMS error of these runs (0.377 rad vs 0.25–0.31), but the smallest tilt (6.7° vs 8.1–15.2°). It gives up tracking to keep its posture.
- **A pre-set negative ankle angle.** The failed 336k run also held the right ankle at −0.66 to −0.85 before landing, so this alone does not explain it.

## What e24 changed, and why we think it worked

`e24_0355000` is a fine-tune of the 342k lean35kfs checkpoint (the one that failed at 734): 342k → 356k iterations, learning rate 1e-5, clip lean35kfs, all earlier flags kept. Only three things were **added**:

1. **Real-state resets**: `--env.real-reset-file data/real_reset/lean35kfs_landing_real_e22e23.npz --env.real-reset-prob 0.2 --env.real-reset-vel clip`. These are 87 states from the three failed lean35kf/kfs landing runs (335k, 339k, 342k), 29 each, at reference frames 695–723. This is our best guess for the main cause: it trains exactly the stretch where the landing reflex went wrong, starting from states the real robot actually reached.
2. **Pelvis push DR**: `--env.dr-push-prob 0.2 --env.dr-push-frames 690 722 --env.dr-push-clips lean35`, on the same stretch. We cannot separate it from item 1.
3. **Waist roll model**: `--env.waist-roll-range 0.72 --env.dr-waist-roll-damping 0 3.6 --env.dr-waist-roll-friction 0 2.0`.

Knee effort-limit scaling (`--env.dr-knee-effort-scale`) was **not** used. The full command with comments is [`scripts/train/train_lean35kfs_e24.sh`](../scripts/train/train_lean35kfs_e24.sh). A later fine-tune that changed several settings at once (another parent checkpoint, a stronger tilt hinge, new reset points, reset probability 0.3, no pushes) gave no checkpoint that met the rule of thumb below (worst tilt error ≥ 16.1° for all of them). It is not an ablation of any single item.

**The landing strategy changed** (first successful run):

- **Before touchdown (721–729) it does not follow the reference.** The right knee is almost straight (0.24 → 0.05 rad; reference 0.39–0.54). The right ankle stays at −0.58 to −0.69 rad (reference +0.3 to +0.5), toes up. The right hip target is positive (+0.42 to +0.86), which gives a hip-extension torque that pushes the leg down.
- **Touchdown (about 730) is heel first, then the foot goes flat.** The knee flexes slowly, 0.17 → 1.00 rad, at |dq| ≤ 7.6 rad/s, while the hip flexes passively (−0.54 → −1.05) against the extension torque. Hip and knee absorb the impact together, eccentrically.
- **After touchdown the hip target stays extended** (−0.16 to +0.22 in 730–744), and a knee-extension torque then brings the knee back from 1.0 to 0.75.
- In the landing window 722–745: right knee |dq| max 7.6 rad/s, right ankle 5.8; right knee target max 1.30 rad. There is no tuck command.

## Safety margins (first successful run, 1095 POLICY steps)

No joint went above 0.8 of its abort threshold. The failed lean35kf/kfs runs had 1, 6, 4 and 4 steps above it. The five closest margins:

| Rank | Quantity | Value / abort threshold | Fraction | Frame |
|---|---|---|---|---|
| 1 | Right knee dq | −12.41 / 16.0 rad/s | 0.776 | 536 (mid-motion; the reference itself needs 10.9) |
| 2 | Left hip yaw dq | −18.28 / 25.6 rad/s | 0.714 | 717 |
| 3 | Left knee dq | 10.68 / 16.0 rad/s | 0.667 | 600 |
| 4 | Tilt error | 13.19 / 20.05° | 0.658 | 750 |
| 5 | Left shoulder pitch dq | 18.56 / 29.6 rad/s | 0.627 | 243 |

- Max |action| 13.1 (abort at 50), no clamping. Max motor temperature during POLICY 55 °C. Policy compute p99 4.94 ms, max lowstate age 2.08 ms.
- **The knee margin is smaller than it looks.** On this robot a knee can change by 15 rad/s within one 50 Hz step (a failed run went −1.67 → −16.83 rad/s from frame 736 to 737). A 3.6 rad/s margin is not "safe".
- The real joints went past the MJCF limits: left ankle roll −0.268 (MJCF ±0.262), right ankle roll −0.326, waist pitch 0.613, waist roll 0.571. The real mechanical limits are wider than in the MJCF.

## Which sim gates predicted the real result

Gate script: [`scripts/train/gate_lean35kfs.sh`](../scripts/train/gate_lean35kfs.sh). All sim2sim options are described in [Export and sim2sim gates](SIM2SIM.md#real-g1-models-and-extra-checks-opt-in).

**Ranked the checkpoints correctly** (on the few checkpoints that were both gated this way and run on the robot):

- **Frame-0 G2 with the full real-G1 model**, ground profile, 36 runs: IMU saturation model + waist coupling + waist roll range 0.72 / damping 1.8 (`--imu-model --waist-coupling --waist-roll-range 0.72 --waist-roll-damping 1.8`).
  - `e24_0355000`: 0 aborts, 0 falls, worst tilt error 13.0°.
  - lean35kfs 342k: 4 aborts (right knee at 733 / 733 / 734 / 746, matching the real abort at 734) and 3 falls, worst tilt error 24.7°.
  - lean35kfs 336k: 1 abort, worst tilt error 22.0°.
- **IMU gravity-estimate error** in the same runs: `e24_0355000` 0.5°, 336k 4.5°, 342k 5.0°.
- **Real-state gate, knee-abort count only** (`realstate_gate.py`, its own PASS criterion): `e24_0355000` 95, the best of all checkpoints gated; 336k 256; 342k 418.

**Misleading:**

- The real-state gate's clean rate, tilt-abort count and fall count. It starts sim from the real states of *other* policies' failed runs, including their observation history. `e24_0355000` never reaches those states on the robot: its joint L2 distance to them is 0.75–1.83 rad, while they are only 0.34–1.06 apart from each other. The same is true for 336k, which this gate rated best and which failed. So these numbers carry no information in either direction.
- The 1 kHz knee peak criterion (the `knee_peak_1khz` fields, ≤ 13 rad/s). `e24_0355000` fails it (17.6 rad/s, on the slack profile), while an earlier lean35kfs checkpoint that failed on the robot passed it: the opposite of the real result.
- Frame-0 G2 without the real-G1 models: almost every failed checkpoint scored 0/36 aborts.
- Standby tilt.
- The formal `--clips` gate verdict is FAIL for `e24_0355000`, because the `slack` profile (0.5 × velocity limit) aborts on the knees. This clip can only run at the `ground` stage.

**Rule of thumb we would use next time.** It was chosen after the fact, on one successful checkpoint, so treat it as a heuristic:

1. Frame-0 G2 with the full real-G1 model: 0/36 aborts, 0 falls, worst tilt error ≤ 14°.
2. IMU gravity-estimate error ≤ about 1°.
3. Use only the knee-abort count of the real-state gate as a hint; do not reject a candidate on its clean rate, tilt or fall counts.

Among the checkpoints of the same fine-tune, 343k–356k, only 355k met both 1 and 2 (the gravity error of 343k / 353k / 354k was 1.6° / 3.2° / 2.7°). The full real-G1 model was never run on the lean35 and lean35kf checkpoints, so we do not know if the rule would have rejected them.

## Deploying it

Follow [Real-robot deployment](DEPLOY_G1.md) step by step: read-only check, debug mode, button test, then the stages from `air`. The dance itself was run at `--stage ground`. Our command on the onboard computer PC2 (over ssh, deployment environment active, repository root; we start it through a small wrapper script that is not part of the repository):

```bash
python scripts/deploy/bridge/g1_runner_bridge.py --iface eth0 --domain 0 --stage ground --allow-ssh \
  --artifacts deploy_artifacts/e24_0355000 \
  --clip deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz \
  --log-dir deploy_logs/real \
  --thermal-stop-temp 78 --max-temp 90 --warn-temp 75
```

- **Temperature thresholds.** These runs used thresholds above the runner defaults (65 / 80 / 70 °C): a smooth thermal stop at 78 °C in POLICY, DAMP above 90 °C, and no start above 75 °C. These values are **UNVERIFIED** against Unitree's motor ratings. In the successful runs the hottest motor in POLICY was 55 °C, so the higher thresholds were never reached. Before the motion starts, the R1+X temperature-rise forecast must stay below the thermal-stop limit (see [Thermal protection](DEPLOY_G1.md#thermal-protection-bridge-runner)). If you are unsure, keep the defaults.
- **Procedure.** Run it as in [Operator procedure (policy standby)](DEPLOY_G1.md#operator-procedure-policy-standby): L2+up → HOLD on the rope → R1+A → STANDBY_PRE, with the feet carrying the weight and the rope just slack enough for the clip, at least 10 s of standby (needed for the temperature forecast) → R1+X. After the motion the robot stands by in STANDBY_POST. Re-hang it before L2+B.
- **Space.** The clip travels about 1 m and turns about 86°. The path length of the pelvis is about 6 m.
- **Sync video (optional).** `--sync-udp HOST:PORT` with `scripts/deploy/bg_sync_server.py` plays the source video in sync with the robot.

## Open items

- Two runs of one checkpoint: we have no second independent checkpoint that succeeded.
- No ablation of real-state resets vs pushes vs the waist roll model.
- No run without the rope.
- The rule of thumb above has not been tested on a new clip.
