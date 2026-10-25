# Method and reproduction details

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

This document collects every actionable number, formula and protocol in the paper (Liao et al., *SuperMimic*, arXiv:2610.28480, CoRL 2026) and lists, item by item, the implementation choices made in this repository. Tags used in the tables: **[paper]** means the paper states it explicitly; **[unspecified]** means the paper does not give it and the reproducers (the upstream repository and this repository) decided it themselves (numbers U1–U20 refer to "[Choices for items the paper leaves unspecified](#choices-for-items-the-paper-leaves-unspecified-u1u20)" below); **[deviation]** means it differs from the paper, or is an addition beyond the paper. The paper PDF is not distributed with the repository.

## Results

The evaluation protocol is the paper's Appendix F (see [Evaluation protocol](#evaluation-protocol)): every clip is played deterministically in full from its first frame, with no noise and no domain randomization. **Normalized return** = Σ per-step reward / (0.1 × number of steps), where 1.0 is perfect tracking; a "failure" means a training termination condition fired.

| Run | Data | Hardware | Observations | Training time | Eval return | Failures |
|---|---|---|---|---|---|---|
| Paper Fig. 4a | LAFAN1, 40 clips | 1×H100 | paper | 1.2 h | 0.82 | – |
| e2 | LAFAN1, 40 clips | 1×RTX 5090 | **paper** | 1.8 h | 0.56 | 9/40 |
| e3 | LAFAN1, 40 clips | 1×RTX 5090 | + anchor-relative pose | 1.66 h / 3.3 h | 0.824 / **0.904** | 5/40 / 3/40 |
| Paper Table A.8 | BONES-SEED, 129.5k clips | 1×H100 | paper | 3 h | >0.90 | 25/129,514 |
| e5 | BONES-SEED, 129.9k clips | 1×RTX 5090 | + anchor-relative pose | 4.2 h / 5.8 h | 0.934 / 0.935 (2000-clip subset) | 6/2000 / 5/2000 |
| `e11c_0220000` (Release) | BONES-SEED, 129,491 clips (`--scope paper`) | – | **paper** | – | 0.9305 (all clips) | 46/129,491 |

- World-model 8-step open-loop normalized error: 0.0065–0.0099 (paper: 0.007–0.011).
- Simulation throughput with 4096 envs is about 2×10⁵ env-steps/s (RTX 5090); after `torch.compile`, one imagination fwd+bwd (batch 2048, H=32) takes 0.33 s.
- 4-GPU data parallelism (explicit gradient all-reduce) reaches 91% efficiency.

## Problem setting and training loop

| Item | Content |
|---|---|
| Formulation [paper] | Goal-conditioned POMDP. Deterministic policy a_t = π_θ(o_t, g_t); actions are normalized joint setpoints |
| Return [paper] | Soft survival weights c_0 = 1, c_{t+1} = c_t (1 − d_{t+1}); J(θ) = E[Σ γ^t c_t r_{t+1}]. The simulator gives hard terminations d ∈ {0,1}; the world model predicts a termination probability d̂ ∈ [0,1] |
| Observation noise [paper] | o_t = ō_t + η_t. The world model predicts only the noise-free ō_t; η_t is sampled outside the model and added afterwards |
| Action noise [paper] | The simulator executes a_t + ε_t, ε_t ~ N(0, 0.1²), independent per dimension. The replay buffer stores the executed noisy actions; this noise only serves to widen the action distribution the world model is fitted on |

Each iteration of Algorithm A.1 has three steps: (1) **collect**: the policy with action noise rolls out in the simulator, transitions are written to a ring buffer, and reset start points are chosen by [return-based reset sampling](#return-based-reset-sampling); (2) **fit the world model**: sample k = 8 step windows, roll f_ω out autoregressively with the recorded actions, and compute the loss per Eq. 1; (3) **improve the policy**: freeze ω, imagine H = 32 steps starting from buffer states, update θ by BPTT per Eq. 2, then fit the critic to GAE targets from the imagined rollouts (Eq. 3). Both ω and θ are updated every iteration. Transitions collected by early policies remain valid for the world model, because the dynamics do not depend on the collecting policy.

## Observations, goals and actions

| Item | Paper | This repository |
|---|---|---|
| Proprioceptive observation [paper] | q − q⁰ (29, noise U±0.01), q̇ (29, U±0.5), IMU angular velocity (3, U±0.2), projected gravity (3, U±0.05), a_{t−1} (29, no noise), 93 dims in total, last 10 frames = 930 dims; no anchor pose and no base linear velocity, i.e. no state estimation is needed | Same as the paper. History is flattened term-major as in mjlab (`[A_t0..A_t9, B_t0..]`); at episode start the 10 frames are back-filled with the first frame |
| Goal g_t | K = 10 future reference frames [paper]; per-frame content and start frame [unspecified U5] | 68 dims per frame = reference joint position − q⁰ (29) + reference joint velocity (29) + reference root projected gravity (3) + reference root angular velocity (body frame, 3) + reference root linear velocity (reference heading frame, 3) + reference root height (1). Frames t..t+9, repeating the last frame past the end of the clip, 680 dims in total, placed in a separate observation group `goal` (no history stacking) |
| Anchor observation | None | **[deviation, off by default]** `--env.anchor-obs` appends a 9-dim BeyondMimic-style anchor-relative pose to the `goal` group (position noise U±0.25, orientation noise U±0.05). See the e2 / e3 comparison in [Results](#results) for its effect; checkpoints trained with it cannot be exported to the real robot |
| Action | q_sp = α a_t + q⁰, with low PD gains as in BeyondMimic [paper]; the value of α and whether to clip [unspecified U6] | α follows mjlab's `G1_ACTION_SCALE` (0.25 · effort / kp, between 0.075 and 0.548 across joint groups); kp = armature·ω², ω = 2π·10 Hz, damping ratio 2; actions are not clipped |
| a_{t−1} and noise [unspecified U9] | – | Both a_{t−1} in the observation and the action smoothness term use the **clean** action output by the policy; ε_t is added only on the execution side (`NoisyJointPositionAction`), and the `a_exec` column in the buffer is used for world-model fitting. At episode start a_{−1} = 0 |
| Input scaling [unspecified] | – | Policy inputs are divided by fixed per-term scales. Observation: q 1, q̇ 5, angular velocity 2, gravity 1, action 1; goal: q 1, q̇ 5, gravity 1, angular velocity 2, linear velocity 1, height 1 |

## Rewards and termination

All reward terms are multiplied by Δt = 0.02 s; the per-step reward upper bound is (1+1+1+1+0.5+0.5) × 0.02 = 0.1. T is the set of 14 tracked bodies from mjlab's G1 configuration (pelvis, left/right hip_roll / knee / ankle_roll, torso, left/right shoulder_roll / elbow / wrist_yaw).

| Reward term [paper Table A.2] | Formula | Weight |
|---|---|---|
| Body position | exp(−mean_b ‖p_b^des − p_b‖² / 0.3²) | 1.0 |
| Body orientation | exp(−mean_b ‖log(R_b^des R_b^⊤)‖² / 0.4²) | 1.0 |
| Body linear velocity | exp(−mean_b ‖v_b^des − v_b‖² / 1.0²) | 1.0 |
| Body angular velocity | exp(−mean_b ‖ω_b^des − ω_b‖² / 3.14²) | 1.0 |
| Root position / orientation | exp(−‖·‖² / 0.3²) / exp(−‖·‖² / 0.4²) | 0.5 / 0.5 |
| Action smoothness | ‖a_t − a_{t−1}‖² | −0.1 |
| Joint limits | Σ_j max(0, q_j − q_j^max) + max(0, q_j^min − q_j) | **−2** (BeyondMimic uses −10) |

- Compared with BeyondMimic, the paper drops the self-collision penalty, on the grounds that this penalty targets reward hacking by stochastic policies, and a deterministic policy has no such loophole in imagination.
- Reference frame [unspecified U7]: follows mjlab's definition. Reference body poses are re-anchored to the robot's anchor, but only the xy translation and yaw are aligned; the body position / orientation terms compare against the re-anchored reference; the root terms compare in the world frame on the anchor **torso_link**; body velocities are compared in the world frame. Rewards are computed analytically with differentiable torch FK and are checked term by term against mjlab's reward functions.
- **Termination [paper]**: pelvis ("robot root (pelvis)") height error |e_z| > 0.25 m, or tilt error e_tilt > 0.8 (the absolute difference between the gravity z components in the reference root frame and the robot pelvis frame, about 78° when the reference is upright). End-effector deviation does **not** trigger termination. Note that the "root" of the rewards is torso_link while the "root" of termination is pelvis; the two use different bodies.
- Episode length [unspecified U15]: training episodes end at the end of the clip with a time-out (not counted as a termination), with an additional 10 s cap (`--env.episode-length-s`). After a clip ends there is no jump to a new clip; neither k-step windows nor imagination start points cross episode boundaries; when imagination runs past the end of the clip, the goal holds the last frame and rollout and reward computation continue.
- Reset noise [unspecified U14]: follows mjlab's tracking-task defaults. Joints ±0.1 rad; root pose xy ±0.05 m, z ±0.01 m, roll/pitch ±0.1, yaw ±0.2; root velocity xy ±0.5, z ±0.2, roll/pitch ±0.52, yaw ±0.78.

## Sim context ξ and domain randomization

| Parameter [paper Table A.3] | Distribution | Implementation in this repository |
|---|---|---|
| Friction coefficient μ | U[0.3, 1.2] | Applied only to foot collision geoms; all foot geoms share one value, resampled every episode [U8] |
| Joint calibration offset | U[−0.01, 0.01] rad | mjlab `dr.encoder_bias`, resampled every episode, acting on both observation (`joint_pos + bias`) and execution (setpoint minus bias). `dr.joint_default_pos` is not used, because it only changes `qpos0` and has no effect on either observations or setpoints [U8] |
| Torso CoM offset | Δx U[±0.025], Δy U[±0.05], Δz U[±0.05] m | `mode="startup"`: fixed per environment. It triggers a `set_const` recomputation; putting it in reset would cause a global recomputation every step |

- ξ has 33 dims (friction 1 + joint offsets 29 + CoM 3); the buffer stores it per step, and it is an input to the world model and the critic.
- The paper **removes** BeyondMimic's root-velocity pushes, on the grounds that a deterministic model cannot fit abrupt state changes. This repository has no pushes either.

## World model

| Item | Paper | This repository |
|---|---|---|
| Components [paper] | ŝ' = f_ω(ŝ, a, ξ); r̂ = r_ω(ŝ, a, g); ō̂ = g_ω(ŝ, ξ); d̂ = d_ω(ŝ, g). Each head can be learned or computed analytically (analytic heads have no loss term) | Dynamics: learned; **observation head: analytic** (the 93-dim observation is computed from ŝ, a_{t−1} and the joint offsets in ξ); **reward head: analytic** (differentiable position FK + velocity FK, matched against MuJoCo to 1e-10 in fp64); **termination head: learned** [U3] |
| State s_t [unspecified U1] | – | Raw MuJoCo (qpos, qvel), 71 dims in total: root position 3 + root quaternion wxyz 4 + joints 29 + root linear velocity 3 + root angular velocity (body frame) 3 + joint velocities 29 |
| Network input [U1] | – | Heading-normalized, 71 dims: root height 1, yaw-removed root rotation in 6D 6, heading-frame root linear velocity 3, body-frame root angular velocity 3, joint positions 29, joint velocities 29. Plus input scales (1, 1, 1, 2, 1, 5) |
| Parameterization [unspecified U4] | – | 70-dim output: velocity increments (dv_h 3, dw_b 3, dq̇ 29) + position corrections (dp_h 3, dθ_b 3, dq 29), followed by **explicit integration**: v' = v + R_h dv_h, p' = p + Δt v' + R_h dp_h, q' = q ⊗ exp(Δt ω' + dθ_b), q_j' = q_j + Δt q̇' + dq. Output scales (0.3, 0.5, 2.0, 0.01, 0.02, 0.02). The last layer is zero-initialized, so training starts from constant-velocity extrapolation |
| Loss Eq. 1 [paper] | k = 8 step autoregressive: Σ_j ‖ŝ_j − s_j‖² + (r̂_j − r_j)² + ‖ō̂_j − ō_j‖² + BCE(d̂_j, d_j) | The observation and reward heads are analytic, so they have no loss terms. State errors are normalized by per-dimension nominal scales and then **summed**: root position 0.05 m, rotation vector 0.1 rad, root linear velocity 0.5 m/s, root angular velocity 1.0 rad/s, joint positions 0.1 rad, joint velocities 2.0 rad/s [U2]. The termination BCE weight is `term_loss_weight` = 1.0. Full BPTT across the 8 steps |
| Termination head | – | Input is the analytic [height error / 0.25, tilt error / 0.8] plus their absolute values, 4 dims in total, through an MLP (64, 64) followed by a sigmoid |
| Precision [paper] | The world model uses TF32 | matmul uses TF32 (mjlab default); state integration and rewards use fp32; bf16 is not used |
| Prediction error | 8-step normalized error 0.007–0.011, measured online on policy rollouts; the definition of "nominal error" [unspecified] | Per-dimension ((ŝ − s)/σ)², averaged first over dimensions, then over the k steps. Because the normalization definitions differ, only trends can be compared. Measured values are in [Results](#results) |

## Policy, critic and BPTT

| Network [paper Table A.4] | Input | Architecture |
|---|---|---|
| Dynamics f_ω | ŝ, a, ξ | MLP (2048, 2048, 2048) |
| Motion encoder | K = 10 future reference frames (680 dims) | MLP (2048, 1024, 512, 256) → z (32 dims, tanh) |
| Shared decoder | z + 930-dim observation history | MLP (2048, 1024, 512, 256) → a (29 dims) |
| Critic V_ψ | ŝ, g, ξ | MLP (256, 128) |

- Activation, normalization, initialization [unspecified U11]: all MLPs use SiLU with no normalization layers; the last dynamics layer is zero-initialized; the last decoder layer uses orthogonal initialization with gain 0.01. About 18M network parameters in total.
- **Critic input [deviation]**: in addition to the heading-normalized state (71) + g (680) + ξ (33), a privileged quantity **e_root (4 dims)** is added: the robot root's position error relative to the reference root (reference heading frame, 3 dims) and the yaw error (1 dim). Without it, neither the normalized state nor g contains xy / yaw drift, and the gradient of V(ŝ_H) with respect to the root rewards (1/5 of the per-step upper bound) is identically zero. The policy does not see e_root.
- **Policy objective Eq. 2 [paper]**: Ĵ = E[Σ_{t<H} γ^t ĉ_t r̂_{t+1} + γ^H ĉ_H V_ψ(ŝ_H, g_H, ξ)]. The action at each step is a_t = π(ō̂_t + η_t, g_t), i.e. the predicted observation plus freshly sampled sensor noise. Gradients flow by first-order BPTT through the world model's Jacobians with respect to ŝ and a. The differences from SHAC are soft survival weights instead of hard terminations, and no EMA target network for the critic.
- **Critic target Eq. 3 [paper]**: R̂_t = r̂_{t+1} + γ(1 − d̂_{t+1})[(1−λ)V_{t+1} + λR̂_{t+1}], R̂_H = V_H. Targets are computed once per rollout and held fixed during fitting.
- Gradient flow [unspecified U10]: ĉ_t keeps its gradient with respect to d̂; the terminal value V(ŝ_H) keeps its gradient with respect to ŝ_H and passes no gradient to ψ; during the policy update ω and ψ are frozen, but backpropagation through their inputs is kept; when fitting the critic, inputs and GAE targets are all detached.
- Imagination start points: (s_0, ξ, clip, frame, history of the previous 9 steps, a_{−1}) are taken from the buffer. When crossing an episode boundary, the history is back-filled with the start-point observation.

## Hyperparameters and schedule

| Category | Paper Table A.5 | This repository's default / CLI |
|---|---|---|
| World model | k = 8; Adam; lr 2e-4; 10 batches × 2048 windows per update; global norm clipping 1.0 | Same (`--wm-trainer.*`). The 10 batches are done as 10 sequential gradient steps [U12] |
| Policy | H = 32; Adam β = (0.9, 0.95); lr same as α_ω; 4096 rollouts per update; clip first by global norm 1.0, then element-wise at 0.1; γ = 0.99 | Same (`--policy-trainer.horizon`, `--policy-trainer.batch-size`). For BONES-SEED training the batch is reduced to 2048 |
| Critic | λ = 0.95; Adam; lr same as α_ω; norm clipping 1.0; 5 epochs × 4 shuffled minibatches per update | Same |
| Collection | 4096 environments per GPU; ε ~ N(0, 0.1²); buffer of 1000 steps per environment | Same (`--env.num-envs`, `--env.action-noise-std`, `--env.replay-capacity`) |
| Schedule | UTD 0.4; 1 policy update per world-model update; cosine decay down to 0.03× at 250k / 320k, then held constant; 320k iterations in total | UTD is implemented with a fractional accumulator, i.e. on average 1 update per 2.5 simulator steps (10 world-model gradient steps + 1 policy update + critic fitting). An "iteration" is read as an **environment step** (reading A, 320k environment steps ≈ 128k updates); see `--schedule.total-iterations`, `--schedule.updates-per-env-step`. The learning-rate schedule is defined proportionally and does not depend on the total length. Before the first update, 40 steps are run without updates to fill the windows [U12] |
| Multi-GPU | Data parallel, gradients averaged across GPUs; lr × √2 with 2 GPUs | torchrun, each rank has its own 4096 environments and buffer, explicit gradient all-reduce, lr × √(number of GPUs), seed = seed + 1000·rank [U13] |

## Return-based reset sampling

**Paper [D.1]**: the motion timeline is split into 1 s bins, and each bin keeps a running estimate G_b of the discounted return starting from there. Reset start points are mixed as follows: 50% uniform over bins; 40% sampled by softmax_b(−25 G_b / G_max); 10% sampled in proportion to each bin's past softmax weights.

**This repository [U14]**:

| Item | Value |
|---|---|
| Bins | One bin per 50 frames (1 s) of each clip, numbered globally across the whole library; the start frame within a bin is sampled uniformly |
| G_max | `gmax_mode="horizon"` (default, `--env.sampler-gmax-mode`): each step's return-to-go is first rescaled to a common scale, G_norm_t = G_t · G_max / A(k_t), A(k) = r_max(1 − γ^k)/(1 − γ), r_max = 0.1, G_max = 10. How k_t is chosen: for episodes that end normally at the end of the clip, the remaining steps of the episode; for episodes that fall, the remaining frames up to the end of the clip (steps lost after the fall count as 0 reward); for episodes that hit the 10 s time-out, the remaining steps of the episode. `const` mode uses a single constant G_max = 10, in which case 94% of the difficulty share lands on the last bin of each clip and the sampler essentially stops working |
| G_b update | Return-to-go is computed backwards at episode end; the update rate adapts to the visit count: α_b = max(0.05, 1/(n_b + 1)) (`--env.sampler-ema-alpha`, `--env.sampler-count-warmstart`), i.e. the first visit takes the observed value directly. The EMA coefficient for the historical weights is 0.01 |
| Time-out truncation | Steps with fewer than 100 remaining steps do not take part in the update (critic bootstrapping is **not** wired into training) |
| State saving | G_b, the historical weights and n_b are all saved in the checkpoint. When resuming with a different motion library, `--resume-reset-sampler` is required |

## Data and filtering

| Item | Paper | This repository |
|---|---|---|
| LAFAN1 | 40 clips, 2.45 h; which clips exactly [unspecified U17] | All 40 CSVs under `g1/` in Hugging Face `lvhaidong/LAFAN1_Retargeting_Dataset`, 2.451 h in total |
| BONES-SEED | G1-retargeted subset: 142,220 → 129,667 after name filtering → 129,514 after feasibility filtering, 264.56 h | `--scope paper`: 129,647 → **129,491**; `--scope d16` (old scope): 130,137 → 129,943 |
| Name filtering | Keyword list adapted from SONIC, not fully published [U16] | SONIC's published full keyword list (including 8 trick-category words) plus words added from the paper's examples (`_box_`, `box_`, `_stair`, `jump_off_50cm`, `off_50cm`, `fence`, `look_over`), matched as substrings against lower-cased file names. Which list the paper actually used cannot be verified; the judgment rests only on matching counts (the counts of the three groups side_roll, fire_in_the_hole and crossed_legs match Table A.8) |
| Feasibility filtering | Elevated support: z_min > 0.20 m for ≥ 0.6 s; ground penetration: z_min < −0.12 m, or < −0.08 m for > 80% of the time; floating feet: both feet off the ground > 0.02 m for ≥ 0.5 s, while the pelvis is upright, at standing height, and the body is nearly static | Same thresholds as the paper. Thresholds the paper does not give [U16]: upright = pelvis gravity z component > 0.9; standing height = pelvis z ∈ [0.60, 0.85] m; static = pelvis velocity < 0.1 m/s and joint velocity RMS < 0.5 rad/s |

### Commands

For BONES-SEED, accept its license on Hugging Face first and download `g1.tar.gz` from `bones-studio/seed`. Then convert it, filter it, and pack it into one library file:

```bash
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/bones_seed/g1 --output-dir data/npz/bones_seed --input-fps 120 --workers 32
uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 24          # -> manifest_paper.csv
uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed \
    --manifest data/npz/bones_seed/manifest_paper.csv --out data/library/bones_seed_compact_paper.pt
```

Train on it (single GPU), or with `torchrun`, one process per GPU:

```bash
uv run sm-train --run-name bones --data.library data/library/bones_seed_compact_paper.pt \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 --schedule.max-hours 20
uv run torchrun --nproc_per_node 4 -m supermimic.scripts.train --run-name bones_4gpu --data.library ...
```

## Evaluation protocol

- **Paper [Appendix F]**: each clip is played once from its first frame; foot friction 1.0; no domain randomization and no observation noise; deterministic physics. Failure = a training termination condition is triggered. Normalized return = undiscounted sum of per-step rewards over the whole episode divided by its maximum possible value. Checkpoints are evaluated every 4000 iterations.
- **This repository** (`scripts/eval_benchmark.py`): action noise and reset noise are likewise turned off, the initial state is exactly the first frame, and episodes have no 10 s cap. Normalized return = Σr / (0.1 × number of steps actually executed); **for failed clips the denominator is also the number of steps actually executed** (the paper does not explain the off-by-one in the denominator, U19), so the number of failures has to be reported separately. It also outputs the step-weighted return, and the mean and maximum pelvis tilt error of each clip (the quantity the real-robot abort is based on).

**Results reported in the paper (for reference)**:

| Experiment | Paper value |
|---|---|
| BONES-SEED, vs. SONIC (128 × L40, 168 h) | After 3 h of training on 1 × H100, exceeds SONIC's normalized return of 0.90. Table A.8 failure counts: SONIC 179 / SuperMimic 1 GPU × 3 h 25 / 2 GPU × 9.8 h 0 (out of 129,514 clips) |
| LAFAN1, vs. PPO-Sim | SuperMimic reaches an evaluation return of 0.82 in 1.2 h on 1 GPU; PPO-Sim does not reach it within 18 h (0.60/0.70/0.72 with 1/4/8 GPUs respectively) |
| Horizon / critic sweep | Without a critic, training fails for H ≤ 16; with a critic, H = 32 reaches 0.82 fastest |
| World-model speed (H100) | More than 10× faster than MuJoCo Warp at 2k–16k environments, 62× faster at 64k environments |
| Push test | Static T-pose, push velocities 0.5–4 m/s; the policy with a critic recovers stability within 0.5–2.4 s |

## Choices for items the paper leaves unspecified (U1–U20)

| ID | Item | Decision |
|---|---|---|
| U1 / U4 | State content and parameterization | 71-dim (qpos, qvel), heading-normalized input, velocity increments + explicit integration |
| U2 | State loss normalization | Nominal scales σ (see [World model](#world-model)), summed over dimensions |
| U3 | Analytic vs. learned heads | Observation and reward heads analytic; termination head learned |
| U5 | Goal content | 68 dims per frame, frames t..t+9, last frame held |
| U6 | α, whether to clip | mjlab `G1_ACTION_SCALE`, no clipping |
| U7 | Reward reference frame and bodies | mjlab re-anchoring definition, 14 bodies, anchor torso_link |
| U8 | Offset semantics and friction scope | `encoder_bias` acts on both ends; friction applies only to the feet and shares one value |
| U9 | Source of a_{t−1} | Clean action; noise added only on the execution side |
| U10 | Critic gradients | Detached during fitting; terminal value keeps its gradient with respect to ŝ_H |
| U11 | Activation and initialization | SiLU, no normalization layers, last dynamics layer zero-initialized, last decoder layer gain 0.01 |
| U12 | UTD and "iteration" | 0.4 updates / environment step; total length counted in environment steps |
| U13 | Multi-GPU buffer | Independent per GPU |
| U14 | Sampler details | Horizon-mode G_max, visit-count update rate, steps with fewer than 100 remaining steps excluded from updates, mjlab reset noise |
| U15 / U20 | Episode length and end of clip | Both clip end and 10 s are treated as time-outs; last frame held in imagination |
| U16 | Keyword list and thresholds | SONIC full list + paper example words; thresholds in the table above |
| U17 | LAFAN1 selection | All 40 clips under HF `g1/` |
| U18 | Velocity encoder features | Not implemented (no velocity interface) |
| U19 | Evaluation length and denominator | Whole clip; denominator is the number of steps actually executed |

## Options that deviate from the paper and known differences

The following training options are **all off** by default. Whenever any of them is used, it should be stated in the report:

| CLI | Effect |
|---|---|
| `--env.anchor-obs` | Appends a 9-dim anchor-relative pose to the goal group (such checkpoints cannot be exported to the real robot; the ONNX export checks this) |
| `--env.tilt-hinge-weight W --env.tilt-hinge-knee K` | Reward term −\|W\|·max(0, tilt_err − K), where tilt_err is the angle between the robot's and the reference's pelvis gravity directions (rad, default K = 0.21); added to both the simulator reward and the imagined reward. The real robot aborts at a tilt error of 20°, whereas the training termination condition is about 78°. The `e13i` fine-tune used W = −24, K = 0.17 |
| `--env.joint-torque-weight W` | **Added only to the imagined reward**: −\|W\|·Σ_j w_j (τ_j / effort_j)², where τ is the PD torque. Default weights: waist roll/pitch 2.0, waist yaw 1.0, knee 1.0, hip roll 0.5, everything else 0.1 (`--env.joint-torque-jw`, `--env.joint-torque-jw-default`). Used to mitigate waist and knee heating on the real robot |
| `--env.dr-kp-range LO HI`, `--env.dr-torso-mass-add LO HI` | Per-episode randomization of the PD stiffness scale and an added torso mass, matching the perturbation range of the sim2sim G2 gate (kp 0.85 / 1.15, +1.8 kg) |
| `--env.sampler-hard-clip-cap F` | Cap on a single clip's share of the 40% difficulty portion |
| `--env.fail-reset-frac F --env.fail-reset-list FILE [--env.fail-reset-window-s 3.0]` | A fraction F of resets start uniformly within the window before known failure frames. FILE is a json produced by the benchmark, or a csv in the format `clip_name,fail_step` |
| `--env.lock-waist` (`--env.lock-waist-refs feasible\|original`, `--env.locked-action-weight -0.1`) | Waist-locked G1 (`mode_machine = 6`): waist roll/pitch are removed from the model, the 29-dim interface is unchanged, actions 13/14 are ignored and their observations are always 0. `feasible` mode zeroes the waist roll/pitch in the reference and redoes FK |
| `--schedule.lr-constant LR` | Use a constant learning rate instead of the cosine schedule when fine-tuning |
| `--env.sampler-gmax-mode const` | Fall back to the old sampler (single constant G_max) |

**Things in the default configuration that differ from the paper or are not in the paper**:
- The critic has an extra privileged input e_root (4 dims).
- Training episodes have a 10 s cap, and mjlab's default reset noise is added.
- The sampler uses horizon-mode G_max and a visit-count update rate (both within what U14 allows, but both are our own reading).
- The hardware is an RTX 5090, not an H100, so GPU hours are not directly comparable.
- BONES-SEED has 129,491 clips after filtering; the paper has 129,514.
- The paper has no real-robot deployment; the deployment stack and the custom-motion fine-tuning are our additions.

**Not implemented**: the velocity interface and its distillation / RL encoder, the PPO-Sim and PPO-WM baselines, the H × critic ablation, the data-length sweep (Fig. 3b), and the SONIC weight comparison (paper Appendix B, G and Table A.6 give the full setup for the multi-interface and velocity interfaces: command ranges, velocity rewards, fall-only termination at h_root < 0.35 m or tilt of about 73°; not implemented in this repository). The push-test script is `scripts/eval_push.py`.

**Implementation notes**:
- mjlab's `csv_to_npz` quaternion slerp returns early when adjacent frames are nearly identical, which makes the orientation lag by one frame. This repository uses its own vectorized slerp (matching scipy to 1e-9).
- MuJoCo Warp's solver-overflow printf is a compile-time constant; it must be turned off after `put_model` and before graph capture (`QuietSimulationCfg`). Otherwise early policies trigger it on every step and the simulation runs 10× slower.
- Under torchrun, MuJoCo Warp always uses the first visible GPU, so `CUDA_VISIBLE_DEVICES` has to be set per rank (`sm-train` handles this automatically).
- mjlab is pinned to main-branch commit `8bb2bc9` rather than 1.6.0 on PyPI (the two differ in their mujoco-warp version and in how `csv_to_npz` handles the last frame).
- Correctness tests (`tests/`): FK, analytic rewards, the observation head and state write-back are all checked element by element against MuJoCo / mjlab; TF32 is turned off in the tests.
