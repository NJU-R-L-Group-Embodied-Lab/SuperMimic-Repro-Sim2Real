"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    # Waist-locked G1 (real robot mode_machine 6, "g1_29dof_lock_waist_rev_1_0"): waist roll/pitch removed from the
    # simulated model, 29-d action/observation interface unchanged (actions 13/14 ignored, q/dq 13/14 read 0).
    lock_waist: bool = False
    lock_waist_refs: str = "feasible"  # feasible: ref waist roll/pitch (+ rates) = 0, bodies by FK | original
    locked_action_weight: float = -0.1  # reward weight of a_13^2 + a_14^2 (dt-scaled like every term)
    # DEVIATION from the paper when != 0 (default off): reward weight of max(0, tilt_err - tilt_hinge_knee) [rad], tilt_err
    # = angle between robot and reference pelvis projected gravity (what the deploy runner aborts on at 20 deg). Applied
    # identically in the simulator reward and the imagined (analytic) reward.
    tilt_hinge_weight: float = 0.0
    tilt_hinge_knee: float = 0.21
    # DEVIATION from the paper when != 0 (default off), imagined reward only: weight of sum_j jw_j (tau_j / effort_j)^2 with
    # tau the PD torque kp (q0 + scale a - q) - kd dq (mdp/rewards.py). jw: joint_torque_jw for the listed joints, else
    # joint_torque_jw_default. Targets the waist / knee heating seen on the real robot.
    joint_torque_weight: float = 0.0
    joint_torque_jw: tuple[tuple[str, float], ...] = (
        ("waist_roll_joint", 2.0), ("waist_pitch_joint", 2.0), ("waist_yaw_joint", 1.0),
        ("left_knee_joint", 1.0), ("right_knee_joint", 1.0), ("left_hip_roll_joint", 0.5), ("right_hip_roll_joint", 0.5),
    )
    joint_torque_jw_default: float = 0.1
    # DEVIATION from the paper when set (default off): per-reset PD stiffness scale range (all actuators) and extra
    # torso mass range [kg], matching the sim2sim G2 robustness axes (kp 0.85/1.15, +1.8 kg).
    dr_kp_range: tuple[float, float] | None = None
    dr_torso_mass_add: tuple[float, float] | None = None
    # Return-based reset sampler (paper D.1, plan D14; envs/mdp/commands.py module docstring).
    # horizon: each step's return-to-go is normalised by the discounted return achievable over its horizon (to the clip
    # end, or the observed 10 s window), so G_max is the "maximum achievable discounted return" per step (fixes the
    # clip-tail bias: under const, the last bin of every clip took most of the hard share); const: single G_max = 10
    # for every step (e1-e11b behaviour).
    sampler_gmax_mode: str = "horizon"
    sampler_softmax_temp: float = 25.0  # paper: softmax(-25 G_b / G_max)
    sampler_ema_alpha: float = 0.05  # G_b EMA coefficient per update (paper: unspecified, U14; plan D14: 0.05)
    # Count-based warm start of G_b: rate max(sampler_ema_alpha, 1/(n_b+1)) with n_b the bin's update count, so the
    # first visit sets G_b to the observed return (plan D14 correction 2026-10-24). n_b is reset by
    # --resume-reset-sampler and saved in checkpoints; a checkpoint without n_b resumed without the reset keeps the
    # plain EMA (n_b treated as saturated), so e11b-style resumes are unchanged.
    sampler_count_warmstart: bool = True
    # DEVIATION from the paper when > 0: no clip may take more than this fraction of the 40% hard share.
    sampler_hard_clip_cap: float = 0.0
    # DEVIATION from the paper when > 0 (failure-point-reset fallback, default off): this fraction of training resets
    # starts uniformly in the fail_reset_window_s seconds before a known failure frame. fail_reset_list: a benchmark
    # json (scripts/eval_benchmark.py output, failed clips with fail_step) or a csv of clip_name,fail_step lines.
    # Must be reported as a deviation from the paper whenever used.
    fail_reset_frac: float = 0.0
    fail_reset_list: Path | None = None
    fail_reset_window_s: float = 3.0

    def __post_init__(self) -> None:
        if self.fail_reset_frac > 0 and self.fail_reset_list is None:
            raise ValueError(f"--env.fail-reset-frac {self.fail_reset_frac} > 0 needs --env.fail-reset-list")


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None
    # Fine-tuning: constant learning rate (world model; policy and critic keep their configured ratio to it) instead
    # of the cosine schedule. Single-GPU value, x sqrt(world_size) like the base rates. Applied from the first update
    # after start/resume (the effective rates are printed at start).
    lr_constant: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    # On resume, reset the sampler statistics (G_b <- G_max, memory <- uniform, visit counts n_b <- 0) instead of loading them; required when
    # the library changed (different bins). Model and optimizer states are still loaded.
    resume_reset_sampler: bool = False
