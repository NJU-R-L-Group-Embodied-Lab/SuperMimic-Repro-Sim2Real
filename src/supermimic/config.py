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
    # DEVIATION from the paper when True (default off): real-G1 waist roll/pitch coupling (parallel linkage) as a physics
    # constraint |roll| + k |pitch| <= c (two limited fixed tendons, supermimic.deploy.waist_coupling; joint ranges
    # unchanged). k, c fitted on the real deploy logs (2026-10-26). Simulator only: world model / imagination unchanged
    # (the world model learns the constraint from the simulator transitions).
    waist_coupling: bool = False
    waist_coupling_k: float = 1.4
    waist_coupling_c: float = 0.95
    # DEVIATION from the paper when set (default off): waist_roll_joint range +-waist_roll_range [rad] in the simulator
    # MJCF instead of +-0.52 (the real G1 reaches |roll| 0.70; supermimic.deploy.waist_roll). The joint-limit reward,
    # the reset clamp and the imagined reward keep the nominal soft limits (0.9 x 0.52) unless
    # waist_roll_model_soft_limits, so only the physics changes. The waist coupling (if on) is unchanged.
    waist_roll_range: float | None = None
    waist_roll_model_soft_limits: bool = False
    # DEVIATION from the paper when set (default off, needs domain_randomization): per-reset waist-roll joint damping
    # [Nm s/rad, abs; PD kd is 1.81] and frictionloss [Nm, abs] ranges (real waist roll ~2x more damped than sim,
    # supermimic.deploy.waist_roll docstring; fitted ranges damping (0, 3.6), friction (0, 2)).
    dr_waist_roll_damping: tuple[float, float] | None = None
    dr_waist_roll_friction: tuple[float, float] | None = None
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
    # DEVIATION from the paper when != 0 (default off), imagined reward only: weight of sum_j max(0, |dq_j| - thr)^2 over
    # joint_vel_hinge_joints [rad/s]. Targets the right-knee over-speed abort seen on the real robot (lean35, frame ~742).
    joint_vel_hinge_weight: float = 0.0
    joint_vel_hinge_joints: tuple[str, ...] = ("left_knee_joint", "right_knee_joint")
    joint_vel_hinge_thr: float = 12.0
    # DEVIATION from the paper when != 0 (default off): penalty -|w| * max(0, |pelvis body-z angular velocity| - thr)^2
    # (dt-scaled) [rad/s], in the simulator reward and the imagined reward. Targets the yaw spins that saturate the real
    # pelvis gyro (500 deg/s) on lean35.
    yaw_rate_hinge_weight: float = 0.0
    yaw_rate_hinge_thr: float = 7.5
    # DEVIATION from the paper when > 0 (default off): saturating-IMU observation model (supermimic.envs.imu_model).
    # Per env and episode, with this probability the policy's base_ang_vel is the gyro clipped to +-imu_gyro_limit and
    # its projected_gravity is integrated from that clipped gyro (complementary gain k ~ U[0, imu_accel_k_max] 1/s,
    # reset to the true gravity at episode start). Imagination: same model from the predicted angular velocity, starting
    # from the true gravity tilted by U[0, imu_init_tilt_deg]. World model / critic / rewards keep the true state.
    imu_sat_prob: float = 0.0
    imu_gyro_limit: float = 8.727
    imu_accel_k_max: float = 0.5
    imu_init_tilt_deg: float = 8.0
    # DEVIATION from the paper when set (default off): per-reset PD stiffness scale range (all actuators) and extra
    # torso mass range [kg], matching the sim2sim G2 robustness axes (kp 0.85/1.15, +1.8 kg).
    dr_kp_range: tuple[float, float] | None = None
    dr_torso_mass_add: tuple[float, float] | None = None
    # DEVIATION from the paper when > 0 (default off): per-episode command latency drawn from {0..N} physics substeps
    # (5 ms each) in the simulator; the real runner -> bridge -> motor path adds a few ms to tens of ms (sim2sim G2 lat axis).
    dr_action_delay_substeps: int = 0
    # DEVIATION from the paper when set (default off): per-reset knee effort-limit scale U[lo, hi] (actuator force range
    # of left/right knee, envs/mdp/events.py). The real knee aborts at frame ~734 of lean35 under weaker push-off.
    dr_knee_effort_scale: tuple[float, float] | None = None
    # DEVIATION from the paper when dr_push_prob > 0 (default off): with this probability per episode, a constant
    # world-frame pelvis force U(-F, F) / torque U(-T, T) per axis for dr_push_steps policy steps, starting at a clip-local
    # reference frame in dr_push_frames (only clips whose name contains one of dr_push_clips; empty = all).
    dr_push_prob: float = 0.0
    dr_push_frames: tuple[int, int] = (690, 730)
    dr_push_steps: tuple[int, int] = (2, 10)
    dr_push_force: tuple[float, float, float] = (60.0, 60.0, 30.0)
    dr_push_torque: tuple[float, float, float] = (8.0, 8.0, 8.0)
    dr_push_clips: tuple[str, ...] = ()
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
    # DEVIATION from the paper when real_reset_prob > 0 (default off): this fraction of training resets starts from a
    # real-robot state of real_reset_file (scripts/data/build_real_reset_states.py; envs/mdp/commands.py docstring),
    # with the motion command at that state's clip frame. real_reset_vel: root linear velocity from the clip ("clip"),
    # the stance-foot-zero-velocity estimate ("foot") or either with prob 1/2 ("mix"). Per-reset perturbations: height
    # U(dz), vertical velocity U(-dvz, dvz), roll/pitch U(-tilt, tilt) deg, joint q / dq noise.
    real_reset_file: Path | None = None
    real_reset_prob: float = 0.0
    real_reset_vel: str = "mix"
    real_reset_dz: tuple[float, float] = (-0.01, 0.01)
    real_reset_dvz: float = 0.2
    real_reset_tilt_deg: float = 3.0
    real_reset_joint_pos_noise: float = 0.02
    real_reset_joint_vel_noise: float = 0.3

    def __post_init__(self) -> None:
        if self.fail_reset_frac > 0 and self.fail_reset_list is None:
            raise ValueError(f"--env.fail-reset-frac {self.fail_reset_frac} > 0 needs --env.fail-reset-list")
        if self.real_reset_prob > 0 and self.real_reset_file is None:
            raise ValueError(f"--env.real-reset-prob {self.real_reset_prob} > 0 needs --env.real-reset-file")
        if not 0.0 <= self.real_reset_prob <= 1.0 or not 0.0 <= self.dr_push_prob <= 1.0:
            raise ValueError("--env.real-reset-prob / --env.dr-push-prob must be in [0, 1]")
        if self.real_reset_vel not in ("clip", "foot", "mix"):
            raise ValueError(f"--env.real-reset-vel {self.real_reset_vel!r} not in clip|foot|mix")
        if self.waist_roll_range is not None and not self.waist_roll_range > 0:
            raise ValueError(f"--env.waist-roll-range {self.waist_roll_range} must be > 0")
        if self.lock_waist and (self.waist_roll_range is not None or self.dr_waist_roll_damping is not None or self.dr_waist_roll_friction is not None):
            raise ValueError("--env.waist-roll-range / --env.dr-waist-roll-* need the unlocked waist (not --env.lock-waist)")
        for n in ("dr_waist_roll_damping", "dr_waist_roll_friction"):
            r = getattr(self, n)
            if r is not None and not 0.0 <= r[0] <= r[1]:
                raise ValueError(f"--env.{n.replace('_', '-')} {r}: needs 0 <= lo <= hi")
        if not 0.0 <= self.imu_sat_prob <= 1.0:
            raise ValueError(f"--env.imu-sat-prob {self.imu_sat_prob} not in [0, 1]")


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
