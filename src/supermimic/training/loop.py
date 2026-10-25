"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.lock_waist import full_joint_constants
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import TrackingRewardCfg, joint_torque_constants
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        if cfg.env.lock_waist:
            from supermimic.envs.lock_waist import LOCKED_JOINT_IDS

            cfg.world_model.state_spec.locked_joints = LOCKED_JOINT_IDS
        else:
            assert not cfg.world_model.state_spec.locked_joints, "locked_joints without --env.lock-waist"
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
            if cfg.schedule.lr_constant is not None:
                cfg.schedule.lr_constant *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                lock_waist=cfg.env.lock_waist,
                lock_waist_refs=cfg.env.lock_waist_refs,
                locked_action_weight=cfg.env.locked_action_weight,
                tilt_hinge_weight=cfg.env.tilt_hinge_weight,
                tilt_hinge_knee=cfg.env.tilt_hinge_knee,
                dr_kp_range=cfg.env.dr_kp_range,
                dr_torso_mass_add=cfg.env.dr_torso_mass_add,
                seed=cfg.seed + 1000 * self.rank,
                extra_command_kwargs=dict(
                    gmax_mode=cfg.env.sampler_gmax_mode,
                    softmax_temp=cfg.env.sampler_softmax_temp,
                    ema_alpha=cfg.env.sampler_ema_alpha,
                    count_warmstart=cfg.env.sampler_count_warmstart,
                    hard_clip_cap=cfg.env.sampler_hard_clip_cap,
                    fail_reset_frac=cfg.env.fail_reset_frac,
                    fail_reset_list=None if cfg.env.fail_reset_list is None else str(cfg.env.fail_reset_list),
                    fail_reset_window_s=cfg.env.fail_reset_window_s,
                ),
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_g1_model(), device=self.device)
        # 29-joint constants (identical to robot.data for the unlocked robot; locked joints filled in otherwise).
        q0_full, soft_full = full_joint_constants(robot)
        reward_cfg = TrackingRewardCfg(
            w_tilt_hinge=cfg.env.tilt_hinge_weight, tilt_hinge_knee=cfg.env.tilt_hinge_knee, w_joint_torque=cfg.env.joint_torque_weight
        )
        pd = None
        if cfg.env.joint_torque_weight != 0.0:
            pd = joint_torque_constants(cfg.env.joint_torque_jw, cfg.env.joint_torque_jw_default, self.device)
            assert torch.allclose(pd["q0"], q0_full.to(pd["q0"]), atol=1e-6), "mjlab default pose != robot default pose"
            assert torch.allclose(pd["q0"], self.lib.default_joint_pos.to(pd["q0"]), atol=1e-6), "mjlab default pose != library q0"
        if cfg.env.lock_waist:
            reward_cfg.locked_action_ids = cfg.world_model.state_spec.locked_joints
            reward_cfg.w_locked_action = cfg.env.locked_action_weight
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, soft_full, q0_full, reward_cfg, pd=pd)
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        self.tail_bins = self.lib.bin_end == self.lib.clip_last[self.lib.bin_clip] + 1  # last bin of each clip
        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)
        # Warm-up counts from this run's start: a resumed run starts with an empty replay buffer too.
        self.start_iteration = self.iteration

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                **self.command.sampler_state(),  # bin_return, bin_memory, bin_visits
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        nb = self.command.bin_return.shape[0]
        if self.cfg.resume_reset_sampler:
            self.command.reset_sampler_state()
            sampler = f"sampler reset (G_b = G_max, memory uniform, n_b = 0 over {nb} bins)"
        else:
            sampler = self.command.load_sampler_state(ck)
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        if self.is_main:
            print(f"[INFO] resumed from {path} at iteration {self.iteration}; {sampler}")

    # ------------------------------------------------------------------ learning rates

    def lr_at(self, iteration: int) -> tuple[float, float]:
        """(world-model lr, policy/critic lr) at an iteration: cosine schedule, or the constant fine-tuning rate."""
        cfg, sch = self.cfg, self.cfg.schedule
        if sch.lr_constant is not None:
            lr = sch.lr_constant
        else:
            lr = cosine_lr(iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
        return lr, lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr

    def apply_lr(self, iteration: int) -> float:
        lr_wm, lr_pi = self.lr_at(iteration)
        self.wm_trainer.set_lr(lr_wm)
        self.policy_trainer.set_lr(lr_pi)  # policy and critic
        return lr_wm

    def effective_lrs(self) -> dict[str, float]:
        """Learning rates currently in the optimizers' param groups."""
        return {
            "world_model": self.wm_trainer.opt.param_groups[0]["lr"],
            "policy": self.policy_trainer.opt.param_groups[0]["lr"],
            "critic": self.policy_trainer.critic_opt.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        # Apply (and report) the learning rates now: after a resume the optimizers hold the checkpoint's rates.
        self.apply_lr(self.iteration)
        if self.is_main:
            mode = f"constant {sch.lr_constant:.3g}" if sch.lr_constant is not None else "cosine"
            lrs = ", ".join(f"{k} {v:.3g}" for k, v in self.effective_lrs().items())
            print(f"[INFO] lr schedule {mode}; effective lr at iteration {self.iteration}: {lrs}", flush=True)
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = self.apply_lr(self.iteration)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    self.writer.add_scalar("sampler/visited_frac", float((self.command.bin_visits > 0).float().mean()), it)
                    if self.command.cfg.sampling_mode == "return":
                        # Share of the hard distribution on the last bin of each clip (measured 94% under
                        # gmax_mode const at 144k) and on the most-favoured clip.
                        hard = self.command.hard_distribution()
                        self.writer.add_scalar("sampler/hard_share_last_bin", float(hard[self.tail_bins].sum()), it)
                        per_clip = torch.zeros(self.lib.num_clips, device=self.device).index_add_(0, self.lib.bin_clip, hard)
                        self.writer.add_scalar("sampler/hard_share_top_clip", float(per_clip.max()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
