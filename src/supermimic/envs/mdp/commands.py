"""Multi-clip motion command with future-frame goals and return-based reset sampling.

Exposes the same attribute API as mjlab's ``MotionCommand`` (``anchor_pos_w``,
``robot_body_pos_w``, ``body_pos_relative_w`` ...) so mjlab's tracking reward
functions work unchanged, plus:

* ``goal``        (N, K*68) future-frame goal features (plan D5)
* ``root_*``      reference pelvis quantities for the paper's root terminations
* ``clip_end``    (N,) True once the clip's last frame has been reached (time-out)
* return-based reset sampling over 1 s bins of the whole library (plan D14)

Return-based sampler, G_max (paper D.1 / plan D14, U14). The paper scores bin b by softmax_b(-25 G_b / G_max) with
G_max "the maximum achievable discounted return". With ``gmax_mode="const"`` (e11/e11b behaviour) G_max is the single
constant r_max / (1 - gamma) = 10, so every step within ~5 s of a clip's end gets a low G no matter how well it is
tracked (the reference simply ends: from k frames before the end at most r_max (1 - gamma^k) / (1 - gamma) can be
earned), and the last bin of every clip absorbs most of the hard share (we measured 94% at 144k updates). With
``gmax_mode="horizon"`` (default) each step's return-to-go is put on the common G_max scale by the maximum discounted
return achievable from that step before the reference ends,

    G_norm_t = G_t * G_max / A(k_t),    A(k) = r_max (1 - gamma^k) / (1 - gamma),

i.e. G_norm_t / G_max is the discount-weighted mean of r_s / r_max over the horizon of k_t steps that step t could
have earned reward over (a perfectly tracked tail bin scores G_max, exactly as a perfectly tracked interior bin does):
  * clip-end episodes: k_t = steps left in the episode (the reference ends there);
  * terminated (fallen) episodes: k_t = frames left to the clip end, so the steps lost to the fall count as zero;
  * truncated (10 s time-out) episodes without a critic bootstrap (D14 drop rule, the trained configuration): k_t =
    steps left in the episode (>= min_remaining_steps), the observed window. Without this, perfectly tracked
    interior bins of long clips score r_max (1 - gamma^m) / (1 - gamma) < G_max and would in turn look ~1 G
    unit harder (x12 softmax weight) than normalised bins of short clips;
  * truncated episodes with a critic bootstrap: unchanged (the bootstrap already covers the rest of the horizon).

G_b update (paper: an EMA, rate unspecified, U14). With ``count_warmstart`` (default) each bin keeps a visit counter
n_b (number of EMA updates it received) and is updated with the count-based rate alpha_b = max(ema_alpha, 1/(n_b+1)):
the first visit sets G_b to the observed (normalised) return, the first 1/ema_alpha visits form a running mean, and
later visits use the fixed rate ema_alpha as before. Without it every update uses ema_alpha, so after a sampler reset
(G_b = G_max) a bin needs ~1/ema_alpha visits to move away from G_max (a 40k-iteration run gives most of the ~1M bins
only a few visits). ``reset_sampler_state`` sets n_b = 0; a checkpoint without counters loaded WITHOUT a reset gets
n_b = VISITS_SATURATED, i.e. alpha_b = ema_alpha exactly as in the run that wrote it (e1-e11b resumes are unchanged).

Deviations from the paper (all default OFF): ``hard_clip_cap`` (no clip may take more than this fraction of the
hard softmax share) and ``fail_reset_frac`` / ``fail_reset_list`` (a fraction of resets placed shortly before known
benchmark failure frames).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from pathlib import Path

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from supermimic.envs.mdp.motion_library import MotionLibrary

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# ---------------------------------------------------------------------- return-based sampler helpers (pure functions)

# Visit count given to every bin of a checkpoint that predates the counters (resumed without a sampler reset): large
# enough that the count-based rate is always ema_alpha, small enough that int32 increments never overflow.
VISITS_SATURATED = 2**30


def achievable_return(k: torch.Tensor, r_max: float, gamma: float) -> torch.Tensor:
    """Maximum discounted return over k remaining reward steps: r_max (1 - gamma^k) / (1 - gamma)."""
    return r_max * (1.0 - torch.pow(torch.as_tensor(gamma, dtype=torch.float64, device=k.device), k.double())) / (1.0 - gamma)


def cap_per_clip(hard: torch.Tensor, bin_clip: torch.Tensor, num_clips: int, cap: float, iters: int = 50) -> torch.Tensor:
    """Water-fill a bin distribution so that no clip holds more than `cap` of the mass (excess goes to the other
    clips in proportion to their mass). cap <= 0 or cap * num_clips <= 1 (infeasible) returns `hard` unchanged."""
    if cap <= 0.0 or cap * num_clips <= 1.0:
        return hard
    p = hard / hard.sum()
    capped = torch.zeros(num_clips, dtype=torch.bool, device=hard.device)
    for _ in range(iters):
        m = torch.zeros(num_clips, dtype=p.dtype, device=p.device).index_add_(0, bin_clip, p)
        over = (m > cap * (1 + 1e-6)) & ~capped
        if not over.any():
            break
        capped |= over
        scale = torch.ones_like(m)
        scale[capped] = cap / m[capped].clamp_min(1e-30)
        free_mass = m[~capped].sum()
        if free_mass > 0:
            scale[~capped] = (1.0 - cap * capped.sum()) / free_mass
        p = p * scale[bin_clip]
    return p


def load_fail_points(path: str | Path) -> list[tuple[str, int]]:
    """(clip name, fail step) pairs from a benchmark json ({"clips": [{name, failed, fail_step}, ...]}, as written
    by scripts/eval_benchmark.py) or a text/csv file of `name,fail_step` lines."""
    import json

    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        clips = d["clips"] if isinstance(d, dict) else d
        return [(c["name"], int(c["fail_step"])) for c in clips if c.get("failed") and int(c.get("fail_step", -1)) > 0]
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, step = [x.strip() for x in line.split(",")[:2]]
        if not step.lstrip("-").isdigit():  # header
            continue
        out.append((name, int(step)))
    return out


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_root_body_index = self.robot.body_names.index(cfg.root_body_name)

        default_joint_pos = self.robot.data.default_joint_pos[0].detach()
        # Waist-locked robot: the entity has 27 joints, everything command-side stays in the 29-joint layout.
        self._locked = tuple(cfg.locked_joint_ids)
        if self._locked:
            from supermimic.envs.lock_waist import expand_joints, joint_slots

            slots = joint_slots(self.robot.joint_names)
            assert sorted(set(range(len(slots) + len(self._locked))) - set(slots)) == sorted(self._locked), (slots, self._locked)
            self._slots = torch.tensor(slots, dtype=torch.long, device=self.device)
            default_joint_pos = expand_joints(default_joint_pos, self._slots, 0.0)
        else:
            assert not cfg.feasible_refs, "feasible_refs needs locked joints"
        self.library = MotionLibrary(
            cfg.npz_files, cfg.body_names, default_joint_pos, device=self.device, bin_frames=cfg.bin_frames,
            compact=cfg.compact_library, store_dtype=torch.float16 if cfg.library_fp16 else torch.float32,
            zero_joints=self._locked if cfg.feasible_refs else (),
        )
        print(f"[INFO] {self.library.summary()}")
        lib = self.library

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._refresh_ref()
        self.body_pos_relative_w = torch.zeros(n, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(n, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Evaluation hook: when >= 0, the next reset of that env starts exactly at this global frame.
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self._init_sampler_state(min(int(math.ceil(env.cfg.episode_length_s / env.step_dt)) + 2, cfg.max_logged_steps))

        for k in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
                  "error_joint_pos", "error_joint_vel", "sampling_entropy", "episode_return_to_go"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    def _init_sampler_state(self, max_ep_steps: int) -> None:
        """Return-based reset sampling state (per-bin G_b and memory, per-env episode logs)."""
        cfg, lib, n = self.cfg, self.library, self.num_envs
        assert cfg.gmax_mode in ("const", "horizon"), cfg.gmax_mode
        self.bin_return = torch.full((lib.num_bins,), cfg.g_max, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.bin_visits = torch.zeros(lib.num_bins, dtype=torch.int32, device=self.device)  # n_b: EMA updates so far
        self.max_ep_steps = max_ep_steps
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=self.device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_frames = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=self.device, dtype=torch.float64)
        self.bootstrap_fn = None  # optional: env_ids -> value estimate for truncated episodes
        # Failure-point resets (deviation from the paper, default off): global frames of known failures.
        self.fail_frames: torch.Tensor | None = None
        if cfg.fail_reset_frac > 0 and not cfg.fail_reset_list:
            raise ValueError(f"fail_reset_frac = {cfg.fail_reset_frac} > 0 needs fail_reset_list (--env.fail-reset-list)")
        if cfg.fail_reset_list:
            self.set_fail_points(load_fail_points(cfg.fail_reset_list))

    # ------------------------------------------------------------------ reference (mjlab MotionCommand API)

    def _refresh_ref(self) -> None:
        """Cache the reference state / tracked-body states of the current target frames."""
        self._ref = self.library.ref(self.frame)
        self._bodies = self.library.bodies(self.frame)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies.lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies.ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w[:, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref.root_pos_w + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref.root_quat_w

    # ------------------------------------------------------------------ robot

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """(N, 29) robot joint positions (locked joints read 0)."""
        if self._locked:
            from supermimic.envs.lock_waist import expand_joints

            return expand_joints(self.robot.data.joint_pos, self._slots, 0.0)
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        if self._locked:
            from supermimic.envs.lock_waist import expand_joints

            return expand_joints(self.robot.data.joint_vel, self._slots, 0.0)
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_root_body_index]

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # ------------------------------------------------------------------ reward bookkeeping for the sampler

    def _log_step_rewards(self, env_mask: torch.Tensor | None = None) -> None:
        """Append this step's simulator reward (and the bin it was earned in) to each env's episode log."""
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_mask is None else env_mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward[ids]
        self.ep_bins[ids, pos] = self.library.bin_of_frame(self.frame[ids])
        self.ep_frames[ids, pos] = self.frame[ids]
        self.ep_len[ids] = torch.minimum(self.ep_len[ids] + 1, torch.full_like(pos, self.max_ep_steps))

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes' discounted returns-to-go into the per-bin estimates."""
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max().item())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        r = r * tmask
        gp = self._gamma_pow[:T]
        # Return-to-go R_t = sum_{s>=t} gamma^{s-t} r_s via a reverse cumulative sum.
        disc = r * gp[None, :]
        rtg = torch.flip(torch.cumsum(torch.flip(disc, [1]), 1), [1]) / gp[None, :]
        # Truncated (timed-out) episodes: bootstrap with the critic if available, otherwise
        # drop the steps too close to the truncation point (plan D14).
        terminated = self._env.termination_manager.terminated[env_ids] if hasattr(self._env, "termination_manager") else torch.ones_like(lens, dtype=torch.bool)
        truncated = ~terminated & ~self.clip_end[env_ids]
        if truncated.any():
            if self.bootstrap_fn is not None:
                v = self.bootstrap_fn(env_ids).double()  # (n,)
                remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0).double()
                rtg = rtg + truncated[:, None] * (cfg.gamma ** remaining) * v[:, None]
            else:
                remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
                tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        if cfg.gmax_mode == "horizon":
            # Put each step on the G_max scale by the return achievable over its horizon (module docstring).
            # Clip-end and truncated episodes: k = steps left in the episode (a clip-end episode rewards the last
            # frame twice: the step that reaches it and the time-out step). Terminated episodes: k = frames left to
            # the clip end (+1 for that time-out step), so a fall forfeits the remaining steps as zero reward.
            steps_left = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(1)
            frames = self.ep_frames[env_ids, :T]
            frames_left = self.library.last_frame_of(frames) - frames + 2
            k = torch.where(terminated[:, None] & ~self.clip_end[env_ids][:, None], frames_left, steps_left)
            scale = cfg.g_max / achievable_return(k, cfg.max_step_reward, cfg.gamma)
            if self.bootstrap_fn is not None:
                scale = torch.where(truncated[:, None], torch.ones_like(scale), scale)
            rtg = rtg * scale
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        mean = sums[touched] / cnts[touched]
        old = self.bin_return[touched]
        new = (1 - cfg.ema_alpha) * old + cfg.ema_alpha * mean
        visits = self.bin_visits[touched]
        if cfg.count_warmstart:
            # Count-based rate alpha_b = max(ema_alpha, 1/(n_b+1)) (module docstring); bins past the warm-up take the
            # fixed-rate branch above unchanged (bit-identical to the plain EMA).
            w = 1.0 / (visits.float() + 1.0)
            new = torch.where(w > cfg.ema_alpha, (1 - w) * old + w * mean, new)
        self.bin_return[touched] = new
        self.bin_visits[touched] = (visits + 1).clamp_max(VISITS_SATURATED)
        self.metrics["episode_return_to_go"][env_ids] = rtg[:, 0].float()
        self.ep_len[env_ids] = 0

    # ------------------------------------------------------------------ sampling

    def set_fail_points(self, points: list[tuple[str, int]]) -> None:
        """Register (clip name, benchmark fail step) pairs for failure-point resets; unknown names are skipped."""
        lib = self.library
        idx = {c.name: i for i, c in enumerate(lib.clips)}
        hits = [(idx[nm], st) for nm, st in points if nm in idx]
        if not hits:
            self.fail_frames = None
            if self.cfg.fail_reset_frac > 0:
                raise ValueError(f"fail-reset: none of the {len(points)} failure points match a library clip")
            print(f"[WARN] fail-reset: none of the {len(points)} failure points match a library clip")
            return
        ci = torch.tensor([h[0] for h in hits], dtype=torch.long, device=self.device)
        st = torch.tensor([h[1] for h in hits], dtype=torch.long, device=self.device)
        # Benchmark step s is rewarded against frame start + s - 1; the fall is detected there.
        self.fail_frames = torch.minimum(lib.clip_start[ci] + st - 1, lib.clip_last[ci])
        print(f"[INFO] fail-reset (deviation from paper): {len(hits)}/{len(points)} failure points, "
              f"frac {self.cfg.fail_reset_frac}, window {self.cfg.fail_reset_window_s} s")

    def reset_sampler_state(self) -> None:
        """G_b <- G_max, memory <- uniform, n_b <- 0 (fresh-start sampler statistics)."""
        self.bin_return.fill_(self.cfg.g_max)
        self.bin_memory.fill_(1.0 / self.library.num_bins)
        self.bin_visits.zero_()

    def sampler_state(self) -> dict[str, torch.Tensor]:
        """Checkpoint entries of the sampler statistics."""
        return {"bin_return": self.bin_return, "bin_memory": self.bin_memory, "bin_visits": self.bin_visits}

    def load_sampler_state(self, ck: dict) -> str:
        """Load sampler statistics from a checkpoint dict (refused if the bins differ). A checkpoint without visit
        counters (written before count_warmstart) gets n_b = VISITS_SATURATED so its bins keep the plain EMA."""
        nb = self.bin_return.shape[0]
        if ck["bin_return"].shape[0] != nb:
            raise ValueError(
                f"checkpoint sampler has {ck['bin_return'].shape[0]} bins, library has {nb}: the library changed; "
                "pass --resume-reset-sampler"
            )
        self.bin_return.copy_(ck["bin_return"])
        self.bin_memory.copy_(ck["bin_memory"])
        if "bin_visits" in ck:
            self.bin_visits.copy_(ck["bin_visits"])
            return "sampler state loaded"
        self.bin_visits.fill_(VISITS_SATURATED)
        return "sampler state loaded (no visit counters in checkpoint: n_b saturated, plain EMA as before)"

    def hard_distribution(self) -> torch.Tensor:
        """The paper's hard-bin distribution softmax_b(-T G_b / G_max), optionally capped per clip."""
        cfg = self.cfg
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        if cfg.hard_clip_cap > 0:
            hard = cap_per_clip(hard, self.library.bin_clip, self.library.num_clips, cfg.hard_clip_cap)
        return hard

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "start":
            clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
            return lib.clip_start[clips]
        f = self._sample_frames_mixture(n)
        if cfg.fail_reset_frac > 0 and self.fail_frames is not None:
            f = self._apply_fail_resets(f)
        return f

    def _apply_fail_resets(self, f: torch.Tensor) -> torch.Tensor:
        """Deviation from the paper (flag, default off): replace a `fail_reset_frac` share of the sampled start
        frames by a frame drawn uniformly from the `fail_reset_window_s` seconds before a known failure frame."""
        lib = self.library
        cfg = self.cfg
        n = f.shape[0]
        use = torch.rand(n, device=self.device) < cfg.fail_reset_frac
        j = torch.randint(0, self.fail_frames.shape[0], (n,), device=self.device)
        ff = self.fail_frames[j]
        cs = lib.clip_start[lib.frame_clip[ff]]
        w = max(1, int(round(cfg.fail_reset_window_s / self._env.step_dt)))  # one reference frame per env step
        back = torch.randint(1, w + 1, (n,), device=self.device)
        g = torch.minimum(ff - back, lib.last_frame_of(ff) - 2).clamp_min(cs)
        return torch.where(use, g, f)

    def _sample_frames_mixture(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "uniform":
            # Uniform over frames (long clips get proportionally more starts), leaving >= 2 steps.
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
        assert cfg.sampling_mode == "return"
        hard = self.hard_distribution()
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        f = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        self.metrics["sampling_entropy"][:] = ent
        return f

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        if self._locked:  # the reference's locked-joint entries have nothing to be written to
            joint_pos, joint_vel = joint_pos[:, self._slots], joint_vel[:, self._slots]
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # Episode bookkeeping for the envs being reset (terminal reward is already in reward_buf).
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        if self.ep_len.sum() > 0 or hasattr(self._env, "reward_buf"):
            self._log_step_rewards(mask)
            self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0

        sampled = self._sample_frames(len(env_ids))
        forced = self.forced_frame[env_ids]
        self.frame[env_ids] = torch.where(forced >= 0, forced, sampled)
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        root_ori = r.root_quat_w.clone()
        root_lin_vel = r.root_lin_vel_w.clone()
        root_ang_vel = r.root_ang_vel_w.clone()
        joint_pos = r.joint_pos.clone()
        joint_vel = r.joint_vel

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + rs[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rs[:, :3]
        root_ang_vel = root_ang_vel + rs[:, 3:]
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, device=self.device)

        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)

    def update_relative_body_poses(self) -> None:
        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        delta_pos_w = robot_anchor_pos
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Clip end must be judged against the clip of the *current* frame (frame+1 may already be
        # the first frame of the next clip in the concatenated library).
        last = self.library.last_frame_of(self.frame)
        if env_ids is None:
            # Per-step path: log the reward earned against the current target frame for the envs
            # that were NOT just reset (their terminal reward was logged in _resample_command).
            reset_buf = getattr(self._env, "reset_buf", None)
            self._log_step_rewards(None if reset_buf is None else ~reset_buf)
            self.frame += 1
        else:
            self.frame[env_ids] += 1
        over = self.frame > last
        if over.any():
            self.clip_end |= over
            self.frame = torch.where(over, last, self.frame)
        self._refresh_ref()
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Deterministic reset to exact reference frames (no noise); used by evaluation."""
        self.ep_len[env_ids] = 0
        self.frame[env_ids] = frame
        self.clip_end[env_ids] = False
        r = self.library.ref(frame)
        self._write_reference_state_to_sim(
            env_ids, r.root_pos_w + self._env.scene.env_origins[env_ids], r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_pos, r.joint_vel
        )

    # ------------------------------------------------------------------ debug visualization (viewer only)

    _ghost_model = None

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the reference pose as a translucent ghost robot (same approach as mjlab's MotionCommand)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            import copy

            m = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(m.ngeom):
                if m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0:
                    m.geom_rgba[gi, 3] = 0  # hide collision geoms
                else:
                    m.geom_rgba[gi] = (0.2, 0.6, 1.0, 0.45)
            self._ghost_model = m
        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        root_pos = self.root_pos_w.cpu().numpy()
        root_quat = self.root_quat_w.cpu().numpy()
        joint_pos = (self.joint_pos[:, self._slots] if self._locked else self.joint_pos).cpu().numpy()
        import numpy as np

        for e in env_indices:
            qpos = np.zeros(self._ghost_model.nq)
            qpos[free_q[0:3]] = root_pos[e]
            qpos[free_q[3:7]] = root_quat[e]
            qpos[joint_q] = joint_pos[e]
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{e}")


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
    npz_files: list[str]
    anchor_body_name: str
    root_body_name: str = "pelvis"
    body_names: tuple[str, ...]
    entity_name: str = "robot"
    future_frames: int = 10
    bin_frames: int = 50
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    # Return-based sampling (plan D14).
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0  # paper: 25
    g_max: float = 10.0  # r_max / (1 - gamma)
    max_step_reward: float = 0.1  # r_max (5 positive terms x dt 0.02)
    # "horizon": each step's return is normalised by the return achievable over its horizon (module docstring;
    # fixes the clip-tail-bin bias of const). "const": single G_max (e1-e11b behaviour).
    gmax_mode: Literal["const", "horizon"] = "horizon"
    # Deviation from the paper, 0 = off: max fraction of the hard (softmax) share any single clip may take.
    hard_clip_cap: float = 0.0
    # Deviation from the paper, 0 = off: fraction of non-"start" resets placed uniformly in the window_s seconds
    # before a known failure frame (fail_reset_list: benchmark json or name,fail_step csv).
    fail_reset_frac: float = 0.0
    fail_reset_list: str | None = None
    fail_reset_window_s: float = 3.0
    gamma: float = 0.99
    ema_alpha: float = 0.05  # G_b EMA rate; with count_warmstart the floor alpha_min of max(alpha_min, 1/(n_b+1))
    # Count-based warm start of G_b (module docstring): first visit sets G_b to the observed return.
    count_warmstart: bool = True
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024
    compact_library: bool = False
    library_fp16: bool = True
    # Waist-locked robot: 29-joint slots with no joint in the entity; feasible_refs zeroes them (q and dq) in the
    # references, body states follow by FK (supermimic.envs.lock_waist).
    locked_joint_ids: tuple[int, ...] = ()
    feasible_refs: bool = False
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)
