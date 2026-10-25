"""Return-based reset sampler (paper D.1, plan D14) and the fine-tuning resume options.

CPU tests drive the real MultiClipMotionCommand bookkeeping (_finalize_episodes / hard_distribution / sampling)
on a stub library and stub env; the GPU test runs the real Trainer resume path on a tiny LAFAN1 env.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.envs.mdp.commands import (
    VISITS_SATURATED,
    MultiClipMotionCommand,
    MultiClipMotionCommandCfg,
    achievable_return,
    cap_per_clip,
    load_fail_points,
)
from supermimic.envs.mdp.motion_library import ClipInfo, MotionLibrary

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
GAMMA = 0.99


class StubLib:
    """Clip/bin layout identical to MotionLibrary's (same index arithmetic, real query methods)."""

    bin_of_frame = MotionLibrary.bin_of_frame
    last_frame_of = MotionLibrary.last_frame_of

    def __init__(self, lens: list[int], bin_frames: int = 50):
        start, clips = 0, []
        for i, n in enumerate(lens):
            clips.append(ClipInfo(f"clip{i}", start, n))
            start += n
        self.clips, self.num_clips, self.num_frames, self.bin_frames = clips, len(clips), start, bin_frames
        self.clip_start = torch.tensor([c.start for c in clips])
        self.clip_len = torch.tensor(lens)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips), self.clip_len)
        bpc = (self.clip_len + bin_frames - 1) // bin_frames
        self.clip_bin_offset = torch.cumsum(bpc, 0) - bpc
        self.num_bins = int(bpc.sum())
        self.bin_clip = torch.repeat_interleave(torch.arange(self.num_clips), bpc)
        local = torch.arange(self.num_bins) - self.clip_bin_offset[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local * bin_frames
        self.bin_end = torch.minimum(self.bin_start + bin_frames, self.clip_start[self.bin_clip] + self.clip_len[self.bin_clip])


def make_cmd(lib: StubLib, n_envs: int, **cfg_kw) -> MultiClipMotionCommand:
    cfg = MultiClipMotionCommandCfg(npz_files=[], anchor_body_name="pelvis", body_names=("pelvis",), **cfg_kw)
    cmd = object.__new__(MultiClipMotionCommand)
    cmd.cfg = cfg
    cmd._env = SimpleNamespace(
        num_envs=n_envs, device="cpu", step_dt=0.02,
        termination_manager=SimpleNamespace(terminated=torch.zeros(n_envs, dtype=torch.bool)),
    )
    cmd.library = lib
    cmd.metrics = {"episode_return_to_go": torch.zeros(n_envs), "sampling_entropy": torch.zeros(n_envs)}
    cmd.clip_end = torch.zeros(n_envs, dtype=torch.bool)
    cmd.frame = torch.zeros(n_envs, dtype=torch.long)
    cmd._init_sampler_state(502)  # 10 s episodes at 50 Hz (+2), as in training
    return cmd


# ---------------------------------------------------------------------- synthetic library and "policy"

LENS = [400] * 16 + [1500] * 2  # 16 short clips (8 bins each) and 2 long ones (episodes there time out)
HARD_CLIPS = (12, 13, 14, 15)  # genuinely hard section: frames [200, 240) tracked poorly, 30% fall at frame 240
HARD_LO, FALL_AT, P_FALL = 200, 240, 0.3
R_GOOD, R_HARD = 0.095, 0.06


def hard_bins(lib: StubLib) -> torch.Tensor:
    """Bins whose return-to-go genuinely contains the hard section or the fall (bins 3 and 4 of the hard clips)."""
    m = torch.zeros(lib.num_bins, dtype=torch.bool)
    for c in HARD_CLIPS:
        o = int(lib.clip_bin_offset[c])
        m[o + 3] = m[o + 4] = True
    return m


def tail_bins(lib: StubLib) -> torch.Tensor:
    return lib.bin_end == lib.clip_last[lib.bin_clip] + 1


def simulate(cmd: MultiClipMotionCommand, n_batches: int = 60, seed: int = 0) -> None:
    """Episodes started uniformly over bins (as the 50% uniform share does), logged exactly as the env logs them:
    a clip-end episode rewards the last frame twice (reaching step + time-out step), a fall ends the episode on the
    fall frame (terminated), and 10 s time-outs truncate."""
    lib, rng = cmd.library, np.random.default_rng(seed)
    n = cmd.num_envs
    clip_last = lib.clip_last.numpy()
    frame_clip = lib.frame_clip.numpy()
    for _ in range(n_batches):
        b = rng.integers(0, lib.num_bins, n)
        f0 = lib.bin_start.numpy()[b] + (rng.random(n) * (lib.bin_end - lib.bin_start).numpy()[b]).astype(int)
        f0 = np.minimum(f0, clip_last[frame_clip[f0]] - 2)
        term = torch.zeros(n, dtype=torch.bool)
        for e in range(n):
            c = int(frame_clip[f0[e]])
            last, cs = int(clip_last[c]), int(lib.clip_start[c])
            frames = list(range(f0[e], last + 1)) + [last]  # clip end: last frame rewarded twice
            end = "clip_end"
            if c in HARD_CLIPS and f0[e] - cs <= FALL_AT and rng.random() < P_FALL:
                frames = list(range(f0[e], cs + FALL_AT + 1))
                end = "fall"
            if len(frames) > cmd.max_ep_steps - 2:
                frames = frames[: cmd.max_ep_steps - 2]
                end = "timeout"
            fr = torch.tensor(frames)
            local = fr - cs
            r = torch.where((c in HARD_CLIPS) & (local >= HARD_LO) & (local < FALL_AT), R_HARD, R_GOOD).float()
            L = len(frames)
            cmd.ep_rewards[e, :L] = r
            cmd.ep_frames[e, :L] = fr
            cmd.ep_bins[e, :L] = lib.bin_of_frame(fr)
            cmd.ep_len[e] = L
            cmd.clip_end[e] = end == "clip_end"
            term[e] = end == "fall"
        cmd._env.termination_manager.terminated = term
        cmd._finalize_episodes(torch.arange(n))


@pytest.fixture(scope="module")
def trained():
    lib = StubLib(LENS)
    out = {}
    for mode in ("const", "horizon"):
        cmd = make_cmd(lib, 64, gmax_mode=mode, ema_alpha=0.2)
        simulate(cmd)
        out[mode] = cmd
    return lib, out


def test_achievable_return():
    k = torch.tensor([1, 2, 50, 10_000])
    a = achievable_return(k, 0.1, GAMMA)
    np.testing.assert_allclose(a.numpy(), [0.1, 0.1 + 0.099, 0.1 * (1 - GAMMA**50) / (1 - GAMMA), 10.0], rtol=1e-9)


def test_const_gmax_puts_hard_share_on_clip_tails(trained):
    """The e1-e11b sampler (single G_max): a perfectly tracked last bin scores ~2 of 10, so the tails take most of
    the hard share although only the hard section is genuinely hard (we measured 94% at 144k updates)."""
    lib, cmds = trained
    hard = cmds["const"].hard_distribution()
    tail, hb = float(hard[tail_bins(lib)].sum()), float(hard[hard_bins(lib)].sum())
    print(f"const: hard share on last bins {tail:.3f}, on genuinely hard bins {hb:.3f}")
    assert tail > 0.6 and tail > 3 * hb


def test_horizon_gmax_puts_hard_share_on_low_return_bins(trained):
    lib, cmds = trained
    cmd = cmds["horizon"]
    hard = cmd.hard_distribution()
    tail, hb = float(hard[tail_bins(lib)].sum()), float(hard[hard_bins(lib)].sum())
    hard_clip_share = float(sum(hard[lib.bin_clip == c].sum() for c in HARD_CLIPS))
    print(f"horizon: hard share on last bins {tail:.3f}, on genuinely hard bins {hb:.3f}, hard clips {hard_clip_share:.3f}")
    assert hb > 0.6 and tail < 0.05 and hard_clip_share > 0.9
    # Perfectly tracked bins (any position, incl. the tails of easy clips) are on the G_max scale: G / G_max = 0.95.
    easy = (lib.bin_clip < 12)
    np.testing.assert_allclose(cmd.bin_return[easy].numpy(), 10 * R_GOOD / 0.1, atol=0.05)  # EMA from G_max = 10
    # The most-favoured bins are the hard section (bins 3 and 4 of each hard clip).
    top = torch.topk(hard, 8).indices
    assert set(top.tolist()) == set(hard_bins(lib).nonzero().squeeze(-1).tolist())


def test_horizon_long_clips_not_biased_by_timeouts(trained):
    """Interior bins of long clips are mostly seen through 10 s time-outs (truncated, D14 drop rule). With the
    single G_max they sit at r_max (1 - gamma^m) / (1 - gamma) < G_max; normalised by the observed horizon they are
    on the same scale as the short clips (otherwise they would look ~1 G harder, x12 softmax weight)."""
    lib, cmds = trained
    long_bins = lib.bin_clip >= 16
    const, horizon = cmds["const"].bin_return[long_bins], cmds["horizon"].bin_return[long_bins]
    print(f"long-clip bins: const G mean {float(const.mean()):.2f} min {float(const.min()):.2f}; "
          f"horizon G mean {float(horizon.mean()):.2f} min {float(horizon.min()):.2f}")
    np.testing.assert_allclose(horizon.numpy(), 10 * R_GOOD / 0.1, atol=0.05)
    assert float(const.min()) < 3.0  # the old tail bin


def test_hard_clip_cap():
    lib = StubLib([400] * 10)
    cmd = make_cmd(lib, 4, hard_clip_cap=0.2)
    cmd.bin_return[: 16] = 0.0  # clips 0 and 1 are very hard
    capped = cmd.hard_distribution()
    per_clip = torch.zeros(lib.num_clips, dtype=capped.dtype).index_add_(0, lib.bin_clip, capped)
    assert abs(float(capped.sum()) - 1) < 1e-5
    assert float(per_clip.max()) <= 0.2 + 1e-5
    np.testing.assert_allclose(per_clip[:2].numpy(), [0.2, 0.2], rtol=1e-4)
    # Within a capped clip the bins keep their relative weights.
    cmd.cfg.hard_clip_cap = 0.0
    raw = cmd.hard_distribution()
    np.testing.assert_allclose((capped[:8] / capped[:8].sum()).numpy(), (raw[:8] / raw[:8].sum()).numpy(), rtol=1e-4)
    # Infeasible cap (cap * clips <= 1) is a no-op.
    assert torch.equal(cap_per_clip(raw, lib.bin_clip, lib.num_clips, 0.05), raw)


def test_fail_resets(tmp_path):
    import json

    lib = StubLib([400] * 4)
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"clips": [
        {"name": "clip1", "failed": True, "fail_step": 300}, {"name": "clip2", "failed": False, "fail_step": -1},
        {"name": "clip3", "failed": True, "fail_step": 20}, {"name": "unknown", "failed": True, "fail_step": 5},
    ]}))
    pts = load_fail_points(p)
    assert pts == [("clip1", 300), ("clip3", 20), ("unknown", 5)]
    csv = tmp_path / "fails.csv"
    csv.write_text("clip,fail_step\nclip1,300\nclip3,20\n")
    assert load_fail_points(csv) == [("clip1", 300), ("clip3", 20)]

    # Off by default: identical frames and identical RNG consumption.
    cmd = make_cmd(lib, 4, sampling_mode="return")
    cmd.set_fail_points(pts)
    torch.manual_seed(0)
    a = cmd._sample_frames(4096)
    torch.manual_seed(0)
    cmd.fail_frames = None
    b = cmd._sample_frames(4096)
    assert torch.equal(a, b)

    cmd = make_cmd(lib, 4, sampling_mode="return", fail_reset_frac=0.25, fail_reset_window_s=1.0, fail_reset_list=str(p))
    torch.manual_seed(0)
    f = cmd._sample_frames(20_000)
    ff1, ff3 = 400 + 299, 1200 + 19
    near = ((f >= ff1 - 50) & (f < ff1)) | ((f >= 1200) & (f < ff3))
    frac = float(near.float().mean())
    assert 0.24 < frac < 0.30, frac  # 25% fail resets + the uniform/hard shares that land there by chance
    assert int(f.max()) <= int(lib.last_frame_of(f).max()) - 2


def test_fail_reset_needs_list(tmp_path):
    """fail_reset_frac > 0 without a usable failure list is an error, not a silent no-op (code review)."""
    from supermimic.config import EnvCfg

    lib = StubLib([400] * 4)
    with pytest.raises(ValueError, match="fail_reset_list"):
        make_cmd(lib, 4, fail_reset_frac=0.1)
    with pytest.raises(ValueError, match="fail-reset-list"):
        EnvCfg(fail_reset_frac=0.1)
    EnvCfg(fail_reset_frac=0.1, fail_reset_list=tmp_path / "x.csv")  # list given: accepted
    EnvCfg()  # default: off
    csv = tmp_path / "nomatch.csv"
    csv.write_text("other_clip,30\n")
    with pytest.raises(ValueError, match="match a library clip"):
        make_cmd(lib, 4, fail_reset_frac=0.1, fail_reset_list=str(csv))
    make_cmd(lib, 4, fail_reset_list=str(csv))  # frac 0: only a warning


def test_start_mode_unchanged(tmp_path):
    """Evaluation path: 'start' sampling draws exactly what it drew before (one randint over clips)."""
    lib = StubLib([400, 120, 900])
    csv = tmp_path / "fails.csv"
    csv.write_text("clip0,11\n")
    cmd = make_cmd(lib, 4, sampling_mode="start", fail_reset_frac=0.5, fail_reset_list=str(csv))
    assert cmd.fail_frames is not None
    torch.manual_seed(3)
    got = cmd._sample_frames(1000)
    torch.manual_seed(3)
    ref = lib.clip_start[torch.randint(0, lib.num_clips, (1000,))]
    assert torch.equal(got, ref)


# ---------------------------------------------------------------------- count-based warm start of G_b (code review)


def clip_end_visit(cmd: MultiClipMotionCommand, clip: int, start_local: int, r: float) -> None:
    """One clip-end episode of constant reward r from clip-local frame start_local: with gmax_mode horizon every step
    normalises to G = g_max * r / r_max, so each touched bin observes exactly that value."""
    lib = cmd.library
    cs, last = int(lib.clip_start[clip]), int(lib.clip_last[clip])
    fr = torch.tensor(list(range(cs + start_local, last + 1)) + [last])
    L = len(fr)
    cmd.ep_rewards[0, :L] = r
    cmd.ep_frames[0, :L] = fr
    cmd.ep_bins[0, :L] = lib.bin_of_frame(fr)
    cmd.ep_len[0] = L
    cmd.clip_end[0] = True
    cmd._env.termination_manager.terminated = torch.zeros(cmd.num_envs, dtype=torch.bool)
    cmd._finalize_episodes(torch.tensor([0]))


def test_count_warmstart_first_visit_then_alpha_min():
    lib = StubLib([400, 400])
    cmd = make_cmd(lib, 1, ema_alpha=0.05)  # count_warmstart defaults on
    assert cmd.cfg.count_warmstart and int(cmd.bin_visits.sum()) == 0
    tail = slice(4, 8)  # bins 4..7 of clip 0 (frames 200..399); bins 0..3 and clip 1 are never visited
    clip_end_visit(cmd, 0, 200, 0.05)
    # First visit: G_b is the observed normalised return (0.05 / r_max * G_max = 5), not 0.95 * 10 + 0.05 * 5.
    torch.testing.assert_close(cmd.bin_return[tail], torch.full((4,), 5.0))
    assert torch.all(cmd.bin_return[:4] == 10.0) and torch.all(cmd.bin_return[8:] == 10.0)
    assert cmd.bin_visits[tail].tolist() == [1] * 4 and int(cmd.bin_visits.sum()) == 4
    # Second visit: rate 1/2 -> running mean of the two observations.
    clip_end_visit(cmd, 0, 200, 0.08)
    torch.testing.assert_close(cmd.bin_return[tail], torch.full((4,), 6.5))
    # Visits 3..20 keep the running mean (rate 1/(n+1) >= alpha_min until n = 19).
    obs = [5.0, 8.0]
    for i in range(18):
        r = 0.06 + 0.002 * i
        clip_end_visit(cmd, 0, 200, r)
        obs.append(r * 100)
    assert cmd.bin_visits[tail].tolist() == [20] * 4
    torch.testing.assert_close(cmd.bin_return[tail], torch.full((4,), float(np.mean(obs))), rtol=1e-5, atol=1e-5)
    # Later visits use alpha_min = ema_alpha.
    for r in (0.02, 0.09):
        g0 = cmd.bin_return[tail].clone()
        clip_end_visit(cmd, 0, 200, r)
        torch.testing.assert_close(cmd.bin_return[tail], 0.95 * g0 + 0.05 * (r * 100))
    # Without the warm start: the first visit only moves G_b by ema_alpha (e1-e11b behaviour).
    off = make_cmd(lib, 1, ema_alpha=0.05, count_warmstart=False)
    clip_end_visit(off, 0, 200, 0.05)
    torch.testing.assert_close(off.bin_return[tail], torch.full((4,), 9.75))
    # The reset zeroes the counters, so the next visit is a first visit again.
    cmd.reset_sampler_state()
    assert int(cmd.bin_visits.sum()) == 0 and torch.all(cmd.bin_return == 10.0)
    clip_end_visit(cmd, 0, 200, 0.07)
    torch.testing.assert_close(cmd.bin_return[tail], torch.full((4,), 7.0))


def test_count_warmstart_saturated_is_plain_ema():
    """Bins loaded from a checkpoint without counters (n_b saturated) update bit-identically to the plain EMA."""
    lib = StubLib([400] * 3 + [1500])
    a = make_cmd(lib, 64, ema_alpha=0.05, count_warmstart=False)
    b = make_cmd(lib, 64, ema_alpha=0.05)
    g0 = torch.rand(lib.num_bins) * 10
    a.bin_return.copy_(g0)
    b.load_sampler_state({"bin_return": g0, "bin_memory": b.bin_memory.clone()})
    assert torch.all(b.bin_visits == VISITS_SATURATED)
    simulate(a, n_batches=5)
    simulate(b, n_batches=5)
    assert torch.equal(a.bin_return, b.bin_return) and not torch.equal(a.bin_return, g0)
    assert torch.all(b.bin_visits == VISITS_SATURATED)  # no overflow past the sentinel


# ---------------------------------------------------------------------- resume options (CPU, stub trainer)


def stub_trainer(lib: StubLib, **cfg_kw):
    from supermimic.config import TrainConfig
    from supermimic.imagination.policy_trainer import PolicyTrainer
    from supermimic.training.loop import Trainer
    from supermimic.world_model.wm_trainer import WorldModelTrainer

    tr = object.__new__(Trainer)
    tr.cfg = TrainConfig(device="cpu", **cfg_kw)
    tr.device, tr.is_main = "cpu", True
    tr.wm, tr.policy, tr.critic = torch.nn.Linear(3, 3), torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    tr.wm_trainer = object.__new__(WorldModelTrainer)
    tr.wm_trainer.opt = torch.optim.Adam(tr.wm.parameters(), lr=2e-4)
    tr.policy_trainer = object.__new__(PolicyTrainer)
    tr.policy_trainer.opt = torch.optim.Adam(tr.policy.parameters(), lr=2e-4)
    tr.policy_trainer.critic_opt = torch.optim.Adam(tr.critic.parameters(), lr=2e-4)
    tr.command = make_cmd(lib, 4)
    return tr


def save_ckpt(tr, path: Path, lr: float = 8e-5, visits: bool = False) -> dict:
    """A checkpoint as written by Trainer.save; visits=False: e1-e11b format (no bin_visits)."""
    for o in (tr.wm_trainer.opt, tr.policy_trainer.opt, tr.policy_trainer.critic_opt):
        for g in o.param_groups:
            g["lr"] = lr
    ck = {
        "iteration": 144_000, "num_updates": 57_000,
        "wm": tr.wm.state_dict(), "policy": tr.policy.state_dict(), "critic": tr.critic.state_dict(),
        "wm_opt": tr.wm_trainer.opt.state_dict(), "policy_opt": tr.policy_trainer.opt.state_dict(),
        "critic_opt": tr.policy_trainer.critic_opt.state_dict(),
        "bin_return": torch.rand(tr.command.library.num_bins) * 10, "bin_memory": torch.rand(tr.command.library.num_bins),
    }
    if visits:
        ck["bin_visits"] = torch.randint(0, 50, (tr.command.library.num_bins,), dtype=torch.int32)
    torch.save(ck, path)
    return ck


def test_resume_loads_or_resets_sampler(tmp_path):
    lib = StubLib([400] * 5)
    ck = save_ckpt(stub_trainer(lib), tmp_path / "ck.pt")

    tr = stub_trainer(lib)
    tr.load(tmp_path / "ck.pt")
    assert torch.equal(tr.command.bin_return, ck["bin_return"]) and torch.equal(tr.command.bin_memory, ck["bin_memory"])
    assert tr.iteration == 144_000
    # Old checkpoint (no counters), no reset: n_b saturated -> plain EMA, the e11b-style resume is unchanged.
    assert torch.all(tr.command.bin_visits == VISITS_SATURATED)
    # New checkpoint: counters restored.
    ck2 = save_ckpt(stub_trainer(lib), tmp_path / "ck2.pt", visits=True)
    tr = stub_trainer(lib)
    tr.load(tmp_path / "ck2.pt")
    assert torch.equal(tr.command.bin_visits, ck2["bin_visits"]) and torch.equal(tr.command.bin_return, ck2["bin_return"])
    assert set(tr.command.sampler_state()) == {"bin_return", "bin_memory", "bin_visits"}

    tr = stub_trainer(lib, resume_reset_sampler=True)
    tr.load(tmp_path / "ck2.pt")
    assert torch.all(tr.command.bin_visits == 0)
    tr = stub_trainer(lib, resume_reset_sampler=True)
    tr.load(tmp_path / "ck.pt")
    assert torch.all(tr.command.bin_visits == 0)
    assert torch.all(tr.command.bin_return == tr.command.cfg.g_max)
    assert torch.allclose(tr.command.bin_memory, torch.full((lib.num_bins,), 1.0 / lib.num_bins))
    assert tr.iteration == 144_000 and torch.equal(tr.wm.weight, ck["wm"]["weight"])

    # Library changed (paper-scope library): bins differ -> loading the old statistics is refused...
    lib2 = StubLib([400] * 4)
    with pytest.raises(ValueError, match="resume-reset-sampler"):
        stub_trainer(lib2).load(tmp_path / "ck.pt")
    # ...and the reset flag makes it work.
    tr = stub_trainer(lib2, resume_reset_sampler=True)
    tr.load(tmp_path / "ck.pt")
    assert torch.all(tr.command.bin_return == 10.0)


def test_constant_lr_overrides_checkpoint_and_cosine(tmp_path):
    from supermimic.training.schedule import cosine_lr

    lib = StubLib([400] * 5)
    save_ckpt(stub_trainer(lib), tmp_path / "ck.pt", lr=8e-5)

    tr = stub_trainer(lib)
    tr.cfg.schedule.lr_constant = 5e-5
    tr.load(tmp_path / "ck.pt")
    assert tr.effective_lrs() == {"world_model": 8e-5, "policy": 8e-5, "critic": 8e-5}  # checkpoint's rates
    for it in (144_000, 200_000, 260_000, 400_000):
        tr.apply_lr(it)
        assert tr.effective_lrs() == pytest.approx({"world_model": 5e-5, "policy": 5e-5, "critic": 5e-5}, rel=1e-12)

    # Policy/critic keep their configured ratio to the world-model rate.
    tr.cfg.policy_trainer.lr = 1e-4
    tr.apply_lr(144_000)
    assert tr.effective_lrs() == pytest.approx({"world_model": 5e-5, "policy": 2.5e-5, "critic": 2.5e-5}, rel=1e-12)

    # Without the override: the cosine schedule (unchanged behaviour).
    tr = stub_trainer(lib)
    tr.apply_lr(144_000)
    want = cosine_lr(144_000, 320_000, 2e-4, 0.03, 250_000 / 320_000)
    assert tr.effective_lrs() == pytest.approx({"world_model": want, "policy": want, "critic": want}, rel=1e-12)


# ---------------------------------------------------------------------- real Trainer (GPU)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_trainer_finetune_resume(tmp_path, capsys):
    """Resume a real run onto a different library with --resume-reset-sampler and --schedule.lr-constant: the
    sampler starts fresh, the constant rate is in all three optimizers before and after training steps."""
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    def small(cfg: TrainConfig) -> TrainConfig:
        cfg.env.num_envs = 16
        cfg.env.replay_capacity = 64
        cfg.world_model.hidden = (64, 64)
        cfg.policy.encoder_hidden = (64, 64)
        cfg.policy.decoder_hidden = (64, 64)
        cfg.wm_trainer.batch_size = 64
        cfg.wm_trainer.batches_per_update = 1
        cfg.wm_trainer.compile = None
        cfg.policy_trainer.batch_size = 32
        cfg.policy_trainer.horizon = 4
        cfg.policy_trainer.critic_epochs = 1
        cfg.policy_trainer.compile = None
        cfg.schedule.warmup_iterations = 12
        cfg.schedule.log_interval = 10
        cfg.schedule.checkpoint_interval = 1000
        return cfg

    cfg = small(TrainConfig(run_name="a", log_root=tmp_path, seed=1))
    cfg.data.clips = ["walk1_subject1"]
    cfg.schedule.total_iterations = 20
    tr = Trainer(cfg)
    tr.run()
    tr.env.close()
    ck = tmp_path / "a" / "ckpt_latest.pt"

    cfg2 = small(TrainConfig(run_name="b", log_root=tmp_path, seed=2, resume=ck))
    cfg2.data.clips = ["walk1_subject1", "dance1_subject1"]
    cfg2.schedule.total_iterations = 40
    cfg2.schedule.lr_constant = 5e-5
    with pytest.raises(ValueError, match="resume-reset-sampler"):
        Trainer(cfg2)
    cfg2.resume_reset_sampler = True
    tr2 = Trainer(cfg2)
    try:
        assert tr2.iteration == 20
        assert torch.all(tr2.command.bin_return == tr2.command.cfg.g_max)
        assert tr2.command.cfg.gmax_mode == "horizon"
        assert int(tr2.command.bin_visits.sum()) == 0
        tr2.run()
        assert tr2.num_updates > 0
        ck2 = torch.load(tmp_path / "b" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        assert torch.equal(ck2["bin_visits"], tr2.command.bin_visits.cpu()) and int(ck2["bin_visits"].max()) > 0
        # Visited bins left G_max after their first visit (warm start), unvisited ones are still at G_max.
        seen = tr2.command.bin_visits > 0
        assert float((tr2.command.bin_return[seen] < 10.0 - 1e-3).float().mean()) > 0.5
        assert torch.all(tr2.command.bin_return[~seen] == 10.0)
        assert torch.isfinite(tr2.command.bin_return).all() and float(tr2.command.bin_return.max()) <= 10.0 + 1e-4
        assert tr2.effective_lrs() == pytest.approx({"world_model": 5e-5, "policy": 5e-5, "critic": 5e-5}, rel=1e-12)
        out = capsys.readouterr().out
        assert "lr schedule constant 5e-05; effective lr at iteration 20: world_model 5e-05, policy 5e-05, critic 5e-05" in out
        assert "sampler reset" in out
    finally:
        tr2.env.close()


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_clip_end_episode_logging_real_env():
    """The horizon k of a clip-end episode assumes the env logs the last frame twice (the step that reaches it and
    the time-out step), i.e. an episode whose first logged frame is f has last - f + 2 logged steps."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1"])
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, action_noise_std=0.0, obs_noise=False,
                                reset_noise=False, domain_randomization=False, record_transitions=False, seed=0)
    env = make_env(args, device="cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        last = int(cmd.library.clip_last[0])
        seen = []
        orig = cmd._finalize_episodes

        def spy(env_ids):
            for e in env_ids[cmd.clip_end[env_ids]].tolist():
                n = int(cmd.ep_len[e])
                seen.append((cmd.ep_frames[e, :n].tolist(), cmd.ep_rewards[e, :n].tolist()))
            return orig(env_ids)

        cmd._finalize_episodes = spy
        cmd.forced_frame[:] = last - 6
        obs, _ = env.reset()
        act = torch.zeros(8, env.action_manager.total_action_dim, device=env.device)
        for _ in range(12):
            env.step(act)
        assert seen, "no clip-end episode"
        for frames, rewards in seen:
            # The command update after a reset advances the frame once: the first rewarded frame is f0 + 1.
            assert frames[0] == last - 5 and len(frames) == last - frames[0] + 2
            assert frames[-1] == frames[-2] == last
        # Normalised returns stay on the G_max scale.
        assert torch.isfinite(cmd.bin_return).all() and float(cmd.bin_return.max()) <= cmd.cfg.g_max + 1e-4
    finally:
        env.close()
