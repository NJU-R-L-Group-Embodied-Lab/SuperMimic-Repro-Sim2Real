"""GAE targets, policy objective, and a short end-to-end training smoke run."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def test_gae_targets_match_hand_computation():
    from supermimic.imagination.objective import gae_targets

    H, n = 4, 3
    g, lam = 0.9, 0.8
    torch.manual_seed(0)
    r = torch.rand(H, n)
    d = torch.rand(H, n) * 0.3
    V = torch.rand(H + 1, n)
    out = gae_targets(r, d, V, g, lam)
    R = V[H].clone()
    for t in range(H - 1, -1, -1):
        R = r[t] + g * (1 - d[t]) * ((1 - lam) * V[t + 1] + lam * R)
        np.testing.assert_allclose(out[t].numpy(), R.numpy(), atol=1e-6)
    # lambda = 1, d = 0 -> Monte-Carlo discounted return with bootstrap.
    out1 = gae_targets(r, torch.zeros_like(d), V, g, 1.0)
    mc = r[0] + g * r[1] + g**2 * r[2] + g**3 * r[3] + g**4 * V[4]
    np.testing.assert_allclose(out1[0].numpy(), mc.numpy(), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_smoke(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="smoke", log_root=tmp_path, seed=1)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.schedule.total_iterations = 60
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 60
    tr = Trainer(cfg)
    try:
        tr.run()
        assert tr.num_updates > 0
        assert (tmp_path / "smoke" / "ckpt_latest.pt").exists()
        # Resume path works.
        cfg2 = TrainConfig(**{**cfg.__dict__, "resume": tmp_path / "smoke" / "ckpt_latest.pt", "run_name": "smoke2"})
        tr.env.close()
        tr2 = Trainer(cfg2)
        assert tr2.iteration == 60
        tr2.env.close()
    finally:
        pass
