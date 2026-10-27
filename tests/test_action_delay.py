"""Command-latency DR (--env.dr-action-delay-substeps): per env, the first d substeps of a policy step still apply the
previous set-point; off by default."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
CLIPS = ["walk1_subject1", "dance1_subject1"]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")


def _env(delay):
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=CLIPS)
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=16, replay_capacity=64, seed=3,
                                action_noise_std=0.0, dr_action_delay_substeps=delay)
    return make_env(args, device="cuda:0")


def _record(env):
    term = env.action_manager.get_term("joint_pos")
    ent = term._entity
    log = []
    orig = ent.set_joint_position_target

    def rec(target, joint_ids=None):
        log.append(target.clone())
        return orig(target, joint_ids=joint_ids)

    ent.set_joint_position_target = rec
    return term, log


def test_off_by_default():
    from supermimic.config import EnvCfg
    from supermimic.envs import SuperMimicEnvCfgArgs

    assert EnvCfg().dr_action_delay_substeps == 0 and SuperMimicEnvCfgArgs(npz_files=[]).dr_action_delay_substeps == 0


def test_delayed_substeps_apply_previous_setpoint():
    env = _env(4)
    try:
        env.reset()
        term, log = _record(env)
        dec = env.cfg.decimation
        n, a = env.num_envs, term.action_dim
        a1 = torch.full((n, a), 0.2, device=env.device)
        a2 = torch.full((n, a), -0.2, device=env.device)
        env.step(a1)
        t1 = log[-1].clone()
        log.clear()
        env.step(a2)
        assert len(log) == dec
        t2 = log[-1]
        d = term._delay
        assert d.min() >= 0 and d.max() <= 4 and len(set(d.tolist())) > 1
        for s in range(dec):
            late = s < d
            assert torch.allclose(log[s][late], t1[late]) and torch.allclose(log[s][~late], t2[~late])
        assert not torch.allclose(t1, t2)
    finally:
        env.close()
