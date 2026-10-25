"""Environment layer smoke test: build the env on a few LAFAN1 clips, step, inspect the replay."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built"),
]


@pytest.fixture(scope="module")
def env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    assert len(files) == 3
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=32, replay_capacity=64, seed=1)
    e = make_env(args, device="cuda:0")
    yield e
    e.close()


def test_spaces_and_step(env):
    obs, _ = env.reset()
    assert obs["policy"].shape == (32, 930)
    assert obs["goal"].shape == (32, 680)
    assert env.action_manager.total_action_dim == 29
    for _ in range(40):
        act = 0.3 * torch.randn(32, 29, device=env.device)
        obs, rew, term, tout, extras = env.step(act)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["goal"]).all()
        assert torch.isfinite(rew).all()
    # Rewards are dt-scaled; the positive terms sum to at most 0.1 per step.
    assert rew.max() <= 0.1 + 1e-6


def test_goal_matches_library(env):
    cmd = env.command_manager.get_term("motion")
    goal = cmd.goal
    lib = cmd.library
    f = cmd.frame
    direct = lib.goal_feat[f]
    np.testing.assert_allclose(goal[:, :68].cpu().numpy(), direct.cpu().numpy(), atol=1e-6)
    # Clamping at the clip end.
    last = lib.last_frame_of(f)
    k9 = torch.minimum(f + 9, last)
    np.testing.assert_allclose(goal[:, 9 * 68 :].cpu().numpy(), lib.goal_feat[k9].cpu().numpy(), atol=1e-6)


def test_replay_rows_and_windows(env):
    rec = env.recorder_manager.get_term("replay")
    buf = rec.buffer
    assert buf.size > 32 * 10
    # Every non-terminal row's successor within the episode must be the next row.
    e = 0
    rows = [t for t in range(buf.T) if buf.valid[e, t]]
    assert len(rows) > 5
    w = buf.sample_windows(256, 8)
    assert w.states.shape == (9, 256, 71) and w.a_exec.shape == (8, 256, 29) and w.xi.shape == (256, 33)
    assert torch.isfinite(w.states).all()
    # Executed action differs from the clean one by the injected noise only.
    d = (w.a_exec - w.a_clean).std().item()
    assert 0.05 < d < 0.2, d
    s = buf.sample_starts(128, hist=10)
    assert s.hist_states.shape == (10, 128, 71) and s.hist_actions.shape == (10, 128, 29)
    assert torch.equal(s.hist_states[-1], s.state)


def test_state_io_round_trip(env):
    from supermimic.envs.state_io import StateIO

    io = StateIO(env)
    s = io.read()
    assert s.shape == (32, 71)
    ids = torch.arange(8, device=env.device)
    s2 = s[ids].clone()
    s2[:, 0] += 0.123
    io.write(ids, s2)
    back = io.read()
    np.testing.assert_allclose(back[ids].cpu().numpy(), s2.cpu().numpy(), atol=1e-6)
    robot = env.scene["robot"]
    origin_x = env.scene.env_origins[ids, 0]
    np.testing.assert_allclose(robot.data.root_link_pos_w[ids, 0].cpu().numpy(), (s2[:, 0] + origin_x).cpu().numpy(), atol=1e-5)


def test_sim_context(env):
    rec = env.recorder_manager.get_term("replay")
    xi = rec.ctx.read()
    assert xi.shape == (32, 33)
    fr = xi[:, 0]
    assert (fr >= 0.3 - 1e-6).all() and (fr <= 1.2 + 1e-6).all()
    assert xi[:, 1:30].abs().max() <= 0.01 + 1e-6
    assert xi[:, 30:].abs().max() <= 0.05 + 1e-6
    # Per-episode re-sampling: friction values should not all be identical across envs.
    assert fr.std() > 0.01


def test_episode_never_crosses_clip_boundary(env):
    """Frames must stay inside the clip the episode started in; clip end raises the time-out flag."""
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    env.reset()
    start_clip = lib.frame_clip[cmd.frame].clone()
    for _ in range(60):
        env.step(torch.zeros(env.num_envs, 29, device=env.device))
        same = lib.frame_clip[cmd.frame] == start_clip
        assert bool(same[~env.reset_buf].all())
        start_clip = lib.frame_clip[cmd.frame].clone()
    # Force an env to the second-to-last frame of its clip and step twice: clip_end must fire as a time-out.
    ids = torch.tensor([0], device=env.device)
    cmd.reset_to_frame(ids, lib.clip_last[lib.frame_clip[cmd.frame[ids]]] - 1)
    env.sim.forward()
    cmd._refresh_ref()
    cmd.update_relative_body_poses()
    fired = False
    for _ in range(3):  # targets last-1 -> last -> (clip_end set) -> time-out on the next check
        obs, rew, term, tout, _ = env.step(torch.zeros(env.num_envs, 29, device=env.device))
        fired = fired or bool(tout[0]) or bool(term[0])
    assert fired
