"""Analytic reward / observation / termination must match mjlab's env computations (plan 2.8, 2.9)."""

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
def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    from mjlab.tasks.tracking import mdp as track_mdp

    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    files = discover_npz(NPZ, names=["walk1_subject1", "fight1_subject2"])
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=16, replay_capacity=64, obs_noise=False, seed=3)
    env = make_env(args, device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False  # make_env turns TF32 on; keep the comparison exact
    robot = env.scene["robot"]
    fk = G1Kinematics(load_g1_model(), device=env.device, dtype=torch.float32)
    heads = AnalyticHeads(
        fk,
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        soft_limits=robot.data.soft_joint_pos_limits[0].clone(),
        default_joint_pos=robot.data.default_joint_pos[0].clone(),
    )
    yield dict(env=env, heads=heads, io=StateIO(env), track_mdp=track_mdp, robot=robot)
    env.close()


def _ref_from_command(cmd):
    from supermimic.world_model.heads import ReferenceFrame

    lib, f = cmd.library, cmd.frame
    return ReferenceFrame(
        body_pos_w=lib.body_pos_w[f], body_quat_w=lib.body_quat_w[f],
        body_lin_vel_w=lib.body_lin_vel_w[f], body_ang_vel_w=lib.body_ang_vel_w[f],
        root_pos_w=lib.root_pos_w[f], root_quat_w=lib.root_quat_w[f],
    )


def test_reward_terms_match_mjlab(setup):
    env, heads, io, mdp, robot = setup["env"], setup["heads"], setup["io"], setup["track_mdp"], setup["robot"]
    from mjlab.envs import mdp as base_mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    env.reset()
    cmd = env.command_manager.get_term("motion")
    worst = {}
    for step in range(25):
        act = 0.5 * torch.randn(env.num_envs, 29, device=env.device)
        env.step(act)
        # Post-step: sim.forward() has run and the command advanced; evaluate both on this instant.
        s = io.read()
        ours = heads.reward(s, env.action_manager.action, env.action_manager.prev_action, _ref_from_command(cmd))
        asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
        asset_cfg.resolve(env.scene)
        theirs = {
            "motion_global_root_pos": 0.5 * mdp.motion_global_anchor_position_error_exp(env, "motion", 0.3),
            "motion_global_root_ori": 0.5 * mdp.motion_global_anchor_orientation_error_exp(env, "motion", 0.4),
            "motion_body_pos": mdp.motion_relative_body_position_error_exp(env, "motion", 0.3),
            "motion_body_ori": mdp.motion_relative_body_orientation_error_exp(env, "motion", 0.4),
            "motion_body_lin_vel": mdp.motion_global_body_linear_velocity_error_exp(env, "motion", 1.0),
            "motion_body_ang_vel": mdp.motion_global_body_angular_velocity_error_exp(env, "motion", 3.14),
            "action_rate_l2": -0.1 * base_mdp.action_rate_l2(env),
            "joint_limit": -2.0 * base_mdp.joint_pos_limits(env, asset_cfg),
        }
        for k, v in theirs.items():
            d = (ours[k] / 0.02 - v).abs().max().item()
            worst[k] = max(worst.get(k, 0.0), d)
    for k, d in worst.items():
        assert d < 2e-5, (k, d, worst)


def test_observation_head_matches_env(setup):
    env, heads, io, robot = setup["env"], setup["heads"], setup["io"], setup["robot"]
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    obs, _ = env.reset()
    rec = env.recorder_manager.get_term("replay")
    q0 = robot.data.default_joint_pos[0]
    hist = []
    for step in range(14):
        act = 0.5 * torch.randn(env.num_envs, 29, device=env.device)
        obs, *_ = env.step(act)
        if env.reset_buf.any():
            pytest.skip("an env reset during the short rollout; rerun")  # keeps the test simple and exact
        s = io.read()
        o = analytic_observation(s, env.action_manager.action, rec.ctx.read(), q0)
        hist.append(o)
        newest = obs["policy"].view(env.num_envs, -1)
        # Newest frame of each term in the term-major layout.
        start = 0
        for name, d in (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29)):
            block = newest[:, start * 10 : (start + d) * 10].reshape(env.num_envs, 10, d)  # time-major inside a term
            env_newest = block[:, -1, :]
            our = o[:, start : start + d]
            np.testing.assert_allclose(our.cpu().numpy(), env_newest.cpu().numpy(), atol=2e-5, err_msg=name)
            start += d
    full = flatten_history_term_major(torch.stack(hist[-10:], 0))
    np.testing.assert_allclose(full.cpu().numpy(), obs["policy"].cpu().numpy(), atol=2e-5)


def test_termination_matches_env(setup):
    env, heads, io = setup["env"], setup["heads"], setup["io"]
    from supermimic.envs.mdp import terminations as sm_term

    env.reset()
    cmd = env.command_manager.get_term("motion")
    for _ in range(10):
        env.step(1.0 * torch.randn(env.num_envs, 29, device=env.device))
        s = io.read()
        ref = _ref_from_command(cmd)
        b = heads.robot_bodies(s)
        f = heads.termination_features(b["root_pos"], b["root_quat"], ref)
        theirs_h = sm_term.bad_root_height(env, "motion", 0.25)
        theirs_t = sm_term.bad_root_tilt(env, "motion", 0.8)
        # Compare the continuous quantities (booleans could flip on ties).
        h_env = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2])
        np.testing.assert_allclose(f[:, 0].cpu().numpy(), h_env.cpu().numpy(), atol=2e-5)
        assert torch.equal((f[:, 0].abs() > 0.25), theirs_h) or (f[:, 0].abs() - 0.25).abs().min() < 1e-4
        assert torch.equal((f[:, 1].abs() > 0.8), theirs_t) or (f[:, 1].abs() - 0.8).abs().min() < 1e-4


def test_anchor_relative_obs_matches_env():
    """The analytic anchor-relative term (from state via FK) equals mjlab's observation terms."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, anchor_relative_obs

    files = discover_npz(NPZ, names=["walk1_subject1"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, obs_noise=False, anchor_obs=True, record_transitions=False, seed=4), "cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device=env.device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        io = StateIO(env)
        obs, _ = env.reset()
        assert obs["goal"].shape == (8, 689)
        cmd = env.command_manager.get_term("motion")
        for _ in range(5):
            obs, *_ = env.step(0.3 * torch.randn(8, 29, device=env.device))
            s = io.read()
            b = heads.robot_bodies(s)
            ai = heads.anchor_index
            ref = _ref_from_command(cmd)
            ours = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
            np.testing.assert_allclose(ours.cpu().numpy(), obs["goal"][:, 680:].cpu().numpy(), atol=3e-5)
    finally:
        env.close()
