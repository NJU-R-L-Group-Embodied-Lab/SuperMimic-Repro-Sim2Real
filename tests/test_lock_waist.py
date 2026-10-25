"""Waist-locked G1 (--env.lock-waist): the locked joints never move, read exactly 0, and the feasible references are
FK-consistent with the locked model."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
CLIPS = ["walk1_subject1", "fight1_subject2", "dance1_subject1"]
LOCK_COLS_71 = [7 + 13, 7 + 14, 42 + 13, 42 + 14]
needs_env = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built"),
]


# ------------------------------------------------------------------------------------------------ model


def test_locked_model_is_29dof_minus_waist_roll_pitch():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    from supermimic.envs.lock_waist import ACTIVE_JOINT_IDS, LOCKED_JOINT_IDS, LOCKED_JOINT_NAMES, lock_waist_spec

    full, locked = get_spec().compile(), lock_waist_spec().compile()
    assert (full.nq, full.nv, full.njnt) == (36, 35, 30)
    assert (locked.nq, locked.nv, locked.njnt) == (34, 33, 28)
    names = [locked.joint(i).name for i in range(1, locked.njnt)]
    full_names = [full.joint(i).name for i in range(1, full.njnt)]
    assert [full_names[i] for i in LOCKED_JOINT_IDS] == list(LOCKED_JOINT_NAMES)
    assert names == [full_names[i] for i in ACTIVE_JOINT_IDS]
    # Same bodies, masses, inertias and body frames (Unitree's lock_waist MJCF differs from the 29-DoF one only in the
    # two deleted joints as well).
    assert full.nbody == locked.nbody
    for f in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(full, f), getattr(locked, f), err_msg=f)
    # Same kinematics as the 29-DoF model with waist roll/pitch at 0.
    rng = np.random.default_rng(0)
    d_full, d_lock = mujoco.MjData(full), mujoco.MjData(locked)
    for _ in range(5):
        q = rng.uniform(full.jnt_range[1:, 0], full.jnt_range[1:, 1])
        q[list(LOCKED_JOINT_IDS)] = 0.0
        d_full.qpos[:7] = d_lock.qpos[:7] = [0.1, -0.2, 0.8, 0.9, 0.1, -0.2, 0.3]
        d_full.qpos[7:] = q
        d_lock.qpos[7:] = q[list(ACTIVE_JOINT_IDS)]
        mujoco.mj_kinematics(full, d_full)
        mujoco.mj_kinematics(locked, d_lock)
        np.testing.assert_allclose(d_lock.xpos, d_full.xpos, atol=1e-12)
        np.testing.assert_allclose(d_lock.xquat, d_full.xquat, atol=1e-12)


def test_world_model_projection_keeps_locked_joints_at_zero():
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    spec = StateSpec(StateSpecCfg(locked_joints=(13, 14)), "cpu")
    s = torch.randn(64, 71)
    s[:, 3:7] = torch.nn.functional.normalize(s[:, 3:7], dim=-1)
    s[:, LOCK_COLS_71] = 0.0
    for _ in range(5):
        s = spec.integrate(s, torch.randn(64, 70) * 3)
    assert (s[:, LOCK_COLS_71] == 0).all()
    a = spec.mask_action(torch.randn(8, 29))
    assert (a[:, [13, 14]] == 0).all() and (a[:, :13] != 0).all()
    assert StateSpec(StateSpecCfg(), "cpu").joint_mask is None  # unlocked: untouched


# ------------------------------------------------------------------------------------------------ references


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_feasible_references_fk_consistent():
    """Feasible references: waist roll/pitch (and rates) 0 in every frame, pelvis unchanged, tracked bodies == MuJoCo
    kinematics of the locked model, full (FK-recomputed) and compact (FK on the fly) libraries agree."""
    torch.backends.cuda.matmul.allow_tf32 = False
    from supermimic.envs.lock_waist import ACTIVE_JOINT_IDS, LOCKED_JOINT_IDS, lock_waist_spec
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    files = discover_npz(NPZ, names=CLIPS[:2])
    q0 = torch.zeros(29)
    orig = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=False)
    full = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=False, zero_joints=LOCKED_JOINT_IDS)
    comp = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=True, store_dtype=torch.float32, zero_joints=LOCKED_JOINT_IDS)
    # The source clips do use the waist (otherwise this test would be vacuous).
    assert orig.joint_pos[:, 14].abs().max() > 0.05
    f = torch.randint(0, full.num_frames, (256,), device="cuda")
    for lib in (full, comp):
        r = lib.ref(f)
        assert (r.joint_pos[:, list(LOCKED_JOINT_IDS)] == 0).all() and (r.joint_vel[:, list(LOCKED_JOINT_IDS)] == 0).all()
        g = lib.goal_window(f, 10).view(-1, 10, 68)
        assert (g[..., [13, 14, 29 + 13, 29 + 14]] == 0).all()
    ro, rf = orig.ref(f), full.ref(f)
    for k in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"):
        assert torch.equal(getattr(ro, k), getattr(rf, k)), k
    other = [j for j in range(29) if j not in LOCKED_JOINT_IDS]
    assert torch.equal(ro.joint_pos[:, other], rf.joint_pos[:, other])
    # Goal features other than q/dq 13/14 are unchanged.
    keep = [i for i in range(68) if i not in (13, 14, 29 + 13, 29 + 14)]
    go, gf = orig.goal_window(f, 10).view(-1, 10, 68), full.goal_window(f, 10).view(-1, 10, 68)
    torch.testing.assert_close(gf[..., keep], go[..., keep], atol=2e-6, rtol=0)  # same math, different kernel batching
    # Full (FK-recomputed) == compact (FK on the fly).
    bf, bc = full.bodies(f), comp.bodies(f)
    torch.testing.assert_close(bc.pos_w, bf.pos_w, atol=1e-6, rtol=0)
    torch.testing.assert_close(bc.lin_vel_w, bf.lin_vel_w, atol=1e-5, rtol=0)
    # Pelvis/legs unchanged, torso no longer matches the original reference (the waist was moving).
    bo = orig.bodies(f)
    pel, lknee, torso = (TRACKED_BODIES.index(n) for n in ("pelvis", "left_knee_link", "torso_link"))
    torch.testing.assert_close(bf.pos_w[:, [pel, lknee]], bo.pos_w[:, [pel, lknee]], atol=2e-5, rtol=0)
    assert (bf.quat_w[:, torso] - bo.quat_w[:, torso]).abs().max() > 1e-2
    # Tracked bodies == MuJoCo kinematics of the *locked* model with the reference pelvis + 27 joints.
    m = lock_waist_spec().compile()
    d = mujoco.MjData(m)
    qpos, qvel = full.ref_qpos_qvel(f[:16])
    qpos, qvel = qpos.double().cpu().numpy(), qvel.double().cpu().numpy()
    names = [m.body(i).name for i in range(m.nbody)]
    bidx = [names.index(n) for n in TRACKED_BODIES]
    res = np.zeros(6)
    for i in range(16):
        d.qpos[:7], d.qpos[7:] = qpos[i, :7], qpos[i, 7:][list(ACTIVE_JOINT_IDS)]
        d.qvel[:6], d.qvel[6:] = qvel[i, :6], qvel[i, 6:][list(ACTIVE_JOINT_IDS)]
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(bf.pos_w[i].cpu().numpy(), d.xpos[bidx], atol=2e-5)
        q_ref = d.xquat[bidx]
        q = bf.quat_w[i].cpu().numpy()
        q = q * np.sign((q * q_ref).sum(-1, keepdims=True))
        np.testing.assert_allclose(q, q_ref, atol=2e-5)
        for k, b in enumerate(bidx):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            np.testing.assert_allclose(bf.lin_vel_w[i, k].cpu().numpy(), res[3:], atol=2e-4)
            np.testing.assert_allclose(bf.ang_vel_w[i, k].cpu().numpy(), res[:3], atol=2e-4)


# ------------------------------------------------------------------------------------------------ env


@pytest.fixture(scope="module")
def lenv():
    torch.backends.cuda.matmul.allow_tf32 = False
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=CLIPS)
    # Training defaults (obs noise, DR incl. encoder bias, reset noise, action noise) + lock.
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=32, replay_capacity=128, seed=5, lock_waist=True)
    env = make_env(args, device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield env
    env.close()


@pytest.mark.parametrize("_", [0], ids=["lock"])
@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_locked_joints_never_move(lenv, _):
    from mjlab.utils.lab_api.math import quat_inv, quat_mul

    from supermimic.envs.state_io import StateIO

    env = lenv
    robot = env.scene["robot"]
    assert robot.num_joints == 27 and env.action_manager.total_action_dim == 29
    assert env.sim.data.qpos.shape[-1] == 34 and env.sim.data.qvel.shape[-1] == 33
    io = StateIO(env)
    rec = env.recorder_manager.get_term("replay")
    cmd = env.command_manager.get_term("motion")
    bnames = list(robot.body_names)
    yaw_b, torso_b = bnames.index("waist_yaw_link"), bnames.index("torso_link")
    obs, _ = env.reset()
    n_resets = 0
    for step in range(80):
        act = 0.5 * torch.randn(env.num_envs, 29, device=env.device)
        act[:, 13:15] = 20.0 * torch.randn(env.num_envs, 2, device=env.device)  # would slam an unlocked waist
        obs, rew, term, tout, _ = env.step(act)
        n_resets += int((term | tout).sum())
        assert torch.isfinite(rew).all() and torch.isfinite(obs["policy"]).all()
        s = io.read()
        assert (s[:, LOCK_COLS_71] == 0).all()
        # torso rigidly attached to waist_yaw_link at the lock pose (constant relative orientation)
        rel = quat_mul(quat_inv(robot.data.body_link_quat_w[:, yaw_b]), robot.data.body_link_quat_w[:, torso_b])
        rel = rel * torch.sign(rel[:, :1])
        if step == 0:
            rel0 = rel[0].clone()
        assert (rel - rel0).abs().max() < 1e-5, (rel - rel0).abs().max()
        # observation: q / dq of 13/14 exactly 0 in every history slot, despite obs noise and encoder bias
        p = obs["policy"].view(env.num_envs, -1)
        jp = p[:, 0:290].view(env.num_envs, 10, 29)
        jv = p[:, 290:580].view(env.num_envs, 10, 29)
        assert (jp[..., 13:15] == 0).all() and (jv[..., 13:15] == 0).all()
        assert jp[..., :13].abs().max() > 0  # the other joints are live
        last_a = p[:, 640:930].view(env.num_envs, 10, 29)[:, -1]
        live = ~(term | tout)
        torch.testing.assert_close(last_a[live], act[live])  # clean 29-d action observed, 13/14 included
        # feasible references
        assert (cmd.joint_pos[:, 13:15] == 0).all() and (cmd.robot_joint_pos[:, 13:15] == 0).all()
        g = obs["goal"].view(env.num_envs, 10, 68)
        assert (g[..., [13, 14, 42, 43]] == 0).all()
    assert n_resets > 0  # resets wrote reference states into the locked sim
    assert robot.data.encoder_bias.abs().max() > 0
    xi = rec.ctx.read()
    assert xi.shape == (env.num_envs, 33) and (xi[:, 1 + 13] == 0).all() and (xi[:, 1 + 14] == 0).all()
    buf = rec.buffer
    assert (buf.s[buf.valid][:, LOCK_COLS_71] == 0).all() and (buf.xi[buf.valid][:, [14, 15]] == 0).all()
    w = buf.sample_windows(64, 8)
    assert w.states.shape == (9, 64, 71) and w.a_exec.shape == (8, 64, 29)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_locked_analytic_heads_match_env(lenv):
    """Imagination heads (29-joint layout) reproduce the locked env's reward terms and observations."""
    from mjlab.envs import mdp as base_mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.tracking import mdp as track_mdp

    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.lock_waist import LOCKED_JOINT_IDS, full_joint_constants
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, analytic_observation

    env = lenv
    torch.backends.cuda.matmul.allow_tf32 = False
    robot = env.scene["robot"]
    q0, soft = full_joint_constants(robot)
    assert q0.shape == (29,) and soft.shape == (29, 2) and q0[13] == 0 and soft[13, 0] < 0 < soft[13, 1]
    rcfg = TrackingRewardCfg(locked_action_ids=LOCKED_JOINT_IDS, w_locked_action=-0.1)
    heads = AnalyticHeads(G1Kinematics(load_g1_model(), device=env.device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, soft, q0, rcfg)
    io = StateIO(env)
    rec = env.recorder_manager.get_term("replay")
    cmd = env.command_manager.get_term("motion")
    env.reset()
    asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    asset_cfg.resolve(env.scene)
    worst = {}
    for _ in range(15):
        obs, *_ = env.step(0.5 * torch.randn(env.num_envs, 29, device=env.device))
        s = io.read()
        b = cmd.library.bodies(cmd.frame)
        r = cmd.library.ref(cmd.frame)
        ref = ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)
        ours = heads.reward(s, env.action_manager.action, env.action_manager.prev_action, ref)
        theirs = {
            "motion_body_pos": track_mdp.motion_relative_body_position_error_exp(env, "motion", 0.3),
            "motion_body_ori": track_mdp.motion_relative_body_orientation_error_exp(env, "motion", 0.4),
            "motion_global_root_ori": 0.5 * track_mdp.motion_global_anchor_orientation_error_exp(env, "motion", 0.4),
            "action_rate_l2": -0.1 * base_mdp.action_rate_l2(env),
            "joint_limit": -2.0 * base_mdp.joint_pos_limits(env, asset_cfg),
            "locked_action_l2": -0.1 * env.action_manager.action[:, 13:15].square().sum(-1),
        }
        for k, v in theirs.items():
            worst[k] = max(worst.get(k, 0.0), (ours[k] / 0.02 - v).abs().max().item())
        tr = env.reward_manager.term_rewards if hasattr(env.reward_manager, "term_rewards") else None
        if tr is not None:
            i = list(env.reward_manager.active_terms).index("locked_action_l2")
            assert torch.isfinite(tr[:, i]).all()
        if not env.reset_buf.any():
            o = analytic_observation(s, env.action_manager.action, rec.ctx.read(), q0)
            newest = obs["policy"].view(env.num_envs, -1)
            np.testing.assert_allclose(o[:, 0:29].cpu().numpy(), newest[:, 9 * 29 : 10 * 29].cpu().numpy(), atol=0.011)  # obs noise 0.01
            assert (o[:, [13, 14, 29 + 13, 29 + 14]] == 0).all()
    for k, d in worst.items():
        assert d < 2e-5, (k, d, worst)
