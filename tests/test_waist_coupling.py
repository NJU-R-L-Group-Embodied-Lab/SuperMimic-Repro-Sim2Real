"""Real-G1 waist roll/pitch coupling (--env.waist-coupling / sim2sim --waist-coupling): |roll| + k |pitch| <= c as two
limited fixed tendons; off by default, joint ranges unchanged."""

import copy
import importlib.util
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.deploy.waist_coupling import ENV_VAR, SOLREF_LIMIT, TENDON_NAMES, WAIST_COUPLING_C as C, WAIST_COUPLING_K as K, add_waist_coupling, coupling_from_env

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
CLIPS = ["walk1_subject1", "dance1_subject1"]
ROLL, PITCH = 13, 14  # G1_JOINT_NAMES order
TOL = 0.02  # tendon limits are soft constraints (quasi-static penetration <= 0.003 rad with SOLREF_LIMIT / SOLIMP_LIMIT)
UNITREE_XML = Path(os.environ.get("G1_UNITREE_XML", Path.home() / "workspace/g1_deploy/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"))


def test_off_by_default(monkeypatch):
    from supermimic.config import EnvCfg
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import robot_cfg

    assert EnvCfg().waist_coupling is False and SuperMimicEnvCfgArgs(npz_files=[]).waist_coupling is False
    assert robot_cfg().spec_fn().compile().ntendon == 0
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert coupling_from_env() is None
    monkeypatch.setenv(ENV_VAR, "1.2,0.9")
    assert coupling_from_env() == (1.2, 0.9)


def test_training_spec_has_tendons_and_unchanged_ranges():
    from supermimic.envs.tracking_env_cfg import robot_cfg

    base = robot_cfg().spec_fn().compile()
    m = robot_cfg(waist_coupling=(K, C)).spec_fn().compile()
    assert m.ntendon == 2 and [m.tendon(i).name for i in range(2)] == list(TENDON_NAMES)
    np.testing.assert_allclose(m.tendon_range, [[-C, C], [-C, C]])
    assert m.tendon_limited.all()
    np.testing.assert_allclose(m.tendon_solref_lim, [SOLREF_LIMIT] * 2)
    np.testing.assert_array_equal(m.jnt_range, base.jnt_range)  # +-0.52 boxes kept
    for f in ("nq", "nv", "nbody", "njnt"):
        assert getattr(m, f) == getattr(base, f)
    with pytest.raises(ValueError):
        robot_cfg(lock_waist=True, waist_coupling=(K, C))


def _waist_pd(m, d, roll_t, pitch_t, n_steps=3000, kp=200.0, kd=5.0, effort=50.0, pitch_effort=50.0, pitch_kp=200.0):
    """Pinned pelvis, waist roll/pitch torque-PD (Unitree torque motors), other motors 0; returns final (roll, pitch).
    pitch_effort / pitch_kp above the real 50 Nm / 200 (test only, with the joint actuatorfrcrange raised) hold the pitch:
    with equal 50 Nm drives the tendon force lets the roll drive slide the pitch down along the envelope edge."""
    jr, jp = (m.joint(n).qposadr[0] for n in ("waist_roll_joint", "waist_pitch_joint"))
    vr, vp = (m.joint(n).dofadr[0] for n in ("waist_roll_joint", "waist_pitch_joint"))
    ar, ap = (m.actuator(n).id for n in ("waist_roll", "waist_pitch"))
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.0
    q7 = d.qpos[:7].copy()
    for _ in range(n_steps):
        d.qpos[:7], d.qvel[:6] = q7, 0.0
        d.ctrl[:] = 0.0
        d.ctrl[ar] = np.clip(kp * (roll_t - d.qpos[jr]) - kd * d.qvel[vr], -effort, effort)
        d.ctrl[ap] = np.clip(pitch_kp * (pitch_t - d.qpos[jp]) - kd * d.qvel[vp], -pitch_effort, pitch_effort)
        mujoco.mj_step(m, d)
    return float(d.qpos[jr]), float(d.qpos[jp])


def _sim2sim():
    spec = importlib.util.spec_from_file_location("sim2sim", ROOT / "scripts/deploy/sim2sim.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # robot-side deps (onnxruntime) missing in this venv
        pytest.skip(f"sim2sim import: {e}")
    return mod


@pytest.mark.skipif(not UNITREE_XML.exists(), reason="Unitree MJCF not found")
def test_sim2sim_models(monkeypatch):
    s2s = _sim2sim()
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert s2s.load_model(s2s.UNITREE_XML).ntendon == 0  # default off
    assert s2s.load_model(s2s.TRAIN_XML).ntendon == 0 if s2s.TRAIN_XML.exists() else True
    monkeypatch.setenv(ENV_VAR, f"{K},{C}")
    m = s2s.load_model(s2s.UNITREE_XML)
    assert m.ntendon == 2 and m.tendon_limited.all()
    if s2s.TRAIN_XML.exists():  # prefixed joint names ("robot/waist_roll_joint")
        mt = s2s.load_model(s2s.TRAIN_XML)
        assert mt.ntendon == 2 and mt.tendon(0).name == "robot/" + TENDON_NAMES[0]
    with pytest.raises(SystemExit):
        s2s.load_model(s2s.UNITREE_XML, lock=True)
    # Physics (Unitree torque-motor model, 1 kHz like G2): roll target -1.5 at pitch ~0.5 stops at the envelope; at pitch 0
    # roll still reaches the -0.52 joint limit.
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    # Real effort (50 Nm both): the roll drive slides pitch down along the edge, still on/inside the envelope.
    roll50, pitch50 = _waist_pd(m, d, -1.5, 0.52)
    print("sim2sim coupled 50 Nm: roll %.3f pitch %.3f envelope %.3f" % (roll50, pitch50, C - K * abs(pitch50)))
    assert abs(roll50) <= C - K * abs(pitch50) + TOL and pitch50 > 0.25, (roll50, pitch50)
    m.actuator_ctrlrange[:] = m.jnt_actfrcrange[:] = [-500, 500]  # test only: hold pitch ~0.5 against the tendon
    roll, pitch = _waist_pd(m, d, -1.5, 0.52, pitch_effort=500.0, pitch_kp=3000.0)
    print("sim2sim coupled: roll %.3f pitch %.3f envelope %.3f" % (roll, pitch, C - K * abs(pitch)))
    assert 0.45 < pitch < 0.56, pitch
    assert abs(roll) <= C - K * abs(pitch) + TOL, (roll, pitch)
    roll0, pitch0 = _waist_pd(m, d, -1.5, 0.0)
    assert abs(pitch0) < 0.05 and roll0 < -0.48, (roll0, pitch0)
    # Uncoupled reference: same drive reaches the joint box at pitch 0.5.
    monkeypatch.delenv(ENV_VAR)
    m0 = s2s.load_model(s2s.UNITREE_XML)
    m0.opt.timestep = 0.001
    m0.actuator_ctrlrange[:] = m0.jnt_actfrcrange[:] = [-500, 500]
    roll_u, pitch_u = _waist_pd(m0, mujoco.MjData(m0), -1.5, 0.52, pitch_effort=500.0, pitch_kp=3000.0)
    print("sim2sim uncoupled: roll %.3f pitch %.3f" % (roll_u, pitch_u))
    assert roll_u < -0.48 and pitch_u > 0.45, (roll_u, pitch_u)


def test_env_var_parsing_and_resolve(monkeypatch):
    from supermimic.deploy.waist_coupling import env_args, resolve_coupling

    for bad in ("1.4", "1.4,0.9,2", "abc", "1.4,-0.5"):
        monkeypatch.setenv(ENV_VAR, bad)
        with pytest.raises(ValueError, match=ENV_VAR):
            coupling_from_env()
    monkeypatch.setenv(ENV_VAR, "yes")
    assert coupling_from_env() == (K, C)
    monkeypatch.setenv(ENV_VAR, "off")
    assert coupling_from_env() is None
    monkeypatch.delenv(ENV_VAR)
    assert resolve_coupling({}) is None and resolve_coupling(None) is None and resolve_coupling({"waist_coupling": False}) is None
    assert resolve_coupling({"waist_coupling": True, "waist_coupling_k": 1.2, "waist_coupling_c": 0.9}) == (1.2, 0.9)
    monkeypatch.setenv(ENV_VAR, "1.3,0.8")  # the env var wins over the checkpoint's setting
    assert resolve_coupling({"waist_coupling": True, "waist_coupling_k": 1.2, "waist_coupling_c": 0.9}) == (1.3, 0.8)
    assert env_args(None) == {} and env_args((1.3, 0.8)) == {"waist_coupling": True, "waist_coupling_k": 1.3, "waist_coupling_c": 0.8}


def test_sim2sim_guards(monkeypatch, tmp_path):
    """G1 refuses a reference made with other waist physics; lw / lw_robust and --lock-waist are refused with coupling."""
    s2s = _sim2sim()
    monkeypatch.setattr(s2s, "REPORTS", tmp_path)
    (tmp_path / "sim2sim").mkdir()
    np.savez(tmp_path / "sim2sim" / "ref_t_x_no-tf32.npz", clips=np.array(["c"]), fail_step=np.array([-1]))
    monkeypatch.setenv(ENV_VAR, f"{K},{C}")
    with pytest.raises(SystemExit, match="waist coupling None"):
        s2s.g1_gate("x", {}, ("c",), ("c",), "t_")
    np.savez(tmp_path / "sim2sim" / "ref_t_y_no-tf32.npz", clips=np.array(["c"]), fail_step=np.array([-1]), waist_coupling=np.array([K, C]))
    monkeypatch.delenv(ENV_VAR)
    with pytest.raises(SystemExit, match="this run None"):
        s2s.g1_gate("y", {}, ("c",), ("c",), "t_")
    for argv in (["--cmd", "lw"], ["--cmd", "lw_robust"], ["--cmd", "g2", "--lock-waist"]):
        monkeypatch.setattr("sys.argv", ["sim2sim.py", *argv, "--waist-coupling"])
        with pytest.raises(SystemExit, match="waist-coupling"):
            s2s.main()


def test_add_waist_coupling_needs_waist_joints():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="b").add_joint(name="other")
    with pytest.raises(ValueError):
        add_waist_coupling(spec)


def _hold_pitch_cpu(m, roll_t, pitch_t, n_steps=600, pitch_gain=100.0):
    """Training MJCF on CPU MuJoCo (same opt: 5 ms, implicitfast, 10 iterations), pelvis pinned, position actuators; the
    pitch actuator made pitch_gain x stiffer with a 500 Nm limit (test only) so the pitch is held against the tendon."""
    m = copy.deepcopy(m)
    jr, jp = m.joint("robot/waist_roll_joint"), m.joint("robot/waist_pitch_joint")
    a_r, a_p = m.actuator("robot/waist_roll_joint").id, m.actuator("robot/waist_pitch_joint").id
    m.actuator_gainprm[a_p, 0] *= pitch_gain
    m.actuator_biasprm[a_p, 1:3] *= pitch_gain
    m.actuator_forcerange[a_p] = [-500, 500]
    m.jnt_actfrcrange[jp.id] = [-500, 500]
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0
    d.qpos[3:7] = [1, 0, 0, 0]
    q7 = d.qpos[:7].copy()
    for _ in range(n_steps):
        d.qpos[:7], d.qvel[:6] = q7, 0.0
        d.ctrl[:] = 0.0
        d.ctrl[a_r], d.ctrl[a_p] = roll_t, pitch_t
        mujoco.mj_step(m, d)
    return float(d.qpos[jr.qposadr[0]]), float(d.qpos[jp.qposadr[0]])


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_env_enforces_envelope():
    """Training env with --env.waist-coupling (mjlab, pelvis pinned, waist roll position target -1.5):
      CPU MuJoCo on the env's mj_model, pitch held ~0.5 by a stiffened pitch actuator -> |roll| <= c - k |pitch| (+ tol),
        while the uncoupled model reaches the -0.52 box; pitch 0 -> roll reaches ~-0.52;
      mujoco_warp (the training simulator), real 50 Nm actuators: envs 0..3 pitch target 0.52 -> the roll drive slides
        the pitch down along the edge, state on/inside the envelope and |roll| < 0.52 box only where it binds; envs 4..7
        pitch target 0 -> roll reaches the -0.52 joint limit;
      then 100 policy steps of random actions through env.step stay finite."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=CLIPS)
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, replay_capacity=64, seed=3, action_noise_std=0.0,
                                domain_randomization=False, waist_coupling=True)
    env = make_env(args, device="cuda:0")
    try:
        m = env.sim.mj_model
        assert m.ntendon == 2 and m.opt.timestep == 0.005
        roll, pitch = _hold_pitch_cpu(m, -1.5, 0.5)
        print("training MJCF (CPU) coupled, pitch held: roll %.3f pitch %.3f envelope %.3f" % (roll, pitch, C - K * abs(pitch)))
        # 100x pitch actuator presses ~80 Nm into the tendon -> ~0.03 rad soft-constraint penetration (realistic 50 Nm
        # drives: <= 0.003, see the warp check below and test_sim2sim_models)
        assert 0.45 < pitch < 0.53 and abs(roll) <= C - K * abs(pitch) + 0.04, (roll, pitch)
        roll0, pitch0 = _hold_pitch_cpu(m, -1.5, 0.0)
        assert abs(pitch0) < 0.03 and roll0 < -0.48, (roll0, pitch0)
        m_off = copy.deepcopy(m)
        m_off.tendon_limited[:] = 0  # same model, coupling disabled
        roll_u, pitch_u = _hold_pitch_cpu(m_off, -1.5, 0.5)
        print("training MJCF (CPU) uncoupled, pitch held: roll %.3f pitch %.3f" % (roll_u, pitch_u))
        assert roll_u < -0.48 and pitch_u > 0.45, (roll_u, pitch_u)

        env.reset()
        robot = env.scene["robot"]
        root = robot.data.root_link_pose_w.clone()
        root[:, 2] = 1.0
        zero_v = torch.zeros(env.num_envs, 6, device=env.device)
        target = robot.data.default_joint_pos.clone()
        target[:, ROLL] = -1.5
        target[:4, PITCH] = 0.52
        target[4:, PITCH] = 0.0
        for _ in range(600):  # 3 s at 5 ms
            robot.write_root_link_pose_to_sim(root)
            robot.write_root_link_velocity_to_sim(zero_v)
            robot.set_joint_position_target(target)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
        q = robot.data.joint_pos
        roll, pitch = q[:, ROLL].cpu().numpy(), q[:, PITCH].cpu().numpy()
        print("training env (warp): roll", np.round(roll, 3), "pitch", np.round(pitch, 3), "envelope", np.round(C - K * np.abs(pitch), 3))
        assert (pitch[:4] > 0.25).all() and (np.abs(roll[:4]) <= C - K * np.abs(pitch[:4]) + TOL).all(), (roll, pitch)
        assert (np.abs(pitch[4:]) < 0.05).all() and (roll[4:] < -0.48).all(), (roll, pitch)

        env.reset()
        g = torch.Generator(device=env.device).manual_seed(0)
        for _ in range(100):
            a = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device, generator=g)
            env.step(a)
            assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
    finally:
        env.close()
