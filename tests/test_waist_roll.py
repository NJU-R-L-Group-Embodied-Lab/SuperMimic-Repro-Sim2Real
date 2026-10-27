"""Real-G1 waist roll model (supermimic.deploy.waist_roll): opt-in wider waist_roll_joint range (training
--env.waist-roll-range, sim2sim --waist-roll-range / SM_WAIST_ROLL) and opt-in waist-roll damping / friction (training DR
--env.dr-waist-roll-damping / -friction, sim2sim --waist-roll-damping / -friction on the Unitree gate models)."""

import copy
import importlib.util
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.deploy.waist_coupling import ENV_VAR as WC_ENV, WAIST_COUPLING_C as C, WAIST_COUPLING_K as K
from supermimic.deploy.waist_roll import (ENV_VAR, NOMINAL_RANGE, apply_actuator_model, env_string, resolve_range, roll_dof,
                                          set_waist_roll_range, waist_roll_from_env)

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
CLIPS = ["walk1_subject1", "dance1_subject1"]
UNITREE_XML = Path(os.environ.get("G1_UNITREE_XML", Path.home() / "workspace/g1_deploy/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"))
R = 0.72


def _roll_range(m):
    j = [i for i in range(m.njnt) if m.joint(i).name.split("/")[-1] == "waist_roll_joint"][0]
    return m.jnt_range[j].tolist()


def test_off_by_default(monkeypatch):
    from supermimic.config import EnvCfg
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import robot_cfg

    e, a = EnvCfg(), SuperMimicEnvCfgArgs(npz_files=[])
    assert e.waist_roll_range is None and e.dr_waist_roll_damping is None and e.dr_waist_roll_friction is None
    assert e.waist_roll_model_soft_limits is False
    assert a.waist_roll_range is None and a.dr_waist_roll_damping is None and a.dr_waist_roll_friction is None
    np.testing.assert_allclose(_roll_range(robot_cfg().spec_fn().compile()), [-NOMINAL_RANGE, NOMINAL_RANGE])
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert waist_roll_from_env() is None and resolve_range({}) is None and resolve_range(None) is None


def test_training_spec_range_only_roll_and_keeps_coupling():
    from supermimic.envs.tracking_env_cfg import robot_cfg

    base = robot_cfg().spec_fn().compile()
    m = robot_cfg(waist_roll_range=R).spec_fn().compile()
    np.testing.assert_allclose(_roll_range(m), [-R, R])
    j = [i for i in range(m.njnt) if m.joint(i).name.endswith("waist_roll_joint")][0]
    other = np.ones(m.njnt, bool)
    other[j] = False
    np.testing.assert_array_equal(m.jnt_range[other], base.jnt_range[other])
    assert m.ntendon == 0
    mc = robot_cfg(waist_coupling=(K, C), waist_roll_range=R).spec_fn().compile()  # both: tendons + wider box
    assert mc.ntendon == 2 and mc.tendon_limited.all()
    np.testing.assert_allclose(mc.tendon_range, [[-C, C], [-C, C]])
    np.testing.assert_allclose(_roll_range(mc), [-R, R])
    with pytest.raises(ValueError):
        robot_cfg(lock_waist=True, waist_roll_range=R)


def test_env_cfg_validation():
    from supermimic.config import EnvCfg

    for kw in ({"waist_roll_range": 0.0}, {"waist_roll_range": -0.7}, {"dr_waist_roll_damping": (2.0, 1.0)},
               {"dr_waist_roll_friction": (-1.0, 1.0)}, {"lock_waist": True, "waist_roll_range": R}):
        with pytest.raises(ValueError):
            EnvCfg(**kw)
    EnvCfg(waist_roll_range=R, dr_waist_roll_damping=(0.0, 3.6), dr_waist_roll_friction=(0.0, 2.0))


def test_env_var_parsing_and_resolve(monkeypatch):
    for bad in ("0.72", "range=abc", "foo=1", "range=-0.7", "damping=-1", "range=0.7,friction=-2"):
        monkeypatch.setenv(ENV_VAR, bad)
        with pytest.raises(ValueError, match=ENV_VAR):
            waist_roll_from_env()
    monkeypatch.setenv(ENV_VAR, "off")
    assert waist_roll_from_env() is None
    monkeypatch.setenv(ENV_VAR, env_string(R, 1.8, 2.0))
    assert waist_roll_from_env() == {"range": R, "damping": 1.8, "friction": 2.0}
    monkeypatch.setenv(ENV_VAR, env_string(None, 1.8))
    assert waist_roll_from_env() == {"range": None, "damping": 1.8, "friction": None}
    assert resolve_range({"waist_roll_range": 0.65}) == 0.65  # no range in the env var: as trained
    monkeypatch.setenv(ENV_VAR, env_string(R))
    assert resolve_range({"waist_roll_range": 0.65}) == R  # the env var wins
    monkeypatch.delenv(ENV_VAR)
    assert resolve_range({"waist_roll_range": None}) is None and env_string(None) == ""


def test_helpers_need_the_joint():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="b").add_joint(name="other")
    with pytest.raises(ValueError):
        set_waist_roll_range(spec, R)
    with pytest.raises(ValueError):
        roll_dof(spec.compile())


def _sim2sim():
    spec = importlib.util.spec_from_file_location("sim2sim", ROOT / "scripts/deploy/sim2sim.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # robot-side deps (onnxruntime) missing in this venv
        pytest.skip(f"sim2sim import: {e}")
    return mod


def _roll_pd(m, roll_t, pitch_t=0.0, n_steps=2500, kp=28.5, kd=1.81, effort=50.0, pitch_kp=3000.0):
    """Pinned pelvis, Unitree torque motors, waist roll torque-PD (training gains), pitch held stiff (test only);
    returns the roll trajectory (1 kHz)."""
    m = copy.deepcopy(m)
    m.opt.timestep = 0.001
    m.actuator_ctrlrange[:] = m.jnt_actfrcrange[:] = [-500, 500]
    d = mujoco.MjData(m)
    jr, jp = (m.joint(n).qposadr[0] for n in ("waist_roll_joint", "waist_pitch_joint"))
    vr, vp = (m.joint(n).dofadr[0] for n in ("waist_roll_joint", "waist_pitch_joint"))
    ar, ap = (m.actuator(n).id for n in ("waist_roll", "waist_pitch"))
    d.qpos[2] = 1.0
    q7 = d.qpos[:7].copy()
    out = []
    for _ in range(n_steps):
        d.qpos[:7], d.qvel[:6] = q7, 0.0
        d.ctrl[:] = 0.0
        d.ctrl[ar] = np.clip(kp * (roll_t - d.qpos[jr]) - kd * d.qvel[vr], -effort, effort)
        d.ctrl[ap] = np.clip(pitch_kp * (pitch_t - d.qpos[jp]) - 50.0 * d.qvel[vp], -500, 500)
        mujoco.mj_step(m, d)
        out.append(float(d.qpos[jr]))
    return np.array(out)


@pytest.mark.skipif(not UNITREE_XML.exists(), reason="Unitree MJCF not found")
def test_sim2sim_models(monkeypatch):
    s2s = _sim2sim()
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(WC_ENV, raising=False)
    m0 = s2s.load_model(s2s.UNITREE_XML, actuator_model=True)
    np.testing.assert_allclose(_roll_range(m0), [-NOMINAL_RANGE, NOMINAL_RANGE])  # default off
    a = roll_dof(m0)
    damp0, fric0 = float(m0.dof_damping[a]), float(m0.dof_frictionloss[a])
    monkeypatch.setenv(ENV_VAR, env_string(R, 1.8, 2.0))
    m = s2s.load_model(s2s.UNITREE_XML, actuator_model=True)
    np.testing.assert_allclose(_roll_range(m), [-R, R])
    assert np.isclose(m.dof_damping[a], damp0 + 1.8) and np.isclose(m.dof_frictionloss[a], 2.0)
    other = np.ones(m.nv, bool)
    other[a] = False
    np.testing.assert_array_equal(m.dof_damping[other], m0.dof_damping[other])
    mg = s2s.load_model(s2s.UNITREE_XML)  # G1-style call: range only
    np.testing.assert_allclose(_roll_range(mg), [-R, R])
    assert np.isclose(mg.dof_damping[a], damp0) and np.isclose(mg.dof_frictionloss[a], fric0)
    if s2s.TRAIN_XML.exists():  # prefixed names
        np.testing.assert_allclose(_roll_range(s2s.load_model(s2s.TRAIN_XML)), [-R, R])
    with pytest.raises(SystemExit):
        s2s.load_model(s2s.UNITREE_XML, lock=True)
    # Physics: roll target -1.0 at pitch 0 reaches ~-0.72 (nominal stops at -0.52).
    monkeypatch.setenv(ENV_VAR, env_string(R))
    mr = s2s.load_model(s2s.UNITREE_XML, actuator_model=True)
    roll_w, roll_n = _roll_pd(mr, -1.0)[-1], _roll_pd(m0, -1.0)[-1]
    print(f"roll at target -1.0: widened {roll_w:.3f} nominal {roll_n:.3f}")
    assert -R - 0.02 < roll_w < -0.68 and -0.54 < roll_n < -0.50
    # With the coupling the envelope still binds: pitch 0.3 -> |roll| <= 0.95 - 1.4 * 0.3 = 0.53.
    monkeypatch.setenv(WC_ENV, f"{K},{C}")
    mc = s2s.load_model(s2s.UNITREE_XML, actuator_model=True)
    assert mc.ntendon == 2
    roll_c = _roll_pd(mc, -1.0, pitch_t=0.3)[-1]
    assert abs(roll_c) <= C - K * 0.3 + 0.02, roll_c
    monkeypatch.delenv(WC_ENV)
    # Extra damping slows the waist-roll step response (0 -> 0.4 target): t63 longer.
    monkeypatch.setenv(ENV_VAR, env_string(R, 1.8))
    md = s2s.load_model(s2s.UNITREE_XML, actuator_model=True)
    t63 = [int(np.argmax(_roll_pd(x, 0.4, n_steps=1500) >= 0.63 * 0.4)) for x in (mr, md)]
    print("t63 [ms] nominal / +1.8 damping:", t63)
    assert t63[1] > t63[0] * 1.1


def test_apply_actuator_model_validation():
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="b")
    b.add_joint(name="waist_roll_joint")
    b.add_geom(size=[0.1, 0, 0])
    m = spec.compile()
    apply_actuator_model(m, 1.0, 0.5)
    assert m.dof_damping[0] == 1.0 and m.dof_frictionloss[0] == 0.5
    with pytest.raises(ValueError):
        apply_actuator_model(m, -1.0)


def test_sim2sim_guards(monkeypatch, tmp_path):
    """G1 refuses a reference made with another waist-roll range; --lock-waist / lw are refused with the waist-roll model."""
    s2s = _sim2sim()
    monkeypatch.delenv(WC_ENV, raising=False)
    monkeypatch.setattr(s2s, "REPORTS", tmp_path)
    (tmp_path / "sim2sim").mkdir()
    np.savez(tmp_path / "sim2sim" / "ref_t_x_no-tf32.npz", clips=np.array(["c"]), fail_step=np.array([-1]))
    monkeypatch.setenv(ENV_VAR, env_string(R))
    with pytest.raises(SystemExit, match="waist roll range None"):
        s2s.g1_gate("x", {}, ("c",), ("c",), "t_")
    np.savez(tmp_path / "sim2sim" / "ref_t_y_no-tf32.npz", clips=np.array(["c"]), fail_step=np.array([-1]), waist_roll_range=np.array(R))
    monkeypatch.delenv(ENV_VAR)
    with pytest.raises(SystemExit, match="this run None"):
        s2s.g1_gate("y", {}, ("c",), ("c",), "t_")
    for argv in (["--cmd", "lw"], ["--cmd", "g2", "--lock-waist"]):
        monkeypatch.setattr("sys.argv", ["sim2sim.py", *argv, "--waist-roll-range", str(R)])
        with pytest.raises(SystemExit, match="waist-roll"):
            s2s.main()
        os.environ.pop(ENV_VAR, None)  # main() sets it directly


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_env():
    """Training env with --env.waist-roll-range 0.72 + coupling + waist-roll DR: GPU model range 0.72 (other joints unchanged),
    soft limits nominal (or widened with waist_roll_model_soft_limits), per-env damping / friction drawn in range on reset
    and only on the waist roll dof; CPU MuJoCo on the env's mj_model, pelvis pinned, roll target -1.0 reaches ~-0.72."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = [str(f) for f in discover_npz(NPZ, names=CLIPS)]
    kw = dict(npz_files=files, num_envs=8, record_transitions=False, seed=0, waist_coupling=True, waist_roll_range=R)
    env = make_env(SuperMimicEnvCfgArgs(**kw, dr_waist_roll_damping=(0.5, 3.6), dr_waist_roll_friction=(0.2, 2.0)), device="cuda:0")
    try:
        robot = env.scene["robot"]
        m = env.sim.mj_model
        jid = m.joint("robot/waist_roll_joint").id
        dof = int(m.jnt_dofadr[jid])
        np.testing.assert_allclose(m.jnt_range[jid], [-R, R], atol=1e-6)
        gpu_rng = env.sim.model.jnt_range
        gpu_rng = gpu_rng[0] if gpu_rng.ndim == 3 else gpu_rng
        np.testing.assert_allclose(gpu_rng[jid].cpu().numpy(), [-R, R], atol=1e-6)
        names = [n.split("/")[-1] for n in robot.joint_names]
        j = names.index("waist_roll_joint")
        soft = robot.data.soft_joint_pos_limits[:, j].cpu().numpy()
        np.testing.assert_allclose(soft, np.tile([-0.9 * NOMINAL_RANGE, 0.9 * NOMINAL_RANGE], (8, 1)), atol=1e-6)
        env.reset()
        damp = env.sim.model.dof_damping.cpu().numpy()
        fric = env.sim.model.dof_frictionloss.cpu().numpy()
        assert damp.shape[0] == 8 and fric.shape[0] == 8
        assert (damp[:, dof] >= 0.5 - 1e-6).all() and (damp[:, dof] <= 3.6 + 1e-6).all() and damp[:, dof].std() > 0.05
        assert (fric[:, dof] >= 0.2 - 1e-6).all() and (fric[:, dof] <= 2.0 + 1e-6).all() and fric[:, dof].std() > 0.02
        other = np.ones(damp.shape[1], bool)
        other[dof] = False
        np.testing.assert_allclose(damp[:, other], np.tile(m.dof_damping[other], (8, 1)), atol=1e-6)
        np.testing.assert_allclose(fric[:, other], np.tile(m.dof_frictionloss[other], (8, 1)), atol=1e-6)
        # CPU physics on the compiled training model (position actuators, 5 ms)
        mm = copy.deepcopy(m)
        d = mujoco.MjData(mm)
        a_r = mm.actuator("robot/waist_roll_joint").id
        d.qpos[2] = 1.0
        d.qpos[3:7] = [1, 0, 0, 0]
        q7 = d.qpos[:7].copy()
        for _ in range(600):
            d.qpos[:7], d.qvel[:6] = q7, 0.0
            d.ctrl[:] = 0.0
            d.ctrl[a_r] = -1.0
            mujoco.mj_step(mm, d)
        roll = float(d.qpos[mm.jnt_qposadr[jid]])
        print(f"training model roll at target -1.0: {roll:.3f}")
        assert -R - 0.02 < roll < -0.66
        for _ in range(20):  # env.step stays finite with the widened range and DR
            obs, *_ = env.step(torch.randn(8, env.action_manager.total_action_dim, device="cuda:0").clamp(-2, 2))
        assert torch.isfinite(obs["policy"]).all()
    finally:
        env.close()
    env = make_env(SuperMimicEnvCfgArgs(**kw, waist_roll_model_soft_limits=True), device="cuda:0")
    try:
        robot = env.scene["robot"]
        j = [n.split("/")[-1] for n in robot.joint_names].index("waist_roll_joint")
        np.testing.assert_allclose(robot.data.soft_joint_pos_limits[0, j].cpu().numpy(), [-0.9 * R, 0.9 * R], atol=1e-6)
    finally:
        env.close()
