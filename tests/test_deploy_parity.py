"""Deploy parity: the numpy DeployCore (+ ONNX policy) must reproduce the training env's policy inputs/outputs.

A deterministic SuperMimic env (exactly the supermimic.eval.benchmark.run_benchmark setup: no noise, no DR,
start sampling, forced start frame, fp32 npz library) runs the torch policy for ~200 steps; the recorded
robot state (joint_pos, joint_vel, pelvis gyro in body frame, pelvis quat) is then replayed through
DeployCore, and its observation / goal / action / joint targets are compared with the env's.

Needs CUDA, the BONES-SEED npz clips and deploy_artifacts/ (python -m supermimic.deploy.export all).
Also runnable as a script:  PYTHONPATH= .venv/bin/python tests/test_deploy_parity.py --ckpt ckpt_0116000
(onnxruntime from outside the venv: SM_ORT_OVERLAY=<dir>).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "deploy_artifacts"
NPZ = ROOT / "data/npz/bones_seed"
# Three long clips + one shorter than STEPS (exercises the goal clamp and the clip end).
CLIPS = ("neutral_idle_loop_001__A081", "walk_sideway_090_stop_003__A037", "wave_R_001__A431", "itching_left_arm_R_003__A121")
STEPS = 200
CKPT = os.environ.get("SM_PARITY_CKPT", "ckpt_0116000")
TOL = {"obs": 1e-4, "goal": 1e-4, "action": 1e-3, "q_target_contract": 1e-5}


def _have_ort() -> bool:
    try:
        from supermimic.deploy.export import import_ort

        import_ort()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not all((NPZ / f"{c}.npz").exists() for c in CLIPS), reason="BONES-SEED npz missing"),
    pytest.mark.skipif(not (ART / CKPT / "policy.onnx").exists(), reason="deploy_artifacts not exported"),
    pytest.mark.skipif(not (ART / "ckpts" / f"{CKPT}.pt").exists(), reason="deploy_artifacts/ckpts checkpoint missing"),
    pytest.mark.skipif(not all((ART / "clips" / f"{c}.npz").exists() for c in CLIPS), reason="deploy_artifacts/clips missing"),
    pytest.mark.skipif(not _have_ort(), reason="onnxruntime missing"),
]


def record_env(ckpt_tag: str, clips: tuple[str, ...], steps: int, device: str) -> dict:
    """Roll the deterministic benchmark env with the torch policy and record everything DeployCore needs."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    ck = ART / "ckpts" / f"{ckpt_tag}.pt"
    policy, cfg = load_policy_from_checkpoint(ck, device)
    args = SuperMimicEnvCfgArgs(  # == run_benchmark
        npz_files=[str(NPZ / f"{c}.npz") for c in clips], num_envs=len(clips), future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        assert [c.name for c in lib.clips] == list(clips)
        assert not lib.compact
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        n = len(clips)
        ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[ids]
        obs, _ = env.reset()
        rec = {k: [] for k in ("obs", "goal", "q", "q_biased", "dq", "gyro", "quat", "proj_grav", "frame", "action", "action_tf32", "q_target", "valid", "done")}
        alive = torch.ones(n, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(steps):
                rec["obs"].append(obs["policy"].clone())
                rec["goal"].append(obs["goal"].clone())
                rec["q"].append(robot.data.joint_pos.clone())
                rec["q_biased"].append(robot.data.joint_pos_biased.clone())
                rec["dq"].append(robot.data.joint_vel.clone())
                rec["gyro"].append(robot.data.root_link_ang_vel_b.clone())
                rec["quat"].append(robot.data.root_link_quat_w.clone())
                rec["proj_grav"].append(robot.data.projected_gravity_b.clone())
                rec["frame"].append(cmd.frame - lib.clip_start[ids])  # first goal row, clip-local
                rec["valid"].append(alive.clone())
                # make_env (mjlab configure_torch_backends) turns on TF32 matmuls, which is what training and
                # run_benchmark use. Drive the env with IEEE fp32 (the reference ONNX must match) and keep the
                # TF32 action on the same inputs for information.
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                rec["action_tf32"].append(policy(obs["policy"], obs["goal"]).clone())
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                act = policy(obs["policy"], obs["goal"])
                obs, _, term_, tout, _ = env.step(act)
                rec["action"].append(act.clone())
                rec["q_target"].append(term._processed_actions.clone())
                rec["done"].append((term_ | tout).clone())
                alive &= ~(term_ | tout)
                if not alive.any():
                    break
        out = {k: torch.stack(v, 1).double().cpu().numpy() if v[0].is_floating_point() else torch.stack(v, 1).cpu().numpy() for k, v in rec.items()}
        out["encoder_bias_max"] = float(robot.data.encoder_bias.abs().max())
        out["clip_len"] = np.array([c.length for c in lib.clips])
        return out
    finally:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        env.close()


def _term_slices(blocks: tuple[tuple[str, int], ...], reps: int):
    o = 0
    for name, d in blocks:
        yield name, slice(o, o + reps * d)
        o += reps * d


def replay_core(ckpt_tag: str, clips: tuple[str, ...], rec: dict) -> dict:
    from supermimic.deploy.core import FUTURE, HISTORY, OBS_TERMS, Clip, DeployConfig, DeployCore, OnnxPolicy, SafetyCfg

    dc = DeployConfig.load(ART / ckpt_tag / "deploy_config.json")
    onnx = OnnxPolicy(ART / ckpt_tag / "policy.onnx")
    nosafe = SafetyCfg(clamp_soft_limits=False, clamp_torque=False, max_tilt_err=np.inf, max_vel_frac=np.inf, max_abs_action=np.inf)
    goal_blocks = (("q", 29), ("dq", 29), ("grav", 3), ("ang_b", 3), ("lin_h", 3), ("height", 1))
    res: dict = {"clips": {}}
    agg: dict[str, float] = {}

    def upd(k, v):
        agg[k] = max(agg.get(k, 0.0), float(v))

    for e, name in enumerate(clips):
        T = int(rec["valid"][e].sum())
        seen = []

        def pol(o, g):
            seen.append((o.copy(), g.copy()))
            return onnx(o, g)

        clip = Clip.load(ART / "clips" / f"{name}.npz")
        core = DeployCore(dc, pol, clip, nosafe)
        core.reset(0)
        r_act, r_qt, r_qt_env_act, r_frame, r_done, r_onnx_env = [], [], [], [], [], []
        for t in range(T):
            r_frame.append(min(core.frame + 1, clip.num_frames - 1))  # first goal row core will use (clamped)
            sr = core.step(rec["q"][e, t], rec["dq"][e, t], rec["gyro"][e, t], rec["quat"][e, t])
            r_act.append(sr.action)
            r_qt.append(sr.q_target)
            r_done.append(sr.done)
            r_qt_env_act.append(core.targets(rec["action"][e, t], rec["q"][e, t], rec["dq"][e, t])[0])
            r_onnx_env.append(onnx(rec["obs"][e, t], rec["goal"][e, t]))
        obs_c = np.stack([s[0] for s in seen])
        goal_c = np.stack([s[1] for s in seen])
        obs_e, goal_e = rec["obs"][e, :T], rec["goal"][e, :T]
        d = {}
        for tn, sl in _term_slices(OBS_TERMS, HISTORY):
            d[f"obs/{tn}"] = float(np.abs(obs_c[:, sl] - obs_e[:, sl]).max())
        d["obs"] = float(np.abs(obs_c - obs_e).max())
        gd = np.abs(goal_c - goal_e).reshape(T, FUTURE, 68)
        o = 0
        for gn, w in goal_blocks:
            d[f"goal/{gn}"] = float(gd[:, :, o : o + w].max())
            o += w
        d["goal"] = float(gd.max())
        d["action(core onnx vs env torch)"] = float(np.abs(np.stack(r_act) - rec["action"][e, :T]).max())
        d["action(onnx on env inputs vs torch)"] = float(np.abs(np.stack(r_onnx_env) - rec["action"][e, :T]).max())
        d["action(core onnx vs env torch TF32) [info]"] = float(np.abs(np.stack(r_act) - rec["action_tf32"][e, :T]).max())
        d["action(torch TF32 vs IEEE, env inputs) [info]"] = float(np.abs(rec["action_tf32"][e, :T] - rec["action"][e, :T]).max())
        d["q_target(core vs env, own actions)"] = float(np.abs(np.stack(r_qt) - rec["q_target"][e, :T]).max())
        d["q_target_contract(core.targets(env action) vs env)"] = float(np.abs(np.stack(r_qt_env_act) - rec["q_target"][e, :T]).max())
        d["encoder_bias(q_biased - q)"] = float(np.abs(rec["q_biased"][e, :T] - rec["q"][e, :T]).max())
        from supermimic.deploy.core import projected_gravity

        pg = np.stack([projected_gravity(q) for q in rec["quat"][e, :T]])
        d["proj_grav(core formula vs env)"] = float(np.abs(pg - rec["proj_grav"][e, :T]).max())
        d["goal_row_offset_mismatches"] = int(np.sum(np.asarray(r_frame) != rec["frame"][e, :T]))
        env_done_step = int(np.argmax(rec["done"][e, :T])) if rec["done"][e, :T].any() else -1
        core_done_step = int(np.argmax(r_done)) if any(r_done) else -1
        res["clips"][name] = {"steps": T, "clip_frames": int(rec["clip_len"][e]), "env_done_step": env_done_step, "core_done_step": core_done_step,
                              "max_abs_action": float(np.abs(rec["action"][e, :T]).max()), **d}
        for k, v in d.items():
            upd(k, v)
    res["max"] = agg
    res["encoder_bias_max"] = rec["encoder_bias_max"]
    return res


def run_parity(ckpt_tag: str = CKPT, clips: tuple[str, ...] = CLIPS, steps: int = STEPS, device: str = "cuda:0") -> dict:
    rec = record_env(ckpt_tag, clips, steps, device)
    res = replay_core(ckpt_tag, clips, rec)
    res["ckpt"] = ckpt_tag
    return res


def check(res: dict) -> list[str]:
    m = res["max"]
    errs = []
    for key, tol in (("obs", TOL["obs"]), ("goal", TOL["goal"]), ("action(core onnx vs env torch)", TOL["action"]),
                     ("q_target_contract(core.targets(env action) vs env)", TOL["q_target_contract"]), ("encoder_bias(q_biased - q)", 1e-9)):
        if not m[key] < tol:
            errs.append(f"{key} = {m[key]:.3e} >= {tol}")
    if m["goal_row_offset_mismatches"] != 0:
        errs.append(f"goal frame offset mismatches: {m['goal_row_offset_mismatches']}")
    return errs


def test_deploy_parity():
    res = run_parity()
    errs = check(res)
    assert not errs, errs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = run_parity(a.ckpt, CLIPS, a.steps, a.device)
    for name, d in res["clips"].items():
        print(f"\n{name}: steps {d['steps']}/{d['clip_frames']} env_done@{d['env_done_step']} core_done@{d['core_done_step']} max|a| {d['max_abs_action']:.2f}")
        for k, v in d.items():
            if k not in ("steps", "clip_frames", "env_done_step", "core_done_step", "max_abs_action"):
                print(f"    {k:52s} {v:.3e}" if isinstance(v, float) else f"    {k:52s} {v}")
    print("\nMAX over clips:")
    for k, v in res["max"].items():
        print(f"    {k:52s} {v:.3e}" if isinstance(v, float) else f"    {k:52s} {v}")
    errs = check(res)
    out = Path(a.out) if a.out else ART / "reports" / f"parity_{a.ckpt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "errors": errs}, indent=1))
    print("\nwrote", out)
    print("PASS" if not errs else "FAIL:\n  " + "\n  ".join(errs))
    sys.exit(1 if errs else 0)
