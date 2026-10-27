"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).

Per clip we also record the pelvis tilt error (angle between the robot's and the reference's pelvis gravity
direction, i.e. the quantity the deploy runtime aborts on at 20 deg), mean and max over the clip.
``lock_waist`` evaluates on the waist-locked robot (``supermimic.envs.lock_waist``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    tilt_mean_deg: float = float("nan")  # pelvis gravity direction vs reference (deploy abort quantity)
    tilt_max_deg: float = float("nan")


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def _tilt_deg(cmd) -> torch.Tensor:
    """(N,) angle [deg] between robot and reference pelvis projected gravity."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    g = cmd.robot.data.gravity_vec_w
    a = quat_apply_inverse(cmd.robot_root_quat_w, g)
    b = quat_apply_inverse(cmd.root_quat_w, g)
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)
    return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0)))


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False,
                  lock_waist: bool = False, lock_waist_refs: str = "feasible", locked_action_weight: float = 0.0,
                  waist_coupling: tuple[float, float] | None = None, waist_roll_range: float | None = None) -> list[ClipResult]:
    """locked_action_weight defaults to 0 so that the benchmark reward is the same as the unlocked one (the term is
    a training-only regulariser); pass the training value to include it. waist_coupling = (k, c): evaluate on the robot
    with the real-G1 waist roll/pitch coupling (supermimic.deploy.waist_coupling); None = uncoupled (default).
    waist_roll_range: waist_roll_joint range +-r (supermimic.deploy.waist_roll); None = MJCF +-0.52 (default)."""
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        lock_waist=lock_waist,
        lock_waist_refs=lock_waist_refs,
        locked_action_weight=locked_action_weight,
        **({} if waist_coupling is None else {"waist_coupling": True, "waist_coupling_k": waist_coupling[0], "waist_coupling_c": waist_coupling[1]}),
        **({} if waist_roll_range is None else {"waist_roll_range": waist_roll_range}),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        tilt_sum = torch.zeros(num_envs, device=device)
        tilt_max = torch.zeros(num_envs, device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            # Tilt of the state reached by this step vs the reference it was rewarded against. For envs that just
            # reset the sim already holds the next clip's start, so their value is taken before the step (see below).
            tilt = _tilt_deg(cmd)
            steps += 1
            done = (term | tout) & active
            live = ~(term | tout)
            tilt_sum += torch.where(live, tilt, torch.zeros_like(tilt))
            tilt_max = torch.where(live, torch.maximum(tilt_max, tilt), tilt_max)
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            tilt_mean_deg=float(tilt_sum[e]) / max(st - 1, 1),
                            tilt_max_deg=float(tilt_max[e]),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                tilt_sum[ids] = 0.0
                tilt_max[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    tm = [r.tilt_max_deg for r in results if r.tilt_max_deg == r.tilt_max_deg]
    if tm:
        tm_sorted = sorted(tm)
        out["tilt_mean_deg"] = sum(r.tilt_mean_deg * r.steps for r in results) / max(tot_steps, 1)
        out["tilt_max_deg_p50"] = tm_sorted[len(tm_sorted) // 2]
        out["tilt_max_deg_p95"] = tm_sorted[min(len(tm_sorted) - 1, int(0.95 * len(tm_sorted)))]
        out["tilt_max_deg_max"] = tm_sorted[-1]
        out["clips_tilt_over_20deg"] = sum(t > 20.0 for t in tm)
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
