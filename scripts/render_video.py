"""Render benchmark rollouts to mp4 (headless, EGL): solid robot = policy, translucent blue = reference.

Same deterministic protocol as scripts/eval_benchmark.py (no noise, no DR, every clip from its first frame);
one env per clip, recording stops when the clip ends, the termination fires, or after --seconds.
Writes <out>/<tag>__<clip>.mp4 and a sidecar .json with per-frame normalised reward and root errors.

  MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg python scripts/render_video.py \
      --ckpt logs/<run>/ckpt_XXXXXXX.pt --clips walk1_subject1 dance1_subject1 --seconds 30 --tag demo
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import tyro

from supermimic.data.motion_convert import g1_xml_path
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint

GHOST_RGBA = (0.35, 0.6, 1.0, 0.35)


@torch.no_grad()
def rollout(ckpt: Path, files: list[str], max_steps: int, device: str, lock_waist: bool | None = None, lock_waist_refs: str | None = None) -> list[dict]:
    """Play each clip once; returns per clip the robot / reference qpos trajectories and per-step rewards.

    lock_waist / lock_waist_refs: None = as the checkpoint was trained. Locked joints are rendered at 0 (the 29-DoF
    scene with waist roll/pitch at 0 is the locked robot's geometry)."""
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    lock = cfg["env"].get("lock_waist", False) if lock_waist is None else lock_waist
    refs = lock_waist_refs or cfg["env"].get("lock_waist_refs", "feasible")
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=len(files),
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        lock_waist=lock,
        lock_waist_refs=refs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n = lib.num_clips
        ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[ids]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[ids]
        origins = env.scene.env_origins
        active = torch.ones(n, dtype=torch.bool, device=device)
        fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
        robot_q, ref_q, rews = [], [], []

        def snapshot():
            robot_q.append(torch.cat([cmd.robot_root_pos_w - origins, cmd.robot_root_quat_w, cmd.robot_joint_pos], -1).cpu())
            ref_q.append(torch.cat([cmd.root_pos_w - origins, cmd.root_quat_w, cmd.joint_pos], -1).cpu())

        snapshot()
        length = torch.zeros(n, dtype=torch.long, device=device)
        for t in range(max_steps):
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            done = (term | tout) & active
            fail_step[term & active] = t + 1
            length[active & ~done] = t + 1
            rews.append(rew.cpu())
            # After an auto-reset the sim holds the next episode's state; those frames are cut via `length`.
            snapshot()
            active &= ~done
            if not active.any():
                break
        robot_q, ref_q, rews = torch.stack(robot_q, 1), torch.stack(ref_q, 1), torch.stack(rews, 1)
        out = []
        for i, clip in enumerate(lib.clips):
            L = int(length[i]) + 1
            out.append({
                "name": clip.name,
                "frames": clip.length,
                "robot_q": robot_q[i, :L].numpy().astype(np.float64),
                "ref_q": ref_q[i, :L].numpy().astype(np.float64),
                "reward": (rews[i, : L - 1] / 0.1).numpy(),
                "fail_step": int(fail_step[i]),
            })
        return out
    finally:
        env.close()


def build_scene(width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.92, 0.93, 0.94], rgb2=[0.80, 0.82, 0.85], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.95, 0.97], rgb2=[0.78, 0.82, 0.87], width=512, height=3072)
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [1, 1]
    mat.texuniform = True
    mat.reflectance = 0.0
    wb = spec.worldbody
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", contype=0, conaffinity=0)
    wb.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True, diffuse=[0.7, 0.7, 0.7])
    wb.add_light(pos=[3, -3, 4], dir=[-1, 1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False, diffuse=[0.3, 0.3, 0.3])
    return spec.compile()


def render(clip: dict, model: mujoco.MjModel, path: Path, width: int, height: int, fps: int, sim_hz: int = 50, camera: str = "fixed") -> None:
    """camera: 'fixed' (azimuth 150 deg) or 'front' (in front of the reference, facing it: azimuth = circular mean of the
    reference heading + 180 deg)."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    d_robot, d_ref = mujoco.MjData(model), mujoco.MjData(model)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 150.0, -12.0
    if camera == "front":
        yaw = _yaw(clip["ref_q"][:, 3:7])
        cam.azimuth = float(np.degrees(np.angle(np.exp(1j * yaw).mean()))) + 180.0
    stride = max(sim_hz // fps, 1)
    rq, gq = clip["robot_q"], clip["ref_q"]
    look = None
    with imageio.get_writer(path, fps=sim_hz / stride, codec="libx264", quality=7, macro_block_size=8,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for t in range(0, len(rq), stride):
            d_robot.qpos[:], d_ref.qpos[:] = rq[t], gq[t]
            mujoco.mj_kinematics(model, d_robot)
            mujoco.mj_kinematics(model, d_ref)
            # Follow the midpoint of robot and reference (smoothed), backing off as they drift apart.
            target = 0.5 * (rq[t, :3] + gq[t, :3])
            target[2] = 0.8
            look = target if look is None else 0.75 * look + 0.25 * target
            sep = float(np.linalg.norm(rq[t, :2] - gq[t, :2]))
            cam.lookat[:] = look
            cam.distance = min(2.6 + 0.6 * sep, 9.0)
            renderer.update_scene(d_robot, camera=cam, scene_option=opt)
            scn = renderer.scene
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(model, d_ref, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for g in range(n0, scn.ngeom):
                scn.geoms[g].rgba[:] = GHOST_RGBA
                scn.geoms[g].matid = -1
            w.append_data(renderer.render())
    renderer.close()


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main(
    ckpt: Path,
    clips: list[str],
    npz_dir: Path = Path("data/npz/lafan1"),
    out: Path = Path("logs/videos"),
    tag: str = "run",
    seconds: float = 30.0,
    width: int = 800,
    height: int = 450,
    fps: int = 25,
    device: str = "cuda:0",
    lock_waist: bool | None = None,
    lock_waist_refs: str | None = None,
    camera: str = "fixed",
) -> None:
    """camera: 'fixed' (azimuth 150 deg) or 'front' (G1 faces the camera, see render())."""
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    assert len(files) == len(clips), f"found {len(files)} of {len(clips)} clips"
    results = rollout(ckpt, files, int(seconds * 50), device, lock_waist, lock_waist_refs)
    model = build_scene(width, height)
    out.mkdir(parents=True, exist_ok=True)
    stride = max(50 // fps, 1)
    for c in results:
        stem = f"{tag}__{c['name']}"
        render(c, model, out / f"{stem}.mp4", width, height, fps, camera=camera)
        rq, gq = c["robot_q"], c["ref_q"]
        xy = np.linalg.norm(rq[:, :2] - gq[:, :2], axis=-1)
        yaw = np.degrees(np.abs(np.angle(np.exp(1j * (_yaw(rq[:, 3:7]) - _yaw(gq[:, 3:7]))))))
        rew = np.concatenate([[c["reward"][0] if len(c["reward"]) else 0.0], c["reward"]])
        meta = {
            "tag": tag, "ckpt": str(ckpt), "clip": c["name"], "clip_frames": c["frames"], "steps": len(rq) - 1,
            "fail_step": c["fail_step"], "fps": 50 / stride,
            "mean_reward": float(np.mean(c["reward"])) if len(c["reward"]) else 0.0,
            "reward": np.round(rew[::stride], 3).tolist(),
            "xy_err": np.round(xy[::stride], 3).tolist(),
            "yaw_err_deg": np.round(yaw[::stride], 1).tolist(),
        }
        (out / f"{stem}.json").write_text(json.dumps(meta))
        print(f"{stem}: {len(rq) - 1} steps, mean reward {meta['mean_reward']:.3f}, fail {c['fail_step']}, final xy err {xy[-1]:.2f} m")


if __name__ == "__main__":
    tyro.cli(main)
