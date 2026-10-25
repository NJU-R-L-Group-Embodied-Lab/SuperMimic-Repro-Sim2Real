"""Compact (FK on the fly, fp16 store) library must reproduce the full library's goals and body states."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")


def test_compact_matches_full():
    torch.backends.cuda.matmul.allow_tf32 = False
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    files = discover_npz(NPZ, names=["walk1_subject1", "jumps1_subject1"])
    q0 = torch.zeros(29)
    full = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=False)
    comp32 = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=True, store_dtype=torch.float32)
    comp16 = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=True, store_dtype=torch.float16)
    f = torch.randint(0, full.num_frames, (512,), device="cuda")
    bf, b32, b16 = full.bodies(f), comp32.bodies(f), comp16.bodies(f)
    np.testing.assert_allclose(b32.pos_w.cpu().numpy(), bf.pos_w.cpu().numpy(), atol=2e-5)
    np.testing.assert_allclose(b32.lin_vel_w.cpu().numpy(), bf.lin_vel_w.cpu().numpy(), atol=2e-4)
    np.testing.assert_allclose(b32.ang_vel_w.cpu().numpy(), bf.ang_vel_w.cpu().numpy(), atol=2e-4)
    sign = torch.sign((b32.quat_w * bf.quat_w).sum(-1, keepdim=True))
    np.testing.assert_allclose((b32.quat_w * sign).cpu().numpy(), bf.quat_w.cpu().numpy(), atol=2e-5)
    # fp16 store: positions to ~1 mm, velocities to ~1e-2 (half precision of the reference state).
    np.testing.assert_allclose(b16.pos_w.cpu().numpy(), bf.pos_w.cpu().numpy(), atol=5e-3)
    np.testing.assert_allclose(b16.lin_vel_w.cpu().numpy(), bf.lin_vel_w.cpu().numpy(), atol=5e-2)
    g_full, g_32 = full.goal_window(f, 10), comp32.goal_window(f, 10)
    np.testing.assert_allclose(g_32.cpu().numpy(), g_full.cpu().numpy(), atol=1e-5)
    assert comp16.summary().startswith("MotionLibrary(compact)")
