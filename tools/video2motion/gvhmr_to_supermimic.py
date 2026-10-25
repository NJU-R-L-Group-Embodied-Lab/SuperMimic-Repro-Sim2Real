"""GVHMR hmr4d_results.pt -> GMR retarget (unitree_g1) -> G1 motion CSV (the motions/custom_dance format).

Headless version of GMR's scripts/gvhmr_to_robot.py plus:
  * optional IK seeding at the human pelvis (--init_from_human) and per-frame IK restarts (--restart_tol) against
    IK local minima (twisted waist / wound-up arm), with a short cosine cross-fade at every branch switch;
  * joints clamped to the limits of the G1 MJCF (--robot_xml / $G1_XML / mjlab's g1.xml, see g1_model.py);
  * floor alignment measured on that model's foot collision geoms: "constant" puts the --contact_pct percentile of
    the per-frame lowest foot point at z=0 (one offset per clip, keeps jump heights and GVHMR's height drift);
    "drift" subtracts a 1 s rolling lower envelope of the feet (smoothed), which removes the drift and keeps hops
    shorter than ~1 s (recommended for training, used for motions/custom_dance);
  * first-frame root xy moved to the origin; quaternions made hemisphere-continuous;
  * output: <out_dir>/<name>.csv (no header, 30 fps, 36 columns: root x y z, root quat x y z w, 29 joint angles in
    Unitree SDK order) and <name>_gmr.pkl (the same motion as a GMR-style pickle).

Run inside the GMR environment (GMR pinned to the commit in docs/VIDEO_TO_MOTION.md#environment-setup; this script uses GMR internals):
  python tools/video2motion/gvhmr_to_supermimic.py --gvhmr_pred <work>/gvhmr/<clip>/hmr4d_results.pt \
      --name <clip> --floor_mode drift --out_dir <work>/csv_drift
"""

import argparse
import os
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import g1_model

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--gvhmr_pred", required=True, help="GVHMR output hmr4d_results.pt")
parser.add_argument("--name", required=True, help="clip name (output file stem)")
parser.add_argument("--out_dir", default=os.path.join(os.environ.get("V2M_WORK", "work/video2motion"), "csv"))
parser.add_argument(
    "--body_models",
    default=None,
    help="folder that contains smplx/SMPLX_NEUTRAL.npz (default: $SMPLX_BODY_MODELS, else $GMR_ROOT/assets/body_models)",
)
parser.add_argument("--robot_xml", default=None, help="G1 29-DoF MJCF for limits/feet (default: $G1_XML, else mjlab's g1.xml)")
parser.add_argument("--contact_pct", type=float, default=10.0, help="constant mode: percentile of per-frame min foot z taken as floor")
parser.add_argument(
    "--floor_mode",
    choices=("constant", "drift"),
    default="drift",
    help="constant: one offset per clip (as in the video); drift: also remove GVHMR's slow height drift "
    "(1 s rolling lower envelope of the feet, smoothed), which keeps hops shorter than ~1 s",
)
parser.add_argument(
    "--init_from_human",
    action="store_true",
    help="seed the IK root at the human pelvis pose and settle frame 0 before the sweep; without it the IK starts "
    "from the robot's default heading and, for clips that start turned away/bent over, can lock into a twisted "
    "waist + wound-up arm local minimum that warm-starting carries through the whole clip",
)
parser.add_argument(
    "--restart_tol",
    type=float,
    default=0.0,
    help=">0: every frame also re-solves with waist+arm joints reset to zero (30 IK rounds) and keeps that "
    "solution when the GMR task error drops by more than this; gets the IK out of local minima the warm start "
    "falls into (e.g. hands crossed in front of the body). 0.1 is a good value",
)
parser.add_argument("--smooth_sigma", type=float, default=0.0, help=">0: Gaussian smoothing (frames) of the joint angles after IK")
args = parser.parse_args()

# GMR is imported after argparse so that --help works without it
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file  # noqa: E402

body_models = args.body_models or os.environ.get("SMPLX_BODY_MODELS")
if body_models is None:
    if "GMR_ROOT" not in os.environ:
        raise SystemExit("pass --body_models or set SMPLX_BODY_MODELS / GMR_ROOT")
    body_models = os.path.join(os.environ["GMR_ROOT"], "assets", "body_models")
if not os.path.isfile(os.path.join(body_models, "smplx", "SMPLX_NEUTRAL.npz")):
    raise SystemExit(f"SMPL-X model missing: {body_models}/smplx/SMPLX_NEUTRAL.npz (download it yourself, see docs/VIDEO_TO_MOTION.md#environment-setup)")

smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(args.gvhmr_pred, body_models)
frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=g1_model.FPS)
retarget = GMR(actual_human_height=human_height, src_human="smplx", tgt_robot="unitree_g1", verbose=False)
gmr_joints = [mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(retarget.model.njnt)]
assert tuple(gmr_joints[1:]) == g1_model.G1_JOINT_NAMES, f"GMR G1 joint order changed: {gmr_joints}"
if args.init_from_human:
    pos, quat = frames[0]["pelvis"]
    q0 = retarget.configuration.data.qpos.copy()
    q0[:3] = pos
    q0[3:7] = (R.from_quat(quat, scalar_first=True) * retarget.rot_offsets1["pelvis"]).as_quat(scalar_first=True)
    retarget.configuration.update(q0)
    for _ in range(20):
        retarget.retarget(frames[0])


def ik_err():
    return retarget.error1() + retarget.error2()


qpos, n_restart, switches = [], 0, []
for f in frames:
    q = retarget.retarget(f).copy()
    if args.restart_tol > 0:
        e = ik_err()
        q_r = q.copy()
        q_r[7 + 12 :] = 0  # waist + arms (Unitree order: 12 leg joints first)
        retarget.configuration.update(q_r)
        for _ in range(30):
            q_r = retarget.retarget(f).copy()
        if ik_err() < e - args.restart_tol:
            q, n_restart = q_r, n_restart + 1
            switches.append(len(qpos))
        retarget.configuration.update(q)
    qpos.append(q)
qpos = np.stack(qpos)  # (T, 36): pos, quat wxyz, 29 dof
if args.restart_tol > 0:
    print(f"  IK restarts kept on {n_restart} frames: {switches}")
# a restart switches IK branch (typically shoulder pitch/yaw near the arm singularity), which is a jump in joint
# space: cross-fade the joints that jump over +-XFADE frames (cosine) so it becomes a short, fast arm swing
XFADE = 5
for k in (k for k in switches if k > 0):
    a0, a1 = max(k - XFADE, 0), min(k + XFADE, len(qpos) - 1)
    jump = np.abs(qpos[k, 7:] - qpos[k - 1, 7:]) > 0.3
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a1 - a0 + 1))[:, None]
    qpos[a0 : a1 + 1, 7:][:, jump] = ((1 - w) * qpos[a0, 7:] + w * qpos[a1, 7:])[:, jump]
if args.smooth_sigma > 0:
    from scipy.ndimage import gaussian_filter1d

    qpos[:, 7:] = gaussian_filter1d(qpos[:, 7:], args.smooth_sigma, axis=0, mode="nearest")
print(f"{args.name}: {len(qpos)} frames @ {fps} fps, human height {human_height:.2f} m")
if abs(float(fps) - g1_model.FPS) > 1e-6:
    raise SystemExit(f"expected {g1_model.FPS} fps from GMR, got {fps}")

# clamp joints to the robot model's limits and measure foot heights on that model
m = g1_model.load_g1(args.robot_xml)
d = mujoco.MjData(m)
lo, hi = g1_model.joint_limits(m)
n_clip = int(((qpos[:, 7:] < lo - 1e-4) | (qpos[:, 7:] > hi + 1e-4)).sum())
qpos[:, 7:] = np.clip(qpos[:, 7:], lo, hi)
feet = g1_model.foot_contact_points(m)
minz = np.empty(len(qpos))
for t, q in enumerate(qpos):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    minz[t] = min(g1_model.sole_z(d, feet))
if args.floor_mode == "constant":
    floor = np.full(len(qpos), np.percentile(minz, args.contact_pct))
else:
    w = int(fps)
    pad = np.pad(minz, w // 2, mode="edge")
    env = np.array([pad[i : i + w].min() for i in range(len(minz))])
    pad = np.pad(env, w // 2, mode="edge")
    floor = np.convolve(pad, np.ones(w) / w, mode="valid")[: len(minz)]
qpos[:, 2] -= floor
minz -= floor
qpos[:, :2] -= qpos[0, :2]

dq = np.diff(qpos[:, 7:], axis=0) * fps
print(
    f"  floor offset [{floor.min():+.3f}, {floor.max():+.3f}] m, foot min z after align: "
    f"p0={minz.min():+.3f} p50={np.median(minz):+.3f} max={minz.max():+.3f}"
)
print(f"  root z [{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}], max |qd| {np.abs(dq).max():.1f} rad/s, clamped {n_clip} joint values")

os.makedirs(args.out_dir, exist_ok=True)
quat_xyzw = g1_model.hemisphere_continuous(qpos[:, [4, 5, 6, 3]])
csv = np.concatenate([qpos[:, :3], quat_xyzw, qpos[:, 7:]], 1)
g1_model.save_csv(os.path.join(args.out_dir, f"{args.name}.csv"), csv)
with open(os.path.join(args.out_dir, f"{args.name}_gmr.pkl"), "wb") as f:
    pickle.dump(
        {"fps": fps, "root_pos": qpos[:, :3], "root_rot": quat_xyzw, "dof_pos": qpos[:, 7:], "local_body_pos": None, "link_body_list": None},
        f,
    )
print(f"  -> {args.out_dir}/{args.name}.csv")
