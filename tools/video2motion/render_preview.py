"""Render a G1 motion CSV (30 fps kinematic reference) next to its source video clip: source | robot.

  MUJOCO_GL=egl python tools/video2motion/render_preview.py <clip.csv> <source.mp4> <out.mp4> [--cam_yaw DEG]

Row t of the CSV is shown next to frame t of the source clip (both 30 fps). The camera looks at the robot from the
front of its initial heading (like a frontal source video) and follows the root xy. Use --cam_yaw when the
dancer starts turned away from the camera. Headless: MUJOCO_GL=egl (NVIDIA) or MUJOCO_GL=osmesa.
"""

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

import g1_model

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("csv")
ap.add_argument("src", help="source video (30 fps)")
ap.add_argument("out")
ap.add_argument("--robot_xml", default=None, help="default: $G1_XML, else mjlab's g1.xml")
ap.add_argument(
    "--cam_yaw",
    type=float,
    default=float(os.environ["CAM_YAW"]) if "CAM_YAW" in os.environ else None,
    help="world heading (deg) the camera looks along +180; default: the robot's heading in row 0 (env CAM_YAW also works)",
)
ap.add_argument("--height", type=int, default=480, help="output height in px (robot panel is 4:3)")
args = ap.parse_args()

m = g1_model.load_g1(args.robot_xml, floor=True)
H, W = args.height // 2 * 2, int(args.height * 4 / 3) // 2 * 2
m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
d = mujoco.MjData(m)
a = np.loadtxt(args.csv, delimiter=",", ndmin=2)
assert a.shape[1] == g1_model.CSV_COLS, a.shape
yaw0 = args.cam_yaw if args.cam_yaw is not None else float(np.degrees(g1_model.yaw_xyzw(a[0, 3:7])))
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.2, -10, yaw0 + 180
reader = imageio.get_reader(args.src)
w = imageio.get_writer(args.out, fps=g1_model.FPS, macro_block_size=1)
s = None
src_iter = iter(reader)
for row in a:
    s = next(src_iter, s)  # hold the last source frame if the video is shorter than the CSV
    g1_model.set_row(m, d, row)
    cam.lookat[:] = [row[0], row[1], 0.75]
    r.update_scene(d, cam)
    img = r.render()
    sw = int(s.shape[1] * H / s.shape[0]) // 2 * 2
    idx_y = (np.arange(H) * s.shape[0] / H).astype(int)
    idx_x = (np.arange(sw) * s.shape[1] / sw).astype(int)
    w.append_data(np.concatenate([s[idx_y][:, idx_x, :3], img], 1))
w.close()
reader.close()
print(args.out, len(a), "frames")
