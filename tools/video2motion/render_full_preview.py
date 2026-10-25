"""Side-by-side check of a clip extended with extend_full.py (shots A wide -> B close-up -> C wide): source | G1.

  MUJOCO_GL=egl python tools/video2motion/render_full_preview.py <clip_full.csv> <clip_full.mp4> <out.mp4> \
      --cuts 271 334 --kept_rows 269 [--from_row 240] [--crop_top 0] [--src_offset 0]

Row i of the CSV = frame i of the (30 fps) clip video. Every frame is labelled with its row / source frame number
and its shot; rows produced by extend_full.py (row >= --kept_rows) are highlighted. Unlike render_preview.py the
camera is fixed (where a static wide-shot camera would be), so the robot's travel reads like the source.
"""

import argparse

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

import g1_model

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("csv")
ap.add_argument("src", help="the whole A+B+C clip video, frame 0 = CSV row 0")
ap.add_argument("out")
ap.add_argument("--cuts", type=int, nargs=2, required=True, help="CSV rows where shot B and shot C begin")
ap.add_argument("--kept_rows", type=int, required=True, help="number of leading rows copied unchanged from the original CSV")
ap.add_argument("--from_row", type=int, default=0, help="first row to render")
ap.add_argument("--src_offset", type=int, default=0, help="frame number of row 0 in a longer source video (labels only)")
ap.add_argument(
    "--front_yaw",
    type=float,
    default=None,
    help="world heading (deg) of a body facing the real camera; default: the robot's heading in row 0",
)
ap.add_argument("--crop_top", type=int, default=0, help="px cropped from the top of each source frame (e.g. a caption or logo band)")
ap.add_argument("--robot_xml", default=None, help="default: $G1_XML, else mjlab's g1.xml")
args = ap.parse_args()

PH, PW = 540, 720
m = g1_model.load_g1(args.robot_xml, floor=True)
m.vis.global_.offwidth, m.vis.global_.offheight = max(PW, m.vis.global_.offwidth), max(PH, m.vis.global_.offheight)
d = mujoco.MjData(m)
a = np.loadtxt(args.csv, delimiter=",", ndmin=2)
assert a.shape[1] == g1_model.CSV_COLS, a.shape
rd = imageio.get_reader(args.src)
r = mujoco.Renderer(m, PH, PW)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 3.4, -10
i0 = args.from_row
front = args.front_yaw if args.front_yaw is not None else float(np.degrees(g1_model.yaw_xyzw(a[0, 3:7])))
cam.azimuth = front + 180
look = a[i0:, :2].mean(0)
w = imageio.get_writer(args.out, fps=g1_model.FPS, macro_block_size=1, quality=8)
n_out = 0
for i, fr in enumerate(rd):
    if i < i0:
        continue
    if i >= len(a):
        break
    g1_model.set_row(m, d, a[i])
    cam.lookat[:] = [look[0], look[1], 0.7]
    r.update_scene(d, cam)
    img = np.ascontiguousarray(r.render())
    s = np.ascontiguousarray(fr[args.crop_top :, :, :3])
    s = cv2.resize(s, (int(s.shape[1] * PH / s.shape[0]) // 2 * 2, PH), interpolation=cv2.INTER_AREA)
    shot = "A wide" if i < args.cuts[0] else ("B close-up (legs inferred)" if i < args.cuts[1] else "C wide")
    new = i >= args.kept_rows
    for im, txt in ((s, f"src {args.src_offset + i}  {shot}"), (img, f"row {i}  {'NEW rows' if new else 'existing rows'}")):
        cv2.putText(im, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0) if new else (255, 255, 255), 2, cv2.LINE_AA)
    w.append_data(np.concatenate([s, img], 1))
    n_out += 1
w.close()
print(args.out, n_out, "frames")
