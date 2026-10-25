"""Person boxes for a clip that contains hard camera cuts, written where GVHMR's demo.py looks for them.

GVHMR's own tracker keeps only the longest YOLO track and smooths boxes across the whole clip, so a cut
(new track id, jump in box size) loses or blurs the person. Here: YOLO detection per frame, largest person,
cuts found from frame differences, boxes interpolated and smoothed only inside each shot.

Run inside the GVHMR environment BEFORE GVHMR's demo.py (demo.py skips its tracker when bbx.pt exists):
  python tools/video2motion/shot_bbx.py <clip.mp4> <V2M_WORK>/gvhmr/<clip>/preprocess

GVHMR is not part of this repository: its helpers are imported at run time from your own checkout ($GVHMR_ROOT).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("clip", help="30 fps video clip (the same file that is passed to GVHMR demo.py)")
ap.add_argument("preprocess_dir", help="GVHMR <output_root>/<clip stem>/preprocess; bbx.pt is written there")
ap.add_argument("--gvhmr_root", default=os.environ.get("GVHMR_ROOT"), help="GVHMR checkout (default: $GVHMR_ROOT)")
ap.add_argument("--yolo_ckpt", default=None, help="default: <gvhmr_root>/inputs/checkpoints/yolo/yolov8x.pt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--conf", type=float, default=0.3, help="YOLO person confidence threshold")
ap.add_argument("--cut_min", type=float, default=25.0, help="a cut needs a mean |frame diff| (8x downsampled) above this ...")
ap.add_argument("--cut_k", type=float, default=6.0, help="... and above cut_k x the clip median")
args = ap.parse_args()
if not args.gvhmr_root:
    raise SystemExit("pass --gvhmr_root or set GVHMR_ROOT")
gvhmr_root = Path(args.gvhmr_root).resolve()
sys.path.insert(0, str(gvhmr_root))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa: E402
from hmr4d.utils.video_io_utils import read_video_np  # noqa: E402

pre = Path(args.preprocess_dir)
video = read_video_np(args.clip)  # (L, H, W, 3) RGB
small = video[:, ::8, ::8].astype(np.float32)
diff = np.abs(np.diff(small, axis=0)).mean((1, 2, 3))
cuts = [int(i) + 1 for i in np.where(diff > max(args.cut_min, args.cut_k * np.median(diff)))[0]]  # first frame of each new shot
print("frame diff median", float(np.median(diff)), "cuts at clip frames", cuts, [float(diff[c - 1]) for c in cuts])

yolo = YOLO(args.yolo_ckpt or str(gvhmr_root / "inputs/checkpoints/yolo/yolov8x.pt"))
boxes = np.zeros((len(video), 4), np.float32)
ok = np.zeros(len(video), bool)
for i in range(0, len(video), args.batch):
    batch = list(video[i : i + args.batch][..., ::-1])  # BGR for ultralytics
    for j, r in enumerate(yolo.predict(batch, classes=0, conf=args.conf, verbose=False, device=args.device)):
        b = r.boxes.xyxy.cpu().numpy()
        if len(b):
            boxes[i + j] = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
            ok[i + j] = True
print("frames without detection:", np.where(~ok)[0].tolist())
bounds = [0, *cuts, len(video)]
for s, e in zip(bounds[:-1], bounds[1:]):
    idx = np.arange(s, e)
    good = idx[ok[s:e]]
    if len(good) == 0:
        raise SystemExit(f"no person detected in shot {s}-{e - 1}")
    for k in range(4):
        boxes[s:e, k] = np.interp(idx, good, boxes[good, k])
        # same smoothing as GVHMR (2x moving average of 5), but never across a cut
        for _ in range(2):
            p = np.pad(boxes[s:e, k], 2, mode="edge")
            boxes[s:e, k] = np.convolve(p, np.ones(5) / 5, mode="valid")
    print(f"shot {s}-{e - 1}: mean box h {np.mean(boxes[s:e, 3] - boxes[s:e, 1]):.0f}px")
bbx_xyxy = torch.from_numpy(boxes).float()
bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
pre.mkdir(parents=True, exist_ok=True)
torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys, "cuts": cuts}, pre / "bbx.pt")
print("->", pre / "bbx.pt")
