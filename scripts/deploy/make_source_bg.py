"""Build background videos of the original motion source (cropped to remove black borders), time-locked 1:1 to a combo reference.

The source is the dance video the custom clips were retargeted from (1440x1080 @30 fps); frames are cropped to the
picture area without the black borders (crop 1400x944 at x=20, y=120). CSV frame i of each custom clip is
source frame START[clip] + i (found by correlating video motion energy with CSV joint speed).
The source video is not distributed with this repo (its rights belong to its owner): pass your own local copy
with --src (or set SM_SOURCE_VIDEO).

Outputs (in --out-dir):
  <clip>.mp4      : exactly the CSV range of each clip, 30 fps, with the matching source audio
  <combo>.mp4     : 50 fps, one output frame per combo reference frame (frame k at t = k / 50 s): clip segments map
                    to the matching source frame and audio; transitions hard-cut at their midpoint (hold the last
                    frame, then the next clip's first frame; no crossfade) and are silent, stand / fade-out segments hold
                    the previous clip's last frame.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

SRC = os.environ.get("SM_SOURCE_VIDEO", "source.mp4")  # --src / SM_SOURCE_VIDEO: local copy of the source video
CROP = (1400, 944, 20, 120)  # w, h, x, y
SRC_FPS = 30.0
SR = 44100
CLIPS = ("shanshenbu", "langzi_tiqiu", "gouxiong_duosuomao", "yaotou_huangnao")  # one <clip>.mp4 each
START = {"shanshenbu": 75, "langzi_tiqiu": 362, "gouxiong_duosuomao": 602, "yaotou_huangnao": 1065,
         "gouxiong_duosuomao_vc": 602}  # *_vc: joint-velocity-capped copy (cap_joint_velocity.py), same frames
CSV_FRAMES = {"shanshenbu": 216, "langzi_tiqiu": 152, "gouxiong_duosuomao": 397, "yaotou_huangnao": 313,
              "gouxiong_duosuomao_vc": 397}
# segment names in motions/custom_dance/shanshenbu_combo4.json use the original internal clip names
ALIAS = {"gouxiong_duosuomao_full_vc": "gouxiong_duosuomao_vc", "gouxiong_duosuomao_full": "gouxiong_duosuomao",
         "yaotou_huangnao_full": "yaotou_huangnao"}


def frames(src: str, w: int, h: int, x: int, y: int):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"crop={w}:{h}:{x}:{y}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.kill()
        p.wait()


def source_audio(src: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def encoder(path: Path, w: int, h: int, fps: float, audio: Path):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", str(audio),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                             str(path)], stdin=subprocess.PIPE)


def main(combo_json: str, out_dir: str = "videos/source", src: str = SRC) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h, x, y = CROP
    for clip in CLIPS:
        s, n = START[clip], CSV_FRAMES[clip]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf",
                        f"select=between(n\\,{s}\\,{s + n - 1}),setpts=N/({SRC_FPS}*TB),crop={w}:{h}:{x}:{y}",
                        "-af", f"atrim=start={s / SRC_FPS:.6f}:duration={n / SRC_FPS:.6f},asetpts=PTS-STARTPTS",
                        "-r", str(SRC_FPS), "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out / f"{clip}.mp4")], check=True)

    meta = json.loads(Path(combo_json).read_text())
    fps = float(meta["fps"])
    name = meta.get("name", Path(combo_json).stem)
    segs = [dict(g, name=ALIAS.get(g["name"], g["name"])) for g in meta["segments"]]
    # per output frame: source frame index
    plan = []
    clip_segs = [g for g in segs if g["kind"] == "clip" and g["name"] in START]
    first, last = clip_segs[0], clip_segs[-1]
    src_of = lambda g, k: START[g["name"]] + int(round((k - g["start"]) / fps * SRC_FPS))
    end_src = lambda g: src_of(g, g["end"] - 1)
    for i, g in enumerate(segs):
        for k in range(g["start"], g["end"]):
            if g["kind"] == "clip" and g["name"] in START:
                plan.append(src_of(g, k))
            elif g["kind"] == "transition":
                prev = next((p for p in reversed(segs[:i]) if p["kind"] == "clip" and p["name"] in START), None)
                nxt = next((p for p in segs[i + 1:] if p["kind"] == "clip" and p["name"] in START), None)
                cut = prev is None or (nxt is not None and k >= (g["start"] + g["end"]) // 2)
                plan.append(START[nxt["name"]] if cut else end_src(prev))
            else:  # stand / fade-out / settle segments: hold the previous clip's last frame (first clip's f0 before it)
                prev = next((p for p in reversed(segs[:i]) if p["kind"] == "clip" and p["name"] in START), None)
                plan.append(end_src(prev) if prev is not None else START[first["name"]])
    assert len(plan) == meta["frames"], (len(plan), meta["frames"])

    # audio: source audio under clip segments, silence elsewhere (5 ms fades at the edges against clicks)
    a_src, nfade = source_audio(src), int(0.005 * SR)
    audio = np.zeros((int(round(meta["frames"] / fps * SR)), 2), np.float32)
    ramp = np.linspace(0, 1, nfade, dtype=np.float32)[:, None]
    for g in clip_segs:
        a0, i0 = int(round(START[g["name"]] / SRC_FPS * SR)), int(round(g["start"] / fps * SR))
        n = min(int(round((g["end"] - g["start"]) / fps * SR)), len(audio) - i0, len(a_src) - a0)
        seg = a_src[a0:a0 + n].copy()
        seg[:nfade] *= ramp
        seg[-nfade:] *= ramp[::-1]
        audio[i0:i0 + n] = seg
    wav = out / f"{name}.f32"
    wav.write_bytes(audio.tobytes())

    need = sorted(set(plan))
    cache, it = {}, frames(src, w, h, x, y)
    for idx, f in enumerate(it):
        if idx in need:
            cache[idx] = f
        if idx >= need[-1]:
            break
    enc = encoder(out / f"{name}.mp4", w, h, fps, wav)
    for p in plan:
        enc.stdin.write(cache[p].tobytes())
    enc.stdin.close()
    enc.wait()
    wav.unlink()
    (out / f"{name}.json").write_text(json.dumps(
        {"source": src, "crop_whxy": CROP, "csv_start_frame": START, "fps": fps, "frames": len(plan),
         "segments": segs}, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--combo-json", required=True)
    ap.add_argument("--out-dir", default="videos/source")
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    main(a.combo_json, a.out_dir, a.src)
