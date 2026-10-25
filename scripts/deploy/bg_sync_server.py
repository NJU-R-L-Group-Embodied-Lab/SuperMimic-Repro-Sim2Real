#!/usr/bin/env python3
"""Background-video player locked to the real-robot runner (stdlib only; runs on the deploy PC or any LAN machine).

  python scripts/deploy/bg_sync_server.py --video <out-dir>/combo_c4_full_vc_b15f20.mp4  # written by scripts/deploy/make_source_bg.py from your own copy of the source video
  python scripts/deploy/g1_runner.py ... --clip <combo>.npz --sync-udp <this host>:15000
  open http://<this host>:8770/ in a browser on the display (F = fullscreen, 1 = native 1:1 pixels / fit)

The video must have one frame per reference frame at the clip fps (scripts/deploy/make_source_bg.py writes 50 fps).
The runner sends "state clip frame t" every tick; the page shows frame 0 (paused) before the motion, plays and
follows the runner's frame during POLICY (seeks when off by > --max-lag-frames), and freezes on the last POLICY
frame afterwards (THERMAL_STOP / STANDBY_POST / END_HOLD / DAMP). Only display: nothing is ever sent back to the runner.
Audio: browsers block autoplay with sound, so click the page or press any key once before the run (HUD shows
"sound on"); until then the video plays muted. M toggles mute.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>G1 background</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}
video{width:100vw;height:100vh;object-fit:contain;display:block}
video.native{width:auto;height:auto;max-width:none;margin:auto;position:absolute;inset:0}
#hud{position:fixed;left:8px;bottom:6px;color:#8f8;font:13px monospace;opacity:.6}</style></head>
<body><video id="v" src="/video.mp4" muted playsinline preload="auto"></video><div id="hud">waiting for runner…</div>
<script>
const FPS=__FPS__, LAG=__LAG__, v=document.getElementById('v'), hud=document.getElementById('hud');
let last=null, lastPolicyFrame=0, hudOn=true, unlocked=false;
// one user gesture (sticky activation) lets later play() calls run with sound; never starts playback by itself
function unlock(){ if(unlocked) return false; unlocked=true; v.muted=false; return true }
document.addEventListener('click',unlock);
document.addEventListener('keydown',e=>{
  const first=unlock();
  if((e.key==='m'||e.key==='M') && !first){v.muted=!v.muted}
  if(e.key==='f'||e.key==='F'){document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen()}
  if(e.key==='1'){v.classList.toggle('native')}
  if(e.key==='h'||e.key==='H'){hudOn=!hudOn;hud.style.display=hudOn?'':'none'}
});
function show(f){const t=f/FPS; if(Math.abs(v.currentTime-t)>0.5/FPS) v.currentTime=t; if(!v.paused) v.pause()}
function apply(m){
  last=m;
  if(m.state==='POLICY' && m.frame>=0){
    lastPolicyFrame=m.frame;
    const t=m.frame/FPS, lag=(v.currentTime-t)*FPS;
    if(v.paused||Math.abs(lag)>LAG){v.currentTime=t}
    if(v.paused){v.play().catch(()=>{})}
  } else if(['THERMAL_STOP','STANDBY_POST','END_HOLD','DAMP'].includes(m.state) && lastPolicyFrame>0){
    show(lastPolicyFrame);
  } else { lastPolicyFrame=0; show(0) }
  hud.textContent=`${m.state} clip ${m.clip} frame ${m.frame}  video ${(v.currentTime*FPS).toFixed(0)}  age ${m.age.toFixed(2)}s  ${v.muted?'MUTED (click / any key for sound)':'sound on'}`;
}
const es=new EventSource('/events'); es.onmessage=e=>apply(JSON.parse(e.data));
es.onerror=()=>{hud.textContent='server lost'};
</script></body></html>"""


class State:
    def __init__(self):
        self.msg = {"state": "NONE", "clip": -1, "frame": -1, "t": 0.0}
        self.rx = 0.0
        self.cv = threading.Condition()


def udp_loop(st: State, port: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    while True:
        data, _ = s.recvfrom(256)
        try:
            state, clip, frame, t = data.decode().split()
            m = {"state": state, "clip": int(clip), "frame": int(frame), "t": float(t)}
        except ValueError:
            continue
        with st.cv:
            st.msg, st.rx = m, time.monotonic()
            st.cv.notify_all()


def make_handler(st: State, video: Path, page: bytes):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", page)
            elif self.path == "/video.mp4":
                self._video()
            elif self.path == "/events":
                self._events()
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _video(self):  # Range support: browsers need it to seek
            size = video.stat().st_size
            a, b = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                lo, _, hi = rng[6:].split(",")[0].partition("-")
                a = int(lo) if lo else max(0, size - int(hi))
                b = int(hi) if (hi and lo) else size - 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            with open(video, "rb") as f:
                f.seek(a)
                left = b - a + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with st.cv:
                        st.cv.wait(timeout=0.5)
                        m = dict(st.msg, age=(time.monotonic() - st.rx) if st.rx else -1.0)
                    self.wfile.write(f"data: {json.dumps(m)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=50.0, help="video frames per reference frame second (clip fps)")
    ap.add_argument("--udp-port", type=int, default=15000)
    ap.add_argument("--http-port", type=int, default=8770)
    ap.add_argument("--max-lag-frames", type=float, default=3.0, help="seek when the video is off by more than this")
    a = ap.parse_args()
    st = State()
    threading.Thread(target=udp_loop, args=(st, a.udp_port), daemon=True).start()
    page = PAGE.replace("__FPS__", str(a.fps)).replace("__LAG__", str(a.max_lag_frames)).encode()
    srv = ThreadingHTTPServer(("0.0.0.0", a.http_port), make_handler(st, a.video.resolve(), page))
    print(f"UDP :{a.udp_port}  page http://{socket.gethostname()}:{a.http_port}/  video {a.video} ({os.path.getsize(a.video) / 1e6:.1f} MB)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
