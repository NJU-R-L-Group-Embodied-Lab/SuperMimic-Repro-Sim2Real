"""READ-ONLY real-robot check through g1_bridge --listen-only (no rt/lowcmd writer exists, nothing is ever published).
  python3 ro_bridge_check.py [--iface eth0] [--secs 10]
Prints lowstate rate, CRC failures, receive gaps (bridge clock, every sample from the ring), mode_machine, max temperature,
and how many rt/lowcmd samples other publishers sent (must be 0 in debug mode)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo layout: g1_common.py in scripts/deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # deploy-tree layout: all in one dir (searched first)
import g1_bridge_io as gb  # noqa: E402
import g1_common as gc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="eth0")
ap.add_argument("--domain", type=int, default=0)
ap.add_argument("--secs", type=float, default=10.0)
a = ap.parse_args()
B = gb.Bridge(a.domain, a.iface, crc=True, hz=500, wd_ms=100, damp_kd=3.0, log_path=Path("/tmp/ro_bridge_check.log"), listen_only=True)
try:
    cur = np.zeros(2, np.uint64)
    buf = np.zeros(gb.RING, gb.RING_DTYPE)
    ts = []
    W = gb.BridgeWatcher(B)
    W.poll()
    lc0 = W.count
    time.sleep(0.5)
    B.read_ring(cur, buf)  # drop the start-up part
    cur[1] = 0
    t_end = time.monotonic() + a.secs
    while time.monotonic() < t_end:
        n = B.read_ring(cur, buf)
        ts.extend(buf["t"][:n].tolist())
        time.sleep(0.05)
    W.poll()
    snap = np.zeros(1, gb.SNAP_DTYPE)
    B.read_snap(snap)
    st = gb.snap_to_state(snap)
    t = np.array(ts)
    g = np.diff(t) * 1e3
    print(f"lowstate {len(t) / (t[-1] - t[0]):.0f} Hz over {t[-1] - t[0]:.1f} s; gap p50 {np.median(g):.3f} p99 {np.percentile(g, 99):.3f} "
          f"max {g.max():.3f} ms; ring entries lost {int(cur[1])}; CRC bad total {int(snap['n_crc_bad'][0])} of {int(snap['n_rx'][0])}")
    print(f"mode_machine {st.mode_machine}  mode_pr {st.mode_pr}  tick {st.tick}  max temp {st.temp.max():.0f} C  "
          f"quat {np.round(st.quat, 3)}  motorstate nonzero {int(np.count_nonzero(st.motorstate))}")
    print(f"rt/lowcmd samples from other publishers during the check: {W.count - lc0}")
finally:
    B.quit()
    print(open("/tmp/ro_bridge_check.log").read().strip())
