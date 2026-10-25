"""Standalone g1_bridge test on DDS domain 2 / lo (never the robot).

Python (unitree_sdk2py) plays the robot: publishes rt/lowstate with the SDK Python CRC (plus some corrupted ones), and
subscribes rt/lowcmd checking the bridge's C++ CRC with the SDK Python CRC. Checks: lowstate CRC parity, snapshot field
parity, ring completeness, command values / modes / mode_machine, q-measured flag, NOW latency, fixed-rate stream,
watchdog damping, QUIT.
  python scripts/deploy/bridge/test_bridge.py   (deploy env, after bridge/build.sh; bridge binaries: see g1_bridge_io.py / SM_BRIDGE_HOME)
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo layout: g1_common.py in scripts/deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # deploy-tree layout: all in one dir (searched first)
import g1_bridge_io as gb  # noqa: E402
import g1_common as gc  # noqa: E402

DOM, IF = 2, "lo"
gc.init_dds(DOM, IF)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

crc = CRC()
fails = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)
    if not ok:
        fails.append(name)


rx = []  # (t, msg) of rt/lowcmd
rx_lock = threading.Lock()


def on_cmd(m):
    with rx_lock:
        rx.append((time.monotonic(), m))


sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub.Init(on_cmd, 0)
pub = ChannelPublisher("rt/lowstate", LowState_)
pub.Init()

rng = np.random.default_rng(0)
state = {"tick": 1000, "stop": False, "n_bad_sent": 0, "q": rng.uniform(-1, 1, 35).astype(np.float32), "keys": 0}


def make_state(bad=False):
    m = unitree_hg_msg_dds__LowState_()
    m.tick = state["tick"]
    m.mode_machine = 5
    m.mode_pr = 0
    for i in range(35):
        ms = m.motor_state[i]
        ms.q = float(state["q"][i])
        ms.dq = float(i * 0.01)
        ms.tau_est = float(-i * 0.1)
        ms.temperature = [30 + i, 40 + i]
        ms.motorstate = i
    m.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    m.imu_state.gyroscope = [0.1, 0.2, 0.3]
    m.imu_state.accelerometer = [0.0, 0.0, 9.81]
    m.imu_state.rpy = [0.01, 0.02, 0.03]
    wr = [0] * 40
    wr[2], wr[3] = state["keys"] & 0xFF, state["keys"] >> 8
    m.wireless_remote = wr
    m.crc = crc.Crc(m) ^ (1 if bad else 0)
    return m


def state_pub():
    k = 0
    while not state["stop"]:
        k += 1
        state["tick"] += 1
        bad = k % 100 == 50
        state["n_bad_sent"] += bad
        pub.Write(make_state(bad))
        time.sleep(0.001)


threading.Thread(target=state_pub, daemon=True).start()
time.sleep(0.5)

log = Path("/tmp/test_bridge.log")
B = gb.Bridge(DOM, IF, crc=True, hz=500, wd_ms=100, damp_kd=3.0, log_path=log, orphan_s=1.0)
time.sleep(1.0)
snap = np.zeros(1, gb.SNAP_DTYPE)
check("snap read", B.read_snap(snap))
r = snap[0]
check("lowstate CRC parity (good accepted)", r["n_rx"] > 300, f"n_rx {r['n_rx']}")
check("lowstate CRC parity (bad dropped)", 5 <= r["n_crc_bad"] <= 15, f"n_crc_bad {r['n_crc_bad']} (sent ~{state['n_bad_sent']})")
st = gb.snap_to_state(snap)
check("snapshot q parity", np.array_equal(st.q, state["q"][:29].astype(float)))
check("snapshot fields", st.mode_machine == 5 and st.temp[3].tolist() == [33, 43] and st.motorstate[7] == 7 and abs(st.acc[2] - np.float32(9.81)) < 1e-9,
      f"temp3 {st.temp[3]} acc {st.acc}")
check("tick moving", bool(r["tick_moving"]) and abs(time.monotonic() - r["t_rx"]) < 0.01, f"age {(time.monotonic() - r['t_rx']) * 1e3:.2f} ms")

# ring: every accepted sample, keys follow
cur = np.zeros(2, np.uint64)
cur[0] = r["n_rx"]  # start from now
buf = np.zeros(gb.RING, gb.RING_DTYPE)
state["keys"] = gc.combo_mask("L2", "B")
time.sleep(0.2)
n = B.read_ring(cur, buf)
ticks = buf["tick"][:n].astype(np.int64)
check("ring contiguous", n > 50 and np.all(np.diff(ticks) >= 1) and int(cur[1]) == 0, f"n {n} lost {cur[1]} tick steps {np.unique(np.diff(ticks))}")
check("ring keys", buf["keys"][n - 1] == gc.combo_mask("L2", "B"))
state["keys"] = 0

# watcher counts nothing before the first command
W = gb.BridgeWatcher(B)
check("no lowcmd before first command", W.count_for(0.3) == 0 and len(rx) == 0)

# commands
N = 29
mode = bytes((1 if i < N and i not in (13, 14) else 0) for i in range(35))
q = np.linspace(-0.5, 0.5, 35)
kp = np.full(35, 40.0)
kd = np.full(35, 2.0)
t_send = time.monotonic()
B.send(q, kp, kd, mode, 5, gb.CMD_NOW)
deadline = time.monotonic() + 1.0
while not rx and time.monotonic() < deadline:
    time.sleep(0.0002)
lat = (rx[0][0] - t_send) * 1e3 if rx else float("nan")
check("first command published", bool(rx), f"latency {lat:.2f} ms")
with rx_lock:
    m = rx[-1][1]
check("lowcmd CRC parity (C++ vs SDK Python)", m.crc == crc.Crc(m), f"{m.crc:#x} vs {crc.Crc(m):#x}")
check("lowcmd values", all(abs(m.motor_cmd[i].q - np.float32(q[i])) < 1e-7 and m.motor_cmd[i].kp == 40 and m.motor_cmd[i].kd == 2 for i in range(35)))
check("lowcmd modes / mode_machine", [m.motor_cmd[i].mode for i in range(35)] == list(mode) and m.mode_machine == 5 and m.mode_pr == 0)

# fixed-rate stream with 500 Hz heartbeats from "the runner"
with rx_lock:
    rx.clear()
t0 = time.monotonic()
lats = []
k = 0
while time.monotonic() - t0 < 2.0:
    k += 1
    if k % 10 == 0:  # NOW command every 20 ms (policy step)
        with rx_lock:
            n0 = len(rx)
        ts = time.monotonic()
        B.send(q, kp, kd, mode, 5, gb.CMD_NOW)
        while True:
            with rx_lock:
                if len(rx) > n0:
                    lats.append((rx[n0][0] - ts) * 1e3)
                    break
            if time.monotonic() - ts > 0.05:
                break
            time.sleep(0.0001)
    else:
        B.send(q, kp, kd, mode, 5, 0)
    time.sleep(0.002)
with rx_lock:
    t = np.array([x[0] for x in rx])
rate = (len(t) - 1) / (t[-1] - t[0])
gaps = np.diff(t) * 1e3
check("stream rate 500 + NOW", 520 < rate < 600, f"{rate:.1f} Hz, gap p99 {np.percentile(gaps, 99):.2f} max {gaps.max():.2f} ms")
check("NOW latency", np.median(lats) < 1.0, f"p50 {np.median(lats):.3f} p99 {np.percentile(lats, 99):.3f} max {max(lats):.3f} ms (n {len(lats)})")
W.poll()
s = B.read_status(np.zeros(1, gb.STATUS_DTYPE))[0]
check("watch counts (self-delivery)", W.count > 0, f"lc_count {W.count} n_pub {s['n_pub']}")
check("bridge-side publish gaps", s["pub_gap_max"] < 0.004, f"max gap {s['pub_gap_max'] * 1e3:.2f} ms, >2 periods {s['n_gap_late']}, slowest Write {s['pub_write_max'] * 1e3:.2f} ms")

# q measured
B.send(q, np.zeros(35), np.full(35, 3.0), mode, 5, gb.CMD_NOW | gb.CMD_Q_MEASURED)
time.sleep(0.02)
with rx_lock:
    m = rx[-1][1]
check("q measured", all((m.motor_cmd[i].q == state["q"][i]) if mode[i] else m.motor_cmd[i].q == 0 for i in range(35)) and m.motor_cmd[0].kp == 0)

# watchdog: go silent
B.send(q, kp, kd, mode, 5, gb.CMD_NOW)
t_sil = time.monotonic()
time.sleep(0.3)
s = B.read_status(np.zeros(1, gb.STATUS_DTYPE))[0]
with rx_lock:
    m = rx[-1][1]
check("watchdog fired", s["wd_fired"] == 1)
check("watchdog damping cmd", all((m.motor_cmd[i].kp == 0 and m.motor_cmd[i].kd == (3.0 if mode[i] else 0.0)) for i in range(35))
      and m.motor_cmd[0].q == state["q"][0] and m.crc == crc.Crc(m))
# latched: a fresh command does not undo it
B.send(q, kp, kd, mode, 5, gb.CMD_NOW)
time.sleep(0.02)
with rx_lock:
    m = rx[-1][1]
check("watchdog latched", m.motor_cmd[0].kp == 0)

# QUIT: stream stops
B.quit()
time.sleep(0.1)
with rx_lock:
    n1 = len(rx)
time.sleep(0.3)
with rx_lock:
    n2 = len(rx)
check("QUIT stops stream", n1 == n2 and B.p.poll() == 0, f"rc {B.p.poll()}")
state["stop"] = True
print(open(log).read())
print("ALL PASS" if not fails else f"FAILED: {fails}")
