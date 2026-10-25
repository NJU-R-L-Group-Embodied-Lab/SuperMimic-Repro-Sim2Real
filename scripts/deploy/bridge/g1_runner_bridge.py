#!/usr/bin/env python3
"""G1 29-DoF real-robot runner for the SuperMimic BONES-SEED tracking policy (ONNX, proprioception only).

BRIDGE VARIANT (internal tag g1_bridge_v1): all rt/lowstate / rt/lowcmd DDS traffic runs in the C++ process bridge/g1_bridge
(unitree_sdk2); this runner reads lowstate from its shared memory and sends commands over a unix socket (g1_bridge_io.py).
Obs / policy / safety / state machine are unchanged from the pure-Python DDS runner it was forked from (internal tag
g1_frozen_v3, cf. scripts/deploy/g1_runner.py). Reason: on the G1's onboard computer (PC2, Orin NX) the Python
DDS (de)serialization + CRC held the GIL ~94% of the time. Extra safety in the bridge: no runner command for --bridge-wd-ms,
runner gone, or SIGTERM -> latched damping (kd 3, kp 0) published by the bridge itself.

Layout: in this repo the file is scripts/deploy/bridge/g1_runner_bridge.py (g1_common.py one directory up, bridge binaries
built next to it by bridge/build.sh). On the robot it was run from a deploy tree <tree>/scripts/deploy/g1_runner.py with
g1_bridge_io.py + g1_common.py beside it, the binaries in <tree>/bridge and supermimic.deploy.core in <tree>/src;
SM_BRIDGE_HOME overrides the binary directory (see g1_bridge_io.py).

Run it on the deploy PC's LOCAL console (not over ssh / Wi-Fi; refused on domain 0 unless --allow-ssh):
  source $G1_DEPLOY_HOME/env.sh   # deploy venv (python3 + numpy + onnxruntime + unitree_sdk2py); default ~/workspace/g1_deploy
  python scripts/deploy/bridge/g1_runner_bridge.py --iface <nic> --domain 0 --stage air \
      --artifacts deploy_artifacts/ckpt_0116000 \
      --clip deploy_artifacts/clips/neutral_idle_loop_001__A081.npz --log-dir deploy_logs/real

State machine (Unitree remote; combos debounced: key word stable >= 3 lowstate msgs, a combo must have been
released >= 100 ms before it can fire again, nothing pressed before the command thread starts is acted on):
  PASSIVE   kp=0 kd=3 on all 29 joints
  --[L2+up]--> INTERP    cosine interpolation measured q -> f0 over --interp-s with stand gains,
                         f0 = clip frame 0 clipped to the soft joint limits (as the training reset does)
  ----------> HOLD      stand gains at f0 (cannot balance on the ground: the hoist carries the robot)
  --[R1+A]--> STANDBY_PRE  only if HOLD >= --hold-min-s, max|q - f0| < --hold-tol, max|dq| < --start-max-dq,
                         tilt vs clip proj_grav[0] < --start-max-tilt-deg, max motor temp < --warn-temp;
                         training gains, DeployCore on static_clip(clip, 0) (frame-0 pose, all rates zero),
                         reset(0) right before the first step, step(advance=False) every tick: the POLICY balances
                         the robot standing still (crew lowers the rope until the feet carry most of the weight, sets the
                         gantry). Documented cap ~2 min (WARN after --standby-warn-s). R1+A again here does nothing.
                         Status line every --status-s (yaw change since entering the state, tilt error, max leg |dq|,
                         temperature, whether R1+X would be accepted); WARN on |yaw change| > --warn-yaw-deg, leg |dq| >
                         --warn-leg-dq for > 0.5 s, temperature > --warn-temp.
                         [L2+up] twice within --confirm-s: END_HOLD blend (stand gains, target = THIS clip's f0) -> HOLD of
                         the same clip (clip index unchanged). THE HOIST MUST CARRY THE ROBOT.
  --[R1+X]--> POLICY    fresh press, only if STANDBY_PRE >= --standby-min-s (3 s) and over the last --standby-window-s (3 s):
                         tilt error vs the reference < --standby-max-tilt-deg and max|dq| < --standby-max-dq
                         (else REFUSED with the reason). Heating forecast: per live motor (max of its two temperature
                         sensors) the slope over the last --temp-rate-window-s of STANDBY_PRE (>= --temp-rate-min-s of
                         samples needed), T_now + max(slope, 0) * clip duration must stay < --thermal-stop-temp (else
                         REFUSED). DeployCore.switch_clip(clip, 0): history and last action are kept, the motion starts
                         from frame 0.
  --(motor temp >= --thermal-stop-temp in POLICY)--> thermal stop: the policy keeps balancing on a synthetic reference
                  (thermal_stop_clip) over --thermal-stop-s from the current reference frame to --thermal-stop-pose:
                  'current' (default) = that same frame held still (joint rates zero, body rates fade out);
                  'upright' = the same (joint rates zero, body rates fade out) but the 3
                  waist joints blend to their frame-0 values (real run pc2bridge11: holding the bent torso of frame 1948
                  heated waist_pitch 72 -> 81 C in 6 s -> DAMP; in sim upright did NOT lower the waist_pitch torque:
                  the policy itself pushes waist_pitch to -0.49 rad); 'frame0' =
                  cosine blend to the clip's frame-0 stand pose (real run pc2bridge10: after a stop at frame 2020 the feet
                  could not get back to the frame-0 stance and the robot fell). Then STANDBY_POST on that static pose. No next clip afterwards (L2+up ignored): re-hoist, then L2+B. > --max-temp still -> DAMP.
  --(clip done)--> STANDBY_POST  switch_clip(static_clip(clip, last frame)), step(advance=False): the policy keeps the
                  robot standing in the last-frame pose. Re-hoist (gantry over the robot, take up slack only) then L2+B.
                  [L2+up] twice within --confirm-s (only with a NEXT --clip): END_HOLD blend (below) to stand gains ->
                  INTERP to the next clip's f0 -> HOLD -> [R1+A] ... THE HOIST MUST CARRY THE ROBOT FROM THE FIRST L2+up.
  --end-hold-legacy (implied by --lock-waist): (clip done) --> END_HOLD instead of STANDBY_POST: starts from the EXACT last
                  policy command (target, training kp/kd); over --end-blend-s the gains ramp linearly to the stand gains and the
                  target moves (cosine) to the clip's last frame clipped to the soft limits; every cycle the target is
                  clamped so that |kp(t) (q_cmd - q)| <= min(1, torque_clamp_factor) * effort_limit. Nothing starts by
                  itself. [L2+up] from END_HOLD (after the blend) interpolates from the CURRENT command target to f0
                  of the NEXT --clip (if any) -> HOLD -> [R1+A] ...
  --lock-waist: policy standby is NOT sim-gated on the waist-locked model (review 2026-10-24: after-motion standby of
                  shoulder_clap stepped / tilt-aborted). R1+A (and R1+X) are REFUSED for clips that are not on the lock-waist
                  STANDBY allow list (g1_common.LOCK_WAIST_STANDBY_ALLOWED_CLIPS, empty; --lock-waist-standby-allow NAME adds
                  one). --lock-waist-direct-start: for those clips R1+A in HOLD starts the motion directly (the pre-standby
                  flow gated in docs/DEPLOY_G1.md#waist-locked-g1-mode_machine--6: hoist carrying), clip end -> END_HOLD.
  [L2+B] / any abort (incl. DeployCore aborts in STANDBY_*) --> DAMP  kp=0 kd=3 forever (restart the program to leave it)
  SIGINT / SIGTERM / SIGHUP / SIGQUIT / exception --> DAMP (performed by the command thread), 1 s of damping
  commands, then exit. After exit NO rt/lowcmd is sent: what the motors then do is UNVERIFIED (see docs/DEPLOY_G1.md#stop-mechanisms-the-only-ones-that-actually-work).

Temperatures: both MotorState_.temperature values of every motor are logged at 50 Hz ("temp", (T, 29, 2)); the
STANDBY_* status line shows waist roll/pitch/yaw and hip pitch/roll (both sensors), the heating rate and forecast, and
POLICY prints the same temperatures every --status-s. --temp-headroom MOTOR:SENSOR:DEG raises every temperature limit
(--max-temp, --warn-temp, --thermal-stop-temp, the forecast) by DEG for that one sensor only (all checks compare
temp - headroom; logs and status lines show the raw readings). Real run pc2bridge09: waist_roll sensor 1 rose 47 -> 72 C
in ~4 s at frame 1403 and fell back to 45 C 40 s later (sensor 0 stayed 41 C): --temp-headroom 13:1:10.

Safety -> DAMP: DeployCore abort (tilt vs reference, joint velocity, non-finite state/IMU/action, |action|),
stale lowstate (> --stale-ms), main loop not producing commands for > (max_overrun + 0.5) cycles in
INTERP/HOLD/STANDBY_PRE/POLICY/STANDBY_POST/END_HOLD (checked by the command thread), policy-step overrun, motor temperature >
--max-temp, tilt > --max-tilt-hold in INTERP/HOLD/END_HOLD, motorstate & --motor-error-mask != 0 (domain 0),
mode_machine change, a foreign rt/lowcmd publisher, --max-write-fail consecutive failed rt/lowcmd writes.

mode_machine: accepted only 5 without --lock-waist or 6 (g1_29dof_lock_waist_rev_1_0) with --lock-waist; latched and sent back.
--lock-waist: motors 13/14 (waist roll/pitch, absent) get mode=0, q=dq=tau=kp=kd=0 in every state; their lowstate q/dq go
to DeployCore unchanged (they read 0, as in sim2sim --lock-waist); they are excluded from the start/pose/temperature/
motor-error/dq checks but watched: |q| > 0.02 rad or temperature != 0 -> DAMP 'waist unexpectedly active'. Clips are
limited to an allow list (g1_common.LOCK_WAIST_ALLOWED_CLIPS or --clip-allow-file); others need --i-know-clip <name>;
HARDER clips are always refused.

Threads: DDS lowstate listener (queue length 0: runs inside the DDS listener, stores the latest msg + receive
time, debounces buttons, only sets flags), 500 Hz command thread (rt/lowcmd from the latest targets, CRC,
watchdogs, signal and DAMP-request handling), 50 Hz policy loop in the main thread, printer thread (the only
thread that touches stdout; also writes <log-dir>/run_<stamp>.console.txt).
"""

from __future__ import annotations

import argparse
import collections
import gc as pygc
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo layout: g1_common.py in scripts/deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # deploy-tree layout: all in one dir (searched first)
import g1_bridge_io as gb  # noqa: E402
import g1_common as gc  # noqa: E402
from supermimic.deploy.core import (  # noqa: E402
    Clip, DeployConfig, DeployCore, OnnxPolicy, SafetyCfg, interpolate_pose, projected_gravity, static_clip,
)

N = gc.NUM_JOINTS
# new states are appended so the int8 state ids of older logs keep their meaning
STATES = ["PASSIVE", "INTERP", "HOLD", "POLICY", "END_HOLD", "DAMP", "STANDBY_PRE", "STANDBY_POST"]
SID = {s: i for i, s in enumerate(STATES)}
POLICY_STATES = ("STANDBY_PRE", "POLICY", "STANDBY_POST")  # DeployCore drives the robot (training gains)
HEARTBEAT_STATES = ("INTERP", "HOLD", "END_HOLD") + POLICY_STATES
STANDBY_STATES = ("STANDBY_PRE", "STANDBY_POST")
LEG_IDX = np.arange(12)  # left/right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
WAIST_IDX = np.arange(12, 15)  # waist yaw, roll, pitch
DAMP_KD = 3.0
# stand gains: per leg hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll; waist x3; arms x14
LEG_KP, LEG_KD = [100, 100, 100, 150, 40, 40], [2, 2, 2, 4, 2, 2]
STAND_KP = np.array(LEG_KP * 2 + [200] * 3 + [40] * 14, float)
STAND_KD = np.array(LEG_KD * 2 + [5] * 3 + [10] * 14, float)
COMBOS = {"interp": gc.combo_mask("L2", "up"), "policy": gc.combo_mask("R1", "A"), "start": gc.combo_mask("R1", "X"),
          "damp": gc.combo_mask("L2", "B")}
COMBO_LABEL = {"interp": "L2+up", "policy": "R1+A", "start": "R1+X"}
# motors shown in the temperature status lines (waist first: run06/07 overheated waist_roll)
TEMP_SHOW = ((13, "wR"), (14, "wP"), (12, "wY"), (0, "hpL"), (6, "hpR"), (1, "hrL"), (7, "hrR"))
_CORE = SafetyCfg()
# --stage profiles: SafetyCfg overrides + tilt guard (rad) in INTERP/HOLD/END_HOLD. clamp_soft_limits stays at
# the core default (False, sim2sim: clamping is out of distribution).
STAGES = {
    "air": dict(max_vel_frac=0.3, max_abs_action=10.0, torque_clamp_factor=0.5, max_tilt_hold=0.35),
    "feet": dict(max_vel_frac=0.3, max_abs_action=10.0, torque_clamp_factor=0.5, max_tilt_hold=0.35),
    "slack": dict(max_vel_frac=0.5, max_abs_action=15.0, torque_clamp_factor=1.0, max_tilt_hold=0.5),
    "ground": dict(max_vel_frac=_CORE.max_vel_frac, max_abs_action=_CORE.max_abs_action,
                   torque_clamp_factor=_CORE.torque_clamp_factor, max_tilt_hold=0.8),
}


# ---------------------------------------------------------------------------------------------- printing
class QueueStream:
    """sys.stdout / sys.stderr replacement: never blocks, never raises; the printer thread does the I/O."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(str(s))
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


class Printer:
    def __init__(self):
        self.q = queue.SimpleQueue()
        self.out = sys.__stdout__
        self.file = None
        self.n_fail = 0
        self.th = threading.Thread(target=self._loop, name="printer", daemon=True)
        self.th.start()

    def open_file(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.q.put(("__open__", path))
        except Exception:  # noqa: BLE001
            pass

    def _loop(self):
        while True:
            s = self.q.get()
            if s is None:
                return
            if isinstance(s, tuple):
                try:
                    self.file = open(s[1], "a", buffering=1)
                except Exception:  # noqa: BLE001
                    self.file = None
                continue
            try:
                self.out.write(s)
                self.out.flush()
            except Exception:  # noqa: BLE001  (EIO / EPIPE / closed stdout: keep running, keep the file log)
                self.n_fail += 1
            if self.file is not None:
                try:
                    self.file.write(s)
                except Exception:  # noqa: BLE001
                    pass

    def close(self, timeout=2.0):
        self.q.put(None)
        self.th.join(timeout)
        try:
            if self.file is not None:
                self.file.close()
        except Exception:  # noqa: BLE001
            pass


class LowcmdWatcher:
    """Counts rt/lowcmd samples in a CHILD process (this file with --lowcmd-watcher-child).

    An in-process rt/lowcmd reader with a Python listener deadlocked the runner in the sim 'foreign' test (all
    threads in futex_wait once a second publisher was active): the DDS receive thread holds the reader while
    waiting for the GIL to run the listener, and the command thread holds the GIL inside Write() while the local
    delivery to that same reader waits for it (diagnosis inferred, gdb attach is not permitted here). In a child
    process nothing shares our GIL. The child prints its running sample count every 0.1 s; read non-blocking.
    """

    def __init__(self, domain: int, iface: str | None):
        self.p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--lowcmd-watcher-child", str(domain), iface or ""],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  start_new_session=True)  # no terminal signals (Ctrl-C / Ctrl-Z) for the child
        self.fd = self.p.stdout.fileno()
        os.set_blocking(self.fd, False)
        self.buf = b""
        self.count = 0
        self.t = 0.0  # child's monotonic time of the latest count (late reports carry their own timestamps)
        self.ready = False
        self.dead = False

    def poll(self):
        if self.dead:
            return
        try:
            b = os.read(self.fd, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            b = b""
        if b == b"":
            self.dead = True
            return
        self.buf += b
        *lines, self.buf = self.buf.split(b"\n")
        for ln in lines:
            if ln == b"ready":
                self.ready = True
            else:
                parts = ln.split()
                if len(parts) == 2 and parts[0].isdigit():
                    self.count, self.t = int(parts[0]), float(parts[1])

    def wait_ready(self, timeout: float) -> bool:
        t_end = time.monotonic() + timeout
        while not self.ready and not self.dead and time.monotonic() < t_end:
            self.poll()
            time.sleep(0.02)
        return self.ready

    def count_for(self, secs: float) -> int:
        self.poll()
        time.sleep(0.15)  # let the child report the current count
        self.poll()
        n0 = self.count
        time.sleep(secs + 0.15)
        self.poll()
        return self.count - n0

    def close(self):
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass


def watcher_child(domain: int, iface: str):
    """--lowcmd-watcher-child: subscribe rt/lowcmd (count only), report on stdout, exit on stdin EOF."""
    gc.init_dds(domain, iface or None)
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    cnt = [0]

    def h(_m):
        cnt[0] += 1

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(h, 0)
    out = sys.stdout
    try:
        out.write("ready\n")
        out.flush()
        while True:
            r, _, _ = select.select([0], [], [], 0.1)
            if r and os.read(0, 64) == b"":
                break
            out.write(f"{cnt[0]} {time.monotonic():.6f}\n")
            out.flush()
    except Exception:  # noqa: BLE001  (parent gone: broken pipe)
        pass
    os._exit(0)


class Runner:
    def __init__(self, args, printer: Printer):
        self.a = args
        self.printer = printer
        self.cfg = DeployConfig.load(args.artifacts / "deploy_config.json")
        assert len(self.cfg.joint_names) == N
        c = self.cfg
        # --lock-waist (mode_machine 6, g1_29dof_lock_waist_rev_1_0): motors 13/14 do not exist. They get mode 0 and all-zero
        # commands in every state, are left out of every pose/temperature/error/dq check (but are watched: non-zero
        # q or temperature -> DAMP 'waist unexpectedly active'); DeployCore still sees their lowstate q = dq = 0.
        self.lock_waist = bool(args.lock_waist)
        self.locked = np.zeros(N, bool)
        if self.lock_waist:
            self.locked[list(gc.LOCKED_WAIST_IDX)] = True
        self.live = ~self.locked
        self.clip_notes = gc.lock_waist_clip_check(args.clip, args.clip_allow_file, args.i_know_clip or ()) if self.lock_waist else []
        self.clips = [Clip.load(p) for p in args.clip]
        # --sync-udp: fire-and-forget (state, clip, frame) datagram every tick for the background-video player
        # (scripts/deploy/bg_sync_server.py); non-blocking, errors ignored, never affects control
        self.sync_addr, self.sync_sock = None, None
        if args.sync_udp:
            host, port = args.sync_udp.rsplit(":", 1)
            self.sync_addr = (host, int(port))
            self.sync_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sync_sock.setblocking(False)
        # training reset clips the reference pose to the soft limits (src/supermimic/envs/mdp/commands.py:277-278)
        self.f0s = [np.clip(cl.joint_pos[0], c.soft_lower, c.soft_upper) for cl in self.clips]
        self.fends = [np.clip(cl.joint_pos[-1], c.soft_lower, c.soft_upper) for cl in self.clips]
        for f in self.f0s + self.fends:
            f[self.locked] = 0.0  # locked joints sit at 0 (Unitree lock-waist model); never commanded anyway
        # policy standby references: frame-0 pose before the motion, last-frame pose after it (all rates zero)
        self.static_pre = [static_clip(cl, 0) for cl in self.clips]
        self.static_post = [static_clip(cl, cl.num_frames - 1) for cl in self.clips]
        self.sb_win: collections.deque = collections.deque()  # STANDBY_PRE (t, tilt err [rad], max|dq| live) for the R1+X check
        self.temp_hist: collections.deque = collections.deque()  # STANDBY_PRE (t, per-motor max of the 2 sensors) for the forecast
        self.thermal: dict | None = None  # thermal stop (set once in POLICY): {frame, motor, temp, clip}
        self.temp_hr = np.zeros((len(c.joint_names), 2))  # --temp-headroom: per (motor, sensor) limit raise [C]
        for spec in args.temp_headroom or ():
            m, k, d = spec.split(":")
            self.temp_hr[int(m), int(k)] = float(d)
        self.pst_print = 0.0  # last POLICY temperature line
        # --lock-waist: policy standby only for clips on the lock-waist STANDBY allow list (empty by default)
        lw_sb = set(gc.LOCK_WAIST_STANDBY_ALLOWED_CLIPS) | set(args.lock_waist_standby_allow or ())
        self.standby_ok = [(not self.lock_waist) or cl.name in lw_sb for cl in self.clips]
        self.confirm: dict | None = None  # double-press confirmation: {kind, t, state_t}
        self.sbs: dict | None = None  # STANDBY_* status tracking (reset on every state entry)
        # max reference joint speed the policy sees in the first 0.5 s of each motion (goal dq, frames 1..25): info only
        self.ref_speed = []
        for cl in self.clips:
            dq = np.abs(cl.goal[1:26, 29:58]).astype(float)
            dq[:, self.locked] = 0.0
            k, j = np.unravel_index(int(np.argmax(dq)), dq.shape) if dq.size else (0, 0)
            self.ref_speed.append((float(dq.max()) if dq.size else 0.0, c.joint_names[j], int(k) + 1,
                                   float(np.abs(cl.goal[0, 29:58]).max())))
        self.policy_fn = OnnxPolicy(args.artifacts / "policy.onnx", threads=args.ort_threads)
        self.n_policy_calls = 0

        def counted(obs, goal):
            self.n_policy_calls += 1
            a = self.policy_fn(obs, goal)
            if args.sim_inject_nan_step is not None and self.n_policy_calls >= args.sim_inject_nan_step:
                a = a * np.nan
            return a

        self.policy = counted
        prof = dict(STAGES[args.stage])
        for k in ("max_vel_frac", "max_abs_action", "torque_clamp_factor", "max_tilt_hold"):
            v = getattr(args, k)
            if v is not None:
                prof[k] = v
        self.profile = prof
        self.max_tilt_hold = prof["max_tilt_hold"]
        self.safety = SafetyCfg(max_vel_frac=prof["max_vel_frac"], max_abs_action=prof["max_abs_action"],
                                torque_clamp_factor=prof["torque_clamp_factor"])
        self.stand_kp = STAND_KP.copy()
        self.stand_kd = STAND_KD.copy()
        self.stand_kd[12:15] = args.stand_waist_kd
        self.stand_kd[15:] = args.stand_arm_kd
        self.clip_idx = 0
        self.core: DeployCore | None = None
        self.end: dict | None = None  # END_HOLD blend (set in run)
        self.lock = threading.RLock()
        self.state = "PASSIVE"
        self.state_t = 0.0
        self.damp_reason: str | None = None
        self.events: list[dict] = []
        self.pending: collections.deque = collections.deque()  # (combo name, receive time)
        self.accept_buttons = False
        self.combo = gc.ComboDebounce(COMBOS, stable_n=args.button_stable_n, release_s=args.button_release_s)
        self.damp_req: str | None = None  # set by the DDS listener, executed by the command thread
        self.sig: int | None = None
        self.sig_r = None
        # latest lowstate (from the C++ bridge's shared memory)
        self.bridge: gb.Bridge | None = None
        self.snap_cmd = np.zeros(1, gb.SNAP_DTYPE)  # command-thread snapshot buffer
        self.snap_main = np.zeros(1, gb.SNAP_DTYPE)  # main-thread snapshot buffer
        self.bstat = np.zeros(1, gb.STATUS_DTYPE)  # command-thread bridge status buffer
        self.ring_cur = np.zeros(2, np.uint64)  # [next ring index, lost ring entries]
        self.ring_buf = np.zeros(gb.RING, gb.RING_DTYPE)
        self.pump_lock = threading.Lock()
        self.q_meas = np.zeros(N)  # latest measured q (logged for PASSIVE / DAMP commands)
        self.cmd_mode = bytes(gb.NM)
        self.bridge_bad: str | None = None
        self.last_state: gc.G1State | None = None
        self.t_rx = 0.0
        self.n_rx = 0
        self.n_crc_bad = 0
        self.cb_errors = 0
        self.cb_last_err = ""
        self.last_tick = None
        self.t_tick_change = 0.0
        self.tick_moving = False
        self.mm_latched: int | None = None
        # rt/lowcmd watch (foreign publishers), counted in a child process
        self.watcher: LowcmdWatcher | None = None
        self.self_delivery: float | None = None
        # command shared with the 500 Hz thread: (q_des or None=measured, kp, kd)
        self.cmd = (None, np.zeros(N), np.full(N, DAMP_KD))
        self.cmd_lock = threading.Lock()
        self.t_main_cmd = time.monotonic()
        self.last_cmd_q: np.ndarray | None = None
        self.n_sent = 0
        self.n_write_fail = 0
        self.write_fail_consec = 0
        self.write_last_err = ""
        self.n_cmd_exc = 0
        self.stop = threading.Event()  # ends the policy loop
        self.cmd_stop = threading.Event()  # ends the 500 Hz command thread (after the final damping second)
        self.cmd_started = False
        self.rt_info: dict = {}
        self.t0 = time.monotonic()
        self.wall0 = time.time()
        self._alloc_logs()

    # ------------------------------------------------------------------ logging
    def _alloc_logs(self):
        cap = int(self.a.cmd_log_max_s * self.a.cmd_hz * 1.15)  # + immediate sends after policy steps
        self.cl_cap = cap
        self.cl_t = np.zeros(cap)
        self.cl_q = np.zeros((cap, N), np.float32)
        self.cl_kp = np.zeros((cap, N), np.float32)
        self.cl_kd = np.zeros((cap, N), np.float32)
        self.cl_s = np.zeros(cap, np.int8)
        self.cl_n = 0
        self.rec_cap = int(self.a.rec_max_s / self.cfg.control_dt)
        self.rec_warned = False
        self.rec: dict[str, list] = {k: [] for k in (
            "t", "deadline", "compute_ms", "state", "clip", "frame", "q", "dq", "gyro", "quat", "tick", "state_age_ms",
            "action", "q_target", "n_clamped", "obs_hist", "cmd_kp_mean", "locked_temp", "locked_mstat", "tilt_err", "temp")}

    def event(self, kind: str, msg: str):
        """Thread-safe and non-blocking (only enqueues): callable from any thread."""
        t = time.monotonic() - self.t0
        self.events.append({"t": round(t, 4), "kind": kind, "msg": msg})
        line = f"[{t:8.3f}] {kind}: {msg}"
        if kind in ("ABORT", "DAMP"):
            line = "\n" + "!" * 80 + "\n" + line + "\n" + "!" * 80
        self.printer.q.put(line + "\n")

    # ------------------------------------------------------------------ state machine
    def set_state(self, new: str, reason: str = ""):
        with self.lock:
            if self.state == "DAMP":
                return False
            old = self.state
            self.state, self.state_t = new, time.monotonic()
        self.event("STATE", f"{old} -> {new}" + (f" ({reason})" if reason else ""))
        return True

    def damp(self, reason: str):
        """Main or command thread only (never the DDS listener / a signal handler). Sends before logging."""
        with self.lock:
            if self.state == "DAMP":
                return
            old = self.state
            self.state, self.state_t, self.damp_reason = "DAMP", time.monotonic(), reason
            with self.cmd_lock:
                self.cmd = (None, np.zeros(N), np.full(N, DAMP_KD))
        if self.cmd_started:
            try:
                self.send()
            except Exception:  # noqa: BLE001
                pass
        self.event("DAMP", f"{old} -> DAMP: {reason}")

    def set_cmd(self, q, kp, kd, expect_state: str):
        """Install new targets unless the state machine moved (e.g. to DAMP) meanwhile. Also the main-loop heartbeat."""
        with self.lock:
            if self.state != expect_state:
                return False
            qq = None if q is None else np.asarray(q, float).copy()
            with self.cmd_lock:
                self.cmd = (qq, np.asarray(kp, float).copy(), np.asarray(kd, float).copy())
            self.t_main_cmd = time.monotonic()
            if qq is not None:
                self.last_cmd_q = qq
        return True

    # ------------------------------------------------------------------ DDS (listener thread: flags only)
    def pump(self):
        """Every lowstate sample the bridge accepted since the last call (mode_machine check, button debouncer, exactly
        what the DDS listener did per message in the pure-Python runner), then the latest snapshot's receive/tick times.
        Command thread (every cycle) and startup only; never blocks."""
        with self.pump_lock:
            try:
                B = self.bridge
                n = B.read_ring(self.ring_cur, self.ring_buf)
                for e in self.ring_buf[:n]:
                    t = float(e["t"])
                    mm = int(e["mode_machine"])
                    if self.mm_latched is not None and mm != self.mm_latched and self.damp_req is None:
                        self.damp_req = f"mode_machine changed {self.mm_latched} -> {mm}"
                    fired = self.combo.update(int(e["keys"]), t)
                    if fired and self.accept_buttons:
                        for name in fired:
                            if name == "damp":
                                if self.damp_req is None:
                                    self.damp_req = "remote L2+B"
                            else:
                                self.pending.append((name, t))
                if B.read_snap(self.snap_cmd):
                    r = self.snap_cmd[0]
                    self.t_rx, self.t_tick_change, self.tick_moving = float(r["t_rx"]), float(r["t_tick_change"]), bool(r["tick_moving"])
                    self.n_rx, self.n_crc_bad = int(r["n_rx"]), int(r["n_crc_bad"])
                    self.q_meas = r["q"][:N].astype(float)
            except Exception as e:  # noqa: BLE001
                self.cb_errors += 1
                self.cb_last_err = repr(e)

    def read_state(self) -> gc.G1State | None:
        """Latest lowstate as gc.extract_state built it (main thread / startup)."""
        if not self.bridge.read_snap(self.snap_main) or self.snap_main["n_rx"][0] == 0:
            return self.last_state  # no consistent copy (writer kept the seqlock busy): previous state; stale check still runs
        r = self.snap_main[0]
        if float(r["t_rx"]) > self.t_rx:  # fresher than the command thread's last pump: same fields on_state kept
            self.t_rx, self.t_tick_change, self.tick_moving = float(r["t_rx"]), float(r["t_tick_change"]), bool(r["tick_moving"])
        self.last_state = gb.snap_to_state(self.snap_main)
        return self.last_state

    def state_age(self, t: float | None = None) -> float:
        t = time.monotonic() if t is None else t
        age = t - self.t_rx
        if self.tick_moving:
            age = max(age, t - self.t_tick_change)
        return age

    def build_cmd(self):
        # locked waist motors 13/14: mode 0 (disabled), never changed afterwards; dq/tau are never written for any motor
        return bytes((1 if i < N and self.live[i] else 0) for i in range(gb.NM))

    def send(self, now: bool = True) -> bool:
        """Hand the latest targets to the bridge (500 Hz thread: now=False, the bridge's own --cmd-hz stream publishes it;
        right after policy steps and in damp(): now=True, the bridge publishes at once)."""
        with self.cmd_lock:
            q, kp, kd = self.cmd
            flags = gb.CMD_NOW if now else 0
            if q is None:  # PASSIVE / DAMP: hold measured position (kp = 0 anyway); the bridge fills in its latest q
                flags |= gb.CMD_Q_MEASURED
                q = self.q_meas
            if self.lock_waist:  # motors 13/14: q = kp = kd = 0 in every state (mode 0 set in build_cmd)
                q, kp, kd = np.where(self.live, q, 0.0), np.where(self.live, kp, 0.0), np.where(self.live, kd, 0.0)
            z = np.zeros(gb.NM - N)
            ok = self.bridge.send(np.concatenate((q, z)), np.concatenate((kp, z)), np.concatenate((kd, z)), self.cmd_mode,
                                  self.mm_latched if self.mm_latched is not None else 0, flags)
            self.n_sent += 1
            if not ok:
                self.write_last_err = f"command to g1_bridge failed: {self.bridge.last_send_err}"
            k = self.cl_n % self.cl_cap
            self.cl_t[k] = time.monotonic() - self.t0
            self.cl_q[k], self.cl_kp[k], self.cl_kd[k], self.cl_s[k] = q, kp, kd, SID[self.state]
            self.cl_n += 1
            if self.cl_n == self.cl_cap:
                self.event("WARN", f"command log full ({self.a.cmd_log_max_s:.0f} s): ring buffer, keeping only the newest")
        return ok

    def _poll_signal(self) -> int | None:
        if self.sig_r is not None:
            try:
                b = os.read(self.sig_r, 64)
                if b:
                    return int(b[-1])
            except (BlockingIOError, InterruptedError):
                pass
            except OSError:
                pass
        return self.sig

    def _pin_cmd_thread(self):
        a, info = self.a, self.rt_info
        if a.cmd_cpu != "none":
            try:
                cpus = sorted(os.sched_getaffinity(0))
                cpu = cpus[-1] if a.cmd_cpu == "auto" else int(a.cmd_cpu)
                os.sched_setaffinity(threading.get_native_id(), {cpu})
                info["cmd_cpu"] = cpu
            except Exception as e:  # noqa: BLE001
                info["cmd_cpu_err"] = repr(e)
        if a.rt_prio > 0:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(a.rt_prio))
                info["cmd_sched"] = f"SCHED_FIFO {a.rt_prio}"
            except Exception as e:  # noqa: BLE001
                info["cmd_sched_err"] = repr(e)
        self.event("RT", f"command thread: {info}")

    def cmd_loop(self):
        a = self.a
        self._pin_cmd_thread()
        period = 1.0 / a.cmd_hz
        hb_lim = (a.max_overrun + 0.5) * self.cfg.control_dt
        dl = time.monotonic()
        W = self.watcher
        B = self.bridge
        bs = B.read_status(self.bstat)
        win_t, win_rx, win_sent, calib, w_dead_noted = dl, (W.count if W else 0), int(bs["n_pub"][0]), [], False
        win_rx_t = W.t if W else 0.0
        while not self.cmd_stop.is_set():
            dl += period
            try:
                sig = self._poll_signal()
                if sig is not None and self.state != "DAMP":
                    try:
                        name = signal.Signals(sig).name
                    except ValueError:
                        name = str(sig)
                    self.damp(f"signal {name}")
                self.pump()
                if self.damp_req is not None and self.state != "DAMP":
                    self.damp(self.damp_req)
                now = time.monotonic()
                bs = B.read_status(self.bstat)
                n_pub = int(bs["n_pub"][0])
                self.n_write_fail, self.write_fail_consec = int(bs["n_pub_fail"][0]), int(bs["fail_consec"][0])
                if self.bridge_bad is None:
                    if not B.alive():
                        self.bridge_bad = f"g1_bridge exited ({B.p.returncode}): NO rt/lowcmd any more"
                    elif bs["wd_fired"][0]:
                        self.bridge_bad = "g1_bridge watchdog fired (runner commands late): bridge publishes latched damping"
                    elif now - float(bs["t_hb"][0]) > a.bridge_hb_ms * 1e-3:
                        self.bridge_bad = f"g1_bridge publisher stalled {(now - float(bs['t_hb'][0])) * 1e3:.0f} ms"
                    if self.bridge_bad is not None:
                        self.event("ERROR", self.bridge_bad)
                        if self.state != "DAMP":
                            self.damp(self.bridge_bad)
                st = self.state
                if st != "DAMP":
                    age = self.state_age(now)
                    if age > a.stale_ms * 1e-3:
                        self.damp(f"stale lowstate: {age * 1e3:.1f} ms since last new state (limit {a.stale_ms} ms)")
                    elif st in HEARTBEAT_STATES and now - self.t_main_cmd > hb_lim:
                        self.damp(f"main loop stalled: no new command for {(now - self.t_main_cmd) * 1e3:.1f} ms in {st} "
                                  f"(limit {hb_lim * 1e3:.0f} ms)")
                # foreign rt/lowcmd publisher watch (child-process count of all rt/lowcmd samples vs our own sends)
                if W is not None:
                    W.poll()
                    if W.dead and not w_dead_noted:
                        w_dead_noted = True
                        self.event("WARN", "rt/lowcmd watcher process died: foreign-publisher watch OFF for the rest of the run")
                if W is not None and not W.dead and now - win_t >= 1.0:
                    d_rx, d_sent = W.count - win_rx, n_pub - win_sent
                    rx_dt, tx_dt = W.t - win_rx_t, now - win_t
                    win_t, win_rx, win_sent, win_rx_t = now, W.count, n_pub, W.t
                    if self.self_delivery is None:
                        calib.append(d_rx / max(d_sent, 1))
                        if len(calib) >= 2:
                            self.self_delivery = 1.0 if max(calib) > 0.5 else 0.0
                            self.event("INFO", f"rt/lowcmd self-delivery ratio {max(calib):.2f} -> {self.self_delivery:.0f} "
                                               "(foreign-publisher watch armed)")
                    else:
                        # Compare rates, each on its own clock: rx over the child's timestamps, tx over ours.
                        if rx_dt > 0.5:
                            rx_rate, tx_rate = d_rx / rx_dt, d_sent / max(tx_dt, 1e-3)
                            thr = tx_rate * 1.3 + 10 if self.self_delivery > 0 else 10
                            if rx_rate > thr and self.state != "DAMP":
                                self.damp(f"foreign rt/lowcmd publisher: {rx_rate:.0f} samples/s received vs {tx_rate:.0f}/s own sends")
                self.send(now=False)
                if self.write_fail_consec >= a.max_write_fail and self.state != "DAMP":
                    self.damp(f"{self.write_fail_consec} consecutive rt/lowcmd write failures ({self.write_last_err})")
            except Exception as e:  # noqa: BLE001
                self.n_cmd_exc += 1
                self.event("ERROR", f"command thread exception {e!r}: {traceback.format_exc(limit=3)}")
                try:
                    self.damp(f"command thread exception {e!r}")
                except Exception:  # noqa: BLE001
                    pass
            rem = dl - time.monotonic()
            if rem > 0:
                time.sleep(rem)
            elif rem < -10 * period:
                dl = time.monotonic()  # do not burst after a long stall

    # ------------------------------------------------------------------ startup
    def foreign_lowcmd(self, secs: float) -> int:
        n = self.watcher.count_for(secs)
        self.watcher.poll()
        if self.watcher.dead:
            raise SystemExit("rt/lowcmd watcher process died; cannot check for another rt/lowcmd publisher")
        return n

    def check_mode(self, when: str):
        a = self.a
        if not (a.domain == 0 or a.sim_try_check_mode):
            return
        try:
            code, name, raw = gc.check_motion_mode(a.check_mode_timeout)
        except Exception as e:  # noqa: BLE001
            code, name, raw = -1, None, repr(e)
        print(f"[startup] MotionSwitcher CheckMode ({when}): code={code} result={raw}", flush=True)
        if name:
            raise SystemExit(f"Unitree motion mode '{name}' is ACTIVE (MotionSwitcher CheckMode). Enter debug mode / release "
                             "the motion mode on the robot (this program never calls ReleaseMode) and retry.")
        if code != 0:
            msg = (f"MotionSwitcher CheckMode did not answer (code {code}); cannot confirm the built-in controller is off "
                   "(UNVERIFIED whether the service is visible from this PC). Relying on the rt/lowcmd check.")
            if a.require_check_mode:
                raise SystemExit(msg + " --require-check-mode is set.")
            print(f"[startup] WARNING {msg}", flush=True)

    def startup(self):
        a = self.a
        if a.domain == 0 or a.sim_try_check_mode:
            gc.init_dds(a.domain, a.iface)  # Python DDS only for the MotionSwitcher CheckMode RPC (no rt/low* topics)
        self.crc_check = a.domain == 0 and not a.no_state_crc
        self.bridge = gb.Bridge(a.domain, a.iface, crc=self.crc_check, hz=a.cmd_hz, wd_ms=a.bridge_wd_ms, damp_kd=DAMP_KD,
                                log_path=a.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.wall0))}.bridge.txt",
                                cpu=a.bridge_cpu, prio=a.bridge_prio, orphan_s=a.bridge_orphan_s)
        print(f"[startup] g1_bridge pid {self.bridge.p.pid}: {' '.join(self.bridge.args)}", flush=True)
        self.watcher = gb.BridgeWatcher(self.bridge)  # the bridge's rt/lowcmd watch reader (all publishers)
        t_end = time.monotonic() + a.wait_state_s
        while self.n_rx == 0 and time.monotonic() < t_end:
            self.pump()
            time.sleep(0.01)
        if self.n_rx == 0:
            raise SystemExit(f"no rt/lowstate within {a.wait_state_s} s on iface={a.iface} domain={a.domain} "
                             f"(CRC failures: {self.n_crc_bad})")
        n0 = self.n_rx
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            self.pump()
            time.sleep(0.02)
        rate = self.n_rx - n0
        s = self.read_state()
        print(f"[startup] lowstate rate {rate} Hz, mode_machine={s.mode_machine} mode_pr={s.mode_pr} tick={s.tick} "
              f"CRC bad={self.n_crc_bad}", flush=True)
        # accepted: {5 without --lock-waist, 6 with --lock-waist}; both refusals below also apply in --sim
        mm, ML, MF = s.mode_machine, gc.MODE_MACHINE_LOCK_WAIST, gc.MODE_MACHINE_FULL
        if mm == ML and not self.lock_waist:
            raise SystemExit(f"mode_machine={mm} = g1_29dof_lock_waist_rev_1_0 (waist roll/pitch LOCKED, motors 13/14 absent). "
                             "Restart with --lock-waist (and a lock-waist checkpoint/clip, see docs/DEPLOY_G1.md#waist-locked-g1-mode_machine--6). Abort.")
        if self.lock_waist and mm != ML:
            raise SystemExit(f"--lock-waist given but mode_machine={mm} (lock-waist G1 reports {ML}"
                             + (f"; {MF} = 3-DoF waist: run WITHOUT --lock-waist" if mm == MF else "") + "). Abort.")
        if not self.lock_waist and mm != MF:
            if a.sim:
                print(f"[startup] WARNING mode_machine {mm} != {MF} (ignored in --sim for an unlocked run)")
            else:
                raise SystemExit(f"mode_machine={mm}, expected {MF} (29-DoF G1 rev 1.0). Abort.")
        if self.lock_waist:
            print(f"[startup] WAIST LOCKED: motors 13/14 now {gc.waist_locked_values(s)}", flush=True)
            wa = gc.waist_active(s)
            if wa:
                raise SystemExit(f"--lock-waist but the waist looks ACTIVE (|q| > {gc.WAIST_ACTIVE_Q} rad or temperature != 0): {wa}. Abort.")
        self.mm_latched = s.mode_machine
        # nobody else may be commanding the motors
        if not self.watcher.wait_ready(10.0):
            raise SystemExit("rt/lowcmd watcher process did not start (cannot check for other controllers)")
        n = self.foreign_lowcmd(a.lowcmd_check_s)
        if n:
            raise SystemExit(f"{n} rt/lowcmd messages from ANOTHER publisher in {a.lowcmd_check_s} s "
                             "(sport/locomotion service or another controller is active). Put the robot in debug mode "
                             "(see docs/DEPLOY_G1.md#g1-model-ai-mode-and-debug-mode) and retry.")
        print(f"[startup] no other rt/lowcmd publisher seen in {a.lowcmd_check_s} s", flush=True)
        self.check_mode("before confirmation")
        if self.crc_check and self.n_crc_bad:
            print(f"[startup] WARNING {self.n_crc_bad} lowstate CRC failures so far (those messages are dropped)")
        tmax = self.teff(s.temp)[self.live].max()
        if tmax > a.max_temp:
            raise SystemExit(f"motor temperature {tmax:.0f} C > --max-temp {a.max_temp}")
        # ORT warm-up (first inference allocates; keep it out of the 50 Hz loop)
        for _ in range(3):
            self.policy_fn(np.zeros(930), np.zeros(680))
        gov = _governor()
        g = projected_gravity(s.quat / max(np.linalg.norm(s.quat), 1e-9))
        f0 = self.f0s[0]
        n_soft = [int(np.count_nonzero((np.abs(f - cl.joint_pos[0]) > 1e-9) & self.live)) for f, cl in zip(self.f0s, self.clips)]
        print("=" * 80)
        print(f"artifacts : {a.artifacts}")
        print(f"clips     : {[c.name + f' ({c.num_frames} fr, {c.num_frames * self.cfg.control_dt:.1f} s)' for c in self.clips]}")
        print(f"start pose: clip frame 0 clipped to soft limits ({n_soft} joints clipped per clip)")
        for cl, (v, jn, fr, v0) in zip(self.clips, self.ref_speed):
            print(f"ref speed : {cl.name}: max reference joint speed over frames 1-25 {v:.2f} rad/s ({jn}, frame {fr}); "
                  f"frame 0 {v0:.2f} rad/s (info only)")
        print(f"iface/dom : {a.iface} / {a.domain}  {'SIM' if a.sim else 'REAL ROBOT'}  auto-sequence={a.auto_sequence}")
        if self.lock_waist:
            print("#" * 80)
            print(f"WAIST LOCK: --lock-waist, mode_machine {self.mm_latched} (g1_29dof_lock_waist_rev_1_0). Motors 13 waist_roll /")
            print("            14 waist_pitch: lowcmd mode=0 q=dq=tau=kp=kd=0 in EVERY state; excluded from start/pose/temperature/")
            print(f"            motor-error/dq checks; |q| > {gc.WAIST_ACTIVE_Q} rad or temperature != 0 on them -> DAMP 'waist unexpectedly active'.")
            print("            Expect the pelvis to lean 7-10 deg forward during POLICY (sim2sim, UNVERIFIED on hardware).")
            for n in self.clip_notes:
                print(f"            clip {n}")
            print("            clip end -> END_HOLD (--end-hold-legacy implied: STANDBY_POST is not sim-gated on this model)")
            for cl, ok in zip(self.clips, self.standby_ok):
                how = ("policy standby allowed (lock-waist STANDBY allow list)" if ok else
                       ("NOT on the lock-waist STANDBY allow list -> R1+A in HOLD STARTS THE MOTION DIRECTLY (--lock-waist-direct-start)"
                        if a.lock_waist_direct_start else
                        "NOT on the lock-waist STANDBY allow list -> R1+A/R1+X REFUSED (no motion; --lock-waist-direct-start = old flow)"))
                print(f"            {cl.name}: {how}")
            print("#" * 80)
        print(f"control   : policy {1 / self.cfg.control_dt:.0f} Hz, lowcmd {a.cmd_hz:.0f} Hz, mode_pr=0 (PR), "
              f"mode_machine latched {self.mm_latched}, ORT threads {a.ort_threads}")
        print(f"STAGE     : {a.stage.upper()}  -> max_vel_frac {self.safety.max_vel_frac}, max_abs_action {self.safety.max_abs_action}, "
              f"torque_clamp_factor {self.safety.torque_clamp_factor} x effort, clamp_soft_limits {self.safety.clamp_soft_limits}, "
              f"core tilt err {np.degrees(self.safety.max_tilt_err):.0f} deg, hold tilt {np.degrees(self.max_tilt_hold):.0f} deg")
        if a.no_policy:
            print("NO-POLICY : --no-policy -> R1+A is ALWAYS refused; this run only does PASSIVE / INTERP / HOLD / DAMP")
        print(f"stand kp  : {self.stand_kp.tolist()}")
        print(f"stand kd  : {self.stand_kd.tolist()}")
        if a.end_hold_legacy:
            print(f"clip end  : {'--lock-waist implies ' if self.lock_waist else ''}--end-hold-legacy -> END_HOLD, {a.end_blend_s} s blend to stand gains (CANNOT balance: hoist must carry)")
        else:
            print(f"clip end  : STANDBY_POST (policy keeps standing in the last-frame pose); L2+up TWICE within {a.confirm_s} s -> "
                  f"{a.end_blend_s} s blend to stand gains -> INTERP to the next clip (hoist must carry)")
        print(f"safety    : stale>{a.stale_ms} ms, overrun>{a.max_overrun} cycles, main-loop heartbeat>{(a.max_overrun + 0.5) * 20:.0f} ms, "
              f"temp>{a.max_temp} C DAMP / >={a.thermal_stop_temp} C in POLICY smooth thermal stop / >{a.warn_temp} C no start (UNVERIFIED thresholds), motor-error mask 0x{a.motor_error_mask:x}")
        for j, k in zip(*np.nonzero(self.temp_hr)):
            print(f"headroom  : {self.cfg.joint_names[j]} sensor {k}: every temperature limit +{self.temp_hr[j, k]:.0f} C (--temp-headroom)")
        print(f"R1+A req  : HOLD>={a.hold_min_s} s, max|q-f0|<{a.hold_tol} rad, max|dq|<{a.start_max_dq} rad/s, "
              f"tilt vs clip<{a.start_max_tilt_deg} deg, temp<{a.warn_temp} C  -> STANDBY_PRE (policy stands still)")
        print(f"R1+X req  : STANDBY_PRE>={a.standby_min_s} s; over the last {a.standby_window_s} s tilt err<{a.standby_max_tilt_deg} deg, "
              f"max|dq|<{a.standby_max_dq} rad/s; temp<{a.warn_temp} C; heating forecast at clip end < {a.thermal_stop_temp} C "
              f"(rate over the last {a.temp_rate_window_s:.0f} s, >= {a.temp_rate_min_s:.0f} s needed)  -> POLICY (motion)")
        print(f"standby   : status every {a.status_s} s; WARN |yaw change|>{a.warn_yaw_deg} deg, leg |dq|>{a.warn_leg_dq} rad/s for "
              f">{a.warn_leg_dq_s} s, temp>{a.warn_temp} C, standby>{a.standby_warn_s:.0f} s (documented cap)")
        print(f"cpu       : governor {gov}; command thread affinity '{a.cmd_cpu}', rt-prio {a.rt_prio}")
        print(f"robot now : proj_grav={np.round(g, 3)}  max|q - f0|={np.abs(s.q - f0)[self.live].max():.3f} rad  max temp={tmax:.0f} C"
              + (" (27 live motors)" if self.lock_waist else ""))
        print("buttons   : L2+up = interpolate to f0 (PASSIVE; END_HOLD -> next clip); in STANDBY_PRE (-> HOLD, same clip) and")
        print(f"            STANDBY_POST (-> next clip) it must be pressed TWICE within {a.confirm_s} s (stand gains: hoist must carry) |")
        print("            R1+A = policy STANDBY (from HOLD) | R1+X = start the MOTION (from STANDBY_PRE) | L2+B = DAMP (any state)")
        print("            (buttons pressed before the command thread starts are ignored; release and press again)")
        print("=" * 80, flush=True)
        if tmax > a.warn_temp:
            print(f"[startup] WARNING max motor temperature {tmax:.0f} C > --warn-temp {a.warn_temp}: R1+A will be refused")
        if not a.sim or a.sim_ask_yes:
            ans = input("Robot on the hoist / area clear / L2+B tested this session? Type 'yes' to start sending rt/lowcmd "
                        "(PASSIVE damping): ")
            if ans.strip() != "yes":
                raise SystemExit("not confirmed")
            # the prompt may have waited minutes: re-check everything that could have changed
            n = self.foreign_lowcmd(a.lowcmd_recheck_s)
            if n:
                raise SystemExit(f"after 'yes': {n} rt/lowcmd messages from ANOTHER publisher in {a.lowcmd_recheck_s} s")
            self.check_mode("after confirmation")
            self.pump()
            s2 = self.read_state()
            if self.teff(s2.temp)[self.live].max() > a.max_temp:
                raise SystemExit(f"after 'yes': motor temperature {self.teff(s2.temp)[self.live].max():.0f} C (minus headroom) > --max-temp {a.max_temp}")
            if self.lock_waist and gc.waist_active(s2):
                raise SystemExit(f"after 'yes': waist unexpectedly active: {gc.waist_active(s2)}")
            if s2.mode_machine != self.mm_latched:
                raise SystemExit(f"after 'yes': mode_machine changed {self.mm_latched} -> {s2.mode_machine}")
            self.pump()
            if self.state_age() > a.stale_ms * 1e-3:
                raise SystemExit(f"after 'yes': lowstate stale ({self.state_age() * 1e3:.0f} ms)")
            print(f"[startup] re-check after 'yes' OK (no foreign rt/lowcmd in {a.lowcmd_recheck_s} s)", flush=True)
        if not a.lowcmd_watch:
            self.watcher.close()
            self.watcher = None
        self.cmd_mode = self.build_cmd()  # the bridge starts its rt/lowcmd stream on our first command
        self.cmd_thread = threading.Thread(target=self.cmd_loop, name="lowcmd", daemon=True)
        # buttons: nothing pressed so far may act (the debouncer also needs a release before a combo fires)
        self.pending.clear()
        self.accept_buttons = True
        if self.sig_r is not None:  # drop signals that arrived during startup (they raised KeyboardInterrupt)
            self._poll_signal()
        # full collection now (it holds the GIL for ~10 ms), then move everything to the permanent generation
        pygc.collect()
        pygc.freeze()
        self.pump()
        self.ring_cur[1] = 0  # count lost ring entries from here on (the 'yes' prompt / warm-up skip some by design)
        self.state_t = time.monotonic()
        self.t_main_cmd = time.monotonic()
        self.cmd_started = True
        self.cmd_thread.start()
        self.event("STATE", f"start in PASSIVE (stage {a.stage})")
        if a.sim and a.sim_band_file:
            Path(a.sim_band_file).touch()
            self.event("SIM", f"elastic band ON ({a.sim_band_file})")

    # ------------------------------------------------------------------ main loop
    def run(self):
        a = self.a
        c = self.cfg
        dt = c.control_dt
        dl = time.monotonic()
        self.band_off_t = None
        interp_from = None
        self.end = None  # END_HOLD blend: dict(q0, f_end, then_interp)
        auto_damp_t = None
        stalled = False
        eff_lim = min(1.0, self.safety.torque_clamp_factor) * c.effort_limit
        while not self.stop.is_set():
            dl += dt
            rem = dl - time.monotonic()
            if rem > 0:
                time.sleep(rem)
            if self.stop.is_set():
                break
            t_start = time.monotonic()
            late = t_start - dl
            if late > a.max_overrun * dt and self.state not in ("DAMP",):
                self.damp(f"policy loop overrun: {late * 1e3:.1f} ms late (> {a.max_overrun} cycles)")
            if (a.sim_stall_at_s is not None and not stalled and self.state == "POLICY"
                    and t_start - self.state_t >= a.sim_stall_at_s):
                stalled = True
                self.event("SIM", f"injecting a {a.sim_stall_s} s main-loop stall in POLICY")
                time.sleep(a.sim_stall_s)
                dl = time.monotonic()
                continue
            st = self.read_state()
            age = self.state_age(t_start)
            state = self.state
            t_in = t_start - self.state_t
            clip = self.clips[self.clip_idx]
            f0 = self.f0s[self.clip_idx]
            action = np.zeros(N)
            q_t = np.full(N, np.nan)
            n_cl = 0
            obs = None
            frame = -1
            tilt_err = np.nan

            # --- checks valid in every non-damp state
            if state != "DAMP":
                merr = np.where(self.live, st.motorstate & a.motor_error_mask, 0)  # locked motors: logged, not checked
                temp = np.where(self.live[:, None], self.teff(st.temp), -np.inf)
                wa = gc.waist_active(st) if self.lock_waist else None
                if wa:
                    self.damp(f"waist unexpectedly active (--lock-waist, |q| > {gc.WAIST_ACTIVE_Q} rad or temperature != 0): {wa}")
                elif temp.max() > a.max_temp:
                    j, k = (int(x) for x in np.unravel_index(np.argmax(temp), temp.shape))
                    self.damp(f"motor temperature {st.temp[j, k]:.0f} C on {c.joint_names[j]} sensor {k} > {a.max_temp + self.temp_hr[j, k]:.0f} C")
                elif a.domain == 0 and np.any(merr != 0):
                    j = int(np.argmax(merr != 0))
                    self.damp(f"motor error word 0x{int(st.motorstate[j]):x} (masked 0x{int(merr[j]):x}) on {c.joint_names[j]}")
                elif not np.all(np.isfinite(st.q)) or not np.all(np.isfinite(st.quat)) or not np.all(np.isfinite(st.gyro)):
                    self.damp("non-finite lowstate")
                elif abs(float(np.linalg.norm(st.quat)) - 1.0) > 0.05:
                    self.damp(f"bad IMU quaternion (norm {float(np.linalg.norm(st.quat)):.3f})")
                elif state in ("INTERP", "HOLD", "END_HOLD"):
                    tilt = _tilt(st.quat)
                    if tilt > self.max_tilt_hold:
                        self.damp(f"tilt {np.degrees(tilt):.0f} deg in {state} > {np.degrees(self.max_tilt_hold):.0f} deg")
            state = self.state

            # --- events (buttons or auto-sequence); stale presses (main loop was late) are dropped
            evs = []
            while self.pending:
                name, t_ev = self.pending.popleft()
                if t_start - t_ev > a.button_max_age_s:
                    self.event("INFO", f"dropped stale button event {name} ({(t_start - t_ev) * 1e3:.0f} ms old)")
                else:
                    evs.append(name)
            if a.auto_sequence and state != "DAMP":
                evs += self.auto_events(state, t_in, st)
            for ev in evs:
                if ev == "interp":
                    if state not in ("PASSIVE", "END_HOLD", "STANDBY_PRE", "STANDBY_POST"):
                        self.event("INFO", f"L2+up ignored in {state}")
                        continue
                    if state in ("END_HOLD", "STANDBY_POST") and self.thermal is not None:
                        self.event("INFO", f"L2+up in {state} ignored after the thermal stop (no next clip: motors too hot); "
                                   "re-hoist (gantry over the robot, take up slack only), then L2+B")
                        continue
                    if state in ("END_HOLD", "STANDBY_POST") and self.clip_idx + 1 >= len(self.clips) and not (
                            state == "END_HOLD" and self.end is not None and self.end.get("then_hold")):
                        self.event("INFO", f"L2+up in {state}: no next clip (all clips played)"
                                   + ("; the policy keeps standing: re-hoist (gantry over the robot, take up slack only), then L2+B"
                                      if state == "STANDBY_POST" else ""))
                        continue
                    if state in STANDBY_STATES:
                        # leaving the policy needs a 2nd L2+up within --confirm-s (L2+up is the habitual first combo of a session)
                        if not self.confirmed("interp", t_start, f"press L2+up again within {a.confirm_s:.0f} s: is the hoist carrying "
                                              f"the robot? ({'back to HOLD of ' + clip.name if state == 'STANDBY_PRE' else 'next clip ' + self.clips[self.clip_idx + 1].name}"
                                              "; stand gains from the 2nd press on: the robot CANNOT balance by itself)"):
                            continue
                        # same gain/target blend as the legacy END_HOLD; STANDBY_POST: then INTERP to the next clip's f0;
                        # STANDBY_PRE: target = this clip's f0, then HOLD of the same clip (clip_idx unchanged)
                        pre = state == "STANDBY_PRE"
                        self.end = dict(q0=(self.last_cmd_q if self.last_cmd_q is not None else st.q).copy(),
                                        f_end=(f0 if pre else self.fends[self.clip_idx]), then_interp=not pre, then_hold=pre)
                        self.set_state("END_HOLD", f"L2+up x2 from {state}: {a.end_blend_s} s blend to stand gains, then "
                                       + (f"HOLD of {clip.name} (same clip)" if pre else f"INTERP to f0 of {self.clips[self.clip_idx + 1].name}"))
                        self.event("WARN", "stand gains from now on: the robot CANNOT balance by itself -> the HOIST MUST CARRY IT")
                        if not a.sim_no_band_end_hold:
                            self.sim_band_on(f"END_HOLD from {state}: stand gains cannot balance")
                        state, t_in = "END_HOLD", 0.0
                        continue
                    if state == "END_HOLD":
                        if self.end is not None and (self.end.get("then_interp") or self.end.get("then_hold")):
                            self.event("INFO", "L2+up in END_HOLD ignored: " + ("HOLD of the same clip" if self.end.get("then_hold") else
                                       "INTERP to the next clip") + " follows the blend automatically")
                            continue
                        if t_in < a.end_blend_s:
                            self.event("INFO", f"L2+up in END_HOLD ignored: gain blend still running ({t_in:.2f} < {a.end_blend_s} s)")
                            continue
                        self.clip_idx += 1
                        clip, f0 = self.clips[self.clip_idx], self.f0s[self.clip_idx]
                        # continue from the target currently commanded (stand gains): no jump to the measured q
                        interp_from = self.last_cmd_q.copy() if self.last_cmd_q is not None else st.q.copy()
                    else:
                        interp_from = st.q.copy()  # PASSIVE: kp = 0, the measured q is the only step-free start
                    self.set_state("INTERP", f"to f0 of {clip.name}")
                    state, t_in = "INTERP", 0.0
                elif ev == "policy":
                    if state == "STANDBY_PRE":
                        self.event("INFO", "R1+A ignored in STANDBY_PRE (the policy is already standing): press R1+X to start the motion")
                        continue
                    if state != "HOLD":
                        self.event("INFO", f"R1+A ignored in {state}")
                        continue
                    ok, why = self.policy_allowed(t_in, st, f0, clip)
                    if not ok:
                        self.event("REFUSED", f"R1+A (policy standby): {why}")
                        continue
                    if not self.standby_ok[self.clip_idx]:
                        if not a.lock_waist_direct_start:
                            self.event("REFUSED", f"R1+A: {self.lw_standby_why(clip)}; this run cannot start {clip.name} (restart with "
                                                  "--lock-waist-direct-start for the pre-standby flow, hoist carrying)")
                            continue
                        # --lock-waist-direct-start: the pre-standby flow (docs/DEPLOY_G1.md#waist-locked-g1-mode_machine--6): motion from frame 0, hoist carrying
                        if a.sim_band_file and self.band_off_t is None and a.sim_band_release_s < 0:
                            self.release_band("at POLICY start (direct start)")
                        self.core = DeployCore(c, self.policy, clip, self.safety)
                        self.core.reset(0)
                        self.set_state("POLICY", f"motion {clip.name} from frame 0 (--lock-waist-direct-start: no policy standby); {why}")
                        state, t_in = "POLICY", 0.0
                        continue
                    if a.sim_band_file and self.band_off_t is None and a.sim_band_release_standby_s < 0:
                        self.release_band("at STANDBY_PRE start (policy takes over)")
                    self.core = DeployCore(c, self.policy, self.static_pre[self.clip_idx], self.safety)
                    self.core.reset(0)
                    self.sb_win.clear()
                    self.temp_hist.clear()
                    self.set_state("STANDBY_PRE", f"policy standing still at frame 0 of {clip.name}; {why}; crew: feet carry the weight, rope just taut (only give the slack the clip needs), gantry ready, "
                                                  f"then R1+X (>= {a.standby_min_s:.0f} s, steady {a.standby_window_s:.0f} s)")
                    state, t_in = "STANDBY_PRE", 0.0
                elif ev == "start":
                    if state != "STANDBY_PRE":
                        hint = " (press R1+A first: policy standby)" if state == "HOLD" else ""
                        self.event("INFO", f"R1+X ignored in {state}{hint}")
                        continue
                    ok, why = self.standby_allowed(t_in, st)
                    if not ok:
                        self.event("REFUSED", f"R1+X (motion start): {why}")
                        continue
                    self.core.switch_clip(clip, 0)  # keeps observation history and last action
                    self.set_state("POLICY", f"motion {clip.name} from frame 0; {why}")
                    state, t_in = "POLICY", 0.0

            # --- per-state targets
            t_c0 = time.monotonic()
            if state == "PASSIVE":
                self.set_cmd(None, np.zeros(N), np.full(N, DAMP_KD), "PASSIVE")
            elif state == "INTERP":
                q_t = interpolate_pose(interp_from, f0, t_in, a.interp_s)
                # soft limits, but never pull the start point (outside the soft range) inwards in one step
                q_t = np.clip(q_t, np.minimum(c.soft_lower, interp_from), np.maximum(c.soft_upper, interp_from))
                self.set_cmd(q_t, self.stand_kp, self.stand_kd, "INTERP")
                if t_in >= a.interp_s:
                    self.set_state("HOLD", f"max|q-f0|={np.abs(st.q - f0)[self.live].max():.3f}")
            elif state == "HOLD":
                q_t = f0.copy()
                self.set_cmd(q_t, self.stand_kp, self.stand_kd, "HOLD")
                if a.sim_band_file and self.band_off_t is None and 0 <= a.sim_band_release_s <= t_in:
                    self.release_band("in HOLD")
            elif state in POLICY_STATES:
                core = self.core
                if state == "STANDBY_PRE" and a.sim_band_file and self.band_off_t is None and 0 <= a.sim_band_release_standby_s <= t_in:
                    self.release_band(f"{a.sim_band_release_standby_s:.1f} s into STANDBY_PRE")
                if state == "POLICY" and self.thermal is None and a.thermal_stop_temp > 0:
                    te = np.where(self.live[:, None], self.teff(st.temp), -np.inf)
                    if float(te.max()) >= a.thermal_stop_temp:
                        j, k = (int(x) for x in np.unravel_index(np.argmax(te), te.shape))
                        self.thermal_stop(core, clip, j, float(st.temp[j, k]), k)
                # the quantity DeployCore's tilt abort compares with max_tilt_err (pelvis gravity vs reference), for the log
                tilt_err = _tilt_vs(st.quat, core.clip.proj_grav[min(core.frame, core.clip.num_frames - 1)])
                r = core.step(st.q, st.dq, st.gyro, st.quat, advance=(state == "POLICY"))
                action, q_t, n_cl, frame = r.action, r.q_target, r.n_clamped, r.frame
                if state == "STANDBY_PRE":
                    self.sb_win.append((t_start, tilt_err, float(np.abs(st.dq)[self.live].max())))
                    while self.sb_win and t_start - self.sb_win[0][0] > a.standby_window_s:
                        self.sb_win.popleft()
                    self.temp_hist.append((t_start, self.teff(st.temp).max(1)))
                    while self.temp_hist and t_start - self.temp_hist[0][0] > a.temp_rate_window_s:
                        self.temp_hist.popleft()
                if state in STANDBY_STATES:
                    self.standby_status(state, t_in, t_start, st, tilt_err, age)
                elif t_start - self.pst_print >= a.status_s:
                    self.pst_print = t_start
                    self.printer.q.put(f"[{t_start - self.t0:8.3f}] STATUS: POLICY {t_in:6.1f} s frame {core.frame}"
                                       + (" (thermal stop)" if self.thermal is not None else "") + f": {self.temp_line(st)}\n")
                if a.log_obs and core.hist is not None:
                    obs = core.hist.copy()
                if r.abort:
                    self.damp(f"DeployCore abort in {state} at frame {r.frame}: {r.abort}")
                elif self.set_cmd(q_t, c.kp, c.kd, state):
                    self.send()
                    if state == "POLICY" and r.done and self.thermal is not None:
                        th = self.thermal
                        why = (f"thermal stop done ({c.joint_names[th['motor']]} {th['temp']:.0f} C at frame {th['frame']}); "
                               "NO next clip: re-hoist (gantry over the robot, take up slack only), then L2+B")
                        if a.end_hold_legacy:
                            self.end = dict(q0=q_t.copy(), f_end=f0, then_interp=False)
                            self.set_state("END_HOLD", why + " (--end-hold-legacy: stand gains, the HOIST MUST CARRY)")
                            if not a.sim_no_band_end_hold:
                                self.sim_band_on("END_HOLD after the thermal stop")
                        else:
                            core.switch_clip(static_clip(th["clip"], th["clip"].num_frames - 1), 0)
                            self.set_state("STANDBY_POST", f"policy keeps standing in the thermal-stop pose ({a.thermal_stop_pose}); " + why)
                    elif state == "POLICY" and r.done:
                        if a.end_hold_legacy:
                            self.end = dict(q0=q_t.copy(), f_end=self.fends[self.clip_idx], then_interp=False)
                            self.set_state("END_HOLD", f"clip {clip.name} done at frame {r.frame}; blending gains over {a.end_blend_s} s "
                                                       "(--end-hold-legacy)")
                            if not a.sim_no_band_end_hold:
                                self.sim_band_on("END_HOLD: a stiff-PD stand cannot balance on its own")
                        else:
                            core.switch_clip(self.static_post[self.clip_idx], 0)
                            self.set_state("STANDBY_POST", f"clip {clip.name} done at frame {r.frame}; policy keeps standing in the "
                                                           "last-frame pose (re-hoist before L2+B"
                                                           + ("; L2+up = next clip" if self.clip_idx + 1 < len(self.clips) else "") + ")")
            elif state == "END_HOLD":
                T = a.end_blend_s
                s = 1.0 if T <= 0 else float(np.clip(t_in / T, 0.0, 1.0))
                kp = (1.0 - s) * c.kp + s * self.stand_kp
                kd = (1.0 - s) * c.kd + s * self.stand_kd
                q_t = self.end["f_end"].copy() if T <= 0 else interpolate_pose(self.end["q0"], self.end["f_end"], t_in, T)
                q_t = np.clip(q_t, st.q - eff_lim / kp, st.q + eff_lim / kp)
                if self.set_cmd(q_t, kp, kd, "END_HOLD") and t_in >= T:
                    if self.end.get("then_interp"):
                        self.clip_idx += 1
                        interp_from = q_t.copy()  # the target just commanded with stand gains: no step
                        self.set_state("INTERP", f"to f0 of {self.clips[self.clip_idx].name} (after the STANDBY_POST blend)")
                    elif self.end.get("then_hold"):
                        # back to HOLD of the SAME clip (clip_idx unchanged); if the blend target was clamped (torque limit),
                        # interpolate from the commanded target instead of stepping to f0
                        if float(np.abs(q_t - f0)[self.live].max()) < 1e-6:
                            self.set_state("HOLD", f"{clip.name} (after the STANDBY_PRE exit blend), max|q-f0|={np.abs(st.q - f0)[self.live].max():.3f}")
                        else:
                            interp_from = q_t.copy()
                            self.set_state("INTERP", f"to f0 of {clip.name} (after the STANDBY_PRE exit blend; target was clamped)")
            elif state == "DAMP":
                if auto_damp_t is None:
                    auto_damp_t = t_start
                if a.sim and a.sim_exit_after_damp is not None and t_start - auto_damp_t >= a.sim_exit_after_damp:
                    self.event("SIM", f"exit {a.sim_exit_after_damp} s after DAMP (--sim-exit-after-damp)")
                    self.stop.set()
            compute_ms = (time.monotonic() - t_c0) * 1e3

            # --- record (50 Hz)
            R = self.rec
            R["t"].append(t_start - self.t0); R["deadline"].append(dl - self.t0); R["compute_ms"].append(compute_ms)
            R["state"].append(SID[self.state]); R["clip"].append(self.clip_idx); R["frame"].append(frame)
            R["q"].append(st.q); R["dq"].append(st.dq); R["gyro"].append(st.gyro); R["quat"].append(st.quat)
            R["tick"].append(st.tick); R["state_age_ms"].append(age * 1e3)
            R["action"].append(action); R["q_target"].append(q_t); R["n_clamped"].append(n_cl)
            R["cmd_kp_mean"].append(float(np.mean(self.cmd[1])))
            R["tilt_err"].append(tilt_err)
            R["temp"].append(st.temp.astype(np.float32))
            if self.sync_sock is not None:
                # thermal stop: the stop reference has its own frame numbers -> the video freezes on the frame it left
                th = self.thermal if self.state == "POLICY" else None
                lbl, fr = ("THERMAL_STOP", th["frame"]) if th is not None else (self.state, frame)
                try:
                    self.sync_sock.sendto(f"{lbl} {self.clip_idx} {fr} {t_start - self.t0:.3f}".encode(), self.sync_addr)
                except OSError:
                    pass
            if self.lock_waist:  # q/dq of 13/14 are in "q"/"dq" (raw lowstate)
                R["locked_temp"].append(st.temp[self.locked].copy()); R["locked_mstat"].append(st.motorstate[self.locked].copy())
            if a.log_obs:
                if len(R["t"]) <= self.rec_cap:
                    R["obs_hist"].append(obs.reshape(-1).astype(np.float32) if obs is not None else np.zeros(930, np.float32))
                elif not self.rec_warned:
                    self.rec_warned = True
                    self.event("WARN", f"--rec-max-s {a.rec_max_s} reached: obs_hist logging stopped (other logs continue)")
            if time.monotonic() - dl > dt * 10:
                dl = time.monotonic()  # resync after a long stall (already damped by the overrun check)

    def release_band(self, when: str):
        try:
            os.remove(self.a.sim_band_file)
        except FileNotFoundError:
            pass
        self.band_off_t = time.monotonic()
        self.event("SIM", f"elastic band OFF ({when})")

    def sim_band_on(self, why: str):
        if not self.a.sim_band_file:
            return
        Path(self.a.sim_band_file).touch()
        self.band_off_t = None
        self.event("SIM", f"elastic band ON ({why})")

    def teff(self, temp: np.ndarray) -> np.ndarray:
        """Temperatures (29, 2) as the limit checks see them: raw minus the --temp-headroom of that sensor."""
        return temp - self.temp_hr

    def thermal_stop(self, core: DeployCore, clip, motor: int, temp: float, sensor: int = 0):
        """POLICY, a motor reached --thermal-stop-temp: hand the policy a reference that stops in --thermal-stop-pose."""
        a = self.a
        frame = min(core.frame, core.clip.num_frames - 1)
        n = max(2, int(round(a.thermal_stop_s / self.cfg.control_dt)))
        target = self.static_pre[self.clip_idx] if a.thermal_stop_pose == "frame0" else static_clip(core.clip, frame)
        if a.thermal_stop_pose == "upright":
            target.goal[:, WAIST_IDX] = self.static_pre[self.clip_idx].goal[0, WAIST_IDX]
            target.joint_pos[:, WAIST_IDX] = self.static_pre[self.clip_idx].joint_pos[0, WAIST_IDX]
        stop = thermal_stop_clip(core.clip, frame, target, n, self.cfg.control_dt)
        core.switch_clip(stop, 0)  # keeps observation history and last action
        self.thermal = {"frame": frame, "motor": motor, "temp": temp, "clip": stop}
        self.event("WARN", f"THERMAL STOP: {self.cfg.joint_names[motor]} sensor {sensor} {temp:.0f} C >= "
                           f"{a.thermal_stop_temp + self.temp_hr[motor, sensor]:.0f} (--thermal-stop-temp + headroom) at "
                           f"frame {frame} of {clip.name}: policy stops in the {a.thermal_stop_pose} pose over {a.thermal_stop_s} s and keeps "
                           f"standing (DAMP still at > {a.max_temp} C); then re-hoist and L2+B")

    def temp_line(self, st) -> str:
        """Both sensors of the waist and hip motors, and the hottest live motor."""
        tm = np.where(self.live, st.temp.max(1), -np.inf)
        j = int(np.argmax(tm))
        parts = [f"{lbl} {st.temp[i, 0]:.0f}/{st.temp[i, 1]:.0f}" for i, lbl in TEMP_SHOW if self.live[i]]
        return "temp C " + " ".join(parts) + f" | max {tm[j]:.0f} {self.cfg.joint_names[j]}"

    def temp_forecast(self, clip) -> tuple[bool, str]:
        """R1+X: per live motor, T_now + max(heating rate over the STANDBY_PRE window, 0) * clip duration < --thermal-stop-temp.
        T = max of the two sensors; T_now = max over the last second (the readings are whole degrees)."""
        a = self.a
        lim = a.thermal_stop_temp if a.thermal_stop_temp > 0 else a.max_temp
        h = list(self.temp_hist)
        if not h or h[-1][0] - h[0][0] < a.temp_rate_min_s:
            span = h[-1][0] - h[0][0] if h else 0.0
            return False, f"heating rate needs >= {a.temp_rate_min_s:.0f} s of STANDBY_PRE (have {span:.1f} s)"
        t = np.array([x[0] for x in h])
        T = np.array([x[1] for x in h])  # (n, 29)
        tc = t - t.mean()
        rate = (tc[:, None] * (T - T.mean(0))).sum(0) / max(float((tc * tc).sum()), 1e-9)
        now = T[t >= t[-1] - 1.0].max(0)
        dur = clip.num_frames * self.cfg.control_dt
        pred = np.where(self.live, now + np.maximum(rate, 0.0) * dur, -np.inf)
        j = int(np.argmax(pred))
        msg = (f"forecast{' (minus headroom)' if self.temp_hr.any() else ''} {self.cfg.joint_names[j]} {now[j]:.0f} C + {max(rate[j], 0.0):.2f} C/s x {dur:.1f} s = {pred[j]:.1f} C "
               f"(limit {lim:.0f}, rate over {t[-1] - t[0]:.0f} s)")
        return bool(pred[j] < lim), msg

    def standby_allowed(self, t_in: float, st) -> tuple[bool, str]:
        """R1+X: STANDBY_PRE long enough and stable (tilt error vs the static reference, joint speed) over the last window."""
        a = self.a
        if not self.standby_ok[self.clip_idx]:
            return False, self.lw_standby_why(self.clips[self.clip_idx])
        if t_in < a.standby_min_s:
            return False, f"STANDBY_PRE for {t_in:.2f} s < {a.standby_min_s} s"
        w = list(self.sb_win)
        need = int(0.8 * a.standby_window_s / self.cfg.control_dt)
        if len(w) < max(need, 1):
            return False, f"only {len(w)} standby samples in the last {a.standby_window_s} s (need {need})"
        te = np.array([x[1] for x in w])
        dqm = np.array([x[2] for x in w])
        if not np.all(np.isfinite(te)):
            return False, "non-finite tilt error in the standby window"
        te_max = float(np.degrees(te.max()))
        if te_max >= a.standby_max_tilt_deg:
            return False, f"standby tilt error max {te_max:.1f} deg over the last {a.standby_window_s} s >= {a.standby_max_tilt_deg} deg"
        if float(dqm.max()) >= a.standby_max_dq:
            adq = np.where(self.live, np.abs(st.dq), 0.0)
            j = int(np.argmax(adq))
            return False, (f"standby max|dq| {dqm.max():.2f} rad/s over the last {a.standby_window_s} s >= {a.standby_max_dq} "
                           f"(now: {self.cfg.joint_names[j]} {adq[j]:.2f})")
        tm = float(self.teff(st.temp)[self.live].max())
        if tm > a.warn_temp:
            return False, f"max motor temperature {tm:.0f} C > --warn-temp {a.warn_temp} (UNVERIFIED field semantics)"
        tok, tmsg = self.temp_forecast(self.clips[self.clip_idx])
        if not tok:
            return False, tmsg
        if a.sim_band_file:
            if self.band_off_t is None:
                return False, "sim elastic band still on"
            if time.monotonic() - self.band_off_t < a.sim_after_band_s:
                return False, f"band released {time.monotonic() - self.band_off_t:.2f} s ago < {a.sim_after_band_s} s"
        return True, (f"standby {t_in:.1f} s, last {a.standby_window_s} s: tilt err max {te_max:.1f} deg, "
                      f"max|dq| {dqm.max():.2f} rad/s; {tmsg}")

    def lw_standby_why(self, clip) -> str:
        return (f"--lock-waist: {clip.name} is not on the lock-waist STANDBY allow list (empty by default; --lock-waist-standby-allow): "
                "policy standby on the waist-locked model failed the review sim gate (shoulder_clap after-motion standby stepped, "
                "tilt error up to 18.7 deg, 20 deg aborts at kp 0.85; idle pre-standby drifted 82 cm)")

    def confirmed(self, kind: str, t: float, prompt: str) -> bool:
        """Double-press confirmation: True on the 2nd press of `kind` within --confirm-s in the same state entry."""
        c = self.confirm
        if c is not None and c["kind"] == kind and c["state_t"] == self.state_t and t - c["t"] <= self.a.confirm_s:
            self.confirm = None
            return True
        self.confirm = {"kind": kind, "t": t, "state_t": self.state_t}
        self.event("CONFIRM", prompt)
        return False

    def standby_status(self, state: str, t_in: float, t: float, st, tilt_err: float, age: float):
        """STANDBY_*: status line every --status-s and edge-triggered WARNs. Only enqueues to the printer (never blocks)."""
        a = self.a
        s = self.sbs
        yaw = _yaw(st.quat)
        leg = float(np.abs(st.dq[LEG_IDX]).max())
        tmax = float(np.where(self.live[:, None], self.teff(st.temp), -np.inf).max())
        if s is None or s["state_t"] != self.state_t:
            s = self.sbs = {"state_t": self.state_t, "yaw_prev": yaw, "dyaw": 0.0, "t_print": t, "leg": collections.deque(),
                            "te": collections.deque(), "leg_over_t": None, "warned": {}, "age_max": 0.0, "dyaw_max": 0.0}
        d = yaw - s["yaw_prev"]
        s["dyaw"] += (d + np.pi) % (2 * np.pi) - np.pi  # unwrapped yaw change since entering the state
        s["yaw_prev"] = yaw
        s["dyaw_max"] = max(s["dyaw_max"], abs(s["dyaw"]))
        s["age_max"] = max(s["age_max"], age)
        s["leg"].append((t, leg))
        s["te"].append((t, tilt_err))
        while s["leg"] and t - s["leg"][0][0] > 3.0:
            s["leg"].popleft()
        while s["te"] and t - s["te"][0][0] > 3.0:
            s["te"].popleft()
        if leg > a.warn_leg_dq:
            s["leg_over_t"] = t if s["leg_over_t"] is None else s["leg_over_t"]
        else:
            s["leg_over_t"] = None
        dyaw_deg = float(np.degrees(s["dyaw"]))
        conds = {
            "yaw": (abs(dyaw_deg) > a.warn_yaw_deg, f"yaw changed {dyaw_deg:+.0f} deg since entering {state} (> {a.warn_yaw_deg:.0f}): "
                    "the motion runs relative to the current heading -> re-aim the gantry or re-hoist"),
            "leg": (s["leg_over_t"] is not None and t - s["leg_over_t"] > a.warn_leg_dq_s,
                    f"leg |dq| > {a.warn_leg_dq} rad/s for > {a.warn_leg_dq_s} s (stepping?) -> re-hoist (gantry over the robot, take up "
                    "slack only), then L2+B"),
            "temp": (tmax > a.warn_temp, f"max motor temperature {tmax:.0f} C > --warn-temp {a.warn_temp} (DAMP at {a.max_temp})"),
            "long": (t_in > a.standby_warn_s, f"{state} for {t_in:.0f} s > {a.standby_warn_s:.0f} s (documented standby cap: long "
                     "unsupported standby is unmeasured on the robot) -> start the motion or re-hoist"),
        }
        warn_now = []
        for k, (on, msg) in conds.items():
            if on:
                warn_now.append(k)
                if k not in s["warned"]:
                    self.event("WARN", msg)
                s["warned"][k] = t  # last time the condition was on
            elif k in s["warned"] and t - s["warned"][k] > a.status_s:
                del s["warned"][k]  # re-arm only after the condition stayed off for --status-s (no WARN flood at the threshold)
        if t - s["t_print"] >= a.status_s:
            s["t_print"] = t
            te = np.array([x[1] for x in s["te"]])
            te_s = f"{np.degrees(tilt_err):4.1f} (3 s max {np.degrees(np.nanmax(te)):4.1f})" if np.isfinite(te).any() else "n/a"
            line = (f"{state} {t_in:6.1f} s: yaw {dyaw_deg:+6.1f} deg, tilt err {te_s} deg, leg max|dq| 3 s "
                    f"{max(x[1] for x in s['leg']):.2f} rad/s, lowstate age max {s['age_max'] * 1e3:.1f} ms; {self.temp_line(st)}")
            if state == "STANDBY_PRE":
                line += "; " + self.temp_forecast(self.clips[self.clip_idx])[1]
            if state == "STANDBY_PRE":
                ok, why = self.standby_allowed(t_in, st)
                line += "; R1+X: " + ("would be ACCEPTED" if ok else f"would be refused ({why})")
            if warn_now:
                line += "  WARN: " + ",".join(warn_now)
            self.printer.q.put(f"[{t - self.t0:8.3f}] STATUS: {line}\n")

    def policy_allowed(self, t_in: float, st, f0, clip) -> tuple[bool, str]:
        a = self.a
        if a.no_policy:
            return False, "--no-policy: this run only does INTERP/HOLD"
        e = np.where(self.live, np.abs(st.q - f0), 0.0)  # locked waist joints excluded
        err = float(e.max())
        if t_in < a.hold_min_s:
            return False, f"HOLD for {t_in:.2f} s < {a.hold_min_s} s"
        if err >= a.hold_tol:
            j = int(np.argmax(e))
            return False, f"max|q - f0| = {err:.3f} rad ({self.cfg.joint_names[j]}) >= {a.hold_tol}"
        qn = float(np.linalg.norm(st.quat))
        if not np.isfinite(qn) or abs(qn - 1.0) > 0.05:
            return False, f"bad IMU quaternion (norm {qn:.3f})"
        adq = np.where(self.live, np.abs(st.dq), 0.0)
        dqm = float(adq.max())
        if dqm >= a.start_max_dq:
            j = int(np.argmax(adq))
            return False, f"max|dq| = {dqm:.2f} rad/s ({self.cfg.joint_names[j]}) >= {a.start_max_dq}"
        g = projected_gravity(st.quat / max(np.linalg.norm(st.quat), 1e-9))
        g0 = clip.proj_grav[0]
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(g, g0) / (np.linalg.norm(g0) + 1e-9), -1, 1))))
        if tilt >= a.start_max_tilt_deg:
            return False, f"tilt vs clip frame 0 = {tilt:.1f} deg >= {a.start_max_tilt_deg}"
        tm = float(self.teff(st.temp)[self.live].max())
        if tm > a.warn_temp:
            return False, f"max motor temperature {tm:.0f} C > --warn-temp {a.warn_temp} (UNVERIFIED field semantics)"
        if a.sim_band_file and a.sim_band_release_s >= 0:
            if self.band_off_t is None:
                return False, "sim elastic band still on"
            if time.monotonic() - self.band_off_t < a.sim_after_band_s:
                return False, f"band released {time.monotonic() - self.band_off_t:.2f} s ago < {a.sim_after_band_s} s"
        return True, f"max|q - f0| = {err:.3f} rad, max|dq| = {dqm:.2f}, tilt vs clip {tilt:.1f} deg"

    def auto_events(self, state: str, t_in: float, st) -> list[str]:
        a = self.a
        if state == "PASSIVE" and t_in >= a.auto_passive_s:
            return ["interp"]
        if state == "HOLD":
            ok, why = self.policy_allowed(t_in, st, self.f0s[self.clip_idx], self.clips[self.clip_idx])
            if ok:
                return ["policy"]
            if t_in > a.auto_hold_timeout_s:
                self.damp(f"auto-sequence: standby start condition not met within {a.auto_hold_timeout_s} s ({why})")
            return []
        if state == "STANDBY_PRE" and t_in >= a.auto_standby_s:
            ok, why = self.standby_allowed(t_in, st)
            if ok:
                return ["start"]
            if t_in > a.auto_standby_s + a.auto_hold_timeout_s:
                self.damp(f"auto-sequence: motion start condition not met within {a.auto_hold_timeout_s} s ({why})")
            return []
        if state == "STANDBY_POST":
            if self.clip_idx + 1 < len(self.clips) and self.thermal is None:
                if t_in >= a.auto_between_s:
                    c = self.confirm  # L2+up must be pressed twice: 1st press now, 2nd 0.5 s later
                    if c is None or c["kind"] != "interp" or c["state_t"] != self.state_t:
                        return ["interp"]
                    if time.monotonic() - c["t"] >= 0.5:
                        return ["interp"]
            elif t_in >= a.auto_end_hold_s:
                if a.sim_band_file and self.band_off_t is not None:
                    self.sim_band_on("auto-sequence: re-hoist before L2+B")
                if t_in >= a.auto_end_hold_s + 1.0:
                    self.damp("auto-sequence finished")
            return []
        if state == "END_HOLD":
            if self.end is not None and (self.end.get("then_interp") or self.end.get("then_hold")):
                return []  # INTERP to the next clip / HOLD follows the blend by itself
            if self.clip_idx + 1 < len(self.clips):
                if t_in >= max(a.auto_between_s, a.end_blend_s + 0.1):
                    return ["interp"]
            elif t_in >= a.auto_end_hold_s:
                self.damp("auto-sequence finished")
        return []

    # ------------------------------------------------------------------ shutdown
    def bridge_meta(self) -> dict:
        if self.bridge is None:
            return {}
        b = self.bridge.read_status(np.zeros(1, gb.STATUS_DTYPE))[0]
        m = {k: (float(b[k]) if b[k].dtype.kind == "f" else int(b[k])) for k in gb.STATUS_DTYPE.names if not k.startswith("_")}
        m.update(send_err=self.bridge.n_send_err, bad=self.bridge_bad, returncode=self.bridge.p.poll())
        return m

    def save(self):
        a = self.a
        a.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.wall0))
        path = a.log_dir / f"run_{stamp}.npz"
        R = {k: np.asarray(v) for k, v in self.rec.items() if len(v)}
        n = min(self.cl_n, self.cl_cap)
        idx = np.arange(self.cl_n - n, self.cl_n) % self.cl_cap  # chronological order (ring buffer)
        meta = {
            "args": {k: str(v) for k, v in vars(a).items()}, "clips": [c.name for c in self.clips],
            "states": STATES, "events": self.events, "damp_reason": self.damp_reason, "n_rx": self.n_rx,
            "n_crc_bad": self.n_crc_bad, "policy_calls": self.n_policy_calls, "joint_names": self.cfg.joint_names,
            "stage": a.stage, "profile": self.profile, "n_sent": self.n_sent, "n_write_fail": self.n_write_fail,
            "n_cmd_exc": self.n_cmd_exc, "cb_errors": self.cb_errors, "cb_last_err": self.cb_last_err,
            "n_lowcmd_rx": self.watcher.count if self.watcher else None, "self_delivery": self.self_delivery, "rt_info": self.rt_info,
            "printer_failures": self.printer.n_fail, "cmd_log_wrapped": self.cl_n > self.cl_cap,
            "joint_lower": self.cfg.joint_lower.tolist(), "joint_upper": self.cfg.joint_upper.tolist(),
            "lock_waist": self.lock_waist, "locked_joints": np.nonzero(self.locked)[0].tolist(), "mode_machine": self.mm_latched,
            "t0_monotonic": self.t0, "end_hold_legacy": bool(a.end_hold_legacy),
            "io": "g1_bridge", "bridge": self.bridge_meta(), "ring_lost": int(self.ring_cur[1]),
        }
        np.savez_compressed(path, **R, cmd_t=self.cl_t[idx], cmd_q=self.cl_q[idx], cmd_kp=self.cl_kp[idx],
                            cmd_kd=self.cl_kd[idx], cmd_state=self.cl_s[idx], meta=json.dumps(meta))
        print(f"[log] {path}  ({len(self.rec['t'])} policy-loop records, {n} lowcmd records)", flush=True)
        summarize(path)
        return path


def _yaw(quat) -> float:
    n = float(np.linalg.norm(quat))
    if not np.isfinite(n) or n < 0.5:
        return 0.0
    w, x, y, z = np.asarray(quat, float) / n
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _tilt(quat) -> float:
    g = projected_gravity(quat / max(np.linalg.norm(quat), 1e-9))
    return float(np.arccos(np.clip(-g[2], -1, 1)))


def _tilt_vs(quat, g_ref) -> float:
    n = float(np.linalg.norm(quat))
    if not np.isfinite(n) or n < 0.5:
        return np.nan
    g = projected_gravity(quat / n)
    return float(np.arccos(np.clip(np.dot(g, g_ref) / (np.linalg.norm(g_ref) + 1e-9), -1, 1)))


def thermal_stop_clip(clip: Clip, frame: int, stand: Clip, n: int, dt: float = 0.02) -> Clip:
    """Reference for the thermal stop: row 0 = `clip` at `frame`, rows 1..n a cosine blend (as make_combo_clip transitions)
    to `stand` row 0 (a static_clip, all rates zero). Goal layout per row (68): q - q0 [0:29] | dq [29:58] | proj grav [58:61]
    | ang vel body [61:64] | lin vel heading [64:67] | height [67]. Joint positions, gravity and height blend; the joint
    rates are the blend's derivative; the body rates fade out with (1 - s). Row n is exactly the static stand row."""
    ga, gb_ = clip.goal[frame].astype(np.float64), stand.goal[0].astype(np.float64)
    u = np.arange(1, n + 1) / n
    s = (0.5 - 0.5 * np.cos(np.pi * u))[:, None]
    ds = (0.5 * np.pi * np.sin(np.pi * u) / (n * dt))[:, None]
    g = np.repeat(ga[None], n, 0)
    g[:, 0:29] = ga[0:29] + s * (gb_[0:29] - ga[0:29])
    g[:, 29:58] = ds * (gb_[0:29] - ga[0:29])
    pg = ga[58:61] + s * (gb_[58:61] - ga[58:61])
    g[:, 58:61] = pg / np.linalg.norm(pg, axis=1, keepdims=True)
    g[:, 61:67] = (1.0 - s) * ga[61:67]
    g[:, 67] = ga[67] + s[:, 0] * (gb_[67] - ga[67])
    jp = clip.joint_pos[frame] + s * (stand.joint_pos[0] - clip.joint_pos[frame])
    pj = clip.proj_grav[frame] + s * (stand.proj_grav[0] - clip.proj_grav[frame])
    pj /= np.linalg.norm(pj, axis=1, keepdims=True)
    return Clip(name=f"{clip.name}@{frame}_thermal_stop", goal=np.concatenate([ga[None], g]).astype(np.float32),
                joint_pos=np.concatenate([clip.joint_pos[frame][None], jp]), proj_grav=np.concatenate([clip.proj_grav[frame][None], pj]))


def _governor() -> str:
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        g = p.read_text().strip()
        return f"{g} ({'writable' if os.access(p, os.W_OK) else 'needs sudo to change'})"
    except Exception:  # noqa: BLE001
        return "unknown"


def summarize(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    t, st = z["t"], z["state"]
    names = meta.get("states", STATES)
    SID = {n: i for i, n in enumerate(names)}  # noqa: N806  (state ids as written by the run that made the log)
    print("[summary] events:")
    for e in meta["events"]:
        print(f"    {e['t']:8.3f}  {e['kind']:<8} {e['msg']}")
    per = np.diff(t) * 1e3
    if len(per) > 5:
        print(f"[summary] policy-loop period ms: mean {per.mean():.3f}  p1 {np.percentile(per, 1):.3f}  p99 {np.percentile(per, 99):.3f}  "
              f"max {per.max():.3f}  |jitter| p99 {np.percentile(np.abs(per - 20.0), 99):.3f}")
        lat = (t - z["deadline"]) * 1e3
        print(f"[summary] wake-up lateness ms: p50 {np.median(lat):.3f}  p99 {np.percentile(lat, 99):.3f}  max {lat.max():.3f};  "
              f"compute ms (STANDBY_*/POLICY): p50/p99/max " + _pct(z["compute_ms"][np.isin(st, [SID[k] for k in POLICY_STATES if k in SID])]))
        print(f"[summary] state age at loop ms: p50 {np.median(z['state_age_ms']):.2f}  p99 {np.percentile(z['state_age_ms'], 99):.2f}  max {z['state_age_ms'].max():.2f}")
    ct = z["cmd_t"]
    if len(ct) > 10:
        cp = np.diff(ct) * 1e3
        print(f"[summary] lowcmd: {len(ct)} msgs in {ct[-1] - ct[0]:.2f} s = {(len(ct) - 1) / (ct[-1] - ct[0]):.1f} Hz (includes immediate sends after policy steps); "
              f"gap p99 {np.percentile(cp, 99):.3f} max {cp.max():.3f} ms; write failures {meta.get('n_write_fail')}  [runner -> g1_bridge handoffs]")
    b = meta.get("bridge") or {}
    if b:
        print(f"[summary] g1_bridge: published {b.get('n_pub')} rt/lowcmd, failed {b.get('n_pub_fail')}, commands rx {b.get('n_cmd_rx')}, "
              f"watchdog fired {b.get('wd_fired')}, problem {b.get('bad')}, ring entries lost {meta.get('ring_lost')}; publish gap max "
              f"{(b.get('pub_gap_max') or 0) * 1e3:.2f} ms ({b.get('n_gap_late')} > 2 periods), slowest Write {(b.get('pub_write_max') or 0) * 1e3:.2f} ms")
    pol = st == SID["POLICY"]
    if pol.any():
        print(f"[summary] POLICY steps {pol.sum()}  max|action| {np.abs(z['action'][pol]).max():.2f}  clamped joints/step mean {z['n_clamped'][pol].mean():.2f}")
        g = np.array([projected_gravity(q / max(np.linalg.norm(q), 1e-9)) for q in z["quat"][pol]])
        msg = f"[summary] POLICY tilt (IMU) max {np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1))).max():.1f} deg"
        if "tilt_err" in z.files and np.isfinite(z["tilt_err"][pol]).any():
            te = np.degrees(np.nanmax(z["tilt_err"][pol]))
            lim = np.degrees(SafetyCfg().max_tilt_err)
            msg += f"; tilt error vs reference max {te:.1f} deg (DeployCore abort > {lim:.0f} deg, margin {lim - te:.1f} deg)"
        print(msg)
        if "joint_lower" in meta:
            lo, hi = np.asarray(meta["joint_lower"]), np.asarray(meta["joint_upper"])
            qp = z["q"][pol]
            at = (qp < lo + 0.02) | (qp > hi - 0.02)
            frac = at.mean(axis=0)
            if frac.max() > 0:
                top = np.argsort(-frac)[:3]
                print("[summary] POLICY time within 0.02 rad of the hard stop: " +
                      ", ".join(f"{meta['joint_names'][j]} {100 * frac[j]:.1f}%" for j in top if frac[j] > 0))
    _standby_summary(z, names, meta)
    if "temp" in z.files and len(z["temp"]):
        T = z["temp"]  # (n, 29, 2)
        jn = meta["joint_names"]
        top = np.argsort(-T.max(axis=(0, 2)))[:5]
        print("[summary] temperature C (sensor 0 / sensor 1: first -> max -> last): " + "; ".join(
            f"{jn[j]} {T[0, j, 0]:.0f}->{T[:, j, 0].max():.0f}->{T[-1, j, 0]:.0f} / {T[0, j, 1]:.0f}->{T[:, j, 1].max():.0f}->{T[-1, j, 1]:.0f}"
            for j in top))
        for name in ("STANDBY_PRE", "POLICY"):
            m = st == SID[name]
            if m.sum() > 50:
                dur = float(t[m][-1] - t[m][0])
                rise = T[m][-1].max(1) - T[m][0].max(1)
                j = int(np.argmax(rise))
                print(f"[summary] {name} {dur:.0f} s: largest rise {jn[j]} {rise[j]:+.0f} C ({rise[j] / max(dur, 1e-9):.2f} C/s, max of both sensors)")
    lk = meta.get("locked_joints") or []
    if meta.get("lock_waist") and lk and len(t):
        cq = z["cmd_q"][:, lk] if len(z["cmd_t"]) else np.zeros((1, len(lk)))
        ck = z["cmd_kp"][:, lk] if len(z["cmd_t"]) else np.zeros((1, len(lk)))
        cd = z["cmd_kd"][:, lk] if len(z["cmd_t"]) else np.zeros((1, len(lk)))
        lt = z["locked_temp"] if "locked_temp" in z.files else np.zeros(1)
        lm = z["locked_mstat"] if "locked_mstat" in z.files else np.zeros(1, np.int64)
        print(f"[summary] WAIST LOCKED (mode_machine {meta.get('mode_machine')}), motors {lk}: lowstate max|q| {np.abs(z['q'][:, lk]).max():.4f} "
              f"max|dq| {np.abs(z['dq'][:, lk]).max():.4f} max temp {lt.max():.0f} mstat seen {sorted({int(x) for x in np.ravel(lm)})}; "
              f"sent max|q|/kp/kd {np.abs(cq).max():.3g}/{np.abs(ck).max():.3g}/{np.abs(cd).max():.3g}")
        if pol.any():
            print(f"[summary] POLICY raw action on locked joints (not applied): max|a| {np.abs(z['action'][pol][:, lk]).max():.2f}")
    print(f"[summary] stage {meta.get('stage')} profile {meta.get('profile')}; rt {meta.get('rt_info')}; "
          f"lowcmd self-delivery {meta.get('self_delivery')}; DDS callback errors {meta.get('cb_errors')}; "
          f"stdout failures {meta.get('printer_failures')}")
    print(f"[summary] damp reason: {meta['damp_reason']}")


def _standby_summary(z, names, meta):
    """Per contiguous STANDBY_PRE / POLICY / STANDBY_POST segment: duration, tilt error vs reference, max|dq| (live joints), max|a|."""
    t, st = z["t"], z["state"]
    if not len(t):
        return
    live = np.ones(z["dq"].shape[1], bool)
    live[meta.get("locked_joints") or []] = False
    lab = np.array([names[i] if 0 <= i < len(names) else "?" for i in st])
    cut = np.nonzero(lab[1:] != lab[:-1])[0] + 1
    tot = {}
    have_te = "tilt_err" in z.files
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(lab)]):
        name = lab[a]
        if name not in POLICY_STATES:
            continue
        dur = float(t[b - 1] - t[a]) + 0.02
        tot[name] = tot.get(name, 0.0) + dur
        dqm = np.abs(z["dq"][a:b][:, live]).max(1)
        te = np.degrees(z["tilt_err"][a:b]) if have_te else np.full(b - a, np.nan)
        te_s = (f"tilt err max {np.nanmax(te):5.2f} mean {np.nanmean(te):5.2f} deg" if np.isfinite(te).any() else "tilt err n/a")
        print(f"[summary] {name:<12} t {t[a]:8.2f} dur {dur:7.2f} s ({b - a} steps): {te_s}; max|dq| max {dqm.max():.2f} "
              f"p99 {np.percentile(dqm, 99):.2f} rad/s; max|a| {np.abs(z['action'][a:b]).max():.2f}")
    if tot:
        print("[summary] totals: " + ", ".join(f"{k} {v:.1f} s" for k, v in tot.items()))


def _pct(x):
    return "n/a" if len(x) == 0 else f"{np.median(x):.2f}/{np.percentile(x, 99):.2f}/{x.max():.2f}"


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--lowcmd-watcher-child":
        watcher_child(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, required=True, help="0 real robot, 1 sim")
    ap.add_argument("--stage", choices=list(STAGES), required=True,
                    help="safety profile: air/feet (vel 0.3, |a|<10, torque 0.5x effort, hold tilt 20 deg), "
                         "slack (0.5, 15, 1.0x, 29 deg), ground (core defaults 0.8, 50, 1.5x, 46 deg)")
    ap.add_argument("--artifacts", type=Path, default=gc.DEFAULT_ARTIFACTS, help="dir with policy.onnx + deploy_config.json")
    ap.add_argument("--clip", type=Path, nargs="+", required=True, help="clip npz(s), played one after another (L2+up between)")
    ap.add_argument("--log-dir", type=Path, default=gc.REPO / "deploy_logs")
    ap.add_argument("--sim", action="store_true", help="simulation run (skips 'yes' prompt; allows --auto-sequence and sim-only test hooks)")
    ap.add_argument("--auto-sequence", action="store_true", help="SIM ONLY: PASSIVE -> INTERP -> HOLD -> STANDBY_PRE -> POLICY -> STANDBY_POST (-> blend -> INTERP next clip ...) "
                         "-> band on -> DAMP -> exit")
    ap.add_argument("--no-policy", action="store_true",
                    help="never start the policy (R1+A always refused): INTERP/HOLD/END tests only, e.g. the first powered run")
    ap.add_argument("--allow-ssh", action="store_true", help="domain 0: allow running from an ssh session (NOT recommended)")
    # profile overrides (default: from --stage)
    ap.add_argument("--max-vel-frac", type=float, default=None)
    ap.add_argument("--max-abs-action", type=float, default=None)
    ap.add_argument("--torque-clamp-factor", type=float, default=None)
    ap.add_argument("--max-tilt-hold", type=float, default=None, help="rad, tilt -> DAMP in INTERP/HOLD/END_HOLD")
    ap.add_argument("--cmd-hz", type=float, default=500.0)
    ap.add_argument("--interp-s", type=float, default=4.0)
    ap.add_argument("--end-blend-s", type=float, default=1.0, help="END_HOLD: gain/target blend duration")
    ap.add_argument("--end-hold-legacy", action="store_true",
                    help="clip done -> END_HOLD (stand-gain blend, cannot balance) instead of STANDBY_POST (policy keeps standing)")
    ap.add_argument("--standby-min-s", type=float, default=3.0, help="R1+X: minimum time in STANDBY_PRE")
    ap.add_argument("--standby-window-s", type=float, default=3.0,
                    help="R1+X: stability window (end of STANDBY_PRE); 3 s catches a stepping robot that 0.5 s missed (review sim)")
    ap.add_argument("--standby-max-tilt-deg", type=float, default=8.0, help="R1+X: max tilt error vs the reference over the window")
    ap.add_argument("--standby-max-dq", type=float, default=1.0, help="R1+X: max |dq| [rad/s] over the window")
    ap.add_argument("--confirm-s", type=float, default=3.0,
                    help="L2+up in STANDBY_PRE / STANDBY_POST (leaves the policy: stand gains) must be pressed twice within this time")
    ap.add_argument("--status-s", type=float, default=3.0, help="STANDBY_*: status line period")
    ap.add_argument("--warn-yaw-deg", type=float, default=15.0, help="STANDBY_*: WARN when the IMU yaw changed more than this since entering the state")
    ap.add_argument("--warn-leg-dq", type=float, default=1.0, help="STANDBY_*: WARN when leg max|dq| [rad/s] stays above this ...")
    ap.add_argument("--warn-leg-dq-s", type=float, default=0.5, help="... for longer than this [s]")
    ap.add_argument("--standby-warn-s", type=float, default=120.0,
                    help="STANDBY_*: WARN after this long (documented standby cap until the lowstate age over a long standby is measured)")
    ap.add_argument("--sync-udp", default=None, metavar="HOST:PORT",
                    help="send 'state clip frame t' over UDP every tick (background-video sync, scripts/deploy/bg_sync_server.py)")
    ap.add_argument("--hold-min-s", type=float, default=1.0)
    ap.add_argument("--hold-tol", type=float, default=0.1, help="max |q - f0| [rad] to allow the policy start")
    ap.add_argument("--start-max-dq", type=float, default=0.5, help="max |dq| [rad/s] to allow the policy start")
    ap.add_argument("--start-max-tilt-deg", type=float, default=10.0, help="max tilt vs clip proj_grav[0] to allow the policy start")
    ap.add_argument("--stale-ms", type=float, default=20.0)
    ap.add_argument("--max-overrun", type=float, default=2.0, help="policy cycles late before DAMP")
    ap.add_argument("--max-temp", type=float, default=80.0, help="motor temperature [C] -> DAMP (UNVERIFIED default)")
    ap.add_argument("--warn-temp", type=float, default=70.0, help="motor temperature [C] above which R1+A is refused (UNVERIFIED)")
    ap.add_argument("--thermal-stop-temp", type=float, default=65.0,  # 15 C below --max-temp: pc2bridge11 went 72 -> 81 C in 6 s after a stop
                    help="POLICY: a live motor at/above this [C] -> smooth stop (--thermal-stop-pose), policy keeps standing; "
                         "also the R1+X heating-forecast limit (<= 0 disables the stop; the forecast then uses --max-temp)")
    ap.add_argument("--temp-headroom", action="append", metavar="MOTOR:SENSOR:DEG",
                    help="raise every temperature limit by DEG for one sensor (0/1) of one motor (repeatable), e.g. 13:1:10")
    ap.add_argument("--thermal-stop-s", type=float, default=1.0, help="thermal stop: blend duration [s]")
    ap.add_argument("--thermal-stop-pose", choices=("current", "upright", "frame0"), default="current",
                    help="thermal stop: hold the reference frame where it triggered (current, default), the same with the waist "
                         "back to its frame-0 values (upright), or blend to the clip's frame-0 pose (frame0)")
    ap.add_argument("--temp-rate-window-s", type=float, default=20.0, help="R1+X forecast: heating-rate window (end of STANDBY_PRE) [s]")
    ap.add_argument("--temp-rate-min-s", type=float, default=10.0, help="R1+X forecast: minimum STANDBY_PRE samples span [s]")
    ap.add_argument("--motor-error-mask", type=lambda s: int(s, 0), default=0xFFFFFFFF,
                    help="domain 0: DAMP if motorstate & mask != 0 (bit meanings UNVERIFIED; 0 disables)")
    ap.add_argument("--max-write-fail", type=int, default=10, help="consecutive failed rt/lowcmd writes -> DAMP")
    ap.add_argument("--stand-arm-kd", type=float, default=10.0, help="stand-gain kd for the 14 arm joints (spec: 10)")
    ap.add_argument("--stand-waist-kd", type=float, default=5.0, help="stand-gain kd for the 3 waist joints (spec: 5)")
    ap.add_argument("--lock-waist", action="store_true",
                    help="waist-locked G1 (lowstate mode_machine 6 = g1_29dof_lock_waist_rev_1_0): REQUIRED when the robot reports 6, "
                         "refused when it reports 5. Motors 13/14 get mode 0 and zero commands; clips restricted (see --i-know-clip)")
    ap.add_argument("--clip-allow-file", type=Path, default=None,
                    help="--lock-waist: replaces the built-in clip allow list (one clip name per line, '#' comments)")
    ap.add_argument("--i-know-clip", action="append", default=None, metavar="CLIP_NAME",
                    help="--lock-waist: accept this clip although it is not on the allow list (explicit approval only; "
                         "HARDER clips are refused regardless)")
    ap.add_argument("--lock-waist-standby-allow", action="append", default=None, metavar="CLIP_NAME",
                    help="--lock-waist: allow policy standby (R1+A -> STANDBY_PRE -> R1+X) for this clip (only after a lock-waist "
                         "standby sim gate passed); default list g1_common.LOCK_WAIST_STANDBY_ALLOWED_CLIPS is empty")
    ap.add_argument("--lock-waist-direct-start", action="store_true",
                    help="--lock-waist: for clips not on the lock-waist STANDBY allow list, R1+A in HOLD starts the motion directly "
                         "(the pre-standby flow of docs/DEPLOY_G1.md#waist-locked-g1-mode_machine--6, hoist carrying); without it R1+A is refused for them")
    ap.add_argument("--expect-mode-machine", type=int, default=None,
                    help="optional cross-check; must equal 5 (default) or 6 (--lock-waist)")
    ap.add_argument("--no-state-crc", action="store_true", help="do not drop lowstate messages with a bad CRC (domain 0)")
    ap.add_argument("--wait-state-s", type=float, default=5.0)
    ap.add_argument("--lowcmd-check-s", type=float, default=1.0)
    ap.add_argument("--lowcmd-recheck-s", type=float, default=0.5, help="foreign rt/lowcmd re-check right after 'yes'")
    ap.add_argument("--no-lowcmd-watch", dest="lowcmd_watch", action="store_false", help="disable the run-time foreign rt/lowcmd watch")
    ap.add_argument("--check-mode-timeout", type=float, default=2.0)
    ap.add_argument("--require-check-mode", action="store_true", help="domain 0: refuse if MotionSwitcher CheckMode does not answer")
    ap.add_argument("--button-stable-n", type=int, default=3)
    ap.add_argument("--button-release-s", type=float, default=0.1)
    ap.add_argument("--button-max-age-s", type=float, default=0.1)
    ap.add_argument("--ort-threads", type=int, default=1)
    ap.add_argument("--bridge-wd-ms", type=float, default=100.0,
                    help="g1_bridge: no runner command for this long -> the bridge publishes latched damping itself")
    ap.add_argument("--bridge-hb-ms", type=float, default=50.0, help="g1_bridge publisher heartbeat older than this -> DAMP")
    ap.add_argument("--bridge-cpu", type=int, default=-1, help="g1_bridge publisher thread CPU (-1 = no pinning)")
    ap.add_argument("--bridge-prio", type=int, default=0, help="g1_bridge publisher thread SCHED_FIFO priority (0 = none)")
    ap.add_argument("--bridge-orphan-s", type=float, default=10.0,
                    help="g1_bridge: damping this long after the runner is gone / SIGTERM, then exit")
    ap.add_argument("--cmd-cpu", default="auto", help="pin the command thread: 'auto' (last allowed CPU), a CPU index, or 'none'")
    ap.add_argument("--rt-prio", type=int, default=50, help="try SCHED_FIFO for the command thread (needs CAP_SYS_NICE/rtprio; 0 = skip)")
    ap.add_argument("--log-obs", action="store_true", help="also log DeployCore.hist (10 frames oldest->newest x 93, frame-major) every POLICY step")
    ap.add_argument("--cmd-log-max-s", type=float, default=900.0, help="lowcmd log ring-buffer length")
    ap.add_argument("--rec-max-s", type=float, default=1800.0, help="--log-obs stops after this many seconds of records")
    # sim-only hooks
    ap.add_argument("--sim-band-file", default=None, help="SIM: flag file read by sim_g1_mujoco.py --band-file")
    ap.add_argument("--sim-band-release-s", type=float, default=-1.0,
                    help="SIM: release the band this long after entering HOLD; <0 = release in the same tick the policy starts")
    ap.add_argument("--sim-after-band-s", type=float, default=1.0, help="SIM (release-s >= 0): stand on own feet this long before the policy may start")
    ap.add_argument("--sim-band-release-standby-s", type=float, default=-1.0,
                    help="SIM: release the band this long after entering STANDBY_PRE; <0 = in the tick STANDBY_PRE starts")
    ap.add_argument("--sim-no-band-end-hold", action="store_true", help="SIM: keep the band off in END_HOLD")
    ap.add_argument("--sim-exit-after-damp", type=float, default=None, help="SIM: exit this many s after entering DAMP")
    ap.add_argument("--sim-inject-nan-step", type=int, default=None, help="SIM: policy returns NaN from this call on")
    ap.add_argument("--sim-stall-at-s", type=float, default=None, help="SIM: block the main loop this long after POLICY starts")
    ap.add_argument("--sim-stall-s", type=float, default=0.5)
    ap.add_argument("--sim-ask-yes", action="store_true", help="SIM: ask the 'yes' confirmation like on the robot")
    ap.add_argument("--sim-try-check-mode", action="store_true", help="SIM: also call MotionSwitcher CheckMode (exercise the code path)")
    ap.add_argument("--auto-passive-s", type=float, default=1.0)
    ap.add_argument("--auto-between-s", type=float, default=1.0)
    ap.add_argument("--auto-standby-s", type=float, default=3.5, help="auto-sequence: R1+X after this long in STANDBY_PRE (if allowed)")
    ap.add_argument("--auto-end-hold-s", type=float, default=3.0)
    ap.add_argument("--auto-hold-timeout-s", type=float, default=10.0)
    a = ap.parse_args()

    sim_only = [a.auto_sequence, a.sim_band_file, a.sim_band_release_standby_s >= 0, a.sim_exit_after_damp is not None, a.sim_inject_nan_step is not None,
                a.sim_stall_at_s is not None, a.sim_ask_yes, a.sim_try_check_mode]
    if a.domain == 0 and (a.sim or any(sim_only)):
        sys.exit("REFUSED: --sim / --auto-sequence / --sim-* hooks are not allowed on DDS domain 0 (real robot)")
    if any(sim_only) and not a.sim:
        sys.exit("--auto-sequence and --sim-* options require --sim")
    if a.domain == 0 and (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")) and not a.allow_ssh:
        sys.exit("REFUSED: running from an ssh session. Run the runner on the deploy PC's local console (a dropped ssh/Wi-Fi "
                 "link must not be able to take the controller down). --allow-ssh overrides (not recommended).")
    want_mm = gc.MODE_MACHINE_LOCK_WAIST if a.lock_waist else gc.MODE_MACHINE_FULL
    if a.expect_mode_machine is not None and a.expect_mode_machine != want_mm:
        sys.exit(f"--expect-mode-machine {a.expect_mode_machine} is inconsistent with {'--lock-waist' if a.lock_waist else 'no --lock-waist'} "
                 f"(accepted: {gc.MODE_MACHINE_FULL} without --lock-waist, {gc.MODE_MACHINE_LOCK_WAIST} with it)")
    a.expect_mode_machine = want_mm
    if not a.lock_waist and (a.clip_allow_file is not None or a.i_know_clip or a.lock_waist_standby_allow or a.lock_waist_direct_start):
        sys.exit("--clip-allow-file / --i-know-clip / --lock-waist-standby-allow / --lock-waist-direct-start only apply with --lock-waist")
    if a.lock_waist:
        a.end_hold_legacy = True  # STANDBY_POST is not sim-gated on the waist-locked model (review 2026-10-24)
    if a.auto_sequence and a.sim_exit_after_damp is None:
        a.sim_exit_after_damp = 1.0

    printer = Printer()
    r = Runner(a, printer)  # before redirecting stdout: a constructor error must reach the terminal
    sys.stdout = QueueStream(printer.q)
    sys.stderr = QueueStream(printer.q)
    printer.open_file(a.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S', time.localtime(r.wall0))}.console.txt")

    # Signals only set flags once commands flow; the command thread performs DAMP. The C-level handler also
    # writes the signal number to a pipe the command thread polls, so a blocked main thread cannot delay it.
    rfd, wfd = os.pipe()
    os.set_blocking(rfd, False)
    os.set_blocking(wfd, False)
    signal.set_wakeup_fd(wfd, warn_on_full_buffer=False)
    r.sig_r = rfd

    def on_signal(signum, _frame):
        r.sig = signum
        r.stop.set()
        if not r.cmd_started:
            raise KeyboardInterrupt(f"signal {signum}")

    # SIGTSTP (Ctrl-Z) would stop the whole process with the last command still applied: treat it as a stop request.
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGTSTP):
        signal.signal(sg, on_signal)
    for sg in (signal.SIGTTIN, signal.SIGTTOU):
        signal.signal(sg, signal.SIG_IGN)
    exit_code = 0
    try:
        r.startup()
        r.run()
    except KeyboardInterrupt as e:
        if r.cmd_started:
            r.damp(f"operator interrupt ({e})")
        else:
            print(f"[startup] interrupted ({e}); nothing was sent", flush=True)
            exit_code = 1
    except SystemExit as e:
        if r.cmd_started:
            r.damp(f"exit: {e}")
        else:
            print(f"[startup] {e}", flush=True)
            exit_code = 1
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if r.cmd_started:
            r.damp(f"exception {e!r}")
        exit_code = 1
    finally:
        if r.cmd_started:
            t_f = time.monotonic()
            reason = f"signal {signal.Signals(r.sig).name}" if r.sig is not None else "shutdown"
            r.damp(reason)
            r.event("SHUTDOWN", "sending damping for 1 s")
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                time.sleep(max(0.0, t_end - time.monotonic()))
            t_j = time.monotonic()
            r.stop.set()
            r.cmd_stop.set()
            r.cmd_thread.join(timeout=1.0)
            meta_b = r.bridge_meta()
            r.bridge.quit()
            r.event("SHUTDOWN", f"command thread stopped (alive={r.cmd_thread.is_alive()}); g1_bridge QUIT (exit "
                                f"{r.bridge.p.poll()}, {meta_b.get('n_pub')} lowcmd published, {meta_b.get('n_pub_fail')} failed): "
                                f"no more rt/lowcmd. damping {t_j - t_f:.3f} s, join {time.monotonic() - t_j:.3f} s")
            if a.sim and a.sim_band_file and os.path.exists(a.sim_band_file):
                os.remove(a.sim_band_file)
            try:
                r.save()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        if r.watcher is not None:
            r.watcher.close()
        if r.bridge is not None and not r.bridge.closed:
            r.bridge.quit()
        printer.close()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
