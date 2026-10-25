#!/usr/bin/env python3
"""READ-ONLY G1 29-DoF state monitor. Subscribes rt/lowstate; NEVER creates an rt/lowcmd publisher.

  source $G1_DEPLOY_HOME/env.sh   # default ~/workspace/g1_deploy
  python scripts/deploy/check_robot.py --iface enp3s0 --domain 0               # live table
  python scripts/deploy/check_robot.py --iface enp3s0 --sign left_knee_joint   # sign check for one joint
  python scripts/deploy/check_robot.py --iface lo --domain 1 --once            # one snapshot (sim)
  python scripts/deploy/check_robot.py --iface enp3s0 --check-mode             # + read-only MotionSwitcher CheckMode
  python scripts/deploy/check_robot.py --iface enp132s0 --lock-waist --check-mode  # waist-locked G1 (mode_machine 6)

Prints mode_machine (expected 5 for the 29-DoF G1 rev 1.0, 6 with --lock-waist = g1_29dof_lock_waist_rev_1_0), mode_pr, message rate and tick rate,
IMU quaternion -> projected gravity (upright ~ (0, 0, -1)), gyro, all 29 joints (q, dq, temperature,
q0, soft limits, limit flag), remote-controller buttons, and lowstate CRC statistics.
Sign mode: press Enter to zero; prints q - q(at Enter) of the chosen joint with the MuJoCo-model hint of
which physical direction is positive. Ctrl-C exits.
--check-mode: one READ-ONLY MotionSwitcherClient.CheckMode() call at start (never SelectMode/ReleaseMode);
shows whether a Unitree motion mode is active (it must be empty before the runner may start).
Temperature: both MotorState_.temperature values are shown; which is which and the unit are UNVERIFIED.
mstat: MotorState_.motorstate in hex; bit meanings UNVERIFIED (the runner DAMPs on motorstate & --motor-error-mask).
--lock-waist: motors 13/14 are flagged LOCKED (expected q = dq = 0, temperature 0) or LOCKED-ACTIVE!! (the runner would
DAMP: 'waist unexpectedly active'); they are left out of the soft/hard-limit flags and the max temperature.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as gc  # noqa: E402
from supermimic.deploy.core import DeployConfig, projected_gravity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", required=True, help="NIC wired to the robot (192.168.123.x), 'lo' for the sim")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain: 0 real robot, 1 sim")
    ap.add_argument("--artifacts", type=Path, default=gc.DEFAULT_ARTIFACTS, help="dir with deploy_config.json")
    ap.add_argument("--mjcf", type=Path, default=gc.DEFAULT_MJCF, help="training MJCF for sign hints")
    ap.add_argument("--sign", default=None, help="joint name (or index) for sign-check mode")
    ap.add_argument("--rate", type=float, default=5.0, help="display refresh Hz")
    ap.add_argument("--once", action="store_true", help="print one snapshot after 1 s and exit")
    ap.add_argument("--lock-waist", action="store_true", help="waist-locked G1: expect mode_machine 6, motors 13/14 LOCKED")
    ap.add_argument("--expect-mode-machine", type=int, default=None, help="default 5, or 6 with --lock-waist")
    ap.add_argument("--no-clear", action="store_true", help="do not clear the terminal between refreshes")
    ap.add_argument("--check-mode", action="store_true", help="read-only MotionSwitcher CheckMode() once at start")
    args = ap.parse_args()
    if args.expect_mode_machine is None:
        args.expect_mode_machine = gc.MODE_MACHINE_LOCK_WAIST if args.lock_waist else gc.MODE_MACHINE_FULL
    locked = set(gc.LOCKED_WAIST_IDX) if args.lock_waist else set()

    cfg = DeployConfig.load(args.artifacts / "deploy_config.json")
    names = cfg.joint_names
    sign_j = None
    if args.sign is not None:
        sign_j = int(args.sign) if args.sign.isdigit() else names.index(args.sign)
    hints = gc.sign_hints(cfg, args.mjcf) if (sign_j is not None or not args.once) else {}

    gc.init_dds(args.domain, args.iface)
    mode_line = "MotionSwitcher: not queried (--check-mode)"
    if args.check_mode:
        try:
            code, name, raw = gc.check_motion_mode(2.0)
            mode_line = (f"MotionSwitcher CheckMode: code={code} result={raw}  -> "
                         + ("NO ANSWER (UNVERIFIED whether reachable from this PC)" if code != 0 else
                            (f"motion mode '{name}' ACTIVE: runner will refuse" if name else "no motion mode active")))
        except Exception as e:  # noqa: BLE001
            mode_line = f"MotionSwitcher CheckMode failed: {e!r}"
        print(mode_line, flush=True)
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC

    crc = CRC()
    lock = threading.Lock()
    st = {"msg": None, "t": 0.0, "n": 0, "crc_bad": 0, "rx_times": [], "ticks": []}

    def on_state(msg):
        t = time.monotonic()
        bad = msg.crc != crc.Crc(msg)
        with lock:
            st["msg"], st["t"] = msg, t
            st["n"] += 1
            st["crc_bad"] += int(bad)
            st["rx_times"].append(t)
            st["ticks"].append(int(msg.tick))
            if len(st["rx_times"]) > 2000:
                del st["rx_times"][:1000], st["ticks"][:1000]

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    zero = {"q": None, "req": sign_j is not None}
    if sign_j is not None and not args.once:
        def stdin_loop():
            for _ in sys.stdin:
                zero["req"] = True
        threading.Thread(target=stdin_loop, daemon=True).start()

    t_start = time.monotonic()
    while not stop.is_set():
        time.sleep(1.0 / args.rate if not args.once else 1.0)
        with lock:
            msg, t_rx, n, nbad = st["msg"], st["t"], st["n"], st["crc_bad"]
            rx = np.array(st["rx_times"][-500:])
            ticks = np.array(st["ticks"][-500:], dtype=np.int64)
        lines = []
        if msg is None:
            lines.append(f"waiting for rt/lowstate on iface={args.iface} domain={args.domain} ... ({time.monotonic() - t_start:.0f} s)")
            _show(lines, args)
            if args.once and time.monotonic() - t_start > 5:
                print("NO LOWSTATE RECEIVED")
                sys.exit(2)
            continue
        s = gc.extract_state(msg)
        hz = (len(rx) - 1) / (rx[-1] - rx[0]) if len(rx) > 2 and rx[-1] > rx[0] else float("nan")
        gaps = np.diff(rx) * 1e3 if len(rx) > 2 else np.zeros(1)
        dtick = np.diff(ticks) if len(ticks) > 2 else np.zeros(1, dtype=np.int64)
        tick_rate = (ticks[-1] - ticks[0]) / (rx[-1] - rx[0]) if len(rx) > 2 and rx[-1] > rx[0] else float("nan")
        g = projected_gravity(s.quat)
        mm_ok = "OK" if s.mode_machine == args.expect_mode_machine else f"!! expected {args.expect_mode_machine}"
        if s.mode_machine == gc.MODE_MACHINE_LOCK_WAIST and not args.lock_waist:
            mm_ok += " -- 6 = g1_29dof_lock_waist_rev_1_0 (waist roll/pitch locked): use --lock-waist here and in the runner"
        elif args.lock_waist and s.mode_machine == gc.MODE_MACHINE_FULL:
            mm_ok += " -- 5 = 3-DoF waist: do NOT use --lock-waist (runner refuses this combination)"
        elif args.lock_waist and s.mode_machine == gc.MODE_MACHINE_LOCK_WAIST:
            mm_ok += " (g1_29dof_lock_waist_rev_1_0, waist roll/pitch locked)"
        lines.append(f"rt/lowstate  iface={args.iface} domain={args.domain}  msgs={n}  rate={hz:6.1f} Hz  "
                     f"gap max/p99={gaps.max():.1f}/{np.percentile(gaps, 99):.1f} ms  age={1e3 * (time.monotonic() - t_rx):.1f} ms")
        lines.append(f"tick={s.tick}  tick rate={tick_rate:.1f}/s  dtick/msg min/med/max={dtick.min()}/{int(np.median(dtick))}/{dtick.max()}  "
                     f"(tick constant => sim or frozen)   CRC bad {nbad}/{n}")
        lines.append(f"mode_machine={s.mode_machine} [{mm_ok}]   mode_pr={s.mode_pr} (0=PR, 1=AB)")
        lines.append(mode_line)
        lines.append(f"IMU quat wxyz=[{_f(s.quat)}]  proj_gravity=[{_f(g)}] (upright ~ [0 0 -1])  tilt={np.degrees(np.arccos(np.clip(-g[2], -1, 1))):5.1f} deg")
        lines.append(f"gyro=[{_f(s.gyro)}] rad/s  acc=[{_f(s.acc)}]  rpy=[{_f(s.rpy)}]")
        btn = gc.pressed(s.keys)
        lines.append(f"remote: keys=0x{s.keys:04x}  pressed={'+'.join(btn) if btn else '-'}   "
                     f"(runner combos: L2+up=interp, R1+A=policy, L2+B=DAMP)")
        if sign_j is None:
            lo, hi = cfg.soft_lower, cfg.soft_upper
            lines.append(f"{'#':>2} {'joint':<27}{'q':>8}{'dq':>8}{'q-q0':>8}{'q0':>8}{'soft_lo':>8}{'soft_hi':>8} {'temp':>9} {'mstat':>6} flag")
            for i, jn in enumerate(names):
                flag = ""
                if i in locked:
                    flag = "LOCKED" if gc.waist_active(s, (i,)) is None else "LOCKED-ACTIVE!! (runner: DAMP 'waist unexpectedly active')"
                elif s.q[i] < cfg.joint_lower[i] or s.q[i] > cfg.joint_upper[i]:
                    flag = "OUT-HARD"
                elif s.q[i] < lo[i] or s.q[i] > hi[i]:
                    flag = "OUT-SOFT"
                tmp = "/".join(str(int(x)) for x in s.temp[i])
                lines.append(f"{i:>2} {jn:<27}{s.q[i]:8.3f}{s.dq[i]:8.3f}{s.q[i] - cfg.default_joint_pos[i]:8.3f}"
                             f"{cfg.default_joint_pos[i]:8.3f}{lo[i]:8.3f}{hi[i]:8.3f} {tmp:>9} {int(s.motorstate[i]):>#6x} {flag}")
            live = [i for i in range(len(names)) if i not in locked]
            lines.append(f"max temperature {s.temp[live].max():.0f} C" + (" (27 live motors)" if locked else "")
                         + " (runner: R1+A refused > 70 C, DAMP > 80 C; field semantics and thresholds UNVERIFIED)")
            if locked:
                lines.append(f"waist LOCKED motors: {gc.waist_locked_values(s)}  -> "
                             + ("OK (absent/locked)" if gc.waist_active(s) is None else "ACTIVE?! runner would refuse / DAMP"))
            extra = [f"motors 29..34 q: " + " ".join(f"{msg.motor_state[i].q:.2f}" for i in range(29, 35))]
            lines += extra
        else:
            if zero["req"]:
                zero["q"], zero["req"] = s.q[sign_j], False
            jn = names[sign_j]
            rel = s.q[sign_j] - zero["q"]
            lines.append("")
            lines.append(f"SIGN CHECK  joint {sign_j} {jn}   (Enter = re-zero)")
            lines.append(f"  q = {s.q[sign_j]:+.4f} rad   q - q(at Enter) = {rel:+.4f} rad = {np.degrees(rel):+7.2f} deg   dq = {s.dq[sign_j]:+.3f}")
            lines.append(f"  model hint: {hints.get(jn, '(mujoco/MJCF unavailable)')}")
            lines.append(f"  q0 = {cfg.default_joint_pos[sign_j]:+.3f}  soft [{cfg.soft_lower[sign_j]:+.3f}, {cfg.soft_upper[sign_j]:+.3f}]  "
                         f"hard [{cfg.joint_lower[sign_j]:+.3f}, {cfg.joint_upper[sign_j]:+.3f}]")
            lines.append("  Move the joint by hand (robot in damping / hoisted) in the hinted direction: q - q(at Enter) must grow POSITIVE.")
        _show(lines, args)
        if args.once:
            break
    sub.Close()
    print("bye (read-only: no rt/lowcmd was ever published)")
    os._exit(0)


def _f(v) -> str:
    return " ".join(f"{x:+.3f}" for x in v)


def _show(lines, args):
    if not args.no_clear and not args.once:
        sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
