#!/usr/bin/env python3
"""Headless unitree_mujoco G1 29-DoF simulator for end-to-end tests of g1_runner.py over DDS (sim only).

Derived from a headless unitree_mujoco G1 sim script of our deploy env setup; reuses Unitree's
simulate_python/unitree_sdk2py_bridge.py unchanged (rt/lowstate @ 200 Hz, rt/lowcmd subscriber) and adds:
  --band-file F     ElasticBand on torso_link is ON while file F exists (the runner creates/removes it)
  --buttons-file F  contents like "L2+up" are written into lowstate.wireless_remote (button injection;
                    the file is re-read every 10 ms, an empty/missing file = no button)
  --log F.npz       pelvis height / tilt / band state / pelvis x, y, yaw and the wall clock (time.monotonic, to align
                    with the runner's t0_monotonic) at 50 Hz, rewritten every second
  --log-tau         also log every physics step: actuator force (after ctrlrange clipping) and the sum of
                    the commanded kp (to locate gain switches), for torque-step analysis (29-motor layout)
  --lock-waist      "G1 29dof with waist locked" (mode_machine 6, g1_29dof_lock_waist_rev_1_0): waist_roll/pitch are
                    deleted from the MJCF with MjSpec (torso welded at q = 0, as Unitree's lock-waist MJCF and
                    sim2sim.py --lock-waist do); lowstate still carries 29 motors, 13/14 report q = dq = tau = 0 and
                    temperature 0, live motors --motor-temp; commands for 13/14 are ignored (and counted if non-zero)
  --mode-machine N  lowstate.mode_machine (default 6 with --lock-waist, else 5; upstream bridge sent 0)
  --waist-active-file F  (--lock-waist) while F exists, motor 14 reports q = 0.05 rad and temperature 25 (fault injection)
  --temp-file F     temperature injection (thermal-stop / forecast tests): lines "motor base rate [offset]"; while F exists
                    motor reports [base + rate * s, base + rate * s - offset] (rounded), s = seconds since F's content last
                    changed; the other motors report --motor-temp
PD is recomputed every 5 ms physics step from the latest LowCmd (like the motor drivers / training);
physics stays frozen at the init pose (state still published) until the first rt/lowcmd arrives.

Env: G1_DEPLOY_HOME (default ~/workspace/g1_deploy) must contain a unitree_mujoco checkout (unitree_mujoco/; its
simulate_python bridge is imported and unitree_robots/g1/scene_29dof.xml is the default --scene).

  source $G1_DEPLOY_HOME/env.sh   # default ~/workspace/g1_deploy
  python scripts/deploy/sim_g1_mujoco.py --band-file /tmp/band --log logs/sim.npz
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import types

import numpy as np

UM = os.path.join(os.path.expanduser(os.environ.get("G1_DEPLOY_HOME", "~/workspace/g1_deploy")), "unitree_mujoco")
ap = argparse.ArgumentParser()
ap.add_argument("--domain", type=int, default=1)
ap.add_argument("--iface", default="lo")
ap.add_argument("--scene", default=os.path.join(UM, "unitree_robots/g1/scene_29dof.xml"))
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--duration", type=float, default=0.0, help="sim seconds, 0 = forever")
ap.add_argument("--init", choices=["zero", "knees_bent"], default="knees_bent")
ap.add_argument("--pelvis-z", type=float, default=0.76)
ap.add_argument("--band-file", default=None)
ap.add_argument("--band-length", type=float, default=1.0)
ap.add_argument("--buttons-file", default=None)
ap.add_argument("--log", default=None)
ap.add_argument("--log-tau", action="store_true")
ap.add_argument("--status-every", type=float, default=1.0)
ap.add_argument("--lock-waist", action="store_true")
ap.add_argument("--mode-machine", type=int, default=None)
ap.add_argument("--motor-temp", type=int, default=None, help="temperature of the live motors (default 30 with --lock-waist, else 0)")
ap.add_argument("--waist-active-file", default=None)
ap.add_argument("--temp-file", default=None)
args = ap.parse_args()
if args.mode_machine is None:
    args.mode_machine = 6 if args.lock_waist else 5
if args.motor_temp is None:
    args.motor_temp = 30 if args.lock_waist else 0
if args.domain == 0:
    sys.exit("refusing to run the simulator on DDS domain 0 (real robot domain)")

# The upstream bridge does `import config` and reads config.ROBOT; provide it without touching upstream files.
cfgmod = types.ModuleType("config")
cfgmod.ROBOT, cfgmod.ROBOT_SCENE, cfgmod.DOMAIN_ID, cfgmod.INTERFACE = "g1", args.scene, args.domain, args.iface
cfgmod.USE_JOYSTICK, cfgmod.ENABLE_ELASTIC_BAND, cfgmod.SIMULATE_DT, cfgmod.PRINT_SCENE_INFORMATION = 0, True, args.dt, False
sys.modules["config"] = cfgmod
sys.path.insert(0, os.path.join(UM, "simulate_python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge  # noqa: E402

import g1_common as gc  # noqa: E402

KNEES_BENT = {}
for side in ("left", "right"):
    KNEES_BENT.update({f"{side}_hip_pitch_joint": -0.312, f"{side}_knee_joint": 0.669, f"{side}_ankle_pitch_joint": -0.363,
                       f"{side}_elbow_joint": 0.6, f"{side}_shoulder_pitch_joint": 0.2})
KNEES_BENT["left_shoulder_roll_joint"], KNEES_BENT["right_shoulder_roll_joint"] = 0.2, -0.2

LOCKED_NAMES = ("waist_roll_joint", "waist_pitch_joint")
NM = gc.NUM_JOINTS  # motors in the lowstate / lowcmd layout (always 29)


def _short(n):
    return (n or "").split("/")[-1]


m_full = mujoco.MjModel.from_xml_path(args.scene)
# motor order = actuator order of the full 29-DoF model (what the upstream bridge assumes)
MOTOR_JOINTS = [_short(mujoco.mj_id2name(m_full, mujoco.mjtObj.mjOBJ_JOINT, m_full.actuator_trnid[i, 0])) for i in range(m_full.nu)]
assert len(MOTOR_JOINTS) == NM, MOTOR_JOINTS
if args.lock_waist:
    spec = mujoco.MjSpec.from_file(args.scene)
    for a_ in list(spec.actuators):
        if _short(a_.target) in LOCKED_NAMES:
            spec.delete(a_)
    for s_ in list(spec.sensors):
        if _short(s_.objname) in LOCKED_NAMES:
            spec.delete(s_)
    for k_ in list(spec.keys):
        spec.delete(k_)
    for j_ in list(spec.joints):
        if _short(j_.name) in LOCKED_NAMES:
            spec.delete(j_)
    m = spec.compile()
    assert m.nu == NM - 2 and not set(LOCKED_NAMES) & {_short(m.joint(i).name) for i in range(m.njnt)}
else:
    m = m_full
m.opt.timestep = args.dt
d = mujoco.MjData(m)
nu = m.nu
# actuator i drives motor ACT_MOTOR[i]; motor sensors are jointpos/jointvel/jointactuatorfrc in actuator order (checked)
ACT_MOTOR = np.array([MOTOR_JOINTS.index(_short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])))
                      for i in range(nu)])
for i in range(nu):
    for blk in range(3):
        sid = blk * nu + i
        assert m.sensor_objid[sid] == m.actuator_trnid[i, 0], ("sensor layout", i, blk)
LOCKED_MOTORS = sorted(set(range(NM)) - set(ACT_MOTOR.tolist()))
if args.init == "knees_bent":
    d.qpos[2] = args.pelvis_z
    for name, val in KNEES_BENT.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = val
mujoco.mj_forward(m, d)
lock = threading.Lock()
buttons = {"b2": 0, "b3": 0, "txt": ""}
waist = {"active": False}
temps = {"txt": "", "t0": 0.0, "rows": []}  # --temp-file: (motor, base, rate, offset)


def temp_of(i):
    for m, base, rate, off in temps["rows"]:
        if m == i:
            v = base + rate * (time.monotonic() - temps["t0"])
            return [int(round(v)), int(round(v - off))]
    return [args.motor_temp, args.motor_temp]


class Bridge(UnitreeSdk2Bridge):
    def __init__(self, *a, **kw):
        self.cmd = np.zeros((5, NM))  # q, dq, kp, kd, tau per MOTOR (29)
        self.n_cmd = 0
        self.n_locked_nonzero = 0  # lowcmds with anything but mode=0 / all-zero on a locked motor
        self.locked_seen = np.zeros(6)  # max |mode, q, dq, kp, kd, tau| seen on locked motors
        super().__init__(*a, **kw)

    def LowCmdHandler(self, msg):
        mcs = msg.motor_cmd
        c = np.array([[mc.q, mc.dq, mc.kp, mc.kd, mc.tau] for mc in mcs[:NM]]).T
        with lock:
            self.cmd[:] = c
            self.n_cmd += 1
            for i in LOCKED_MOTORS:
                v = np.abs([mcs[i].mode, mcs[i].q, mcs[i].dq, mcs[i].kp, mcs[i].kd, mcs[i].tau])
                self.locked_seen = np.maximum(self.locked_seen, v)
                if v.any():
                    self.n_locked_nonzero += 1

    def PublishLowState(self):
        ls = self.low_state
        ls.wireless_remote[2] = buttons["b2"]
        ls.wireless_remote[3] = buttons["b3"]
        ls.mode_machine = args.mode_machine
        if not args.lock_waist:
            if args.motor_temp or temps["rows"]:
                for i in range(NM):
                    ls.motor_state[i].temperature = temp_of(i)
            super().PublishLowState()
            return
        sd = self.mj_data.sensordata
        for k in range(nu):
            ms = ls.motor_state[int(ACT_MOTOR[k])]
            ms.q, ms.dq, ms.tau_est = sd[k], sd[nu + k], sd[2 * nu + k]
            ms.temperature = [args.motor_temp, args.motor_temp]
        for i in LOCKED_MOTORS:  # absent motors: zeros (as the real waist-locked G1 reports)
            ms = ls.motor_state[i]
            ms.q, ms.dq, ms.tau_est, ms.temperature = 0.0, 0.0, 0.0, [0, 0]
        if waist["active"]:  # fault injection: motor 14 "answers"
            ls.motor_state[14].q, ls.motor_state[14].temperature = 0.05, [25, 25]
        b = 3 * nu  # imu_quat, imu_gyro, imu_acc follow the motor sensors (same as the upstream bridge)
        for k in range(4):
            ls.imu_state.quaternion[k] = sd[b + k]
        for k in range(3):
            ls.imu_state.gyroscope[k] = sd[b + 4 + k]
            ls.imu_state.accelerometer[k] = sd[b + 7 + k]
        self.low_state_puber.Write(ls)


def button_loop():
    while True:
        txt = ""
        if args.buttons_file and os.path.exists(args.buttons_file):
            try:
                txt = open(args.buttons_file).read().strip()
            except OSError:
                pass
        tt = ""
        if args.temp_file and os.path.exists(args.temp_file):
            try:
                tt = open(args.temp_file).read().strip()
            except OSError:
                tt = temps["txt"]
        if tt != temps["txt"]:
            rows = []
            for ln in tt.splitlines():
                f = ln.split("#")[0].split()
                if len(f) >= 3:
                    rows.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]) if len(f) > 3 else 0.0))
            temps["txt"], temps["t0"], temps["rows"] = tt, time.monotonic(), rows
            print(f"[sim] temperature injection -> {rows or 'off'}", flush=True)
        wa = bool(args.waist_active_file) and os.path.exists(args.waist_active_file)
        if wa != waist["active"]:
            waist["active"] = wa
            print(f"[sim] waist fault injection {'ON (motor 14 q=0.05, temp 25)' if wa else 'OFF'}", flush=True)
        if txt != buttons["txt"]:
            names = [t for t in txt.replace(",", "+").split("+") if t]
            try:
                buttons["b2"], buttons["b3"] = gc.keys_from_names(names)
                print(f"[sim] buttons -> {names or '-'}", flush=True)
            except KeyError as e:
                print(f"[sim] unknown button {e}", flush=True)
            buttons["txt"] = txt
        time.sleep(0.01)


ChannelFactoryInitialize(args.domain, args.iface)
bridge = Bridge(m, d)
threading.Thread(target=button_loop, daemon=True).start()
band = ElasticBand()
band.length = args.band_length
band_body = m.body("torso_link").id
pelvis = m.body("pelvis").id
print(f"[sim] scene={args.scene} nu={nu} dt={m.opt.timestep} domain={args.domain} iface={args.iface} "
      f"init={args.init} band_file={args.band_file} band_length={args.band_length} mode_machine={args.mode_machine} "
      f"lock_waist={args.lock_waist} locked motors={LOCKED_MOTORS} motor_temp={args.motor_temp} mass={m.body_subtreemass[0]:.3f}", flush=True)

log = {"t": [], "pelvis_z": [], "up_z": [], "band": [], "n_cmd": [], "wall": [], "pelvis_x": [], "pelvis_y": [], "yaw": []}
tlog = {"tau_t": [], "tau": [], "pd": [], "kp_sum": [], "tau_wall": []}


def save_log():
    if args.log:
        log["locked_cmd_nonzero"] = [bridge.n_locked_nonzero]
        tmp = args.log + ".tmp.npz"
        extra = {}
        if args.log_tau and tlog["tau_t"]:
            extra = {"tau_t": np.asarray(tlog["tau_t"]), "tau": np.asarray(tlog["tau"], np.float32),
                     "kp_sum": np.asarray(tlog["kp_sum"], np.float32),
                     "pd": np.asarray(tlog["pd"], np.float32), "tau_wall": np.asarray(tlog["tau_wall"])}
        np.savez(tmp, **{k: np.asarray(v) for k, v in log.items()}, **extra)
        os.replace(tmp, args.log)


def on_term(*_):
    save_log()
    print("[sim] exit (signal)", flush=True)
    os._exit(0)


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)

print("[sim] physics frozen until first rt/lowcmd arrives ...", flush=True)
while bridge.n_cmd == 0:
    time.sleep(0.001)
print("[sim] first LowCmd received, starting physics", flush=True)

t_start = time.perf_counter()
next_status, next_save, steps = args.status_every, 1.0, 0
band_on_prev = None
while True:
    t0 = time.perf_counter()
    band_on = bool(args.band_file) and os.path.exists(args.band_file) if steps % 10 == 0 else band_on_prev
    if band_on != band_on_prev:
        print(f"[sim] t={d.time:7.2f}s elastic band {'ON' if band_on else 'OFF'}", flush=True)
        band_on_prev = band_on
    with lock:
        q, dq = d.sensordata[:nu], d.sensordata[nu:2 * nu]
        c = bridge.cmd[:, ACT_MOTOR]  # commands for locked (absent) motors are ignored
        d.ctrl[:] = c[4] + c[2] * (c[0] - q) + c[3] * (c[1] - dq)
        d.xfrc_applied[band_body, :3] = band.Advance(d.qpos[:3], d.qvel[:3]) if band_on else 0.0
        mujoco.mj_step(m, d)
        if args.log_tau:
            tlog["tau_t"].append(d.time)
            tlog["tau_wall"].append(time.monotonic())
            t29, p29 = np.zeros(NM), np.zeros(NM)
            t29[ACT_MOTOR], p29[ACT_MOTOR] = d.actuator_force[:nu], d.ctrl[:nu]  # commanded PD before ctrlrange clipping
            tlog["tau"].append(t29)
            tlog["kp_sum"].append(float(c[2].sum()))  # applied (live) motors only
            tlog["pd"].append(p29)
    steps += 1
    if steps % 4 == 0:
        log["t"].append(d.time)
        log["pelvis_z"].append(d.xpos[pelvis, 2])
        log["up_z"].append(d.xmat[pelvis].reshape(3, 3)[2, 2])  # cos(tilt) of the pelvis z axis
        log["band"].append(band_on)
        log["n_cmd"].append(bridge.n_cmd)
        log["wall"].append(time.monotonic())
        log["pelvis_x"].append(d.xpos[pelvis, 0])
        log["pelvis_y"].append(d.xpos[pelvis, 1])
        xm = d.xmat[pelvis]
        log["yaw"].append(float(np.arctan2(xm[3], xm[0])))  # heading of the pelvis x axis
    if d.time >= next_status:
        wall = time.perf_counter() - t_start
        print(f"[sim] t={d.time:7.2f}s rtf={d.time / wall:4.2f} pelvis_z={d.xpos[pelvis, 2]:.3f} "
              f"tilt={np.degrees(np.arccos(np.clip(d.xmat[pelvis][8], -1, 1))):5.1f}deg band={int(band_on)} lowcmd_rx={bridge.n_cmd}"
              + (f" locked_cmd_nonzero={bridge.n_locked_nonzero} max|mode,q,dq,kp,kd,tau|={np.round(bridge.locked_seen, 4).tolist()}"
                 if LOCKED_MOTORS else ""), flush=True)
        next_status += args.status_every
    if d.time >= next_save:
        save_log()
        next_save += 1.0
    if args.duration > 0 and d.time >= args.duration:
        break
    rem = m.opt.timestep - (time.perf_counter() - t0)
    if rem > 0:
        time.sleep(rem)
save_log()
print("[sim] exit", flush=True)
os._exit(0)
