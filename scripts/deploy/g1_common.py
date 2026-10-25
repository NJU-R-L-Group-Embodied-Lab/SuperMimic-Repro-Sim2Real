"""Shared helpers for the G1 29-DoF real-robot tools (check_robot.py, g1_runner.py, sim_g1_mujoco.py).

Runs in the deploy venv (python3.10 on the deploy PC, python3.8 on the robot's onboard computer; set up under
$G1_DEPLOY_HOME, default ~/workspace/g1_deploy, activated by $G1_DEPLOY_HOME/env.sh): numpy + unitree_sdk2py (+ mujoco, optional).
No torch. supermimic.deploy.core is imported by adding <repo>/src to sys.path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

NUM_JOINTS = 29
NUM_MOTORS_HG = 35
DEFAULT_ARTIFACTS = REPO / "deploy_artifacts" / "ckpt_0116000"
DEFAULT_MJCF = REPO / "deploy_artifacts" / "models" / "g1_train_scene.xml"

# ---------------------------------------------------------------------------------------------- waist-locked G1
# lowstate.mode_machine -> model, from unitree_rl_gym/resources/robots/g1_description/README.md (docs/DEPLOY_G1.md#g1-model-ai-mode-and-debug-mode):
#   5 = g1_29dof_rev_1_0 (3-DoF waist), 6 = g1_29dof_lock_waist_rev_1_0 (waist roll/pitch fixed, motors 13/14 absent).
MODE_MACHINE_FULL = 5
MODE_MACHINE_LOCK_WAIST = 6
LOCKED_WAIST_IDX = (13, 14)  # waist_roll_joint, waist_pitch_joint
# A locked/absent motor reads q = dq = 0 and temperature 0/0 on the real robot (check_robot, 2026-10-23). Anything
# else means the waist is NOT what we think it is.
WAIST_ACTIVE_Q = 0.02  # rad
# --lock-waist clip policy (docs/SIM2SIM.md#sim2sim-evaluation-of-the-waist-locked-g1, ckpt_0144000): only these run without an override;
# the pelvis leans 7-10 deg forward on all of them. idle_loop_002 / itching / walk_sideway need --i-know-clip <name>
# (tilt margin to the 20 deg abort only ~3 deg; walk_sideway false-aborts under kp/mass perturbation).
LOCK_WAIST_ALLOWED_CLIPS = ("neutral_idle_loop_001__A081", "shoulder_clap_004__A041_M")
# never with a locked waist (sim2sim: jog_sideway fell, dance/jog tilt-aborted); no override flag on purpose.
LOCK_WAIST_FORBIDDEN_CLIPS = ("wave_R_001__A431", "jog_sideway_045_loop_002__A025", "dance_hiphop_alt_stump_R_fast_001__A318_M")
# --lock-waist: clips allowed to use POLICY STANDBY (R1+A -> STANDBY_PRE -> R1+X). Empty until a lock-waist standby sim gate
# passes (review 2026-10-24: shoulder_clap after-motion standby 0/6, idle pre-standby R1+X check 79% / 82 cm drift).
LOCK_WAIST_STANDBY_ALLOWED_CLIPS: tuple[str, ...] = ()


def waist_locked_values(st, idx=LOCKED_WAIST_IDX) -> str:
    return "  ".join(f"m{i} q={st.q[i]:+.4f} dq={st.dq[i]:+.4f} temp={'/'.join(str(int(x)) for x in st.temp[i])} "
                     f"mstat=0x{int(st.motorstate[i]):x}" for i in idx)


def waist_active(st, idx=LOCKED_WAIST_IDX) -> str | None:
    """None if motors 13/14 look absent/locked (|q| <= WAIST_ACTIVE_Q, temperature 0), else a description."""
    bad = [i for i in idx if not np.isfinite(st.q[i]) or abs(st.q[i]) > WAIST_ACTIVE_Q or np.any(st.temp[i] != 0)]
    return None if not bad else waist_locked_values(st, bad)


def lock_waist_clip_check(clip_paths, allow_file=None, i_know=()) -> list[str]:
    """--lock-waist clip gate. Returns printable notes; raises SystemExit for a clip that is not allowed."""
    allowed = set(LOCK_WAIST_ALLOWED_CLIPS)
    src = "built-in list"
    if allow_file is not None:
        allowed = {ln.split("#")[0].strip() for ln in Path(allow_file).read_text().splitlines()} - {""}
        src = f"allow file {allow_file}"
    notes = []
    for p in clip_paths:
        name = Path(p).stem
        if name in LOCK_WAIST_FORBIDDEN_CLIPS:
            raise SystemExit(f"REFUSED: clip {name} is a HARDER clip and is never run with --lock-waist (sim2sim: falls / tilt "
                             "aborts on the waist-locked model). There is no override.")
        if name in allowed:
            notes.append(f"{name}: allowed ({src})")
        elif name in i_know:
            notes.append(f"{name}: NOT on the lock-waist allow list, accepted because of --i-know-clip "
                         "(sim2sim: tilt margin ~3 deg to the 20 deg abort)")
        else:
            raise SystemExit(f"REFUSED: clip {name} is not on the --lock-waist allow list ({src}: {sorted(allowed)}). Pass "
                             f"--i-know-clip {name} only with explicit approval (docs/DEPLOY_G1.md#waist-locked-g1-mode_machine--6).")
    return notes

# ---------------------------------------------------------------------------------------------- buttons
# Bit index in the uint16 formed by wireless_remote[2] | wireless_remote[3] << 8.
# Verified against the cloned sources (see docs/DEPLOY_G1.md#runner-state-machine-and-remote-controller-buttons):
#   unitree_sdk2_python/example/wireless_controller/wireless_controller.py  parse_botton(data1=wr[2], data2=wr[3])
#   unitree_sdk2_python/unitree_sdk2py/utils/joystick.py                    extract()/combine() (MSB-first strings)
#   unitree_sdk2_python/example/h2/low_level/h2_ankle_swing_example.py      struct "<H" at offset 2, A = 1 << 8
#   unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py                 key_map + wireless_remote[2]/[3] packing
# X (R1+X = start the motion from policy standby) re-checked 2026-10-24 in the same four files: wireless_controller.py
# X = (data2 >> 2) & 1 (data2 = wr[3] -> bit 8 + 2 = 10); joystick.py X(button2[5]) with button2 = f'{wr[3]:08b}' (MSB first,
# index 5 = bit 2 -> 10); the bridge packs wr[3] MSB first as [left, down, right, up, Y, X, B, A] (X = bit 2 -> 10) and its
# key_map has "X": 10; h2_ankle_swing_example.py gamepad_X = btn_value & (1 << 10). On the real remote: UNVERIFIED (6.1).
BUTTON_BITS = {
    "R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
    "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15,
}


def remote_keys(wireless_remote) -> int:
    return int(wireless_remote[2]) | (int(wireless_remote[3]) << 8)


def pressed(keys: int) -> list[str]:
    return [n for n, b in BUTTON_BITS.items() if keys >> b & 1]


def combo_mask(*names: str) -> int:
    m = 0
    for n in names:
        m |= 1 << BUTTON_BITS[n]
    return m


def keys_from_names(names) -> tuple[int, int]:
    """(wireless_remote[2], wireless_remote[3]) for a set of button names (used by the sim to inject presses)."""
    k = combo_mask(*names) if names else 0
    return k & 0xFF, (k >> 8) & 0xFF


class ComboEdge:
    """Edge-triggered combo detector: fires once when all buttons of the combo become pressed together.
    (Kept for reference; the runner uses ComboDebounce.)"""

    def __init__(self, combos: dict[str, int]):
        self.combos = combos
        self.prev = 0

    def update(self, keys: int) -> list[str]:
        fired = [name for name, m in self.combos.items() if (keys & m) == m and (self.prev & m) != m]
        self.prev = keys
        return fired


class ComboDebounce:
    """Debounced combo detector for the runner.

    * the key word must be identical for >= `stable_n` consecutive messages before it counts;
    * a combo fires once when it becomes (stably) held, and only if it was ARMED: armed means the combo was
      seen (stably) released for >= `release_s`. Nothing is armed at construction, so a combo that is held
      when the program starts never fires until it has been released and pressed again;
    * combos listed in `no_arm` (DAMP) fire whenever they are stably held (idempotent, safe direction),
      at most once per `release_s` while held.
    """

    def __init__(self, combos: dict[str, int], stable_n: int = 3, release_s: float = 0.1, no_arm=("damp",)):
        self.combos, self.stable_n, self.release_s, self.no_arm = combos, stable_n, release_s, set(no_arm)
        self.cand, self.cand_n = None, 0
        self.armed = {k: False for k in combos}
        self.released_t = {k: None for k in combos}
        self.last_fire = {k: -1e9 for k in combos}

    def update(self, keys: int, t: float) -> list[str]:
        if keys == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = keys, 1
        if self.cand_n < self.stable_n:
            return []
        fired = []
        for name, m in self.combos.items():
            if (keys & m) == m:
                self.released_t[name] = None
                if name in self.no_arm:
                    if t - self.last_fire[name] >= self.release_s:
                        fired.append(name)
                        self.last_fire[name] = t
                elif self.armed[name]:
                    fired.append(name)
                    self.armed[name] = False
            else:
                if self.released_t[name] is None:
                    self.released_t[name] = t
                if t - self.released_t[name] >= self.release_s:
                    self.armed[name] = True
        return fired


# ---------------------------------------------------------------------------------------------- state
@dataclass
class G1State:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    # (29, 2) int16 MotorState_.temperature as reported. UNVERIFIED which of the two values is the winding
    # and which the housing/driver temperature, and the unit (assumed deg C); the runner uses the max of both.
    temp: np.ndarray
    gyro: np.ndarray
    quat: np.ndarray  # w, x, y, z
    acc: np.ndarray
    rpy: np.ndarray
    tick: int
    mode_machine: int
    mode_pr: int
    keys: int
    motorstate: np.ndarray  # (29,) MotorState_.motorstate word; bit meanings UNVERIFIED (0 assumed = healthy)


def extract_state(msg) -> G1State:
    ms = msg.motor_state
    q = np.fromiter((ms[i].q for i in range(NUM_JOINTS)), float, NUM_JOINTS)
    dq = np.fromiter((ms[i].dq for i in range(NUM_JOINTS)), float, NUM_JOINTS)
    tau = np.fromiter((ms[i].tau_est for i in range(NUM_JOINTS)), float, NUM_JOINTS)
    temp = np.array([list(ms[i].temperature) for i in range(NUM_JOINTS)], dtype=float)
    mstat = np.fromiter((ms[i].motorstate for i in range(NUM_JOINTS)), np.int64, NUM_JOINTS)
    imu = msg.imu_state
    return G1State(
        q=q, dq=dq, tau=tau, temp=temp,
        gyro=np.asarray(imu.gyroscope, float), quat=np.asarray(imu.quaternion, float),
        acc=np.asarray(imu.accelerometer, float), rpy=np.asarray(imu.rpy, float),
        tick=int(msg.tick), mode_machine=int(msg.mode_machine), mode_pr=int(msg.mode_pr),
        keys=remote_keys(msg.wireless_remote), motorstate=mstat,
    )


def check_motion_mode(timeout: float = 2.0):
    """READ-ONLY query of Unitree's motion switcher service (MotionSwitcherClient.CheckMode, api 1001).

    Returns (code, name, raw): code 0 = answered; name = active motion mode ('' / None = no mode active, i.e.
    the built-in locomotion controller is released). Never calls SelectMode/ReleaseMode. Requires DDS to be
    initialised. Whether the service is reachable from an external PC on 192.168.123.x is UNVERIFIED.
    API checked in unitree_sdk2_python/unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py and
    example/g1/low_level/g1_low_level_example.py (which, unlike us, calls ReleaseMode in a loop).
    """
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    c = MotionSwitcherClient()
    c.SetTimeout(timeout)
    c.Init()
    code, res = c.CheckMode()
    name = res.get("name") if isinstance(res, dict) else None
    return code, name, res


def init_dds(domain: int, iface: str | None):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(domain, iface) if iface else ChannelFactoryInitialize(domain)


# ---------------------------------------------------------------------------------------------- sign hints
def sign_hints(cfg, mjcf: Path = DEFAULT_MJCF) -> dict[str, str]:
    """Human-readable 'what does +q do' per joint, from the training MJCF at the q0 pose (pelvis frame:
    +x forward, +y robot-left, +z up). Uses mujoco if available; otherwise returns {}."""
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return {}
    try:
        m = mujoco.MjModel.from_xml_path(str(mjcf))
    except Exception as e:  # noqa: BLE001
        print(f"[sign] cannot load {mjcf}: {e}")
        return {}
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0  # identity pelvis orientation
    names = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        names[nm.split("/")[-1]] = j
    for i, jn in enumerate(cfg.joint_names):
        d.qpos[m.jnt_qposadr[names[jn]]] = cfg.default_joint_pos[i]
    mujoco.mj_forward(m, d)
    words = {0: ("forward", "backward"), 1: ("to robot-left", "to robot-right"), 2: ("up", "down")}
    out = {}
    for jn in cfg.joint_names:
        j = names[jn]
        axis, anchor = d.xaxis[j].copy(), d.xanchor[j].copy()
        b = m.jnt_bodyid[j]
        # farthest geom (of the moving subtree) from the joint axis = the point that visibly moves
        geoms = [g for g in range(m.ngeom) if _is_desc(m, m.geom_bodyid[g], b)]
        lever = lambda g: np.linalg.norm(np.cross(axis, d.geom_xpos[g] - anchor))  # noqa: E731
        tip = max(geoms, key=lever)
        tip_name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[tip]) or "").split("/")[-1]
        tip_name = f"farthest point of {tip_name}"
        v = np.cross(axis, d.geom_xpos[tip] - anchor)
        order = np.argsort(-np.abs(v))
        parts = [words[k][0] if v[k] > 0 else words[k][1] for k in order[:2] if abs(v[k]) > 0.25 * np.abs(v).max()]
        ax = " ".join(f"{a:+.2f}" for a in axis)
        sgn = lambda k: 1 if axis[k] > 0 else -1  # noqa: E731
        rot = ""
        if abs(axis[2]) > 0.8:  # yaw
            rot = " (yaw: CCW seen from above, link turns to robot-left)" if sgn(2) > 0 else " (yaw: CW seen from above)"
        elif abs(axis[1]) > 0.8:  # pitch about the left axis: front goes down
            rot = " (pitch: link's front edge goes DOWN, back edge up)" if sgn(1) > 0 else " (pitch: front edge goes UP)"
        elif abs(axis[0]) > 0.8:  # roll about the forward axis: left side goes up
            rot = " (roll: link's LEFT side goes UP, right side down)" if sgn(0) > 0 else " (roll: left side goes DOWN)"
        out[jn] = f"axis(pelvis)=[{ax}]{rot}; +q moves {tip_name} {' & '.join(parts)}"
    return out


def _is_desc(m, k: int, b: int) -> bool:
    while k > 0:
        if k == b:
            return True
        k = m.body_parentid[k]
    return False


def now() -> float:
    return time.monotonic()
