"""Python side of the C++ DDS bridge (bridge/g1_bridge.cpp, protocol in bridge/g1_bridge_shm.h).

The bridge process does all rt/lowstate / rt/lowcmd traffic; this module reads its shared memory through the ctypes
client (bridge/libg1bridge_client.so: seqlock / ring readers with proper memory barriers) and sends commands to it over a
unix datagram socket. No DDS, no CRC and no per-sample Python callbacks in the runner.

Env: SM_BRIDGE_HOME = directory holding the built g1_bridge + libg1bridge_client.so (bridge/build.sh). Default:
<tree>/bridge when that exists (deploy-tree layout: this file copied to <tree>/scripts/deploy/), else this file's own
directory (repo layout: scripts/deploy/bridge/, where build.sh puts the binaries).
"""

from __future__ import annotations

import ctypes
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

import g1_common as gc

NM = 35
RING = 256
_TREE_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "bridge"
BRIDGE_DIR = Path(os.environ.get("SM_BRIDGE_HOME") or (_TREE_BRIDGE_DIR if _TREE_BRIDGE_DIR.is_dir() else Path(__file__).resolve().parent))

SNAP_DTYPE = np.dtype([
    ("t_rx", "<f8"), ("t_tick_change", "<f8"), ("n_rx", "<u8"), ("n_crc_bad", "<u8"),
    ("tick", "<u4"), ("tick_moving", "<u4"), ("mode_machine", "<u4"), ("mode_pr", "<u4"),
    ("q", "<f8", NM), ("dq", "<f8", NM), ("tau", "<f8", NM), ("temp", "<f8", (NM, 2)),
    ("motorstate", "<u4", NM), ("_pad0", "<u4"),
    ("quat", "<f8", 4), ("gyro", "<f8", 3), ("acc", "<f8", 3), ("rpy", "<f8", 3),
    ("wireless_remote", "u1", 40),
], align=True)
RING_DTYPE = np.dtype([("idx", "<u8"), ("t", "<f8"), ("tick", "<u4"), ("mode_machine", "<u4"), ("keys", "<u4"), ("_pad", "<u4")],
                      align=True)
STATUS_DTYPE = np.dtype([
    ("t_hb", "<f8"), ("n_pub", "<u8"), ("n_pub_fail", "<u8"), ("fail_consec", "<u8"), ("n_cmd_rx", "<u8"),
    ("t_last_cmd", "<f8"), ("lc_count", "<u8"), ("lc_t", "<f8"), ("ready", "<u4"), ("pub_started", "<u4"),
    ("wd_fired", "<u4"), ("quit", "<u4"), ("bridge_pid", "<u4"), ("_pad", "<u4"), ("wd_s", "<f8"),
    ("pub_gap_max", "<f8"), ("n_gap_late", "<u8"), ("pub_write_max", "<f8"),
], align=True)
CMD_MAGIC = 0x47314231
CMD_NOW, CMD_Q_MEASURED, CMD_QUIT = 1, 2, 4
CMD_HDR = struct.Struct(f"<IIQII{NM}s5x")
CMD_SIZE = CMD_HDR.size + 3 * NM * 8


def _load_client(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.g1b_open.restype = ctypes.c_void_p
    lib.g1b_open.argtypes = [ctypes.c_char_p]
    lib.g1b_close.argtypes = [ctypes.c_void_p]
    lib.g1b_sizes.argtypes = [ctypes.c_void_p]
    lib.g1b_snap_offsets.argtypes = [ctypes.c_void_p]
    lib.g1b_status_offsets.argtypes = [ctypes.c_void_p]
    lib.g1b_read_snap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.g1b_read_ring.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.g1b_read_status.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    sz = np.zeros(5, np.int64)
    lib.g1b_sizes(sz.ctypes.data)
    want = (SNAP_DTYPE.itemsize, RING_DTYPE.itemsize, STATUS_DTYPE.itemsize, CMD_SIZE)
    if tuple(int(x) for x in sz[:4]) != want:
        raise SystemExit(f"bridge layout mismatch: C sizes {sz[:4].tolist()} vs Python {list(want)}")
    for fn, dt in ((lib.g1b_snap_offsets, SNAP_DTYPE), (lib.g1b_status_offsets, STATUS_DTYPE)):
        off = np.zeros(64, np.int64)
        n = fn(off.ctypes.data)
        py = [dt.fields[k][1] for k in dt.names]
        if off[:n].tolist() != py:
            raise SystemExit(f"bridge layout mismatch: C offsets {off[:n].tolist()} vs Python {py}")
    return lib


class Bridge:
    """Starts and talks to one g1_bridge process (child, own session: Ctrl-C goes to the runner only)."""

    def __init__(self, domain: int, iface: str | None, crc: bool, hz: float, wd_ms: float, damp_kd: float,
                 log_path: Path, cpu: int = -1, prio: int = 0, orphan_s: float = 10.0,
                 bridge_dir: Path = BRIDGE_DIR, start_timeout: float = 10.0, listen_only: bool = False):
        self.lib = _load_client(bridge_dir / "libg1bridge_client.so")
        tag = f"g1b_{os.getpid()}"
        self.shm_name = "/" + tag
        self.sock_path = f"/tmp/{tag}.sock"
        args = [str(bridge_dir / "g1_bridge"), "--domain", str(domain), "--shm", self.shm_name, "--sock", self.sock_path,
                "--hz", f"{hz:g}", "--wd-ms", f"{wd_ms:g}", "--damp-kd", f"{damp_kd:g}", "--orphan-s", f"{orphan_s:g}",
                "--cpu", str(cpu), "--prio", str(prio)]
        if iface:
            args += ["--iface", iface]
        if crc:
            args += ["--crc"]
        if listen_only:
            args += ["--listen-only"]
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "CYCLONEDDS_URI")}  # SDK libs via rpath
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "ab")
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log, env=env, start_new_session=True)
        self.args = args
        self.shm = None
        t_end = time.monotonic() + start_timeout
        st = np.zeros(1, STATUS_DTYPE)
        while time.monotonic() < t_end:
            if self.p.poll() is not None:
                raise SystemExit(f"g1_bridge exited with {self.p.returncode} at startup (log {log_path})")
            if self.shm is None:
                self.shm = self.lib.g1b_open(self.shm_name.encode()) or None
            if self.shm is not None:
                self.lib.g1b_read_status(self.shm, st.ctypes.data)
                if st["ready"][0]:
                    break
            time.sleep(0.01)
        else:
            self.kill()
            raise SystemExit(f"g1_bridge not ready within {start_timeout} s (log {log_path})")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.connect(self.sock_path)
        self.sock.setblocking(False)
        self.seq = 0
        self.n_send_err = 0
        self.last_send_err = ""
        self.closed = False

    # ------------------------------------------------------------------ reads
    def read_snap(self, buf: np.ndarray) -> bool:
        """Latest snapshot into buf (np.zeros(1, SNAP_DTYPE), one per calling thread). False if no consistent copy."""
        return self.lib.g1b_read_snap(self.shm, buf.ctypes.data) == 0

    def read_ring(self, cur: np.ndarray, out: np.ndarray) -> int:
        """cur: np.zeros(2, np.uint64) = [next index, lost count] (updated); out: np.zeros(RING, RING_DTYPE)."""
        return self.lib.g1b_read_ring(self.shm, cur.ctypes.data, out.ctypes.data, len(out), cur[1:].ctypes.data)

    def read_status(self, buf: np.ndarray) -> np.ndarray:
        self.lib.g1b_read_status(self.shm, buf.ctypes.data)
        return buf

    def alive(self) -> bool:
        return self.p.poll() is None

    # ------------------------------------------------------------------ commands
    def send(self, q, kp, kd, mode: bytes, mode_machine: int, flags: int) -> bool:
        """q/kp/kd: length-35 float64 arrays (q ignored with CMD_Q_MEASURED); mode: 35 bytes."""
        self.seq += 1
        b = CMD_HDR.pack(CMD_MAGIC, flags, self.seq, mode_machine, 0, mode) + np.concatenate((q, kp, kd)).astype("<f8").tobytes()
        try:
            self.sock.send(b)
            return True
        except OSError as e:
            self.n_send_err += 1
            self.last_send_err = repr(e)
            return False

    def quit(self, timeout: float = 2.0):
        """Stop the bridge's rt/lowcmd stream right away and let it exit."""
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.send(CMD_HDR.pack(CMD_MAGIC, CMD_QUIT, self.seq + 1, 0, 0, bytes(NM)) + bytes(3 * NM * 8))
        except OSError:
            pass
        try:
            self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.kill()
        try:
            self.log.close()
        except Exception:  # noqa: BLE001
            pass

    def kill(self):
        try:
            self.p.kill()
            self.p.wait(1.0)
        except Exception:  # noqa: BLE001
            pass
        for f in (self.sock_path, "/dev/shm" + self.shm_name):
            try:
                os.unlink(f)
            except OSError:
                pass


class BridgeWatcher:
    """Same interface as g1_runner.LowcmdWatcher, fed by the bridge's rt/lowcmd watch reader (all publishers)."""

    def __init__(self, bridge: Bridge):
        self.b = bridge
        self.buf = np.zeros(1, STATUS_DTYPE)
        self.count = 0
        self.t = time.monotonic()
        self.ready = True
        self.dead = False

    def poll(self):
        if self.dead:
            return
        if not self.b.alive():
            self.dead = True
            return
        s = self.b.read_status(self.buf)
        self.count, self.t = int(s["lc_count"][0]), time.monotonic()  # live counter: no reporting delay

    def wait_ready(self, timeout: float) -> bool:
        self.poll()
        return not self.dead

    def count_for(self, secs: float) -> int:
        self.poll()
        n0 = self.count
        time.sleep(secs)
        self.poll()
        return self.count - n0

    def close(self):
        pass  # the bridge keeps running (it is our rt/lowcmd publisher); only the runner stops looking


def snap_to_state(s: np.ndarray, n: int = gc.NUM_JOINTS) -> gc.G1State:
    """G1State (as gc.extract_state builds it from a LowState_) from a bridge snapshot record."""
    r = s[0]
    wr = r["wireless_remote"]
    return gc.G1State(
        q=r["q"][:n].astype(float), dq=r["dq"][:n].astype(float), tau=r["tau"][:n].astype(float),
        temp=r["temp"][:n].astype(float), gyro=r["gyro"].astype(float), quat=r["quat"].astype(float),
        acc=r["acc"].astype(float), rpy=r["rpy"].astype(float), tick=int(r["tick"]), mode_machine=int(r["mode_machine"]),
        mode_pr=int(r["mode_pr"]), keys=int(wr[2]) | (int(wr[3]) << 8), motorstate=r["motorstate"][:n].astype(np.int64),
    )
