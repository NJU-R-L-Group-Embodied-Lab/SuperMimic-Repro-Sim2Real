// Shared-memory / socket protocol between g1_bridge (C++, unitree_sdk2 DDS I/O) and the Python runner
// (scripts/deploy/g1_runner.py, via the ctypes client libg1bridge_client.so built from g1_bridge_client.cpp).
//
//   bridge -> runner : POSIX shm /dev/shm/<name>, layout G1BShm below.
//                      snap   : latest CRC-good rt/lowstate (seqlock, written by the DDS listener thread)
//                      ring   : per-sample (t, tick, mode_machine, remote keys) of every accepted lowstate, so the runner
//                               runs its button debouncer / mode_machine check on every sample as before
//                      status : counters of the lowcmd publisher / watchdog / rt/lowcmd watch (single atomics)
//   runner -> bridge : unix datagram socket <sock>, one G1BCmd per datagram (latest wins).
//
// All times are CLOCK_MONOTONIC seconds (= Python time.monotonic() on Linux).
#pragma once
#include <stdint.h>

#define G1B_MAGIC 0x47314231u  // "G1B1"
#define G1B_VERSION 1u
#define G1B_NM 35               // motors in unitree_hg LowState_/LowCmd_
#define G1B_RING 256            // power of two

// ---------------------------------------------------------------------------------------- bridge -> runner
typedef struct {
  double t_rx;           // receive time of this sample
  double t_tick_change;  // last time lowstate.tick changed
  uint64_t n_rx;         // accepted samples so far
  uint64_t n_crc_bad;    // dropped (bad CRC) samples so far
  uint32_t tick;
  uint32_t tick_moving;  // 1 once tick has changed at least once
  uint32_t mode_machine;
  uint32_t mode_pr;
  double q[G1B_NM];
  double dq[G1B_NM];
  double tau[G1B_NM];
  double temp[G1B_NM][2];
  uint32_t motorstate[G1B_NM];
  uint32_t _pad0;
  double quat[4];  // w x y z
  double gyro[3];
  double acc[3];
  double rpy[3];
  uint8_t wireless_remote[40];
} G1BSnap;

typedef struct {
  uint64_t idx;  // sample index (= n_rx - 1 at write time)
  double t;
  uint32_t tick;
  uint32_t mode_machine;
  uint32_t keys;  // wireless_remote[2] | wireless_remote[3] << 8
  uint32_t _pad;
} G1BRingEntry;

typedef struct {
  double t_hb;            // publisher thread heartbeat (every loop)
  uint64_t n_pub;         // rt/lowcmd Write() calls
  uint64_t n_pub_fail;    // Write() returned false
  uint64_t fail_consec;   // consecutive failed writes
  uint64_t n_cmd_rx;      // G1BCmd datagrams received from the runner
  double t_last_cmd;      // receive time of the last G1BCmd
  uint64_t lc_count;      // rt/lowcmd samples seen by the watch reader (all publishers, incl. our own if delivered)
  double lc_t;            // time of lc_count
  uint32_t ready;         // 1 once readers + socket are up
  uint32_t pub_started;   // 1 once the rt/lowcmd stream runs (first G1BCmd)
  uint32_t wd_fired;      // latched: runner silent > watchdog (or gone) -> bridge publishes damping itself
  uint32_t quit;          // 1 after a QUIT command (no more rt/lowcmd from the bridge)
  uint32_t bridge_pid;
  uint32_t _pad;
  double wd_s;            // watchdog limit
  double pub_gap_max;     // largest interval between consecutive rt/lowcmd writes since the stream started [s]
  uint64_t n_gap_late;    // intervals > 2 periods
  double pub_write_max;   // slowest Write() call [s]
} G1BStatus;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint64_t snap_seq;   // seqlock: odd while the listener writes snap
  G1BSnap snap;
  uint64_t ring_head;  // next sample index to be written
  G1BRingEntry ring[G1B_RING];
  G1BStatus status;
} G1BShm;

// ---------------------------------------------------------------------------------------- runner -> bridge
#define G1B_CMD_NOW 1u          // publish right away (policy step, DAMP) in addition to the fixed-rate stream
#define G1B_CMD_Q_MEASURED 2u   // q = latest measured lowstate q (PASSIVE / DAMP)
#define G1B_CMD_QUIT 4u         // stop publishing and exit

typedef struct {
  uint32_t magic;
  uint32_t flags;
  uint64_t seq;
  uint32_t mode_machine;
  uint32_t _pad;
  uint8_t mode[G1B_NM];
  uint8_t _pad2[5];
  double q[G1B_NM];
  double kp[G1B_NM];
  double kd[G1B_NM];
} G1BCmd;  // 904 bytes

// ---------------------------------------------------------------------------------------- lock-free readers
// Seqlock read of the latest snapshot. Returns 0 on success, -1 if no consistent copy within `tries`.
static inline int g1b_snap_read(const G1BShm* s, G1BSnap* out, int tries) {
  for (int k = 0; k < tries; k++) {
    uint64_t a = __atomic_load_n(&s->snap_seq, __ATOMIC_ACQUIRE);
    if (a & 1u) continue;
    __builtin_memcpy(out, (const void*)&s->snap, sizeof(G1BSnap));
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    uint64_t b = __atomic_load_n(&s->snap_seq, __ATOMIC_RELAXED);
    if (a == b) return 0;
  }
  return -1;
}

// Copy ring entries with idx in [*from, head) (at most the newest G1B_RING, at most `max`). Advances *from past the
// entries returned; entries overwritten while being read are skipped (*n_lost counts every skipped index).
static inline int g1b_ring_read(const G1BShm* s, uint64_t* from, G1BRingEntry* out, int max, uint64_t* n_lost) {
  uint64_t head = __atomic_load_n(&s->ring_head, __ATOMIC_ACQUIRE);
  uint64_t i = *from;
  if (head > i + G1B_RING) { *n_lost += head - G1B_RING - i; i = head - G1B_RING; }
  int n = 0;
  for (; i < head && n < max; i++) {
    const G1BRingEntry* e = &s->ring[i & (G1B_RING - 1)];
    uint64_t a = __atomic_load_n(&e->idx, __ATOMIC_ACQUIRE);
    G1BRingEntry c;
    __builtin_memcpy(&c, (const void*)e, sizeof c);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    uint64_t b = __atomic_load_n(&e->idx, __ATOMIC_RELAXED);
    if (a != i || b != i) { (*n_lost)++; continue; }
    out[n++] = c;
  }
  *from = i;
  return n;
}
