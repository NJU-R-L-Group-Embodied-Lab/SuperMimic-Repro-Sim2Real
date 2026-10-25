// libg1bridge_client.so: lock-free shm readers for the Python runner (ctypes). Plain C ABI, no DDS.
#include <fcntl.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "g1_bridge_shm.h"

extern "C" {

// Map the bridge's shm read-only. NULL if absent / not yet initialised / wrong version.
void* g1b_open(const char* name) {
  int fd = shm_open(name, O_RDONLY, 0);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, sizeof(G1BShm), PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return nullptr;
  const G1BShm* s = static_cast<const G1BShm*>(p);
  if (__atomic_load_n(&s->magic, __ATOMIC_ACQUIRE) != G1B_MAGIC || s->version != G1B_VERSION) {
    munmap(p, sizeof(G1BShm));
    return nullptr;
  }
  return p;
}

void g1b_close(void* p) {
  if (p) munmap(p, sizeof(G1BShm));
}

// sizes: snap, ring entry, status, cmd, shm
void g1b_sizes(long long* out) {
  out[0] = sizeof(G1BSnap);
  out[1] = sizeof(G1BRingEntry);
  out[2] = sizeof(G1BStatus);
  out[3] = sizeof(G1BCmd);
  out[4] = sizeof(G1BShm);
}

// field offsets, in the order of the Python dtypes (checked at startup)
int g1b_snap_offsets(long long* out) {
  int k = 0;
  out[k++] = offsetof(G1BSnap, t_rx);
  out[k++] = offsetof(G1BSnap, t_tick_change);
  out[k++] = offsetof(G1BSnap, n_rx);
  out[k++] = offsetof(G1BSnap, n_crc_bad);
  out[k++] = offsetof(G1BSnap, tick);
  out[k++] = offsetof(G1BSnap, tick_moving);
  out[k++] = offsetof(G1BSnap, mode_machine);
  out[k++] = offsetof(G1BSnap, mode_pr);
  out[k++] = offsetof(G1BSnap, q);
  out[k++] = offsetof(G1BSnap, dq);
  out[k++] = offsetof(G1BSnap, tau);
  out[k++] = offsetof(G1BSnap, temp);
  out[k++] = offsetof(G1BSnap, motorstate);
  out[k++] = offsetof(G1BSnap, _pad0);
  out[k++] = offsetof(G1BSnap, quat);
  out[k++] = offsetof(G1BSnap, gyro);
  out[k++] = offsetof(G1BSnap, acc);
  out[k++] = offsetof(G1BSnap, rpy);
  out[k++] = offsetof(G1BSnap, wireless_remote);
  return k;
}

int g1b_status_offsets(long long* out) {
  int k = 0;
  out[k++] = offsetof(G1BStatus, t_hb);
  out[k++] = offsetof(G1BStatus, n_pub);
  out[k++] = offsetof(G1BStatus, n_pub_fail);
  out[k++] = offsetof(G1BStatus, fail_consec);
  out[k++] = offsetof(G1BStatus, n_cmd_rx);
  out[k++] = offsetof(G1BStatus, t_last_cmd);
  out[k++] = offsetof(G1BStatus, lc_count);
  out[k++] = offsetof(G1BStatus, lc_t);
  out[k++] = offsetof(G1BStatus, ready);
  out[k++] = offsetof(G1BStatus, pub_started);
  out[k++] = offsetof(G1BStatus, wd_fired);
  out[k++] = offsetof(G1BStatus, quit);
  out[k++] = offsetof(G1BStatus, bridge_pid);
  out[k++] = offsetof(G1BStatus, _pad);
  out[k++] = offsetof(G1BStatus, wd_s);
  out[k++] = offsetof(G1BStatus, pub_gap_max);
  out[k++] = offsetof(G1BStatus, n_gap_late);
  out[k++] = offsetof(G1BStatus, pub_write_max);
  return k;
}

int g1b_read_snap(const void* p, void* out) { return g1b_snap_read(static_cast<const G1BShm*>(p), static_cast<G1BSnap*>(out), 1000); }

int g1b_read_ring(const void* p, unsigned long long* from, void* out, int max, unsigned long long* n_lost) {
  uint64_t f = *from, l = *n_lost;
  int n = g1b_ring_read(static_cast<const G1BShm*>(p), &f, static_cast<G1BRingEntry*>(out), max, &l);
  *from = f;
  *n_lost = l;
  return n;
}

// Status fields are independent counters: one acquire fence, then a plain copy.
void g1b_read_status(const void* p, void* out) {
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  memcpy(out, (const void*)&static_cast<const G1BShm*>(p)->status, sizeof(G1BStatus));
}
}
