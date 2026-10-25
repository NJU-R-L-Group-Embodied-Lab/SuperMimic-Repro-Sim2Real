// g1_bridge: C++ DDS I/O for the G1 runner (scripts/deploy/g1_runner.py --bridge).
//
// Why: on the G1's onboard Orin NX (PC2) the Python cyclonedds (de)serialization + SDK Python CRC of 1 kHz rt/lowstate and
// 500 Hz rt/lowcmd held the GIL ~94% of the time (py-spy), starving the policy loop (overrun DAMP / 23 deg tilt in standby).
// This process does ALL rt/lowstate / rt/lowcmd traffic; the runner keeps obs, policy, safety and the state machine.
//
//   rt/lowstate (listener, every sample): CRC check (--crc), seqlock snapshot + per-sample ring in shm (g1_bridge_shm.h)
//   rt/lowcmd   (main thread)          : fixed --hz stream of the latest runner command + an immediate publish for
//                                        commands flagged NOW; nothing is written before the first runner command
//   rt/lowcmd   (watch reader)         : counts every sample (foreign-publisher watch, as the runner's watcher child did)
//   watchdog                           : no runner command for > --wd-ms, runner gone (stdin EOF) or SIGTERM once
//                                        publishing -> LATCHED damping (kp 0, kd --damp-kd, q measured) on the motors the
//                                        runner had enabled; runner gone/SIGTERM: damping for --orphan-s, then exit
//   QUIT command                       : stop publishing at once and exit (the runner already sent its final damping second)
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "g1_bridge_shm.h"

using namespace unitree::robot;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

static double mono() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Copied from unitree_sdk2 (https://github.com/unitreerobotics/unitree_sdk2),
// example/g1/low_level/g1_ankle_swing_example.cpp.
// Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD-3-Clause;
// see LICENSES/unitree_sdk2-BSD-3-Clause.txt and LICENSES/THIRD_PARTY_NOTICES.md.
static uint32_t Crc32Core(const uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

struct Args {
  std::string iface, shm, sock;
  int domain = -1;
  bool crc = false, listen_only = false;
  double hz = 500.0, wd_ms = 100.0, damp_kd = 3.0, orphan_s = 10.0;
  int cpu = -1, prio = 0;
};

static void store_d(double* p, double v) { __atomic_store(p, &v, __ATOMIC_RELAXED); }

static G1BShm* S = nullptr;
static Args A;
static volatile sig_atomic_t g_sigterm = 0;

static void on_term(int) { g_sigterm = 1; }

// ------------------------------------------------------------------------------------------- rt/lowstate listener
static uint64_t g_n_rx = 0, g_n_bad = 0;
static uint32_t g_last_tick = 0;
static bool g_have_tick = false, g_tick_moving = false;
static double g_t_tick_change = 0.0;

static void snap_begin() {
  __atomic_store_n(&S->snap_seq, S->snap_seq + 1, __ATOMIC_RELAXED);  // odd: writing (single writer)
  __atomic_thread_fence(__ATOMIC_RELEASE);
}
static void snap_end() { __atomic_store_n(&S->snap_seq, S->snap_seq + 1, __ATOMIC_RELEASE); }

static void on_lowstate(const void* p) {
  const LowState_& m = *static_cast<const LowState_*>(p);
  const double t = mono();
  if (A.crc && m.crc() != Crc32Core(reinterpret_cast<const uint32_t*>(&m), (sizeof(LowState_) >> 2) - 1)) {
    g_n_bad++;
    snap_begin();
    S->snap.n_crc_bad = g_n_bad;
    snap_end();
    return;
  }
  const uint32_t tick = m.tick();
  if (!g_have_tick || tick != g_last_tick) {
    if (g_have_tick) g_tick_moving = true;
    g_have_tick = true;
    g_last_tick = tick;
    g_t_tick_change = t;
  }
  g_n_rx++;
  const auto& wr = m.wireless_remote();
  snap_begin();
  G1BSnap& s = S->snap;
  s.t_rx = t;
  s.t_tick_change = g_t_tick_change;
  s.n_rx = g_n_rx;
  s.n_crc_bad = g_n_bad;
  s.tick = tick;
  s.tick_moving = g_tick_moving ? 1u : 0u;
  s.mode_machine = m.mode_machine();
  s.mode_pr = m.mode_pr();
  const auto& ms = m.motor_state();
  for (int i = 0; i < G1B_NM; i++) {
    s.q[i] = ms[i].q();
    s.dq[i] = ms[i].dq();
    s.tau[i] = ms[i].tau_est();
    s.temp[i][0] = ms[i].temperature()[0];
    s.temp[i][1] = ms[i].temperature()[1];
    s.motorstate[i] = ms[i].motorstate();
  }
  const auto& imu = m.imu_state();
  for (int k = 0; k < 4; k++) s.quat[k] = imu.quaternion()[k];
  for (int k = 0; k < 3; k++) {
    s.gyro[k] = imu.gyroscope()[k];
    s.acc[k] = imu.accelerometer()[k];
    s.rpy[k] = imu.rpy()[k];
  }
  memcpy(s.wireless_remote, wr.data(), 40);
  snap_end();
  // ring: invalidate, fill, publish index, advance head
  const uint64_t idx = g_n_rx - 1;
  G1BRingEntry* e = &S->ring[idx & (G1B_RING - 1)];
  __atomic_store_n(&e->idx, ~0ull, __ATOMIC_RELAXED);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  e->t = t;
  e->tick = tick;
  e->mode_machine = m.mode_machine();
  e->keys = uint32_t(wr[2]) | (uint32_t(wr[3]) << 8);
  __atomic_store_n(&e->idx, idx, __ATOMIC_RELEASE);
  __atomic_store_n(&S->ring_head, idx + 1, __ATOMIC_RELEASE);
}

static void on_lowcmd_watch(const void*) {
  G1BStatus& st = S->status;
  store_d(&st.lc_t, mono());
  __atomic_fetch_add(&st.lc_count, 1, __ATOMIC_RELEASE);
}

// ------------------------------------------------------------------------------------------- setup helpers
static int usage() {
  fprintf(stderr,
          "usage: g1_bridge --domain D --shm NAME --sock PATH [--iface IF] [--crc] [--listen-only] [--hz 500] [--wd-ms 100] [--damp-kd 3]\n"
          "                 [--orphan-s 10] [--cpu N] [--prio P]\n");
  return 2;
}

static bool parse(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&](void) -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    const char* v = nullptr;
    if (k == "--crc") { A.crc = true; continue; }
    if (k == "--listen-only") { A.listen_only = true; continue; }
    if (!(v = val())) return false;
    if (k == "--iface") A.iface = v;
    else if (k == "--domain") A.domain = atoi(v);
    else if (k == "--shm") A.shm = v;
    else if (k == "--sock") A.sock = v;
    else if (k == "--hz") A.hz = atof(v);
    else if (k == "--wd-ms") A.wd_ms = atof(v);
    else if (k == "--damp-kd") A.damp_kd = atof(v);
    else if (k == "--orphan-s") A.orphan_s = atof(v);
    else if (k == "--cpu") A.cpu = atoi(v);
    else if (k == "--prio") A.prio = atoi(v);
    else return false;
  }
  return A.domain >= 0 && !A.shm.empty() && !A.sock.empty() && A.hz > 0 && A.wd_ms > 0;
}

static G1BShm* map_shm(const std::string& name) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) { perror("shm_open"); return nullptr; }
  if (ftruncate(fd, sizeof(G1BShm)) != 0) { perror("ftruncate"); return nullptr; }
  void* p = mmap(nullptr, sizeof(G1BShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { perror("mmap"); return nullptr; }
  G1BShm* s = static_cast<G1BShm*>(p);
  memset(s, 0, sizeof(G1BShm));
  for (int i = 0; i < G1B_RING; i++) s->ring[i].idx = ~0ull;
  s->status.bridge_pid = uint32_t(getpid());
  s->status.wd_s = A.wd_ms * 1e-3;
  s->version = G1B_VERSION;
  __atomic_store_n(&s->magic, G1B_MAGIC, __ATOMIC_RELEASE);
  return s;
}

static int open_sock(const std::string& path) {
  int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) { perror("socket"); return -1; }
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  if (path.size() >= sizeof(a.sun_path)) { fprintf(stderr, "socket path too long\n"); return -1; }
  strcpy(a.sun_path, path.c_str());
  unlink(path.c_str());
  if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) { perror("bind"); return -1; }
  return fd;
}

static void rt_setup() {
  if (A.cpu >= 0) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(A.cpu, &cs);
    int r = pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    fprintf(stderr, "[bridge] publisher thread affinity cpu %d: %s\n", A.cpu, r ? strerror(r) : "ok");
  }
  if (A.prio > 0) {
    sched_param sp{};
    sp.sched_priority = A.prio;
    int r = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    fprintf(stderr, "[bridge] publisher thread SCHED_FIFO %d: %s\n", A.prio, r ? strerror(r) : "ok");
  }
}

// ------------------------------------------------------------------------------------------- main
int main(int argc, char** argv) {
  if (!parse(argc, argv)) return usage();
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, SIG_IGN);  // the runner owns Ctrl-C; it tells us to QUIT
  struct sigaction sa{};
  sa.sa_handler = on_term;
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGHUP, &sa, nullptr);

  S = map_shm(A.shm);
  if (!S) return 1;
  const int sfd = open_sock(A.sock);
  if (sfd < 0) return 1;

  if (A.iface.empty()) ChannelFactory::Instance()->Init(A.domain);
  else ChannelFactory::Instance()->Init(A.domain, A.iface);
  ChannelSubscriber<LowCmd_> watch("rt/lowcmd");
  watch.InitChannel(on_lowcmd_watch, 0);
  ChannelSubscriber<LowState_> sub("rt/lowstate");
  sub.InitChannel(on_lowstate, 0);  // 0: handler runs in the DDS listener (no SDK queue hiding stale samples)
  ChannelPublisher<LowCmd_> pub("rt/lowcmd");
  // writer exists from the start (creating it blocks 50-100 ms); it writes nothing before the first command.
  // --listen-only: no rt/lowcmd writer at all, commands are ignored (read-only checks on the real robot)
  if (!A.listen_only) pub.InitChannel();
  rt_setup();
  G1BStatus& st = S->status;
  __atomic_store_n(&st.ready, 1u, __ATOMIC_RELEASE);
  fprintf(stderr, "[bridge] up%s: domain %d iface '%s' crc %d hz %.0f wd %.0f ms damp kd %.1f shm %s sock %s pid %d\n", A.listen_only ? " (LISTEN ONLY: no rt/lowcmd writer)" : "", A.domain,
          A.iface.c_str(), int(A.crc), A.hz, A.wd_ms, A.damp_kd, A.shm.c_str(), A.sock.c_str(), int(getpid()));

  G1BCmd cmd{};       // latest runner command
  G1BCmd rx{};
  bool started = false, have_cmd = false, parent_gone = false, quit = false;
  double t_orphan = 0.0;
  const double period = 1.0 / A.hz, wd = A.wd_ms * 1e-3;
  double deadline = mono() + period;
  LowCmd_ out{};
  G1BSnap snap{};
  double t_last_pub = 0.0;

  auto publish = [&](void) {
    const bool damping = __atomic_load_n(&st.wd_fired, __ATOMIC_RELAXED) != 0;
    const bool need_q = damping || (cmd.flags & G1B_CMD_Q_MEASURED);
    bool have_q = need_q && g1b_snap_read(S, &snap, 100) == 0 && snap.n_rx > 0;
    out.mode_pr() = 0;
    out.mode_machine() = uint8_t(cmd.mode_machine);
    auto& mc = out.motor_cmd();
    for (int i = 0; i < G1B_NM; i++) {
      const bool on = cmd.mode[i] != 0;
      mc[i].mode() = on ? 1 : 0;
      mc[i].dq() = 0.0f;
      mc[i].tau() = 0.0f;
      if (damping) {
        mc[i].q() = on && have_q ? float(snap.q[i]) : 0.0f;
        mc[i].kp() = 0.0f;
        mc[i].kd() = on ? float(A.damp_kd) : 0.0f;
      } else {
        mc[i].q() = need_q ? (on && have_q ? float(snap.q[i]) : 0.0f) : float(cmd.q[i]);
        mc[i].kp() = need_q && !have_q ? 0.0f : float(cmd.kp[i]);  // no measurement yet: never pull towards q = 0
        mc[i].kd() = float(cmd.kd[i]);
      }
    }
    out.crc() = Crc32Core(reinterpret_cast<const uint32_t*>(&out), (sizeof(LowCmd_) >> 2) - 1);
    bool ok = false;
    const double tw = mono();
    try {
      ok = pub.Write(out);
    } catch (...) {
      ok = false;
    }
    const double tw1 = mono();
    if (tw1 - tw > st.pub_write_max) store_d(&st.pub_write_max, tw1 - tw);
    if (t_last_pub > 0.0) {
      const double gap = tw - t_last_pub;
      if (gap > st.pub_gap_max) store_d(&st.pub_gap_max, gap);
      if (gap > 2.0 * period) __atomic_fetch_add(&st.n_gap_late, 1, __ATOMIC_RELAXED);
    }
    t_last_pub = tw;
    __atomic_fetch_add(&st.n_pub, 1, __ATOMIC_RELAXED);
    if (ok) {
      __atomic_store_n(&st.fail_consec, 0, __ATOMIC_RELAXED);
    } else {
      __atomic_fetch_add(&st.n_pub_fail, 1, __ATOMIC_RELAXED);
      __atomic_fetch_add(&st.fail_consec, 1, __ATOMIC_RELAXED);
    }
  };

  while (true) {
    double now = mono();
    store_d(&st.t_hb, now);
    double rem = deadline - now;
    if (rem < 0) rem = 0;
    timespec to{time_t(rem), long((rem - double(time_t(rem))) * 1e9)};
    pollfd pf[2] = {{sfd, POLLIN, 0}, {0, POLLIN, 0}};
    int nfd = parent_gone ? 1 : 2;
    int pr = ppoll(pf, nfd, &to, nullptr);
    now = mono();
    bool now_flag = false;
    if (pr > 0 && (pf[0].revents & POLLIN)) {
      while (true) {
        ssize_t n = recv(sfd, &rx, sizeof rx, 0);
        if (n < 0) break;
        if (n != ssize_t(sizeof rx) || rx.magic != G1B_MAGIC) {
          fprintf(stderr, "[bridge] dropped bad command datagram (%zd bytes)\n", n);
          continue;
        }
        if (rx.flags & G1B_CMD_QUIT) { quit = true; break; }
        if (A.listen_only) continue;
        cmd = rx;
        have_cmd = true;
        if (rx.flags & G1B_CMD_NOW) now_flag = true;
        __atomic_fetch_add(&st.n_cmd_rx, 1, __ATOMIC_RELAXED);
        store_d(&st.t_last_cmd, now);
      }
    }
    if (nfd == 2 && pr > 0 && (pf[1].revents & (POLLIN | POLLHUP | POLLERR))) {
      char b[64];
      ssize_t n = read(0, b, sizeof b);
      if (n <= 0) {
        parent_gone = true;
        fprintf(stderr, "[bridge] runner gone (stdin EOF)\n");
      }
    }
    if (quit) {
      __atomic_store_n(&st.quit, 1u, __ATOMIC_RELEASE);
      fprintf(stderr, "[bridge] QUIT: %lu lowcmd published, %lu failed\n", (unsigned long)st.n_pub, (unsigned long)st.n_pub_fail);
      break;
    }
    if (have_cmd && !started) {
      started = true;
      __atomic_store_n(&st.pub_started, 1u, __ATOMIC_RELEASE);
      fprintf(stderr, "[bridge] first runner command: rt/lowcmd stream starts\n");
      deadline = now;  // publish right away
    }
    if (!started) {
      if (parent_gone || g_sigterm) break;  // nothing was ever published
      if (now >= deadline) deadline += period;
      continue;
    }
    // watchdog (latched)
    if (!st.wd_fired) {
      const char* why = nullptr;
      if (now - st.t_last_cmd > wd) why = "runner silent";
      else if (parent_gone) why = "runner gone";
      else if (g_sigterm) why = "SIGTERM";
      if (why) {
        __atomic_store_n(&st.wd_fired, 1u, __ATOMIC_RELEASE);
        fprintf(stderr, "[bridge] WATCHDOG (%s, last command %.1f ms ago): latched damping kd %.1f\n", why,
                (now - st.t_last_cmd) * 1e3, A.damp_kd);
      }
    }
    if ((parent_gone || g_sigterm) && t_orphan == 0.0) t_orphan = now;
    if (t_orphan > 0.0 && now - t_orphan > A.orphan_s) {
      fprintf(stderr, "[bridge] exit after %.1f s of damping without a runner\n", A.orphan_s);
      break;
    }
    if (now >= deadline || now_flag) publish();
    if (now >= deadline) {
      deadline += period;
      if (now - deadline > 10 * period) deadline = now + period;  // no burst after a stall
    }
  }
  close(sfd);
  unlink(A.sock.c_str());
  shm_unlink(A.shm.c_str());
  fflush(stderr);
  _exit(0);  // skip DDS teardown (the SDK's static destructors may block)
}
