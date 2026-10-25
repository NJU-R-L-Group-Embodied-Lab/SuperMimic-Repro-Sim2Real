#!/usr/bin/env bash
# End-to-end DDS tests of g1_runner.py against the headless unitree_mujoco G1 sim (domain 1, lo). SIM ONLY.
# Usage: bash scripts/deploy/run_sim_e2e.sh <test> [extra runner args]      (default stage: STAGE=ground)
#   auto      PASSIVE -> INTERP (band) -> HOLD -> STANDBY_PRE idle (band off) 2 s -> POLICY idle -> STANDBY_POST 1 s
#             -> L2+up: END_HOLD blend (band on) -> INTERP -> HOLD -> STANDBY_PRE -> POLICY shoulder_clap -> STANDBY_POST
#             -> band on -> DAMP; prints torque steps at every state change (sim --log-tau, aligned on the monotonic clock)
#   auto_legacy  the same with --end-hold-legacy (clip done -> END_HOLD stand-gain blend, as before the standby change)
#   standby   buttons, walk_forward_stop_001__A026: L2+up -> HOLD -> R1+X (ignored in HOLD) -> R1+A -> STANDBY_PRE
#             (band released 2 s in) -> R1+X 0.5 s and 2.3 s in (refused: < 3 s) -> R1+A 5 s in (must not start) -> R1+X 20 s in
#             -> POLICY -> STANDBY_POST -> L2+up (no next clip: ignored) -> L2+B (STATUS lines every 3 s in STANDBY_*)
#   standby_exit  buttons, walk_forward_stop: L2+up -> HOLD -> R1+A -> STANDBY_PRE (band off 2 s in) -> L2+up once (prompt only)
#             -> 4 s later L2+up once (prompt again: the first one expired) -> 0.7 s later L2+up (confirmed) -> END_HOLD blend
#             (band on) -> HOLD of the SAME clip -> R1+A -> STANDBY_PRE -> R1+X -> POLICY -> STANDBY_POST -> L2+B
#   standby_post2 buttons, idle + shoulder_clap: ... R1+X -> POLICY idle -> STANDBY_POST -> L2+up once (prompt only) -> 4 s later
#             L2+up once (prompt again) -> 0.7 s later L2+up (confirmed) -> END_HOLD (band on) -> INTERP -> HOLD shoulder_clap -> L2+B
#   standby_nan  buttons: L2+up -> HOLD -> R1+A -> STANDBY_PRE (band off 2 s in), policy returns NaN from call 150 -> DAMP
#   kill      like auto, but the simulator is SIGKILLed 3 s into the first POLICY clip (stale-state watchdog)
#   nan       like auto, policy returns NaN from call 50 on (DeployCore abort)
#   sigterm   auto-sequence, a 2nd runner is started during POLICY (must refuse), then SIGTERM -> DAMP 1 s -> exit
#   sighup    auto-sequence, SIGHUP 3 s into POLICY -> DAMP 1 s -> exit
#   stdout    auto-sequence with stdout+stderr into a pipe whose reader exits 12 s in (mid-run): must keep running
#   stall     auto-sequence, main loop blocked for 0.5 s 2 s into POLICY -> command-thread heartbeat DAMP
#   foreign   auto-sequence, a raw rt/lowcmd publisher starts 2 s into POLICY -> DAMP (foreign publisher watch)
#   buttons   no auto-sequence: button presses are injected through the sim (L2+up, R1+A, L2+B)
#   prebutton 'yes' prompt (--sim-ask-yes); L2+up / R1+A pressed before 'yes' and L2+up held through 'yes' must
#             not act; a fresh L2+up afterwards must work; L2+B
#   waist_active  (LOCK=1) auto-sequence; 2 s into POLICY the sim makes motor 14 report q=0.05/temp 25 -> DAMP 'waist unexpectedly active'
#   waist_active_start (LOCK=1) the fault is present before the runner starts -> runner refuses, nothing published
#   refuse_nolock  sim in lock-waist mode (mode_machine 6), runner WITHOUT --lock-waist -> must refuse, nothing published
#   refuse_lock5   sim unlocked (mode_machine 5), runner WITH --lock-waist -> must refuse, nothing published
# Env: LOCK=1 (sim --lock-waist + runner --lock-waist --lock-waist-direct-start; LW_DIRECT=0 drops the latter), CKPT=ckpt_0144000 (default ckpt_0116000), STAGE, ARM_KD, SIM_DT,
#      CLIPS="a.npz b.npz" (default idle + shoulder_clap), E2E_NAME (log dir name)
#      SM_REPO (repo root; default: derived from this script's location),
#      G1_DEPLOY_HOME (deploy env dir holding env.sh = python3.10 venv with unitree_sdk2py + unitree_mujoco; default ~/workspace/g1_deploy)
set -u
TEST=${1:-auto}; shift || true
REPO=${SM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"
# shellcheck disable=SC1091
source "${G1_DEPLOY_HOME:-$HOME/workspace/g1_deploy}/env.sh"
L=$REPO/deploy_logs/sim_e2e/${E2E_NAME:-$TEST}
rm -rf "$L"; mkdir -p "$L"
BAND=$L/band.flag; BTN=$L/buttons.txt
A=$REPO/deploy_artifacts/${CKPT:-ckpt_0116000}
C=$REPO/deploy_artifacts/clips
case $TEST in
  standby|standby_nan|standby_exit) CLIPS=${CLIPS:-"$C/walk_forward_stop_001__A026.npz"} ;;
  *) CLIPS=${CLIPS:-"$C/neutral_idle_loop_001__A081.npz $C/shoulder_clap_004__A041_M.npz"} ;;
esac
LOCK=${LOCK:-0}
SIM_LOCK=(); RUN_LOCK=()
# --lock-waist: policy standby is refused for clips off the (empty) lock-waist standby allow list -> pre-standby direct start
# (LW_DIRECT=0: without --lock-waist-direct-start -> R1+A must be REFUSED)
[ "$LOCK" = 1 ] && SIM_LOCK=(--lock-waist) && RUN_LOCK=(--lock-waist --lock-waist-direct-start)
[ "$LOCK" = 1 ] && [ "${LW_DIRECT:-1}" = 0 ] && RUN_LOCK=(--lock-waist)
case $TEST in
  refuse_nolock) SIM_LOCK=(--lock-waist); RUN_LOCK=() ;;
  refuse_lock5)  SIM_LOCK=(); RUN_LOCK=(--lock-waist) ;;
  waist_active_start) touch "$L/waist_active.flag" ;;
esac
echo "TEST=$TEST LOCK=$LOCK sim=[${SIM_LOCK[*]:-}] runner=[${RUN_LOCK[*]:-}] artifacts=$A" | tee "$L/setup.txt"

# dt 0.002 = Unitree MJCF native step; at 0.005 the explicit PD with kd>=2 on light links (wrists, ankle roll) is unstable
python scripts/deploy/sim_g1_mujoco.py --dt "${SIM_DT:-0.002}" --band-file "$BAND" --buttons-file "$BTN" --log "$L/sim.npz" --log-tau \
    --waist-active-file "$L/waist_active.flag" "${SIM_LOCK[@]}" > "$L/sim.log" 2>&1 < /dev/null &
SIM=$!
echo "$SIM" > "$L/sim.pid"
sleep 3
# arm kd: the spec's 10 makes the arms chatter at 4-5 rad/s in this sim (explicit PD, dt 2 ms, light links), which the
# runner's policy-start check (max|dq| < 0.5) then refuses; ARM_KD=10 reproduces it. Real-robot behaviour: UNVERIFIED.
RUN=(python -u scripts/deploy/g1_runner.py --iface lo --domain 1 --sim --stage "${STAGE:-ground}" --stand-arm-kd "${ARM_KD:-2}"
     --artifacts "$A" --log-dir "$L" --sim-band-file "$BAND" "${RUN_LOCK[@]}")
# the motion has started (STATE event; the startup summary also contains "-> POLICY")
wait_policy() { for _ in $(seq 1 600); do grep -qE -- "STATE: (STANDBY_PRE|HOLD) -> POLICY" "$L/runner.log" 2>/dev/null && break; sleep 0.1; done; }
wait_state() { for _ in $(seq 1 900); do grep -qF -- "STATE: $1" "$L/runner.log" 2>/dev/null && break; sleep 0.1; done; }
press() { echo "$(date +%T.%N) press $1" >> "$L/steps.txt"; echo "$1" > "$BTN"; sleep 0.3; : > "$BTN"; sleep "$2"; }
case $TEST in
  auto)    "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 ;;
  auto_legacy) "${RUN[@]}" --clip $CLIPS --auto-sequence --end-hold-legacy "$@" > "$L/runner.log" 2>&1 ;;
  standby) "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 --sim-band-release-standby-s 2 "$@" > "$L/runner.log" 2>&1 &
           R=$!
           sleep 3
           press "L2+up" 4.3         # INTERP 4 s -> HOLD
           press "R1+X" 1.2          # HOLD: must be ignored
           press "R1+A" 0.2          # -> STANDBY_PRE (band released 2 s later)
           press "R1+X" 1.5          # ~0.5 s into standby: REFUSED (< 3 s)
           press "R1+X" 2.7          # ~2.3 s into standby: REFUSED (< 3 s)
           press "R1+A" 14.4         # ~5.3 s into standby: must NOT start the motion
           press "R1+X" 12           # ~20 s into standby: -> POLICY (8.84 s clip) -> STANDBY_POST
           press "L2+up" 11.5        # ~3 s into STANDBY_POST: no next clip -> ignored; STANDBY_POST ~15 s
           press "L2+B" 3            # DAMP
           wait "$R" ;;
  standby_exit)
           "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 --sim-band-release-standby-s 2 "$@" > "$L/runner.log" 2>&1 &
           R=$!
           sleep 3
           press "L2+up" 5.5         # INTERP 4 s -> HOLD >= 1 s
           press "R1+A" 4.5          # -> STANDBY_PRE (band released 2 s later)
           press "L2+up" 4           # ~5 s into standby: CONFIRM prompt only
           press "L2+up" 0.4         # 4.3 s later: the 1st press expired -> prompt again
           press "L2+up" 4           # 0.7 s later: confirmed -> END_HOLD (band on) -> HOLD of the same clip
           press "R1+A" 9            # HOLD of walk_forward_stop again -> STANDBY_PRE (band released 2 s later)
           press "R1+X" 12           # ~9 s into standby -> POLICY (8.84 s clip) -> STANDBY_POST ~3 s
           press "L2+B" 3
           wait "$R" ;;
  standby_post2)
           "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 --sim-band-release-standby-s 2 "$@" > "$L/runner.log" 2>&1 &
           R=$!
           sleep 3
           press "L2+up" 5.5         # INTERP 4 s -> HOLD >= 1 s
           press "R1+A" 9            # -> STANDBY_PRE (band released 2 s later)
           press "R1+X" 0.2          # ~9 s into standby -> POLICY idle
           wait_state "POLICY -> STANDBY_POST"
           sleep 2
           press "L2+up" 4           # CONFIRM prompt only, stays in STANDBY_POST
           press "L2+up" 0.4         # 4.3 s later: prompt again
           press "L2+up" 7           # 0.7 s later: confirmed -> END_HOLD (band on) -> INTERP -> HOLD shoulder_clap
           press "L2+B" 3
           wait "$R" ;;
  standby_nan)
           "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 --sim-band-release-standby-s 2 --sim-inject-nan-step 150 "$@" > "$L/runner.log" 2>&1 &
           R=$!
           sleep 3
           press "L2+up" 5.5         # INTERP 4 s -> HOLD >= 1 s
           press "R1+A" 6            # STANDBY_PRE; NaN from policy call 150 (3 s) -> DAMP
           wait "$R" ;;
  nan)     "${RUN[@]}" --clip $CLIPS --auto-sequence --sim-inject-nan-step 50 "$@" > "$L/runner.log" 2>&1 ;;
  stall)   "${RUN[@]}" --clip $CLIPS --auto-sequence --sim-stall-at-s 2 --sim-stall-s 0.5 "$@" > "$L/runner.log" 2>&1 ;;
  kill)    "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 &
           R=$!; wait_policy
           sleep 3; echo "killing sim pid $SIM at $(date +%T.%N)" | tee -a "$L/kill.txt"; kill -9 "$SIM"
           wait "$R" ;;
  sigterm|sighup)
           "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 &
           R=$!; wait_policy
           if [ "$TEST" = sigterm ]; then
             sleep 1
             # a second controller must detect the first one's rt/lowcmd and exit without publishing
             python -u scripts/deploy/g1_runner.py --iface lo --domain 1 --sim --stage ground --artifacts "$A" --log-dir "$L/second" \
                 --clip $CLIPS "${RUN_LOCK[@]}" > "$L/second_runner.log" 2>&1
             echo "second runner exit code $?" | tee -a "$L/kill.txt"
             SIG=TERM
           else
             sleep 3; SIG=HUP
           fi
           echo "SIG$SIG runner pid $R at $(date +%T.%N)" | tee -a "$L/kill.txt"; kill -"$SIG" "$R"
           wait "$R" ;;
  stdout)  ( "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" 2>&1; echo "runner exit code $?" > "$L/exit.txt" ) \
               | ( timeout 12 cat > "$L/runner.log"; echo "reader closed at $(date +%T.%N)" > "$L/reader.txt" ) ;;
  foreign) "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 &
           R=$!; wait_policy
           sleep 2
           python - > "$L/foreign.log" 2>&1 <<'EOF'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
ChannelFactoryInitialize(1, "lo")
p = ChannelPublisher("rt/lowcmd", LowCmd_); p.Init()
m = unitree_hg_msg_dds__LowCmd_()  # all zero: kp = kd = 0 (sim only)
t0 = time.monotonic(); n = 0
print(f"foreign publisher start {time.strftime('%T')}", flush=True)
while time.monotonic() - t0 < 1.5:
    p.Write(m); n += 1; time.sleep(0.002)
print(f"foreign publisher sent {n}", flush=True)
EOF
           wait "$R" ;;
  waist_active)
           "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 &
           R=$!; wait_policy
           sleep 2; echo "waist fault ON at $(date +%T.%N)" | tee -a "$L/kill.txt"; touch "$L/waist_active.flag"
           wait "$R" ;;
  waist_active_start|refuse_nolock|refuse_lock5)
           "${RUN[@]}" --clip $CLIPS --auto-sequence "$@" > "$L/runner.log" 2>&1 ;;
  buttons) "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 "$@" > "$L/runner.log" 2>&1 &
           R=$!
           sleep 3
           press "R1+A" 1            # ignored in PASSIVE
           press "L2+up" 4.3         # INTERP 4 s -> HOLD
           press "R1+A" 1.5          # refused: HOLD < 1 s
           press "R1+A" 4            # POLICY idle clip
           press "L2+B" 3            # DAMP
           wait "$R" ;;
  prebutton)
           mkfifo "$L/stdin"
           "${RUN[@]}" --clip $CLIPS --sim-exit-after-damp 1.0 --sim-ask-yes "$@" < "$L/stdin" > "$L/runner.log" 2>&1 &
           R=$!
           exec 3> "$L/stdin"
           for _ in $(seq 1 200); do grep -q "Type 'yes'" "$L/runner.log" 2>/dev/null && break; sleep 0.1; done
           echo "prompt seen $(date +%T.%N)" >> "$L/steps.txt"
           press "L2+up" 0.5; press "R1+A" 0.5                       # before 'yes': must be ignored
           echo "L2+up" > "$BTN"; sleep 0.3                          # held THROUGH 'yes'
           echo "yes" >&3; echo "yes sent $(date +%T.%N)" >> "$L/steps.txt"
           sleep 1.5; : > "$BTN"; echo "L2+up released $(date +%T.%N)" >> "$L/steps.txt"
           sleep 2                                                   # still PASSIVE expected
           press "L2+up" 5.5                                         # fresh press -> INTERP -> HOLD
           press "L2+B" 3
           exec 3>&-
           wait "$R" ;;
esac
echo "runner exit code $?"
kill "$SIM" 2>/dev/null; sleep 0.5
grep -E "lowcmd_rx|first LowCmd|locked_cmd" "$L/sim.log" | tail -n 1 | sed 's/^/[sim last status] /'
grep -q "first LowCmd received" "$L/sim.log" || echo "[sim] NO rt/lowcmd was ever received by the sim"
python - "$L" "$LOCK" <<'EOF'
import os, sys, numpy as np
L, LOCK = sys.argv[1], sys.argv[2] == "1"
if not os.path.exists(f"{L}/sim.npz"):
    sys.exit("[sim log] no sim.npz (physics never started)")
z = np.load(f"{L}/sim.npz")
t, pz, up, band = z["t"], z["pelvis_z"], z["up_z"], z["band"]
if len(t) == 0:
    sys.exit("[sim log] empty (physics never started)")
if "locked_cmd_nonzero" in z.files:
    print(f"[sim log] lowcmds with a non-zero field on a locked motor: {int(z['locked_cmd_nonzero'][0])}")
print(f"[sim log] {len(t)} samples, t_end {t[-1]:.2f} s; pelvis_z min {pz.min():.3f} final {pz[-1]:.3f}; "
      f"max pelvis tilt {np.degrees(np.arccos(np.clip(up, -1, 1))).max():.1f} deg")
nb = ~band
if nb.any():
    i0 = np.argmax(nb)
    print(f"[sim log] band released at sim t={t[i0]:.2f}; after release: pelvis_z min {pz[i0:].min():.3f}, "
          f"max tilt {np.degrees(np.arccos(np.clip(up[i0:], -1, 1))).max():.1f} deg")
if "tau_t" not in z.files:
    sys.exit()
# torque-step analysis at the gain switches (physics dt 2 ms -> Nm/ms = |dtau| / 2)
n = min(len(z["tau_t"]), len(z["tau"]), len(z["pd"]), len(z["kp_sum"]))  # the sim may be killed between appends
tt, tau, pd, kps = z["tau_t"][:n], z["tau"][:n].astype(float), z["pd"][:n].astype(float), z["kp_sum"][:n].astype(float)
dms = np.diff(tt)[:, None] * 1e3
rate_f = np.abs(np.diff(tau, axis=0)) / dms     # actuator force (after ctrlrange clipping)
rate_pd = np.abs(np.diff(pd, axis=0)) / dms     # commanded PD torque (before clipping)
STAND = 1820.0 if LOCK else 2220.0  # sum of stand kp; with --lock-waist the runner sends kp 0 to motors 13/14 (2 x 200)
dk = np.abs(np.diff(kps))
sw = np.nonzero(dk > 1e-3)[0]
segs = []  # contiguous runs of kp changes = one switch / blend
for i in sw:
    if segs and i - segs[-1][1] <= 15:  # blend: one kp change per 20 ms policy step = 10 physics steps
        segs[-1][1] = i
    else:
        segs.append([i, i])
names = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r", "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
         "w_yaw", "w_roll", "w_pitch"] + [f"arm{k}" for k in range(14)]
def rep(lbl, a, b):
    a, b = max(a, 0), min(b, len(rate_f))
    rf, rp = rate_f[a:b], rate_pd[a:b]
    jf, jp = np.unravel_index(np.argmax(rf), rf.shape)[1], np.unravel_index(np.argmax(rp), rp.shape)[1]
    print(f"[torque] {lbl:<34} max dtau/dt: actuator {rf.max():6.2f} Nm/ms ({names[jf]}), PD cmd {rp.max():6.2f} Nm/ms ({names[jp]})")
for a, b in segs:
    k0, k1 = kps[a], kps[b + 1]
    kind = "PASSIVE->stand" if k0 < 1 else ("stand->policy (HOLD->POLICY)" if abs(k0 - STAND) < 1 and k1 > 1 and abs(k1 - STAND) > 1 and b - a < 3
            else ("policy->stand blend (END_HOLD)" if abs(k1 - STAND) < 1 and b - a >= 3 else ("policy->stand STEP" if abs(k1 - STAND) < 1 else
            ("->DAMP" if k1 < 1 else f"{k0:.0f}->{k1:.0f}"))))
    print(f"[torque] kp switch at sim t={tt[a]:.3f}..{tt[b + 1]:.3f}  kp_sum {k0:.0f} -> {k1:.0f}  ({kind})")
    if "DAMP" in kind or "PASSIVE" in kind:
        continue
    rep("  onset (t-0.02 .. t+0.02 s)", a - 10, a + 10)
    rep("  whole switch (t-0.02 .. end+0.2 s)", a - 10, b + 100)
pol = np.zeros(len(kps), bool)
for a, b in segs:
    if abs(kps[a] - STAND) < 1 and abs(kps[b + 1] - STAND) > 1 and kps[b + 1] > 1:
        nxt = [s for s in segs if s[0] > b]
        e = nxt[0][0] if nxt else len(kps) - 1
        pol[b + 30:e - 5] = True  # inside POLICY, away from the switches
if pol[:-1].any():
    rf = rate_f[pol[:-1]]
    print(f"[torque] during POLICY (baseline): actuator dtau/dt p99 {np.percentile(rf.max(1), 99):.2f} max {rf.max():.2f} Nm/ms")
EOF
# per state change: torque rate around every runner STATE event, per-state baselines, pelvis drift (runner log + sim log
# aligned on time.monotonic: runner meta t0_monotonic, sim tau_wall / wall)
python - "$L" <<'EOF'
import glob, json, sys, numpy as np
L = sys.argv[1]
runs = sorted(glob.glob(f"{L}/run_*.npz"))
try:
    z = np.load(f"{L}/sim.npz")
except Exception:
    sys.exit()
if not runs or "tau_wall" not in z.files:
    sys.exit()
r = np.load(runs[-1]); meta = json.loads(str(r["meta"]))
if "t0_monotonic" not in meta:
    sys.exit()
t0 = meta["t0_monotonic"]
n = min(len(z["tau_wall"]), len(z["tau"]), len(z["tau_t"]))
tw, tau = z["tau_wall"][:n], z["tau"][:n].astype(float)
rate = np.abs(np.diff(tau, axis=0)) / (np.diff(z["tau_t"][:n])[:, None] * 1e3)
tw1 = tw[1:]
names = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r", "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
         "w_yaw", "w_roll", "w_pitch"] + [f"arm{k}" for k in range(14)]
def win(a, b):
    m = (tw1 >= a) & (tw1 < b)
    if not m.any():
        return None, ""
    seg = rate[m]
    return float(seg.max()), names[int(np.argmax(seg.max(0)))]
for e in meta["events"]:
    if e["kind"] in ("STATE", "DAMP") and "->" in e["msg"]:
        tm = t0 + e["t"]
        v, j = win(tm - 0.02, tm + 0.08)
        tr = e["msg"].split(" (")[0].split(": ")[0]
        if v is not None:
            print(f"[handoff] {tr:<34} t={e['t']:8.3f}  max dtau/dt -20..+80 ms: {v:6.2f} Nm/ms ({j})")
# per-state segments from the runner's 50 Hz record
sn = meta["states"]; st = r["state"]; t = r["t"]
lab = np.array([sn[i] for i in st])
cut = np.nonzero(lab[1:] != lab[:-1])[0] + 1
w50 = z["wall"] if "wall" in z.files else None
band = z["band"]
for a, b in zip(np.r_[0, cut], np.r_[cut, len(lab)]):
    nm = lab[a]
    if nm not in ("STANDBY_PRE", "POLICY", "STANDBY_POST"):
        continue
    ta, tb = t0 + t[a], t0 + t[b - 1] + 0.02
    m = (tw1 >= ta + 0.2) & (tw1 < tb - 0.1)  # away from the hand-offs
    line = f"[segment] {nm:<12} {t[b - 1] - t[a] + 0.02:6.2f} s"
    if m.any():
        rf = rate[m].max(1)
        line += f"  dtau/dt p99 {np.percentile(rf, 99):5.2f} max {rf.max():5.2f} Nm/ms"
    if w50 is not None:
        k = np.nonzero((w50 >= ta) & (w50 < tb) & ~band[: len(w50)])[0]  # band off only (unsupported)
        if len(k) > 1:
            xy = np.stack([z["pelvis_x"][k], z["pelvis_y"][k]], 1)
            dxy = np.linalg.norm(xy - xy[0], axis=1)
            yaw = np.degrees(np.unwrap(z["yaw"][k]) - z["yaw"][k[0]])
            tilt = np.degrees(np.arccos(np.clip(z["up_z"][k], -1, 1)))
            line += (f"  | unsupported {w50[k[-1]] - w50[k[0]]:5.1f} s: pelvis xy drift max {dxy.max() * 100:5.1f} cm final {dxy[-1] * 100:5.1f} cm, "
                     f"yaw drift max {np.abs(yaw).max():5.1f} final {yaw[-1]:+5.1f} deg, tilt max {tilt.max():4.1f} deg, z min {z['pelvis_z'][k].min():.3f}")
    print(line)
EOF
