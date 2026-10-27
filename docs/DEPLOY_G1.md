# Real-robot deployment (sim2real) on the Unitree G1

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

> **Safety warning**: the runner sends low-level commands (`rt/lowcmd`) directly to the G1's joint motors. A mistake in the model, the parameters, the network conditions or the operation can make the robot fall, damage it, or injure people. **Every step requires a hoist/gantry and one person holding the remote controller, ready to press L2+B at any moment.** Use at your own risk.

This document is the operating procedure for running this repository's tracking policies on a Unitree G1 29-DoF (EDU, low-level control in debug mode), written for the operator standing next to the robot. It was written while deploying on our own G1, and it keeps the checkpoint numbers of that time (the early e11 policy `ckpt_0116000`, etc.) and the measured numbers as the basis for each safety rule. Everything that has not been verified on the real robot is marked **UNVERIFIED** and collected in the [UNVERIFIED checklist](#unverified-checklist-confirm-each-item-before-going-on-the-robot) section. Facts that depend on your machine (`mode_machine`, network card, IP, whether the temperature thresholds are appropriate, etc.) must be re-confirmed one by one on your own robot before going on the robot.

Overall flow:

```
training checkpoint (.pt) --export--> deploy_artifacts/<name>/{policy.onnx, deploy_config.json} + clips/<clip>.npz
        |                                              |
        +-- sim2sim gates (same DeployCore)            +-- copy to the deployment computer or the robot's onboard computer PC2
                                                             |
             g1_bridge (C++, unitree_sdk2: rt/lowstate, rt/lowcmd, CRC, watchdog)
                   ^  shared memory + unix socket  v
             g1_runner_bridge.py (Python: state machine, 50 Hz policy, 500 Hz commands, safety checks, thermal protection, logging)
```

**Real-robot record:**

- `ckpt_0116000` (`mode_machine = 5`, without `--lock-waist`) ran the following with the old flow (R1+A starts the motion directly), all completed with no abort: idle suspended in the air, idle with feet on the ground, and, with the rope slightly slack (`slack`), idle_loop_002, itching, shoulder_clap, walk_sideway, walk_forward (3 times).
- The four-segment continuous "shanshenbu (闪身步)" motion `combo_c4_full_vc_b15f20` (about 46 s) of `e13i_0291000` ran to completion once on the onboard computer PC2 with the bridge runner at the `ground` stage (rope fully slack but still attached), with a maximum tilt error of 16.3°. In two other runs waist_roll exceeded 80 °C, the runner entered DAMP and the robot fell mid-motion; thermal protection was added for this reason (see [Thermal protection](#thermal-protection-bridge-runner)).
- The spin / landing dance `shanshen_duosuomao_lean35kfs_pad2` (21.9 s, two fast spins and a one-foot landing) of `e24_0355000` (v1.1 Release) ran to completion twice in a row on PC2 with the bridge runner at the `ground` stage (rope attached), using the policy-standby flow below. All real attempts on this dance with 9 earlier checkpoints had failed (aborted, or did not start). The failures (gyro saturation in the spins, waist roll coupling and range, a knee "tuck" reflex after landing), the training changes, the safety margins, which sim gates predicted the result and the exact command are in [Real-robot result: lean35kfs](REAL_ROBOT_LEAN35KFS.md). It is one checkpoint run twice, without ablation.
- The policy-standby flow described below (R1+A → STANDBY_PRE → R1+X) has passed sim2sim and DDS end-to-end simulation tests, and was used in the `e24_0355000` real runs above. The earlier real-robot runs of `ckpt_0116000` used the old flow.

**Model names**: the defaults of `--artifacts` / `--ckpt` / `CKPT=` in the scripts (`ckpt_0116000`, plus `ckpt_0100000` and `ckpt_0144000` used in the waist-lock part) are our early e11 policies and are **not** in the [GitHub Release](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases). When using a Release model, every runner / sim2sim / e2e command must explicitly pass `--artifacts deploy_artifacts/<name>`, `--ckpt <name>` or `CKPT=<name>` (for example `e13i_0291000`), together with the matching `--clip`. The run-time-generated `deploy_logs/` and `deploy_artifacts/reports/` are not distributed with the repository.

## Prerequisites: hardware, software and files

**Hardware**: G1 29-DoF; a hoist or gantry; the remote controller (paired, sufficiently charged); a computer to run the runner. The runner can run on two kinds of computer:

- An external deployment computer, connected by Ethernet cable to the robot's 192.168.123.x subnet. We use Ubuntu 22.04 x86.
- The robot's own onboard computer PC2 (Orin NX, aarch64).

**Software**: the deployment side does **not** need torch / mjlab, only:

- Python 3.8+ (PC2 ships with 3.8; our x86 deployment computer uses 3.10), `numpy`, `onnxruntime`. `mujoco` is only needed for simulation tests and sim2sim.
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) (`unitree_sdk2py`) and its dependency CycloneDDS (we use 0.10.5).
- The C++ bridge needs [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2). Simulation tests additionally need [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (the official G1 29-DoF MJCF).

| File | Purpose |
|---|---|
| `src/supermimic/deploy/core.py` | `DeployCore`: training-consistent observations, goals, action → joint targets, clamping and abort criteria. Depends only on numpy + onnxruntime; shared by sim2sim and the real robot |
| `scripts/deploy/bridge/g1_runner_bridge.py` | **The recommended runner**: state machine, 50 Hz policy, safety checks, thermal protection, logging. Sends/receives DDS through the C++ `g1_bridge` |
| Other files in `scripts/deploy/bridge/` | `g1_bridge.cpp` + `build.sh` (C++ DDS bridge), `g1_bridge_io.py`, the read-only check `ro_bridge_check.py`, unit tests `test_bridge.py`, end-to-end test `run_sim_e2e_bridge.sh` |
| `scripts/deploy/g1_runner.py` | Pure-Python DDS runner, the predecessor of the bridge runner. Same state machine but **no thermal protection**; kept for reference |
| `scripts/deploy/check_robot.py` | **Read-only** state monitor. Subscribes only to `rt/lowstate` and never creates an `rt/lowcmd` publisher. `--check-mode` performs one read-only MotionSwitcher query |
| `scripts/deploy/g1_common.py` | Button bits, button debouncing, lowstate parsing, joint sign hints, read-only CheckMode, waist-lock clip whitelist |
| `scripts/deploy/sim_g1_mujoco.py`, `run_sim_e2e.sh` | Headless unitree_mujoco simulation (elastic band toggle, button injection, per-step torque log) and end-to-end tests for the pure-Python runner. Testing only; refuses to run on domain 0 |
| `scripts/deploy/bg_sync_server.py`, `make_source_bg.py` | Optional: when the runner is given `--sync-udp HOST:PORT`, plays the original dance video on screen in sync with the robot's current frame (background video sync) |
| `deploy_artifacts/<ckpt>/` | `policy.onnx` + `deploy_config.json` (generated by `python -m supermimic.deploy.export`, or downloaded from the Release) |
| `deploy_artifacts/clips/*.npz` | Per-clip goal tables (generated by `export --cmd clips`) |

## Deployment environment: `G1_DEPLOY_HOME` and `env.sh`

The scripts use the environment variable `G1_DEPLOY_HOME` to point at the deployment environment directory; the default is `~/workspace/g1_deploy`.

- Put an `env.sh` in that directory that activates the venv above, sets up CycloneDDS and does `unset PYTHONPATH`.
- `unitree_mujoco/` also goes in this directory. sim2sim can alternatively be pointed directly at `scene_29dof.xml` with `G1_UNITREE_XML`.

In every terminal, first run:

```bash
export G1_DEPLOY_HOME=/path/to/g1_deploy   # defaults to ~/workspace/g1_deploy when unset
source $G1_DEPLOY_HOME/env.sh
cd /path/to/SuperMimic-Repro-Sim2Real
```

You write `env.sh` yourself, for example:

```bash
source /path/to/deploy_venv/bin/activate          # venv with numpy, onnxruntime, unitree_sdk2py (and mujoco)
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
unset PYTHONPATH
```

## Bridge runner and the onboard computer PC2 (recommended)

**Why a bridge.** On PC2, pure-Python DDS send/receive + CRC (1 kHz `rt/lowstate`, 500 Hz `rt/lowcmd`) takes about 94% of the GIL time and starves the policy loop. We saw policy-loop overruns leading to DAMP, and a 23° tilt error during standby.

So all `rt/lowstate` / `rt/lowcmd` traffic was moved into a C++ process `g1_bridge` (based on unitree_sdk2); the structure is shown in the flow diagram at the top of this document. Observations, policy, safety checks and the state machine are the same as in `g1_runner.py`. The bridge adds its own layer of protection: if any of the following happens, the bridge **itself** latches and publishes damping commands (kp 0, kd 3):

- the runner has not sent a command for more than `--bridge-wd-ms` (100 ms);
- the runner process disappears;
- the bridge receives SIGTERM.

After the runner disappears, the bridge keeps damping for `--bridge-orphan-s` (10 s) and then exits.

**Build and self-test** (on the deployment computer or PC2, with the deployment environment activated):

```bash
bash scripts/deploy/bridge/build.sh /path/to/unitree_sdk2       # produces scripts/deploy/bridge/g1_bridge and libg1bridge_client.so
python scripts/deploy/bridge/test_bridge.py                      # DDS domain 2 / lo, does not touch the robot: CRC, snapshot, command, watchdog, QUIT
CKPT=e13i_0291000 bash scripts/deploy/bridge/run_sim_e2e_bridge.sh auto   # end-to-end simulation (unitree_mujoco, domain 1 / lo)
```

The binaries are looked up in `scripts/deploy/bridge/` by default; if they live elsewhere, set the directory with `SM_BRIDGE_HOME`.

**Read-only check**: run it after connecting to the robot and **before every session on the robot**. It publishes no `rt/lowcmd`.

```bash
python scripts/deploy/bridge/ro_bridge_check.py --iface <nic> --secs 10   # on PC2 usually eth0 (the default)
```

It prints:

- lowstate rate and interval distribution, CRC error count, `mode_machine`, maximum temperature;
- the number of `rt/lowcmd` messages sent by **other publishers** during the check. In debug mode this must be 0. Non-zero means Unitree's built-in motion control is still running, and the runner will refuse to start.

We have seen a badly seated Ethernet cable degrade the robot's LAN: the lowstate rate dropped and burst gaps appeared, and re-seating the cable fixed it. So compare the readings with your usual values every time.

**Running**: the arguments are the same as for `g1_runner.py`, plus `--bridge-*` and the thermal-protection arguments.

```bash
python scripts/deploy/bridge/g1_runner_bridge.py --iface <nic> --domain 0 --stage <air|feet|slack|ground> \
  --artifacts deploy_artifacts/<ckpt> \
  --clip deploy_artifacts/clips/<clip>.npz \
  --log-dir deploy_logs/real
```

`--stage` is required. The first time on the robot, follow the [Staged safety profiles](#staged-safety-profiles---stage-required) and [On-robot procedure](#on-robot-procedure) sections and loosen step by step starting from `air`.

**Running on PC2:**

- **Network and Python**: PC2's robot-side network port is on the 192.168.123.x subnet (the interface is usually `eth0`); log in from an external computer via ssh. The deployment-side code (runner, `g1_common.py`, `g1_bridge_io.py`, `supermimic/deploy/core.py`) is kept compatible with PC2's built-in Python 3.8.
- **ONNX model**: the onnxruntime on PC2 is old, so use `policy_ir9.onnx` from the Release package (the same graph, ONNX IR version 9) and copy it as `policy.onnx` in the deployment package.
- **ssh and stop mechanisms**: PC2 has no local display or keyboard and can only be run via ssh, so `--allow-ssh` is required. In this case **the local Ctrl-C stop mechanism is not reliable**: when ssh disconnects the runner receives SIGHUP and enters DAMP; if the runner runs inside tmux / setsid, an ssh disconnect does not affect it. **The primary stop mechanism is the remote controller's L2+B.** Open another ssh session and be ready to `kill` the runner at any time (after the runner disappears, the bridge sends damping on its own).
- **Unitree built-in motion mode**: after boot the robot may be in a built-in motion mode (for example `ai`), and the runner will refuse to start because it detects another `rt/lowcmd` publisher. First enter debug mode as described in [G1 model, `ai` mode and debug mode](#g1-model-ai-mode-and-debug-mode), then confirm with `ro_bridge_check.py` that there are 0 other publishers.
- **CPU frequency governor**: the governor reverts to the default after PC2 reboots; set it back to `performance` with sudo when needed.

## Preparing an external deployment computer

**Wired network card** (not needed when running on PC2):

- Realtek RTL8125-series cards on some motherboards have no driver on older kernels, and the kernel log shows something like `r8169 ...: unknown chip XID ...`. The fix is to download the r8125 driver source from Realtek's website and build/install it with DKMS. With Secure Boot enabled, the DKMS-built module must first be enrolled with MOK before it can load: run `mokutil --import`, reboot, and choose Enroll MOK in the "Perform MOK management" screen. Follow your distribution's documentation for the exact steps.
- A simpler option is a USB 3.0 gigabit adapter (RTL8153 / AX88179, etc., with in-kernel drivers), which works when plugged in.
- We recommend creating a separate NetworkManager connection for this card and setting `never-default`. Internet traffic then still goes over Wi-Fi and the default route is not changed.

Give the card the static address **`192.168.123.222/24`**, then run `ping 192.168.123.161`. Whether 161 really is the motion-control board's address depends on your robot's manual (**UNVERIFIED**). Note the interface name with `ip -br addr`; it is written `<nic>` below.

**Other preparation:**

- **DDS**: the real robot uses domain 0 and `--iface <nic>`; simulation uses domain 1 and `--iface lo`. **Never run simulation on the network card connected to the robot.**
- **CPU frequency governor**: we recommend `performance` (`sudo cpupower frequency-set -g performance`).
- **Real-time priority**: requires rtprio permission (`ulimit -r` > 0 or CAP_SYS_NICE). Without it the runner's attempt at `SCHED_FIFO` gets a `PermissionError`, which is printed in the startup summary. All our timing data was measured without real-time priority.
- **sudo-free optimizations the runner already applies**: the command thread is pinned to the last CPU (`--cmd-cpu auto`); ONNX Runtime uses 1 thread with busy-waiting disabled; `gc.collect(); gc.freeze()` is executed before entering the loop.
- **Do not run it at the same time as training or evaluation.**
- **Only run the runner on the deployment computer's local console** (display + keyboard), not over ssh or Wi-Fi.
  - Remote desktops such as ToDesk **also do not count as a local console**: they also go over Wi-Fi, so when the connection drops the operator loses the keyboard as a stop mechanism, and the runner can only detect ssh sessions, not remote desktops.
  - On domain 0 the runner refuses to start when it detects an ssh session. `--allow-ssh` overrides this; it is not recommended on an external computer and is required on PC2 (see the previous section).

## Deployment conventions (identical to training, checked by the export/parity tests)

| Item | Convention |
|---|---|
| Rates | Policy 50 Hz; command thread about 500 Hz (about 526 Hz including the extra send after each policy step) |
| Joint order | MJCF order = Unitree motor indices 0..28; `mode_pr = 0` |
| Observation | 93 proprioceptive dims per frame, 10-frame history, term-major, oldest to newest. The first frame fills all 10 slots; the previous action is initialized to 0 |
| Goal | When the robot is at frame k, take rows k+1..k+10 of the goal table |
| Command | `q_target = a·scale + q0`, `dq = 0`, `tau = 0`, kp/kd use the training values |
| Soft-limit clamping | **Off by default** (core default `clamp_soft_limits=False`); see below |
| Torque clamping | `\|kp·(q_t − q) − kd·dq\| ≤ torque_clamp_factor × effort_limit`, with the factor set by `--stage` |
| Start pose f0 | Frame 0 of the clip **clipped to the soft limits first**, consistent with the training reset (`commands.py` clips the reference joint positions to `soft_joint_pos_limits`). The startup summary prints how many joints were clipped for each clip |
| Perturbations not present in training | Latency, PD gain randomization, joint friction, external pushes |

**Why soft-limit clamping is off:**

- In training, the target exceeds 0.9× the soft limits in 61–74% of steps and the hard limits in 25–41% (ankles by up to 1.2 rad), and training never clamps.
- With clamping turned on in sim2sim, the walk_sideway error grows by 2.35× and the robot falls.

The cost is that a joint can be pushed into its mechanical stop, where it is then constrained by the torque clamp. The run summary reports, per joint, the fraction of time spent close to the hard limit.

## Stop mechanisms: the only ones that actually work

| Mechanism | How it works | When it does **not** work |
|---|---|---|
| **Remote L2+B** | The button state reaches the runner with `rt/lowstate`. The DDS callback only sets a flag; the command thread enters DAMP (kp=0, kd=3). A combination must be stable across 3 consecutive lowstate messages, so from press to DAMP takes about 4–6 ms, plus the remote's own refresh interval | The runner has exited, hung or crashed; lowstate is interrupted; the remote has no battery or is not paired. Whether the Unitree firmware itself responds to L2+B in debug mode is **UNVERIFIED** |
| **Local keyboard Ctrl-C** (SIGINT; SIGTERM, SIGHUP, SIGQUIT, SIGTSTP are handled the same way) | The signal handler only sets a flag; the command thread executes DAMP, sends damping commands for another 1 s, then exits | The runner has already exited. **After exiting, no more lowcmd is sent** (with the bridge runner, the bridge damps for another `--bridge-orphan-s`) |
| **Hoist** | The only mechanism that does not depend on software | When the rope is slack enough that the robot can fall |
| Hardware e-stop / power button | — | Whether the robot has an independent e-stop and whether the power button cuts power immediately are both **UNVERIFIED**. Do not treat it as a stop mechanism until verified |

- **What the motors do after lowcmd stops** (hold the last command? go into damping? lose power?) is **UNVERIFIED**. This determines what happens when the runner is `kill -9`'d or crashes, and it must be measured on the hoist with the kill -9 test in the [On-robot procedure](#on-robot-procedure).
- **Ctrl-Z (SIGTSTP)**: would normally suspend the process while the motors keep the last command. The runner now treats it as a stop request (enters DAMP, then exits), but do not use it as a stop key. You can run `stty susp undef` to disable Ctrl-Z completely in that terminal.

Two hard rules:

- **Before the runner exits, the robot must already be supported by the hoist or lowered into a safe posture.** In DAMP the joints are soft and the robot slumps; what the robot does after the runner exits is UNVERIFIED.
- **At the start of every session on the robot, actually press L2+B once** and confirm the runner prints `DAMP: ... remote L2+B`.

## Runner state machine and remote-controller buttons

R1+A only lets the **policy** take over and stand still in place (STANDBY_PRE). After the crew has lowered the rope and positioned the gantry, the operator presses **R1+X** to start the motion. After the motion ends, the policy keeps the robot standing in place (STANDBY_POST) and does not switch back to standing gains.

```
PASSIVE --[L2+up]--> INTERP --(4 s)--> HOLD --[R1+A]--> STANDBY_PRE --[R1+X]--> POLICY --(clip ends)--> STANDBY_POST
 (damping)           (standing gains,   (standing gains, (policy stands in place,  (motion)              (policy stands in place,
                      hoist carries)     hoist carries)   no time limit)                                   no time limit)
STANDBY_POST --[L2+up ×2 (twice within 3 s), only if there is another --clip]--> END_HOLD (1 s blend to standing gains) --> INTERP (next clip) --> HOLD --[R1+A]--> ...
STANDBY_PRE  --[L2+up ×2 (twice within 3 s)]--> END_HOLD (1 s blend to standing gains, target = this clip's f0) --> HOLD (same clip)
                (standing gains again from the second L2+up press; the robot cannot balance itself: the hoist must carry it first)
POLICY --(any motor ≥ 65 °C, bridge runner only)--> smooth thermal stop --> STANDBY_POST (no next clip allowed afterwards)
any state --[L2+B] / any abort condition (including DeployCore aborts in STANDBY_*)--> DAMP (permanent, only a program restart leaves it)
SIGINT/SIGTERM/SIGHUP/SIGQUIT/exception --> DAMP, keep sending damping commands for 1 s, then exit
--end-hold-legacy: clip ends --> END_HOLD (old flow, standing gains, cannot balance itself), no STANDBY_POST
--lock-waist: implies --end-hold-legacy; clips not on the waist-lock standby whitelist (empty by default) cannot enter policy standby (both R1+A and R1+X are refused);
              only with --lock-waist-direct-start does R1+A in HOLD start the motion directly (old flow)
```

| State | Commands sent |
|---|---|
| PASSIVE | All 29 joints kp=0, kd=3, q set to the current measurement |
| INTERP | Standing gains; cosine interpolation to f0 over `--interp-s` (4 s), limited to the soft limits (joints whose starting point is already outside the soft limits are not pulled back in one step). Entered from PASSIVE, it starts from the measured q; entered from END_HOLD, it starts from **the currently commanded target**, without jumping to the measured q |
| HOLD | Standing gains, target f0 |
| STANDBY_PRE | Training kp/kd. Within the same 50 Hz cycle in which R1+A is pressed, a `DeployCore` is created with reference `static_clip(clip, 0)` (frame 0's pose, gravity direction and height, all velocity terms 0); after `reset(0)` the first step runs immediately, and every subsequent cycle runs `step(advance=False)`, so the reference frame does not advance. **The policy actively balances and the robot stands in place.** No hard time limit; **by procedure no more than 2 min** (WARN printed at `--standby-warn-s` 120 s). DeployCore's abort criteria remain active |
| POLICY | Training kp/kd. Within the same cycle in which R1+X is pressed, `core.switch_clip(clip, 0)` runs: **the observation history and previous action are not cleared**, and the reference advances from frame 0 of the clip |
| STANDBY_POST | Training kp/kd. Immediately after the clip's last frame, `switch_clip(static_clip(clip, last frame))` + `step(advance=False)`; the policy stands in place in the last-frame pose. Likewise no more than 2 min by procedure. Not entered with `--lock-waist` |
| END_HOLD | Starts from **the last policy command exactly as it was** (target and training kp/kd unchanged). Over `--end-blend-s` (1 s), kp/kd blend linearly to the standing gains and the target moves to the last frame by cosine interpolation (clipped to the soft limits). Every cycle the target is limited so that `\|kp(t)·(q_cmd − q)\| ≤ min(1, factor) × effort_limit`. **Never starts a motion automatically.** Entered only in three cases (see below the table) |
| DAMP | kp=0, kd=3, held until the program exits |

The three cases in which END_HOLD is entered:

1. In STANDBY_POST, L2+up pressed twice within 3 s and there is another clip: after the blend it **automatically** enters INTERP.
2. In STANDBY_PRE, L2+up pressed twice within 3 s: the target is **this clip's** f0, and after the blend it **automatically** returns to HOLD of the same clip with the clip index unchanged. If the target was cut by the torque clamp during the blend, it first INTERPs to f0 and then enters HOLD.
3. With `--end-hold-legacy` (always the case with `--lock-waist`), it is entered as soon as the clip ends.

**Standing gains** (legs in the order hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll):

| Part | kp | kd |
|---|---|---|
| Legs | 100, 100, 100, 150, 40, 40 | 2, 2, 2, 4, 2, 2 |
| Waist (3) | 200 | 5 (`--stand-waist-kd`) |
| Arms (14) | 40 | 10 (`--stand-arm-kd`) |

In simulation, arm kd=10 makes the wrists, shoulder yaw and elbows oscillate at high frequency at 4–5 rad/s while the position error stays small, so R1+A's `max|dq| < 0.5` condition refuses to start. With kd=2, |dq| ≤ 0.05, so simulation tests default to `--stand-arm-kd 2`. Whether the real robot also oscillates is **UNVERIFIED**; observe it during the on-robot procedure.

**Buttons:**

| Combination | Effect |
|---|---|
| **L2 + up** | PASSIVE → INTERP; END_HOLD under `--end-hold-legacy` (after the blend) → INTERP. **In STANDBY_PRE and STANDBY_POST it must be pressed twice within 3 s (`--confirm-s`)**: the first press only prints `press L2+up again within 3 s: is the hoist carrying the robot?`, the second one executes. In STANDBY_POST (with another clip), it runs the END_HOLD blend → INTERP to the next clip; in STANDBY_PRE, it runs the END_HOLD blend → HOLD of the same clip. In both cases standing gains apply from the second press, so **the hoist must carry the robot first**. L2+up is ignored when there is no next clip or right after a thermal stop. It is also ignored in other states |
| **R1 + A** | HOLD → **STANDBY_PRE** (policy standby, does not start the motion). Before pressing, lower the rope until the feet carry most of the body weight and the rope is just taut. Start conditions are listed below the table; if not met, `REFUSED` and the reason are printed. Pressing R1+A again in STANDBY_PRE does **not** start the motion; it only prints `press R1+X to start the motion` |
| **R1 + X** | STANDBY_PRE → **POLICY** (start the motion). Must be a fresh press (debounced like the other combinations: released for ≥ 100 ms before pressing again). Start conditions are listed below the table; if not met, `REFUSED: R1+X (motion start): <reason>` is printed. Pressing R1+X in HOLD or other states only prints `R1+X ignored`. With `--lock-waist`, clips not on the waist-lock standby whitelist are always refused |
| **L2 + B** | Any state → DAMP |

**R1+A start conditions** (all must hold):

- HOLD ≥ 1 s (`--hold-min-s`);
- max|q − f0| < 0.1 rad (`--hold-tol`);
- max|dq| < 0.5 rad/s (`--start-max-dq`);
- pelvis tilt differs from the clip's frame-0 reference by < 10° (`--start-max-tilt-deg`);
- motor temperature ≤ `--warn-temp` (70 °C, **UNVERIFIED**).

**R1+X start conditions** (all must hold):

- at least 3 s spent in STANDBY_PRE (`--standby-min-s`);
- over the last 3 s (`--standby-window-s`), tilt error always < 8° (`--standby-max-tilt-deg`) and max|dq| always < 1.0 rad/s (`--standby-max-dq`);
- motor temperature ≤ `--warn-temp`;
- the bridge runner additionally applies a temperature-rise forecast; see [Thermal protection](#thermal-protection-bridge-runner).

Changing the stability window from 0.5 s to 3 s is backed by simulation. With a 0.5 s window, a robot that is walking or turning gets through 11.1% of the time at kp 0.85 and 28.6% at kp 0.9 + 1.8 kg; with a 3 s window these drop to 0.7% and 0%. A stationary robot is never refused except during the first 3 s. No yaw-rate limit was added, because a 1°/3 s limit would wrongly refuse a robot standing still at kp 1.0 5.6% of the time.

**Button-handling rules:**

- A button word only counts if it stays unchanged across 3 consecutive lowstate messages.
- A combination must be released for at least 100 ms before it can trigger again; L2+B is the exception and triggers while held.
- **Any button pressed before the runner starts sending lowcmd (including during the 'yes' prompt) is ignored.** A combination held down through 'yes' will not trigger; release it and press again.
- Button events that are processed more than 100 ms late because of main-loop delay are dropped, and this is printed.

Button bits: R1=0, L1=1, start=2, select=3, R2=4, L2=5, F1=6, F2=7, A=8, B=9, X=10, Y=11, up=12, right=13, down=14, left=15; button word = `wireless_remote[2] | wireless_remote[3] << 8`. X=10 was checked one by one against four SDK / simulation sources (`wireless_controller.py`, `joystick.py`, the simulation bridge's key_map, `h2_ankle_swing_example.py`). That the real remote sends the same bits is **UNVERIFIED** and must be verified in the button test, **including R1+X**.

**Standby status line** (one line every 3 s in STANDBY_PRE / STANDBY_POST, interval set by `--status-s`) is emitted through the print thread's queue, so it never blocks the control loop, and is also written to `console.txt`:

```
[  57.300] STATUS: STANDBY_PRE   12.0 s: yaw   +1.2 deg, tilt err  3.1 (3 s max  3.4) deg, leg max|dq| 3 s 0.04 rad/s, temp max 41 C, lowstate age max 2.1 ms; R1+X: would be ACCEPTED
```

| Field | Meaning |
|---|---|
| yaw | Change in IMU yaw since entering the state (unwrapped, no jump at ±180°). The motion is executed relative to the current heading, so however much the robot has turned, the motion direction is off by that much |
| tilt err | Current and maximum-over-the-last-3-s angle between the pelvis gravity direction and the reference (DeployCore aborts at 20°) |
| leg max\|dq\| | Maximum angular velocity of the 12 leg joints over the last 3 s |
| temp max, lowstate age max | Maximum since entering the state (sampled at 50 Hz) |
| R1+X | STANDBY_PRE only: whether R1+X would be accepted right now, and the reason if not |

**WARN**: printed once when the condition appears; counting restarts only 3 s after the condition clears, so hovering around a threshold does not flood the console. The `WARN:` at the end of the status line lists the conditions that still hold. These are warnings only and **do not enter DAMP automatically**:

| WARN | Threshold | Action |
|---|---|---|
| Yaw change | > 15° (`--warn-yaw-deg`) | Re-align the gantry, or re-hang the robot |
| Leg joint speed (possibly stepping) | \|dq\| > 1 rad/s for > 0.5 s (`--warn-leg-dq`, `--warn-leg-dq-s`) | Re-hang, then L2+B |
| Motor temperature | > `--warn-temp` | — |
| Standby duration | > 120 s (`--standby-warn-s`) | End the standby as soon as possible |

## Operator procedure (policy standby)

| Step | Remote | Runner state | What the crew / operator does |
|---|---|---|---|
| 1 | — | PASSIVE | The hoist carries the robot, which is hanging. After typing `yes`, the joints are soft and damped |
| 2 | **L2 + up** | INTERP → HOLD | Moves into the clip's frame-0 pose within 4 s. HOLD uses standing gains and **cannot balance itself**; the hoist must keep carrying the robot |
| 3a | — | HOLD | **Before pressing R1+A**: the crew slowly lowers the rope until **the feet carry most of the body weight and the rope is just taut** (the rope can still catch the robot but carries only a small part of the weight). Do this slowly; the rope must not slacken so far that the robot starts to tip |
| 3 | **R1 + A** | STANDBY_PRE | Press with the feet flat and carrying most of the weight. The policy takes over and the robot stands still in place. The console prints `STANDBY_PRE ... crew: slack the rope (gantry ready), then R1+X`, then one `STATUS` line every 3 s |
| 4 | — | STANDBY_PRE | The crew lets the rope go from "just taut" to slack and positions the gantry. Watch the console: no DAMP, no `WARN`; watch the robot: no drifting, stepping or turning. **The whole standby must not exceed 2 min**; the bridge runner's temperature-rise forecast requires **at least 10 s of standby** |
| 5 | **R1 + X** | POLICY | Once the rope is set, the gantry is in position, the area is clear and `STATUS` shows `R1+X: would be ACCEPTED`, the operator **confirms manually** and presses. If refused, the console prints the reason; wait until the robot is steady and press again |
| 6 | — | STANDBY_POST | Entered automatically when the motion finishes; **the policy keeps the robot standing** (last-frame pose). Likewise no more than 2 min; move on to step 7 as soon as possible |
| 7 | — | STANDBY_POST | **Re-hang**, in this exact order: (1) first move the gantry **directly above** the robot; (2) take up only the **slack** in the rope, **do not lift or drag** the robot (the policy is still running, and pulling at an angle or lifting will destabilize it); (3) as soon as the rope is taut, go to step 8 |
| 8 | **L2 + B** | DAMP | Press **immediately** once the rope is taut. The joints go soft and the robot hangs on the hoist; then Ctrl-C to exit |
| 7' (multiple clips) | **L2 + up ×2** | END_HOLD → INTERP → HOLD | After step 7, with the hoist **already carrying the robot**, press twice within 3 s. Standing gains apply from the second press, so the robot cannot balance itself. After the 1 s blend it automatically interpolates to frame 0 of the next clip, then go back to step 3a |
| Leave standby | **L2 + up ×2** | STANDBY_PRE → END_HOLD → HOLD | When the rope or gantry cannot be prepared and you need to abandon this motion: first re-hang as in step 7 (hoist carrying), then press L2+up twice within 3 s. After a 1 s blend to standing gains it returns to HOLD of **the same clip**; you can restart from step 3a or press L2+B |

Notes:

- **R1+A is only "policy standby"; R1+X starts the motion.** Both combinations use R1; make sure A has been released before pressing R1+X.
- **Do not enter standby when the feet carry no load** (suspended, the `air` stage, or when the hoist carries most of the weight). Simulation results:
  - With the rope carrying ≤ 50% of the body weight, standby is normal (kp 1.0: tilt error ≤ 4.7°, max|dq| ≤ 1.2 rad/s).
  - **At 80%, 2 of 4 clips bounce** (3 at kp 0.85): the pelvis rises 6–8 cm, max|dq| 11–12 rad/s.
  - Replacing the rope with a constant 80% upward force, all 8 runs hit a tilt abort within 1.6–4.9 s.

  So do not press R1+A while the hoist carries more than about half of the body weight.
- L2+B works at any time during standby, and DeployCore aborts also enter DAMP. **If DAMP happens while the rope is being let out, the robot collapses**, so let the rope out slowly, and keep the gantry/rope in a position where it can catch the robot until R1+X.
- **Standby time limit 2 min** (by procedure; STANDBY_PRE and STANDBY_POST are timed separately). The longer the standby, the more chances for a watchdog (lowstate stale 20 ms, main-loop timeout) to fire falsely, and with the rope slack a false trigger means a fall. In simulation, one 21.7 ms lowstate gap occurred after 150 s in HOLD.
- **Watch `STATUS` during standby.** In sim2sim with kp nominal or 15% high:
  - both 60 s and 300 s standbys stay in place, with horizontal drift ≤ 7.6 cm;
  - but the robot **slowly turns in place** (0.08–0.5°/s; within 300 s up to 35° for idle and up to 164° for the ib_dodge start pose).

  **With kp 15% low, the robot steps in place, walks (up to 12 m) and turns a lot.** Looping the idle clip as-is behaves the same way, which shows this is the policy's own sensitivity to stiffness, not a consequence of the standby method. The true stiffness of the real motors is **UNVERIFIED**.
- **When finishing, do not press L2+B with the hoist slack**: first make the hoist carry the robot as in step 7, then press L2+B. In simulation, attaching the rope directly with the robot 1.42 m from the hoist point (about 270 N, about 37° off vertical) produced max|dq| 13.7 rad/s and a 16.2° tilt error within 0.5 s. So always move the gantry directly above first, and only take up the slack.
- `--end-hold-legacy` restores the old behaviour (clip ends → END_HOLD, standing gains); in that case **the hoist must already carry the robot before the clip ends**.

## Staged safety profiles `--stage` (required)

| stage | When to use | Velocity abort (× velocity limit) | \|a\| abort | Torque clamp | INTERP/HOLD/END_HOLD tilt abort |
|---|---|---|---|---|---|
| `air` | Suspended, feet off the ground | 0.3 | 10 | 0.5 × effort | 20° |
| `feet` | Feet on the ground, hoist carrying | 0.3 | 10 | 0.5 × effort | 20° |
| `slack` | Rope slightly slack | 0.5 | 15 | 1.0 × effort | 29° |
| `ground` | Rope fully slack (still attached) | 0.8 (core default) | 50 | 1.5 × effort | 46° |

- The tilt abort in POLICY / standby (> 20° from the reference) is the same at every stage.
- The `STAGE` line of the startup summary prints the values in effect. When needed, they can be overridden individually with `--max-vel-frac`, `--max-abs-action`, `--torque-clamp-factor`, `--max-tilt-hold`.
- The stage profiles **cannot** remove the torque jump when the policy takes over from HOLD. In simulation the knee sees about 11–14 Nm/ms under both `air` and `ground`.

## On-robot procedure

Every step requires: the hoist attached; one person holding the remote ready to press L2+B; one person at the deployment computer's **local** keyboard ready to Ctrl-C (on PC2, ready with `kill` in another ssh session); the surroundings cleared. **Only move to the next step after the current one passes.**

The commands below are written for `g1_runner_bridge.py`. `g1_runner.py` takes the same arguments but has no thermal protection.

**1. Before every session on the robot (every single time)**

1. Check the battery level (in the Unitree App or on the remote); do not go on the robot with a low battery. The threshold is per the manual (**UNVERIFIED**).
2. Check the network: `ping 192.168.123.161`; `python scripts/deploy/check_robot.py --iface <nic> --once` sees lowstate. On PC2, run `ro_bridge_check.py`.
3. Hoist the robot with its feet off the ground.
4. Do the L2+B button test from step 2 below.

**2. Read-only check, enter debug mode, button test (suspended)**

1. The robot is in Unitree's built-in damping or zero-torque state. **Open another terminal and keep the read-only monitor running:**
   ```bash
   python scripts/deploy/check_robot.py --iface <nic> --domain 0 --check-mode
   ```
   Confirm each item:
   - `mode_machine` (should be 5) and `mode_pr`, and write them down;
   - `rate` about 500 Hz, tick increasing, `CRC bad` is 0;
   - the `MotionSwitcher CheckMode` line: at this point it should show some motion mode as active, or `NO ANSWER`;
   - with the robot upright `proj_gravity ≈ [0 0 −1]`, and the x component is positive when it leans forward (`proj_gravity = R^T·[0,0,−1]`);
   - gyro ≈ 0 at rest; gyro z is positive when rotating counter-clockwise seen from above;
   - temperature: what each of the two values is and their units are both **UNVERIFIED**;
   - `mstat` is shown in hex and should be 0x0 when normal (**UNVERIFIED**).
2. **Enter debug mode** so that Unitree's built-in motion control stops publishing `rt/lowcmd`; see [G1 model, `ai` mode and debug mode](#g1-model-ai-mode-and-debug-mode) for the steps. Afterwards `CheckMode` should show `no motion mode active`. If it shows `NO ANSWER`, this computer may not be able to see that service (**UNVERIFIED**), and you can only rely on the runner's rt/lowcmd detection.
3. Test the buttons **only after entering debug mode**. Before that, pressing L2+up or L2+B may trigger Unitree's own stand-up or damping functions (**UNVERIFIED**). With check_robot running, press L2+up, R1+A, **R1+X**, L2+B in turn; the `pressed=` column must show the corresponding key names, e.g. `['R1', 'X']` for R1+X.

**3. Sign check and kill -9 test (suspended)**

1. Sign check:
   ```bash
   python scripts/deploy/check_robot.py --iface <nic> --domain 0 --sign left_knee_joint
   ```
   Press Enter to zero, then move the joint in the direction given by `model hint`; `q − q(at Enter)` must increase. For each leg check hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll, then check waist_yaw and, on both sides, shoulder_pitch, shoulder_roll, elbow.
2. **kill -9 test.** It must be done once; it determines what happens when the runner crashes:
   1. Start the runner as in step 4 (`--stage air`) and enter HOLD.
   2. In **another terminal**, run `kill -9 <runner PID>`. Find the PID with `ps -eo pid,cmd | grep g1_runner`; **do not use `pkill -f`**.
   3. With check_robot and by eye, record the robot's state over the next 5 s: does it hold its pose, go soft, or lose power? With the bridge runner, also record whether the bridge's damping takes over.
   4. Record the result in the UNVERIFIED checklist.

   **Until you have the result, always assume "the robot is uncontrolled once the runner dies".**

**4. Suspended, INTERP/HOLD only (`--stage air`, feet about 5 cm off the ground)**

```bash
python scripts/deploy/bridge/g1_runner_bridge.py --iface <nic> --domain 0 --stage air \
  --artifacts deploy_artifacts/<ckpt> \
  --clip deploy_artifacts/clips/<clip>.npz \
  --log-dir deploy_logs/real
```

1. Check the startup summary (STAGE, CheckMode, temperature, `max|q − f0|`), then type `yes`. The runner immediately checks again (no other rt/lowcmd, CheckMode, temperature, mode_machine, lowstate freshness), then enters PASSIVE. The joints should be "soft and damped".
2. Press L2+B, confirm DAMP is printed, then Ctrl-C to exit. Restart the runner and type `yes` again.
3. Press L2+up; the robot moves to f0 within 4 s. `INTERP -> HOLD (max|q-f0|=...)` should be below 0.1 rad.
4. In HOLD, watch for humming or oscillation in the wrists, elbows and shoulder yaw, and look at dq in the log. If there is any, add `--stand-arm-kd 2 --stand-waist-kd 2`.
5. Press L2+B, then Ctrl-C to exit.

**Do not do policy standby or motions while suspended**: feet without load are outside the training distribution, and standby bounces when the hoist carries more than about half of the body weight.

**5. Feet on the ground, hoist carrying (`--stage feet`)**

- **Rule one**: from this step on, press R1+A only when the feet are flat, carry most of the body weight and the rope is just taut (the hoist carries only a small part of the weight, no more than about half).
- **Rule two**: press R1+X only after the rope has been set as this stage requires, the area is clear and `STATUS` shows that R1+X would be accepted. In the `feet` stage, the rope is not let out further before R1+X; it stays "just taut".
- **Run only idle clips at this stage.** `feet` uses a 0.5× torque clamp and a 0.3× velocity limit, which is too tight for other clips: in sim2sim, walk_sideway suspended already reaches |a| 9.1 and dq 0.28× the velocity limit, right against the 10 / 0.3 thresholds; under a 0.5× torque clamp, walk_sideway falls 2 times out of 40 and jog_sideway 8 times out of 8.
- **HOLD and END_HOLD cannot keep their balance on the ground**: ankle stiffness 2 × 40 = 80 Nm/rad is far below m·g·h ≈ 240 Nm/rad, so the robot falls like an inverted pendulum. In simulation without the hoist, both tip over after about 1–1.3 s.
- After the clip ends, STANDBY_POST is entered and the policy keeps balancing (sim2sim: at nominal kp, no fall and no abort for 30–120 s). **Before pressing L2+B, make the hoist carry the robot again.**
- When going from STANDBY_POST to the next clip (L2+up ×2), the blend to standing gains starts at the second press, so **the hoist must already carry the robot before pressing** (the runner prints `WARN: ... the HOIST MUST CARRY IT`).

**6. Rope slightly slack (`--stage slack`) → rope fully slack (`--stage ground`, still attached)**

Run one clip at a time and look at the log after every run. Run e11's BONES-SEED SAFE clips in this order:

1. `idle_loop_002__A060`
2. `itching_left_arm_R_003__A121`
3. `shoulder_clap_004__A041_M`
4. `walk_sideway_090_stop_003__A037`

Move to the next stage only if all clips of the previous stage satisfy:

- tilt always well below 20°;
- no abort of any kind;
- no joint resting against a hard limit for long.

Other notes:

- `walk_sideway` has little velocity margin under `slack`: in the worst case (kp 0.85, torso +1.8 kg) dq reaches 0.43× the velocity limit, against a threshold of 0.5.
- The HARDER group (wave_R, jog_sideway, dance_hiphop) is not run at this stage.
- The restrictions from step 5 on HOLD, STANDBY_POST → L2+up and END_HOLD apply at every stage.
- Long motions such as the shanshenbu (闪身步) combo, which move forward/backward by more than 1 m, can only be run at the `ground` stage, and the rope must be let out so far that it cannot catch the robot. So before starting, always confirm the thermal-protection parameters (see [Thermal protection](#thermal-protection-bridge-runner)) and start with cool motors as thermal protection requires.

## Abort conditions and startup checks

Every abort enters DAMP, prints the reason in a `!!!!` banner and writes it to the log.

| Condition | Default threshold | Checked by |
|---|---|---|
| lowstate stale: by receive time, or the tick stops advancing (checked only when the tick changes) | 20 ms (`--stale-ms`) | 500 Hz command thread |
| Main-loop heartbeat: in INTERP/HOLD/STANDBY_PRE/POLICY/STANDBY_POST/END_HOLD the main loop has produced no new command for too long | (2 + 0.5) × 20 ms = 50 ms | Command thread |
| Policy loop overrun | 2 cycles past the deadline (`--max-overrun`) | Policy loop |
| DeployCore abort: tilt difference > 20°, velocity, \|a\|, non-finite state or IMU, quaternion norm off from 1 by more than 0.05 | Velocity and \|a\| thresholds per `--stage`; in standby the tilt difference is relative to the static reference (frame 0 / last frame) | STANDBY_PRE, POLICY, STANDBY_POST |
| Motor temperature | > 80 °C (`--max-temp`, **UNVERIFIED**) | All states |
| Motor temperature (bridge runner thermal stop, **does not enter DAMP**) | ≥ 65 °C in POLICY (`--thermal-stop-temp`) | POLICY |
| Motor error word `motorstate & --motor-error-mask != 0` | Mask defaults to 0xFFFFFFFF, bit meanings **UNVERIFIED**; checked only on domain 0 | All states |
| Pelvis tilt in INTERP/HOLD/END_HOLD | See `--stage` | These three states |
| `mode_machine` differs from the value locked at startup | — | DDS callback sets a flag, command thread acts on it |
| With `--lock-waist`, motor 13/14 \|q\| > 0.02 rad or temperature ≠ 0 (`waist unexpectedly active`) | — | All states |
| Another rt/lowcmd publisher appears during the run | Samples received within 1 s > 1.3 × own sends + 10 | Counted in a subprocess, decided by the command thread |
| Consecutive rt/lowcmd send failures | 10 (`--max-write-fail`) | Command thread |
| L2+B; SIGINT/SIGTERM/SIGHUP/SIGQUIT; exception | — | Command thread |
| (bridge) runner sends no command for more than 100 ms, runner disappears, bridge receives SIGTERM | `--bridge-wd-ms` | The C++ bridge sends damping itself |

**Why the rt/lowcmd monitor runs in a subprocess:**

- When a subscriber with a Python callback was attached to `rt/lowcmd` in the same process, the whole runner deadlocked in simulation and lowcmd stopped being sent.
- The presumed cause: the DDS receive thread holds the reader and waits for the GIL, while the command thread holds the GIL and waits in `Write()` for local delivery to that same reader. This cause has not been directly confirmed.
- Limitation: against its own send rate of about 526 Hz, this monitor can only detect other publishers faster than about 170 Hz; slower ones can only be caught by the startup checks.

**Startup checks**: if any one fails, the runner exits immediately without sending any command.

1. lowstate received within 5 s. On domain 0, messages with a bad CRC are dropped.
2. `mode_machine` accepts only two combinations: 5 without `--lock-waist`, or 6 with `--lock-waist`. Once accepted, the value is locked and written back into every lowcmd. With `--lock-waist`, abnormal 13/14 readings also refuse startup.
3. No other rt/lowcmd publisher within 1 s.
4. On domain 0, call `MotionSwitcherClient().CheckMode()`. This is a **read-only** call; the runner **never calls ReleaseMode / SelectMode**. A motion mode is active → refuse; no answer → warn (refuse instead with `--require-check-mode`).
5. Temperature ≤ `--max-temp`; warn above `--warn-temp` (R1+A and R1+X will then be refused).
6. Type `yes`, after which items 3 and 4, as well as temperature, mode_machine and lowstate freshness, are **re-checked** immediately.
7. On domain 0, refuse to start when an ssh session is detected (unless `--allow-ssh` is given).

Other notes:

- The startup summary prints `ref speed` per clip: the maximum reference joint velocity within the first 0.5 s after the motion starts and the corresponding joint name. **It is only a hint and never refuses.** A large value means the robot has to exert force quickly right after R1+X (for example shoulder_clap 5.10 rad/s, left elbow).
- `--sim`, `--auto-sequence` and all `--sim-*` options (including the stall injection used for testing) are rejected outright on domain 0.

## Thermal protection (bridge runner)

Long motions (continuous dances of about 45 s) noticeably heat the hip, knee and waist motors; one 46 s combo motion raised the hip temperature by about 22 °C. On the real robot, waist_roll exceeded 80 °C **mid-motion** twice, the runner entered DAMP and the robot fell forward: the motion moves forward/backward by more than 1 m, so the rope could only be very slack and could not catch it. One of these was caused by 108 s of standby before the motion, during which the motors had already heated up.

**DAMP mid-motion means a fall**, so the bridge runner adds two layers of protection before the 80 °C DAMP:

| Mechanism | Default | Behaviour |
|---|---|---|
| R1+A / R1+X start threshold | `--warn-temp` 70 °C | If any motor exceeds it, entering standby and starting the motion are refused; WARN is printed during standby |
| **R1+X temperature-rise forecast** | Limit `--thermal-stop-temp` 65 °C; slope window `--temp-rate-window-s` 20 s; at least `--temp-rate-min-s` 10 s of standby data | For every online motor (taking the larger of its two temperature sensors), extrapolate using the temperature slope over the last 20 s of STANDBY_PRE: `T_now + max(slope, 0) × clip duration` must be < 65 °C, otherwise `REFUSED` is printed. So **stand by for at least 10 s before pressing R1+X** |
| **Smooth thermal stop** | Any online motor ≥ 65 °C in POLICY (`--thermal-stop-temp`, ≤ 0 disables it) | **Does not enter DAMP**: over `--thermal-stop-s` (1 s) the reference is blended from the current frame to a static reference, **the policy keeps balancing**, then STANDBY_POST is entered. After a thermal stop, going to the next clip is **not allowed** (L2+up is ignored); re-hang as in step 7 of the operator procedure, then press L2+B |
| DAMP | `--max-temp` 80 °C | Enters DAMP in any state |

`--thermal-stop-pose` determines the posture the thermal stop ends in:

| Value | Meaning |
|---|---|
| `current` (default) | Stop in the pose of the frame at which it triggered (joint velocities zeroed, body velocities faded out) |
| `upright` | Same as above, but the 3 waist joints return to their frame-0 values |
| `frame0` | Cosine blend back to the clip's frame-0 standing pose |

Why `current` is the default:

- `frame0` was tried on the real robot: after stopping mid-motion, the feet could not return to the frame-0 stance and the robot fell.
- Another real run stopped and held on a frame with the torso bent far forward; waist_pitch rose from 72 °C to 81 °C within 6 s and the runner entered DAMP. `upright` was added for this, but in simulation it did not reduce the waist_pitch torque (the policy itself pushes waist_pitch to about −0.49 rad), so the default is still `current`.
- 65 °C, 15 °C below 80 °C, is the margin taken from that experience of "9 °C rise within 6 s after stopping".

`--temp-headroom MOTOR:SENSOR:DEG` relaxes all temperature thresholds (`--max-temp`, `--warn-temp`, `--thermal-stop-temp`, the rise forecast) for one temperature sensor of one motor. Use it only when you have confirmed that sensor reads high and you understand the consequences.

Operating advice (from real-robot experience):

- **Start cool**: the maximum temperature at the start should be around 45 °C. Keep standby as short as possible, just long enough for the 10 s rise forecast and for the crew to set the rope.
- The 21.9 s `shanshen_duosuomao_lean35kfs_pad2` runs of `e24_0355000` used `--thermal-stop-temp 78 --max-temp 90 --warn-temp 75` (higher than the defaults, **UNVERIFIED** against the motor ratings; the gap between thermal stop and DAMP is 12 °C instead of 15 °C). The hottest motor in POLICY reached only 55 °C, so none of them triggered. See [Real-robot result: lean35kfs](REAL_ROBOT_LEAN35KFS.md#deploying-it). For long motions such as the 46 s combo, keep the defaults.
- Let the motors cool down fully between consecutive runs. After a fall, we observed the same standby heating rate become about 2.4× the original; check the mechanics (waist) before continuing.
- Simulation tests: `run_sim_e2e_bridge.sh thermal` (inject 75 °C during POLICY → thermal stop → STANDBY_POST), `thermal_buttons`, `forecast_refuse`, `forecast_ok`.
  - `thermal` / `forecast_*` use `deploy_artifacts/e13i_0291000` and `deploy_artifacts/clips/combo_c4_full_vc_b15f20.npz` by default. The latter must be concatenated and exported under this name as described in [Custom motions: CSV → training and deployment](CUSTOM_MOTIONS.md).
  - From the repository root you can also use `CKPT=<directory name> CLIPS=$PWD/deploy_artifacts/clips/<clip>.npz` to specify a different model and clip.
  - sim2sim gate for the thermal-stop pose: first generate test clips with `scripts/deploy/make_thermal_stop_clips.py`, then run `sim2sim.py --cmd g2`.

## G1 model, `ai` mode and debug mode

**`mode_machine`**: the model codes given by Unitree's G1 description-package README (`unitree_rl_gym/resources/robots/g1_description/README.md`; the same table appears in `xr_teleoperate/assets/g1/README.md`):

| MJCF/URDF | mode_machine | Hip roll gear ratio | Legs | Waist | Arms |
|---|:-:|:-:|:-:|:-:|:-:|
| g1_23dof | 1 | 14.5 | 6×2 | 1 | 5×2 |
| g1_29dof / with_hand | 2 | 14.5 | 6×2 | 3 | 7×2 |
| g1_29dof_lock_waist | 3 | 14.5 | 6×2 | 1 | 7×2 |
| g1_23dof_rev_1_0 | 4 | 22.5 | 6×2 | 1 | 5×2 |
| **g1_29dof_rev_1_0 / with_hand_rev_1_0** | **5** | 22.5 | 6×2 | **3** | 7×2 |
| **g1_29dof_lock_waist_rev_1_0** | **6** | 22.5 | 6×2 | **1** | 7×2 |
| g1_dual_arm | 9 | – | 0 | 0 | 7×2 |

- Models with and without hands share the same code, so mode_machine **does not distinguish whether dexterous hands are fitted**.
- This repository's policies are trained on mode 5 (3-DoF waist).
- Our robot initially read 6 (motors 13/14 did not respond); after confirming with Unitree, the waist was enabled, and it now reads 5 with normal readings on 13/14, and the directions have been checked one by one. **If your robot reports 5, follow the general procedure without `--lock-waist`**; if it reports 6, see the next section.
- **A lowcmd is only executed if its `mode_machine` equals the value in lowstate** (answer by a Unitree collaborator in unitree_rl_lab issue #6). The runner locks the lowstate value at startup and writes it back into every lowcmd.
- A user reported that "after unlocking the waist, mode_machine can become 5", without saying how to unlock it (**UNVERIFIED**); confirm with Unitree or your dealer.

**`ai` mode**:

- `MotionSwitcher CheckMode` returning `{'form': '0', 'name': 'ai'}` means Unitree's built-in main motion-control service `ai_sport` is running (it handles damping, zero torque, standing, walking, etc. on the remote). It only says the service is running, not which FSM state the robot is currently in.
- As long as `ai_sport` is running it may be publishing `rt/lowcmd`, which conflicts with our commands, so the runner refuses to start in this case.

**Two ways out**:

| Method | Notes |
|---|---|
| **Remote L2+R2** (enter "debug mode") | The approach of the unitree_rl_gym deployment docs: boot while hoisted, wait for the robot to enter zero-torque mode, then press L2+R2; the robot enters debug mode with the joints damped. It does not actively hold its pose, so **it must be done while hoisted**. A third-party guide says it sometimes takes several presses. Whether CheckMode returns an empty name afterwards is **UNVERIFIED**; go by the real readings |
| SDK `MotionSwitcherClient().ReleaseMode()` | The official example calls it repeatedly while CheckMode's name is non-empty, until it is empty. The physical behaviour on the G1 at the moment of release is **not verified** (on the Go2, someone reported joint torque briefly dropping to 0 and the legs going limp); treat it as "may suddenly go limp", and **it must be hoisted**. There are reports that the high-level API stops working after the call, and `SelectMode('ai')` or a robot reboot is needed to restore it |

**The runner and check_robot themselves never call ReleaseMode / SelectMode.** On our robot, L2+R2 sometimes failed to clear `ai` mode; in the end it was resolved by the operator manually calling `MotionSwitcherClient().ReleaseMode()` once with the robot **hoisted**.

**Order for entering debug mode while hoisted** (all **UNVERIFIED**; go by the real readings):

1. Hoist the robot, feet off the ground, nobody standing around, remote in hand. In another terminal, keep the read-only monitor `check_robot.py --iface <nic> --domain 0 --check-mode` running (add `--lock-waist` on a waist-locked robot).
2. Confirm the robot is in zero-torque state (the boot default; the joints can be moved by hand). If not, first press L2+B to enter damping. **Do not** press L2+up or R2+A, which make the built-in motion control stand up.
3. Press **L2+R2** (hold L2 and tap R2, several times if necessary); the joints should become damped.
4. Restart check_robot (`--check-mode` only queries once at startup). CheckMode should be `no motion mode active` (empty name); no other `rt/lowcmd` publisher within 1 s (you can use `ro_bridge_check.py`); write down mode_machine.
5. Only after everything in step 4 passes, do the button test, sign check and kill -9 test.
6. To restore the built-in motion control, reboot the robot (`SelectMode('ai')` is a call this project does not make).

## Waist-locked G1 (`mode_machine = 6`)

Applies only when the robot reports 6. For the full simulation evaluation, see the waist-locked G1 sim2sim evaluation in [SIM2SIM.md](SIM2SIM.md).

**Basis:**

- The waist-locked URDF differs from the 29-DoF one only in that `waist_roll_joint` and `waist_pitch_joint` change type from revolute to fixed (origin rpy=0, same mass and inertia). The MJCF removes these two joints and their actuators (nu=27). The SDK example's comment for motors 13 and 14 is `INVALID for g1 23dof/29dof with waist locked`.
- The existing policies were trained on a 3-DoF waist and do rely on waist_pitch for balance (for clips such as itching and wave, the joint with the largest torque is waist_pitch, about 33–36 Nm).
- In waist-locked sim2sim, `ckpt_0144000` has 0 falls and 0 false aborts on the SAFE clips, with a 27-joint RMSE 1.20× (worst 1.41×) that of the same conditions without the waist lock. The cost is **7–10° of forward pelvis lean (peak 15–17°)**, leaving only about 3–4° of margin to POLICY's 20° tilt abort. walk_sideway at kp 0.85 / torso +1.8 kg produced 7 false tilt aborts at 20–22°.
- Therefore, until there is a checkpoint trained specifically for the waist-locked model, use `ckpt_0144000`, and the only allowed clip order is **neutral_idle → shoulder_clap**. itching / walk_sideway (and idle_loop_002) are only run with a waist-locked checkpoint or after explicit approval; HARDER clips are never run on a waist-locked robot.

**Behaviour of `--lock-waist`** (runner and check_robot):

| Item | Behaviour |
|---|---|
| mode_machine | Only two combinations are accepted: 5 without `--lock-waist`, or 6 with `--lock-waist`. Any mismatch refuses startup, also under `--sim` |
| lowcmd for 13/14 | `mode=0`, `q=dq=tau=kp=kd=0` in every state (the most conservative choice; whether the firmware ignores it or reports an error is **UNVERIFIED**) |
| Observation | lowstate q/dq for 13/14 are passed through as-is (they read 0). The policy still outputs a 29-dim action; the targets for 13/14 are only logged, not sent |
| Checks | 13/14 are excluded from R1+A's `max\|q−f0\|`, `max\|dq\|` and temperature checks, and from the run-time temperature and motorstate checks |
| Monitoring | q/dq/temperature/motorstate of 13/14 are logged every step. At any time \|q\| > 0.02 rad or temperature ≠ 0 → **DAMP `waist unexpectedly active`**; if this shows up at startup or in the re-check after 'yes', startup is refused outright |
| Clip whitelist | By default only `neutral_idle_loop_001__A081` and `shoulder_clap_004__A041_M` are allowed (`g1_common.LOCK_WAIST_ALLOWED_CLIPS`, replaceable with `--clip-allow-file`). Other SAFE clips must be allowed one by one with `--i-know-clip <name>`, only after explicit approval, only at the `slack` stage, and only after idle and shoulder_clap. wave_R / jog_sideway / dance_hiphop are **always refused, with no override switch** |
| Policy standby | **Standby on the waist-locked model did not pass the simulation gate**: STANDBY_POST after shoulder_clap passed 0/6, with stepping and 20° tilt aborts. So `--lock-waist` **implies `--end-hold-legacy`**, and the waist-lock standby whitelist `g1_common.LOCK_WAIST_STANDBY_ALLOWED_CLIPS` is **empty by default**. `--lock-waist-standby-allow <name>` may only be used after the `sim2sim.py --cmd standby --lock-waist --ckpt ckpt_0144000` gate passes. For clips not on the whitelist, R1+A / R1+X are both refused; with **`--lock-waist-direct-start`**, R1+A in HOLD **starts the motion directly** (old flow), and the hoist must carry the robot |
| Summary | At startup, prints a `WAIST LOCK` banner, the current 13/14 readings, and which path each clip takes. The run summary prints the maximum 13/14 readings, the maximum q/kp/kd actually sent (should be 0), and, for POLICY, **the maximum tilt error and the margin to the 20° abort** |
| check_robot | Expects mode 6 and marks 13/14 `LOCKED` (`LOCKED-ACTIVE!!` if the readings are abnormal). Gives a hint if it sees 6 without `--lock-waist` |

**Commands** (old flow: R1+A in HOLD starts the motion directly, clip ends → END_HOLD, both only while the hoist carries the robot; the stages are likewise loosened step by step starting from `air`):

```bash
python scripts/deploy/g1_runner.py --iface <nic> --domain 0 --lock-waist --lock-waist-direct-start --stage air \
  --artifacts deploy_artifacts/ckpt_0144000 \
  --clip deploy_artifacts/clips/neutral_idle_loop_001__A081.npz \
  --log-dir deploy_logs/real
# shoulder_clap after idle (END_HOLD after idle ends, the hoist must already carry the robot; after the blend press L2+up to go to the next clip)
python scripts/deploy/g1_runner.py --iface <nic> --domain 0 --lock-waist --lock-waist-direct-start --stage feet \
  --artifacts deploy_artifacts/ckpt_0144000 \
  --clip deploy_artifacts/clips/neutral_idle_loop_001__A081.npz deploy_artifacts/clips/shoulder_clap_004__A041_M.npz \
  --log-dir deploy_logs/real
```

- A 7–10° forward pelvis lean is expected, but it eats into the tilt-abort margin. After every run, check `tilt error vs reference max ... margin ...` in the summary; **do not move to the next stage when the margin is below about 5°**. Whether the real robot leans this much too is **UNVERIFIED**.
- The sign check skips waist_roll / waist_pitch.

## Logs and summaries

Every run produces, under `--log-dir`:

- `run_YYYYmmdd_HHMMSS.npz`: the main log.
- `run_YYYYmmdd_HHMMSS.console.txt`: a complete copy of the console output, including the `STATUS` line every 3 s during standby. It keeps being written after the terminal disconnects.
- `run_YYYYmmdd_HHMMSS.bridge.txt`: the C++ bridge process's log (bridge runner only).

Keys in the npz:

| Category | Keys |
|---|---|
| 50 Hz records | `t, deadline, compute_ms, state, clip, frame, q, dq, gyro, quat, tick, state_age_ms, action, q_target, n_clamped, cmd_kp_mean`; with `--log-obs` also `obs_hist`; recording stops after `--rec-max-s` |
| Command log (one row per lowcmd) | `cmd_t, cmd_q, cmd_kp, cmd_kd, cmd_state`. This is a ring buffer, keeping 900 s by default (`--cmd-log-max-s`); a WARN is printed when it fills |
| `meta` (JSON) | Arguments, events, DAMP reason, send-failure count, DDS callback error count, stdout failure count, CPU pinning result, `states` (state number table, STANDBY_PRE=6, STANDBY_POST=7), `t0_monotonic`, etc. |

A summary is printed automatically at the end of the run: one line per STANDBY_PRE / POLICY / STANDBY_POST segment (duration, tilt error max/mean, max|dq| max/p99, max|a|), with a final line giving the total time in each state. To re-print the summary:

```bash
# logs from g1_runner.py
python -c "import sys; sys.path.insert(0,'scripts/deploy'); import g1_runner as r; r.summarize('deploy_logs/real/run_XXXX.npz')"
# logs from g1_runner_bridge.py (only reads the npz, does not start the bridge)
python -c "import sys; sys.path[:0]=['scripts/deploy','scripts/deploy/bridge']; import g1_runner_bridge as r; r.summarize('deploy_logs/real/run_XXXX.npz')"
```

Check each item after every run:

1. Whether the DAMP reason is exactly the expected one.
2. Policy cycle |jitter| p99 < 2 ms; state age p99 only a few milliseconds; lowcmd about 500 Hz (about 526 Hz including the extra sends).
3. `max|action|`, maximum tilt, `time within 0.02 rad of the hard stop`.
4. After the run, look at the temperatures once more with check_robot.

## Simulated end-to-end tests (DDS)

Usage:

- Bridge runner: `bash scripts/deploy/bridge/run_sim_e2e_bridge.sh <test> [extra runner arguments]`
- Pure-Python runner: `bash scripts/deploy/run_sim_e2e.sh <test>`

Logs are in `deploy_logs/sim_e2e/<test>/`. The test environment is the official unitree_mujoco G1 29-DoF MJCF (domain 1, `lo`), physics step 0.002, default `STAGE=ground`. An elastic band is attached during PASSIVE/INTERP/HOLD/END_HOLD and removed once policy standby is entered. Available environment variables: `CKPT`, `CLIPS`, `STAGE`, `ARM_KD`, `LOCK=1` (waist lock), `LW_DIRECT=0`, `SIM_DT`, `E2E_NAME`, `G1_DEPLOY_HOME`.

| Test | What it checks | Our results (e11 / early ckpt) |
|---|---|---|
| `auto` / `auto_legacy` | Full flow (idle + shoulder_clap, including standby and the END_HOLD blend); prints the torque rate of change at each state transition | 939 steps with no abort, POLICY tilt error max 5.1° |
| `standby` / `standby_exit` / `standby_post2` | R1+X ignored in HOLD; R1+X refused with < 3 s of standby; R1+A in STANDBY_PRE does not start the motion; L2+up double-press confirmation (a single press only prints a hint and expires after 3 s); STANDBY_PRE → HOLD of the same clip | As expected; standby tilt error ≤ 3.5°, drift ≤ 2.4 cm |
| `standby_nan` / `nan` | Policy returns NaN | `DeployCore abort ... non-finite action` → DAMP |
| `kill` | Kill the simulator during the policy | `stale lowstate: ~21 ms` → DAMP |
| `sigterm` / `sighup` | Signals | DAMP, exits with code 0 after 1.00 s of damping; a second runner started at the same time refuses to start and publishes no commands |
| `stdout` | stdout read end closed mid-run | Write failures are swallowed, the run completes normally, console.txt is complete |
| `stall` | Main loop blocked for 0.5 s | `main loop stalled: no new command for ~51 ms` → DAMP |
| `foreign` | A bare rt/lowcmd publisher appears mid-policy | `foreign rt/lowcmd publisher` → DAMP |
| `buttons` / `prebutton` | Button injection; buttons pressed before 'yes' and buttons held through 'yes' have no effect | As expected |
| `thermal` / `thermal_buttons` / `forecast_refuse` / `forecast_ok` | Thermal stop, L2+up ignored after a thermal stop, rise forecast refuses / allows (bridge only) | See [Thermal protection](#thermal-protection-bridge-runner) |
| `waist_active` / `waist_active_start` / `refuse_nolock` / `refuse_lock5` (`LOCK=1`) | Waist-lock monitoring and mode_machine consistency | DAMP / startup refused, no commands published |

**Torque rate of change at transitions** (simulation, maximum over all joints of |Δτ|/Δt, in Nm/ms):

| Transition | Rate |
|---|---|
| HOLD → STANDBY_PRE (formerly HOLD → POLICY, mainly at the knees) | About 11–14 (17.8–21.9 in sim2sim at 1 kHz) |
| STANDBY_PRE → POLICY | ≤ 1.7 (8.6–9.9 for ib_dodge, which has a high initial velocity) |
| POLICY → STANDBY_POST | ≤ 2.5 |
| END_HOLD 1 s blend | ≤ 0.6 (15 without the blend) |

The jump when the policy takes over from HOLD has not been addressed; doing so would need a gain blend at policy start, which needs its own evaluation.

**One unexplained overrun**: in one run, the first STANDBY_PRE step took 96 ms to compute, triggering `policy loop overrun` → DAMP (re-running the same test 3 times gave 2–4 ms; other jobs were running on the machine at the time). On the real robot the rope is just taut at this moment, so the hoist should catch the robot on DAMP. This is one of the reasons step 3a insists on the rope being "just taut".

## sim2sim findings relevant to the real robot

For the full sim2sim gates, see [Export and sim2sim gates](SIM2SIM.md).

- **On the ground, HOLD or END_HOLD falls**: holding a pose open-loop cannot balance, and both tip over after about 1–1.3 s. So press R1+A only when the hoist can catch the robot, and enter END_HOLD only while the hoist is supporting it.
- Policy-standby gate (`sim2sim.py --cmd standby`; grid of latency 0/10/20 ms × friction 0.6/1.0 × kp 0.85/1.0/1.15 × torso +0/+1.8 kg):
  - **192/192 pass at kp 1.0 and 1.15**, with standby tilt error ≤ 5.15° and horizontal drift ≤ 4.8 cm.
  - **Only 24/96 pass at kp 0.85**: no falls, but stepping in place, walking (up to 5 m), turning (up to 180°), and 11 aborts during motions. With kp ≥ 0.95, drift ≤ 4.1 cm.
  - In 300 s long standbys, yaw keeps accumulating, up to 164°.
  - With the R1+X check actually enforced, 32/32 pass at kp 1.0, and the check first passes at 3.04–3.14 s of standby. At kp 0.85 the 3 s check blocks most "start while walking" cases, but **cannot prevent the motion itself from failing at low stiffness** (still 4 aborts).
- 0–20 ms latency did not change the results on the SAFE clips; faster clips were not verified.
- hip_roll torque limit: 139 Nm in both training and `deploy_config`, 88 Nm in the Unitree MJCF; the actual motor spec is **UNVERIFIED**.

## UNVERIFIED checklist (confirm each item before going on the robot)

1. Whether the wired network card works; whether the motion-control board's IP is 192.168.123.161; the lowstate rate and intervals on PC2 (`ro_bridge_check.py`).
2. The `mode_machine` your robot reports (5 = movable waist; 6 = waist locked), and whether it stays unchanged after entering debug mode (if it changes, the runner enters DAMP). On a waist-locked robot, also confirm: whether the lock angle is 0; whether the all-zero lowcmd for 13/14 is ignored by the firmware; the motorstate values of 13/14; whether the real pelvis also leans forward 7–10°.
3. How to enter debug mode (L2+R2); whether the built-in motion control stops publishing afterwards; whether this computer can reach the MotionSwitcher service (whether CheckMode answers).
4. lowstate rate 500 Hz, whether the tick increments, whether the CRC is valid. If the CRC is invalid, `--no-state-crc` is needed.
5. Whether `imu_state` is the pelvis IMU, whether the quaternion order is w,x,y,z, and whether the gyro is in the body frame.
6. Remote-controller button bits (**including R1+X**, X = bit 10); whether L2+B, L2+up and R1+X also trigger Unitree's own functions in debug mode.
7. **Motor behaviour after lowcmd stops** (kill -9 test), and the robot's state after the runner exits.
8. Whether there is a hardware e-stop; power-button behaviour.
9. Battery level threshold.
10. What each of the two motor temperature values is and their units; whether the 65 / 70 / 80 °C thresholds are appropriate.
11. Bit meanings of `motorstate`, and whether it is 0 when normal. If it is not 0, use `--motor-error-mask` to mask only bits confirmed to be harmless; do not turn the check off entirely.
12. Velocity limits (37 rad/s for waist and ankles); whether the hip_roll torque limit is 88 or 139 Nm.
13. Whether arm kd=10 and waist kd=5 oscillate on the real robot.
14. Signs of all joints.
15. Effect of real-robot latency on the policy; whether the firmware clips targets outside the joint range.
16. **Real-robot behaviour of policy standby**: the true stiffness of the real motors (with kp 15% low, standby steps and walks); yaw accumulation during long standbys; how much body weight the rope actually carries when "just taut" (≤ 50% is fine, 80% bounces).
17. **Motor-temperature WARN during standby** (`--warn-temp`, STANDBY_* states) cannot be triggered in simulation (simulated temperature is 0) and has only been code-reviewed; the thermal stop and rise forecast have `thermal` / `forecast_*` injection tests.
18. **lowstate gaps during long standbys**: with the feet carrying the weight and the rope just taut, do one 5 min policy standby and record `lowstate age max` from `STATUS` and the maximum state age from the summary. Until then, **standby is limited to 2 min by procedure**.

## Troubleshooting and known issues

| Symptom | Cause / fix |
|---|---|
| The runner reports another rt/lowcmd publisher at startup, or CheckMode shows `ai` | Unitree's built-in motion control is still running. Hoist the robot and enter debug mode (L2+R2, several times if necessary). If it does not clear, have the operator manually call `MotionSwitcherClient().ReleaseMode()` once with the robot hoisted, then confirm with `ro_bridge_check.py` that there are 0 other publishers |
| `mode_machine=6 ... Restart with --lock-waist`, or `--lock-waist given but mode_machine=5` | The model and the arguments do not match. Confirm the model and follow the corresponding procedure; do not change arguments just to get past the check |
| CheckMode returns 3102 (`RPC_ERR_CLIENT_SEND`) / `NO ANSWER` | The MotionSwitcher service is not visible (always the case in simulation). The runner only warns; with `--require-check-mode` it refuses instead |
| lowstate rate drops, burst gaps appear, `stale lowstate` DAMP | The robot's LAN is degraded (badly seated Ethernet cable). Re-seat the cable, and run `ro_bridge_check.py` before every session to compare |
| Policy loop overruns on PC2, large tilt error during standby | Pure-Python DDS saturates the GIL. Switch to the bridge runner; make sure the CPU governor is `performance` and do not run other jobs at the same time |
| onnxruntime on PC2 fails to load `policy.onnx` | The IR version is too high. Use `policy_ir9.onnx`, or run `python -c "import onnx; m=onnx.load('policy.onnx'); m.ir_version=9; onnx.save(m,'policy_ir9.onnx')"` |
| R1+A refused (`max\|dq\|` ≥ 0.5) | High-frequency arm oscillation in HOLD. Add `--stand-arm-kd 2 --stand-waist-kd 2` |
| R1+X refused | See the reason at the end of the `STATUS` line: standby shorter than 3 s, too much shaking over the last 3 s, temperature too high, or the rise forecast exceeds 65 °C (bridge runner). Press again once the robot is steady or has cooled down; the rise forecast requires at least 10 s of standby |
| Yaw / leg-speed WARN during standby | Re-align the gantry; if stepping, re-hang as in step 7 of the operator procedure, then L2+B |
| `SCHED_FIFO` reports `PermissionError` | No rtprio permission. Does not affect operation; all timing data was measured in this situation |
| Thermal stop during a motion | The next clip cannot be run. Re-hang, then L2+B; let the motors cool fully (and check the waist mechanics) before running again |

Still unresolved:

- The torque jump when the policy takes over from HOLD (about 11–14 Nm/ms).
- Low-rate (below about 170 Hz) foreign rt/lowcmd publishers cannot be detected at run time; the deadlock cause is only presumed.
- On the ground, HOLD can still only be held up by the hoist.
- The standby time limit is currently only a procedure, not a hard abort.
- Standby is sensitive to motor stiffness; the root fix is to add gain randomization in training. Yaw accumulation during long standbys is also not addressed.
- Policy standby on the waist-locked model did not pass the simulation gate and is disabled by default.
