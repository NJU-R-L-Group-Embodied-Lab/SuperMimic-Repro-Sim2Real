# Custom motion data

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

`motions/custom_dance/` holds the reference motions used by the shanshenbu (闪身步) demo; `tools/video2motion/` is a tool that turns a single-person dance video into G1 motion CSVs of this kind. This page first describes the motion data released with the repository, then the steps from CSV to training and deployment. The video-to-motion pipeline itself is described in [Video to motion](VIDEO_TO_MOTION.md).

## Custom motion data (motions/custom_dance)

These are the reference motions used by the shanshenbu (闪身步) demo:
- Four dance clips: retargeted to the G1 from the same third-party dance video. The video is 1440×1080, 30 fps, fixed camera, mostly wide shots with close-up shots interspersed.
- One four-clip continuous motion: the reference that the released `e13i_0291000` policy runs on the real robot, resampled to 30 fps.

For the format, see "Output format and checks" in [Video to motion](VIDEO_TO_MOTION.md).

| File | Motion | Frames (30 fps) | Duration | Notes |
|---|---|---|---|---|
| `shanshenbu.csv` | shanshenbu (闪身步) | 216 | 7.2 s | |
| `langzi_tiqiu.csv` | langzi tiqiu (浪子踢球) | 152 | 5.1 s | |
| `gouxiong_duosuomao.csv` | gouxiong duosuomao (狗熊哆嗦毛, full version) | 397 | 13.2 s | Runs from the first full-body wide shot through the close-up shot into the second wide shot, until the dancer comes to a standstill; contains several small hops. The left knee has one velocity spike of 17.5 rad/s, which exceeds the real-robot speed limit, so joint-velocity capping must be applied before concatenation (step 2 of "[CSV → training and deployment](#csv--training-and-deployment)") |
| `yaotou_huangnao.csv` | yaotou huangnao (摇头晃脑, full version) | 313 | 10.4 s | Extended until the dancer comes to a standstill |
| `shanshenbu_combo4.csv` (+ `.json`) | Four-clip continuous motion | 1373 | 45.8 s | The 50 Hz deployment reference of `e13i_0291000`, `combo_c4_full_vc_b15f20` (2288 frames, 45.74 s), resampled to 30 fps. Order: 1 s static stance → shanshenbu → langzi tiqiu → gouxiong duosuomao (after joint-velocity capping, `_vc`) → yaotou huangnao → 1 s static stance. Cosine transitions of 1.5 s, with the last transition 2.0 s. Segment boundaries (frame indices at 30 fps) are written in the `.json` |

**How they were made**

1. The video was cut into 4 segments by motion, keeping only the wide-shot parts in which the full body is visible.
2. Each segment was processed with the "One-command run" flow (see [Video to motion](VIDEO_TO_MOTION.md)), with three differences from the current defaults:
   - GVHMR with `-s`;
   - `gvhmr_to_supermimic.py` with GMR's original initialization, i.e. `RETARGET_ARGS=""`, without `--init_from_human` / `--restart_tol`;
   - grounding with `--floor_mode drift`.

   The G1 model at the time was an MJCF converted from the Unitree `g1_29dof_rev_1_0` URDF, with four 5 mm-radius sole spheres per foot. In the default pose its lowest sole point is the same as that of the mjlab G1 (capsule soles), so switching to the mjlab model only introduces millimeter-level differences.
3. For `gouxiong_duosuomao` and `yaotou_huangnao`, the first wide shot cuts to a close-up in the middle of the motion, so `extend_full.py` (see "Tips and lessons learned" in [Video to motion](VIDEO_TO_MOTION.md)) was used to extend them until the dancer comes to a standstill in the next wide shot:

   | Clip | Existing rows | Arguments | Result |
   |---|---|---|---|
   | `gouxiong_duosuomao` | 269 | `--cuts 271 334 --feet steps --place cam --human_height 1.70` | 397 rows (13.2 s) |
   | `yaotou_huangnao` | 185 | `--cuts 187 252 --feet fixed --place feet` | 313 rows (10.4 s) |

   These two files are released directly in their extended versions. In the close-up parts only the upper body is estimated from the video; the legs are inferred.
4. For joint-velocity capping, concatenation into `shanshenbu_combo4`, training and deployment, see "[CSV → training and deployment](#csv--training-and-deployment)".

**Usage notes**

- The segment names in `shanshenbu_combo4.json` keep our original internal clip names: `gouxiong_duosuomao_full_vc` is `gouxiong_duosuomao` after joint-velocity capping (i.e. `gouxiong_duosuomao_vc`), and `yaotou_huangnao_full` is `yaotou_huangnao` in this directory.
- The 1 s static stance used for concatenation has no separate file. It is simply the first 30 rows of `shanshenbu_combo4.csv` (30 identical stance frames); extract it with `head -n 30 shanshenbu_combo4.csv > stand_idle.csv`.
- `shanshenbu_combo4.csv` can be converted directly to npz for viewing or training, but with two limitations: converting back to 50 Hz is a second resampling (50→30→50 Hz), and it has no `segment_*` metadata. To reproduce the deployment clip, re-concatenate it from the four dance clips following the recipe in "[CSV → training and deployment](#csv--training-and-deployment)".

**Source and copyright**

- The copyright of the source video belongs to its original author / publisher, and the video is not distributed with this repository. This directory distributes only the retargeted robot joint trajectories and contains no video frames or audio. If you are the rights holder of the source video and want this data removed, please contact us by opening an issue.
- These CSVs are released under the MIT license like the rest of the repository, but only to the extent of the rights we hold in the trajectory data itself.
- The pipeline that generated them depends on GVHMR and SMPL-X, both of which are restricted to non-commercial use; see "License notice (read first)" in [Video to motion](VIDEO_TO_MOTION.md).

## CSV → training and deployment

The CSV can come from `motions/custom_dance/`, from the video-to-motion pipeline ([Video to motion](VIDEO_TO_MOTION.md)), or from any retargeting tool that outputs the same format.

**1. CSV → 50 Hz npz** (output name = CSV file name without extension). The 1 s static stance `stand_idle` used for concatenation has no separate file; take it from the first 30 rows (30 identical stance frames) of the four-clip continuous motion CSV:

```bash
mkdir -p data/raw/custom_dance
cp motions/custom_dance/*.csv data/raw/custom_dance/
head -n 30 motions/custom_dance/shanshenbu_combo4.csv > data/raw/custom_dance/stand_idle.csv
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/custom_dance --output-dir data/npz/custom_dance --input-fps 30
# -> data/npz/custom_dance/{shanshenbu, langzi_tiqiu, gouxiong_duosuomao, yaotou_huangnao, shanshenbu_combo4, stand_idle}.npz
```

**2. (Optional) Joint-velocity capping**: the real-robot runner aborts when a joint velocity exceeds a certain fraction of its limit. If the reference motion has spikes above the limit (our `gouxiong_duosuomao` has one of 17.5 rad/s on the left knee), smooth them locally first:

```bash
uv run python scripts/deploy/cap_joint_velocity.py --src data/npz/custom_dance/gouxiong_duosuomao.npz \
    --dst data/npz/custom_dance/gouxiong_duosuomao_vc.npz --deploy-config deploy_artifacts/e13i_0291000/deploy_config.json
```

`--deploy-config` is only used to read each joint's `velocity_limit`. It does not depend on the checkpoint (these are robot constants), so any `deploy_config.json` works: for example `deploy_artifacts/e13i_0291000/deploy_config.json` from the Release, or one produced by first running `--cmd config` from "Export the deployment bundle" in [Export and sim2sim gates](SIM2SIM.md) on any checkpoint.

**3. (Optional) Concatenate into one continuous reference**: the clips are rigidly aligned (horizontal position and heading) and joined with cosine transition segments; velocities and rigid-body states are recomputed over the whole sequence. Below is the recipe for the `e13i_0291000` deployment clip (a 1 s static stance `stand_idle` at the start and at the end, 1.5 s transitions, the last transition 2.0 s, ending back in the stance; ending on a frame in motion makes the STANDBY_POST after the motion unstable):

```bash
uv run python scripts/deploy/make_combo_clip.py --npz-dir data/npz/custom_dance --out-dir data/npz/custom_combo \
    --clips stand_idle shanshenbu langzi_tiqiu gouxiong_duosuomao_vc yaotou_huangnao stand_idle \
    --name combo_c4_full_vc_b15f20 --blend-s 1.5 --last-blend-s 2.0
```

Our original `combo_c4_full_vc_b15f20` is 50 Hz, 2288 frames, 45.74 s. The `stand_idle` above is converted from a 30-frame CSV, so its frame count may differ slightly from the original 50 frames; the rebuilt result therefore approximates the original clip but is not guaranteed to match it frame for frame. `shanshenbu_combo4.npz` (already converted in step 1) can be used directly, but see the two limitations under "Usage notes" above. With the name `combo_c4_full_vc_b15f20`, the `run_sim_e2e_bridge.sh thermal|forecast_*` tests can use it directly (they default to this clip name).

There are also options such as `--end-settle-s` (decays the reference velocity to 0 at the end, so that the policy can stand by in place after the motion) and `--fade-out-s`; see `--help`. A `<name>.json` recording the construction parameters is written as well.

**4. Training / fine-tuning**: a small number of clips can be used directly with `--data.npz-dir data/npz/custom_combo`; to mix with BONES-SEED, put the needed npz files in one directory (symlinks are fine), provide a manifest with `name,keep` columns, and pack them with `build_library.py`. Our approach is to fine-tune from the paper-scope checkpoint pretrained on BONES-SEED, for example:

```bash
uv run sm-train --run-name custom_ft --resume logs/bones/ckpt_XXXXXXX.pt --resume-reset-sampler \
  --data.library data/library/bones_plus_custom.pt --schedule.total-iterations <total steps, larger than the starting step> \
  --env.tilt-hinge-weight -24 --env.tilt-hinge-knee 0.17      # tilt hinge that deviates from the paper; needed for jumping motions
```

**5.** Export the deployment bundle and run the sim2sim gates: see [Export and sim2sim gates](SIM2SIM.md). **6.** Go to the real robot: see [Real-robot deployment (sim2real)](DEPLOY_G1.md).
