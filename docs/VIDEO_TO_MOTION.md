# Video to motion pipeline

> 中文概览见 [README.md](../README.md)。English overview: [README_EN.md](../README_EN.md).

## Pipeline overview

`tools/video2motion/` turns a single-person dance video into a Unitree G1 29-DoF reference motion in the same format as `motions/custom_dance/` (CSV, no header, 30 fps, 36 columns). After that it feeds into this repository's conversion, training and deployment flow; see [CSV → training and deployment](CUSTOM_MOTIONS.md#csv--training-and-deployment). The four dances in `motions/custom_dance/` were made with these scripts. You can also use any other retargeting tool that outputs the same CSV format.

```
video (any frame rate)
  │ ffmpeg -vf fps=30                       ← GVHMR / GMR assume 30 fps by default and do not resample on their own
  ▼
30 fps video
  │ [optional] shot_bbx.py                  ← with hard shot cuts: per-shot person boxes (GVHMR env)
  ▼
GVHMR tools/demo/demo.py -s                 ← YOLO + ViTPose + GVHMR, static-camera assumption, world-frame SMPL-X parameters
  │ hmr4d_results.pt
  ▼
gvhmr_to_supermimic.py                      ← GMR IK retargeting to unitree_g1 (GMR env)
  │   + joint-limit clipping (G1 MJCF)
  │   + foot-to-floor grounding (drift / constant), first-frame xy zeroed
  ▼
<clip>.csv (36 columns, 30 fps)  ── render_preview.py → source video | G1 side-by-side preview
  │ [optional] extend_full.py               ← complete motion cut off by close-up shots (experimental)
  ▼
scripts/data/csv_to_npz_batch.py → 50 Hz npz → training / joint velocity limiting / concatenation / deployment
```

`gvhmr_to_supermimic.py` is a headless version of GMR's own `scripts/gvhmr_to_robot.py`. It does three extra things: handles IK local minima, clips joint limits, and grounds the feet.

| File | Runs in | Purpose |
|---|---|---|
| `run_pipeline.sh` | bash (calls both environments) | One-command flow: 30 fps transcode → [shot_bbx] → GVHMR `-s` → CSVs with both grounding modes → preview |
| `shot_bbx.py` | GVHMR | With hard shot cuts, detects and smooths person boxes per shot and writes GVHMR's `preprocess/bbx.pt` |
| `gvhmr_to_supermimic.py` | GMR | GVHMR result → GMR retargeting → limit clipping + grounding → CSV |
| `render_preview.py` | GMR | Source video \| G1 side-by-side preview |
| `extend_full.py` | GMR | Experimental: completes motion cut off by close-up shots |
| `render_full_preview.py` | GMR | Comparison preview for `extend_full.py` results (fixed camera, shot labels) |
| `g1_model.py` | GMR | Shared G1 utilities: joint order, model loading, foot collision points, CSV read/write |

## License notes (read first)

| Component | License | How this repository uses it |
|---|---|---|
| `tools/video2motion/*` (this repository's code) | MIT | Released with the repository |
| [GVHMR](https://github.com/zju3dv/GVHMR) code and weights | **Education, research and non-profit use only**; commercial use requires contacting the authors separately (see its `LICENSE`) | **Not distributed with this repository**; users clone it themselves. `run_pipeline.sh` calls its `demo.py` as a subprocess; `shot_bbx.py` imports two helper functions from the user's GVHMR directory at runtime |
| Pretrained weights GVHMR depends on (HMR2, ViTPose, YOLOv8, etc.) | Their own licenses (e.g. ultralytics/YOLOv8 is AGPL-3.0) | Users download them following the GVHMR docs |
| [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models | MPI model licenses (registration required, **non-commercial scientific research only**) | **Not distributed with this repository**; users download them after registering |
| [smplx](https://github.com/vchoutas/smplx) Python package | See the `LICENSE` in its repository (tied to the SMPL-X model license) | Installed as a GMR dependency |
| [GMR](https://github.com/YanjieZe/GMR) | MIT | Users clone and install it themselves; this repository's code imports its Python package |
| G1 model (MJCF / meshes) | Unitree's G1 description files are BSD-3-Clause; the G1 in mjlab follows mjlab's license and notices | Not distributed with this repository; read from the mjlab or GMR installation |

**Bottom line**:
- This repository's glue code is MIT, but the whole pipeline depends on GVHMR and SMPL-X. So **running this pipeline, and using the intermediate results it produces (SMPL-X parameters), are subject to these non-commercial licenses**.
- Whether the resulting robot trajectories can be used for your purpose is for you to judge; comply with the licenses above.
- **Copyright in a source video belongs to its author.** Only run this pipeline on videos you have the right to use, and do not redistribute source video frames or preview videos.
- The licenses of all components are summarized in [LICENSES/THIRD_PARTY_NOTICES.md](../LICENSES/THIRD_PARTY_NOTICES.md).

## Environment setup

Three environments are needed:
- One separate Python environment each for GVHMR and GMR (GVHMR's dependency versions are older and conflict with GMR);
- This repository's own uv environment, used to resolve the G1 model path and for the later conversion.

The following uses Linux + NVIDIA GPU as the example. System dependencies:
- `git`, `ffmpeg`;
- An NVIDIA driver that supports CUDA 12.1 (needed by GVHMR);
- For headless rendering, EGL (ships with the NVIDIA driver) or OSMesa (`libosmesa6`).

External tools can go in any directory; below it is referred to as `$V2M_HOME`:

```bash
export V2M_HOME=$HOME/v2m          # any directory
mkdir -p $V2M_HOME
```

**GVHMR (pinned commit + one-line patch)**

- Version: commit **`ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`** of `https://github.com/zju3dv/GVHMR`.
- The only local change: delete the unused `from turtle import forward` on line 1 of `hmr4d/utils/body_model/body_model.py`. That line imports tkinter, which fails on a headless server.

```bash
cd $V2M_HOME
git clone https://github.com/zju3dv/GVHMR && cd GVHMR
git checkout ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
sed -i '/^from turtle import forward$/d' hmr4d/utils/body_model/body_model.py   # the only local change
git diff --stat                                                                   # should show only 1 line deleted

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt     # includes torch 2.3.0+cu121, pytorch3d, ultralytics, etc. (see GVHMR docs/INSTALL.md)
pip install -e .
conda deactivate
```

Then download the weights following GVHMR's `docs/INSTALL.md` and put them in `GVHMR/inputs/checkpoints/`. We only use the static-camera mode (`-s`), so DPVO is not needed.

```
inputs/checkpoints/
├── body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz   # SMPL-X, registration required at smpl-x.is.tue.mpg.de
├── body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl     # SMPL, registration required at smpl.is.tue.mpg.de
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt                                     # also used by shot_bbx.py
```

**GMR (pinned commit, unmodified)**

- Version: commit **`25f56db79ac411258b377b6d4ccba0c8ff3372ac`** of `https://github.com/YanjieZe/GMR`, with no modifications.
- `smplx` is installed as its dependency. GMR's `uv.lock` pins it to commit `1265df7ba545e8b00f72e7c557c766e15c71632f` of `https://github.com/vchoutas/smplx`.
- Be sure to pin this commit: `gvhmr_to_supermimic.py` uses GMR internals (`configuration`, `rot_offsets1`, `error1/error2`), and another commit may require code changes.

Pick either of the two installation methods:

```bash
cd $V2M_HOME
git clone https://github.com/YanjieZe/GMR && cd GMR
git checkout 25f56db79ac411258b377b6d4ccba0c8ff3372ac

# (a) uv: install into GMR/.venv per GMR's uv.lock, then use GMR_PY=$V2M_HOME/GMR/.venv/bin/python
uv sync

# (b) conda: environment name gmr
conda create -y -n gmr python=3.10 && conda activate gmr
pip install -e .
pip install --force-reinstall --no-deps "smplx @ git+https://github.com/vchoutas/smplx@1265df7ba545e8b00f72e7c557c766e15c71632f"
conda install -y -c conda-forge libstdcxx-ng      # recommended by the GMR docs (MuJoCo rendering under conda)
conda deactivate
```

The GMR environment already has what this tool needs: `mujoco`, `numpy`, `scipy`, `torch` (only used for `torch.load`; the CPU build is fine), `imageio[ffmpeg]` and `opencv-python`.

**SMPL-X model (download it yourself)**

Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and accept the license, then download the model in npz format. GMR uses the neutral model `SMPLX_NEUTRAL.npz` when reading GVHMR results:

```bash
mkdir -p $V2M_HOME/GMR/assets/body_models/smplx
ln -s $V2M_HOME/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
      $V2M_HOME/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

You can also put it elsewhere and point to it with `--body_models <dir>` or `SMPLX_BODY_MODELS=<dir>`, as long as `<dir>/smplx/SMPLX_NEUTRAL.npz` exists. This repository does not contain or distribute any SMPL / SMPL-X files.

**G1 model**

This G1 29-DoF MJCF is used for joint-limit clipping, foot height computation and preview rendering. By default it uses the mjlab G1 that this repository trains with; its joint order matches the CSV and its foot collision geoms are capsules. When `G1_XML` is not set, `run_pipeline.sh` resolves it automatically like this:

```bash
cd <this repo> && uv sync
export G1_XML=$(uv run python -c 'from supermimic.data.motion_convert import g1_xml_path; print(g1_xml_path())')
```

You can also substitute another G1 29-DoF MJCF, as long as the joint order is the same and the feet have sphere or capsule collision geoms, e.g. a model converted from Unitree's `g1_29dof_rev_1_0` URDF. GMR's bundled `g1_mocap_29dof.xml` has joint limits narrower than the real robot and is not recommended as `G1_XML`.

**Environment variables**

| Variable | Default | Purpose |
|---|---|---|
| `GVHMR_ROOT` | none (required) | GVHMR directory (pinned commit + patch) |
| `GMR_ROOT` | none (required) | GMR directory; SMPL-X defaults to `$GMR_ROOT/assets/body_models/smplx/` |
| `V2M_WORK` | `<this repo>/work/video2motion` | Work and output directory |
| `GVHMR_PY` / `GVHMR_ENV` | empty / `gvhmr` | Python of the GVHMR environment; when `GVHMR_PY` is empty, `conda run -n $GVHMR_ENV python` is used |
| `GMR_PY` / `GMR_ENV` | empty / `gmr` | Python of the GMR environment; when `GMR_PY` is empty, `conda run -n $GMR_ENV python` is used |
| `G1_XML` | mjlab `g1.xml` resolved via `uv run` | G1 MJCF |
| `SMPLX_BODY_MODELS` | `$GMR_ROOT/assets/body_models` | Directory containing `smplx/SMPLX_NEUTRAL.npz` |
| `GPU` | `0` | GPU used by GVHMR / YOLO |
| `SHOTS` | `0` | `1` = run `shot_bbx.py` first (when the video has hard shot cuts) |
| `RETARGET_ARGS` | `--init_from_human --restart_tol 0.1` | Extra arguments passed to `gvhmr_to_supermimic.py`; set to an empty string to use GMR's original initialization |
| `PREVIEW` | `1` | `0` = do not render previews |
| `MUJOCO_GL` | `egl` | Headless rendering backend, `egl` or `osmesa` |

## One-command run

First cut the video into clips with **a single person, whole body visible**; 5–20 s per clip is recommended (see [Tips and lessons](#tips-and-lessons) below). Then run:

```bash
export GVHMR_ROOT=$V2M_HOME/GVHMR GMR_ROOT=$V2M_HOME/GMR
# GMR installed with uv: export GMR_PY=$V2M_HOME/GMR/.venv/bin/python
tools/video2motion/run_pipeline.sh clips/my_dance.mp4 [clips/another.mp4 ...]
```

Outputs for each clip `<c>` (the input file name without its extension):

| Path | Contents |
|---|---|
| `$V2M_WORK/video30/<c>.mp4` | 30 fps copy, i.e. the video GVHMR actually processes; CSV row t = frame t of this video |
| `$V2M_WORK/gvhmr/<c>/hmr4d_results.pt`, `gvhmr/<c>.log` | GVHMR result and log |
| `$V2M_WORK/csv_drift/<c>.csv` (+ `<c>_gmr.pkl`) | **Recommended**: version with GVHMR's height drift removed |
| `$V2M_WORK/csv_constant/<c>.csv` (+ `<c>_gmr.pkl`) | Version using a single floor offset for the whole clip |
| `$V2M_WORK/preview/<c>.mp4` | Source video \| G1 side-by-side preview (drift version) |

Convert to the 50 Hz npz used for training:

```bash
uv run python scripts/data/csv_to_npz_batch.py --input-dir work/video2motion/csv_drift \
    --output-dir data/npz/my_dance --input-fps 30 --pattern '*.csv'
```

For joint velocity limiting and smoothing, multi-clip concatenation, training / fine-tuning and deployment, see [CSV → training and deployment](CUSTOM_MOTIONS.md#csv--training-and-deployment).

## Step-by-step commands

When you need to tune a single step, you can run the steps by hand. Below, `$GVHMR_PY` / `$GMR_PY` are the pythons of the two environments, and `W=$V2M_WORK`.

```bash
# 0) Convert to 30 fps (GVHMR always tags the video it writes as 30 fps, and GMR also reads it as 30 fps without resampling;
#    a 29.97 fps source is only 0.1% off, but 25/60 fps sources become 0.83x / 2x speed if not converted)
ffmpeg -i in.mp4 -vf fps=30 -an -c:v libx264 -crf 17 -pix_fmt yuv420p $W/video30/clip.mp4

# 1) (optional) With hard shot cuts: detect person boxes per shot and write them where GVHMR will read them (must run before demo.py)
$GVHMR_PY tools/video2motion/shot_bbx.py $W/video30/clip.mp4 $W/gvhmr/clip/preprocess --gvhmr_root $GVHMR_ROOT

# 2) GVHMR, static camera (must run inside the GVHMR directory; it finds weights by relative path)
(cd $GVHMR_ROOT && $GVHMR_PY tools/demo/demo.py --video $W/video30/clip.mp4 --output_root $W/gvhmr -s)

# 3) GMR retargeting + limits + grounding → CSV
$GMR_PY tools/video2motion/gvhmr_to_supermimic.py --gvhmr_pred $W/gvhmr/clip/hmr4d_results.pt \
    --name clip --floor_mode drift --out_dir $W/csv_drift --init_from_human --restart_tol 0.1

# 4) Preview
MUJOCO_GL=egl $GMR_PY tools/video2motion/render_preview.py $W/csv_drift/clip.csv $W/video30/clip.mp4 $W/preview/clip.mp4
```

Main arguments of `gvhmr_to_supermimic.py`:

| Argument | Default | Description |
|---|---|---|
| `--floor_mode drift` | `drift` | Per frame, takes the lowest point of the two feet's collision geoms, applies a 1 s sliding minimum (lower envelope) and then 1 s smoothing, and subtracts the result as the floor height. This removes GVHMR's slow height drift (measured at up to 3–20 cm) while keeping jumps shorter than about 1 s. **Recommended for training** |
| `--floor_mode constant` | | Subtracts a single constant for the whole clip, namely the `--contact_pct` percentile (default 10%) of the lowest point. Keeps jump height, but also keeps the drift, so the feet may penetrate the floor or float |
| `--init_from_human` | off | Initializes the IK root from the human pelvis pose, with 20 extra iterations on frame 0. If the dancer starts with their back or side to the camera, or bent at the waist, without this flag the whole clip may get locked into a "twisted waist + circling arms" local minimum |
| `--restart_tol 0.1` | 0 (off) | Each frame additionally re-solves for 30 iterations with the waist and arms reset to zero, and adopts the new solution if the GMR task error drops by more than this value; at switch points, joints that jump get a ±5-frame cosine blend. Handles IK branch flips when the hands cross in front of the body |
| `--smooth_sigma` | 0 | Gaussian smoothing of joint angles (unit: frames); usually not needed |
| `--robot_xml` / `--body_models` | `$G1_XML` / `$SMPLX_BODY_MODELS`, otherwise `$GMR_ROOT/assets/body_models` | See [Environment setup](#environment-setup) above |

The script prints the following statistics: frame count, estimated human height, floor offset range, lowest foot point, root height range, maximum joint velocity, and number of clipped joint values.

## Output format and checks

Both `motions/custom_dance/*.csv` and the LAFAN1 G1 retargeted CSVs use this format:
- Plain-text CSV, **no header**, comma-separated, `%.6f`, one frame per row.
- **30 fps** (specify the frame rate with `--input-fps` when converting), z up, units m / rad.
- 36 columns in total:

| Columns | Contents |
|---|---|
| 0–2 | Pelvis (root) position x, y, z in the world frame (m); the floor is at z = 0, and the xy of frame 0 is (0, 0) |
| 3–6 | Pelvis orientation quaternion, **x, y, z, w** (scalar last, adjacent rows in the same hemisphere; `supermimic.data.motion_convert` converts it to MuJoCo's wxyz when reading) |
| 7–35 | 29 joint angles (rad), in the same order as the G1 MJCF / Unitree SDK motor indices 0–28 / `supermimic.data.motion_convert.G1_JOINT_NAMES`: left leg 6, right leg 6, waist yaw/roll/pitch, left arm 7, right arm 7; already clipped to the G1 MJCF joint limits |

- `scripts/data/csv_to_npz_batch.py --input-fps 30` converts this CSV directly. It checks the joint order, uses FK to compute joint/root velocities and the poses and velocities of 30 bodies, and outputs a 50 Hz npz.
- The pipeline additionally writes `<c>_gmr.pkl` next to the CSV. This is a GMR-style pickle of the same motion (`fps`, `root_pos`, `root_rot` (xyzw), `dof_pos`); this repository does not use it.

Before training or running on the robot, we recommend checking the following (the table lists the thresholds we use):

| Check | Threshold / notes |
|---|---|
| Joint limits | No limit violations. Watch out when a joint sits against its limit (within 0.02 rad) on >3% of frames; this is common for ankle roll and waist pitch during deep squats |
| Floor penetration | Lowest point of both feet ≥ −1 cm |
| Joint velocity | p99 around 8–16 rad/s. Spikes > 15–20 rad/s are usually estimation errors caused by fast arm swings or motion blur |
| Root velocity | < 3 m/s |
| Pelvis tilt (on the robot) | Motions with large sideways or backward lean are high risk on the real robot; before deployment use the checks in [Export and sim2sim gates](SIM2SIM.md) |

## Preview rendering

```bash
# Source video | G1, camera follows the root xy, viewing from directly in front of the initial heading
MUJOCO_GL=egl $GMR_PY tools/video2motion/render_preview.py clip.csv video30/clip.mp4 preview.mp4 [--cam_yaw DEG] [--height 480]

# extend_full.py results: fixed camera, with shot labels (A wide / B close-up / C wide) and the added rows marked
MUJOCO_GL=egl $GMR_PY tools/video2motion/render_full_preview.py clip_full.csv clip_full.mp4 preview_full.mp4 \
    --cuts 271 334 --kept_rows 269 --from_row 240 [--crop_top 0] [--front_yaw DEG]
```

- When the dancer starts with their back or side to the camera, the default camera direction is wrong. Use `--cam_yaw` (or the environment variable `CAM_YAW`) to give an angle: when the robot's heading equals that angle, it faces the camera head-on. You can estimate this angle from GVHMR's in-camera heading, or just try a few values.
- Without a display, use `MUJOCO_GL=egl` (NVIDIA) or `MUJOCO_GL=osmesa`.
- Preview videos contain source video frames. Unless you own the rights to the source video, **do not distribute them publicly**.
- Simulation videos of trained policies are rendered with `scripts/render_video.py`; see [Evaluation and visualization](../README_EN.md#evaluation-and-visualization) in README_EN.md.

## Tips and lessons

| Topic | Advice |
|---|---|
| Choosing clips | Best results come from clips with a single person whose whole body (including both feet) stays in frame throughout, with a camera as fixed as possible. For handheld phone follow shots (horizontal panning), `-s` still gives usable body poses; only the root translation is affected. GVHMR's built-in SimpleVO (without `-s`) gives a different root trajectory, but not necessarily a better one |
| Hard shot cuts | When a clip has hard cuts between wide and close-up shots, GVHMR's built-in tracking keeps only the longest track and smooths the person box across shots, so it easily loses the person. In that case use `SHOTS=1` (`shot_bbx.py`) to detect and smooth per shot |
| Legs not visible in close-ups | For these parts the lower body GVHMR gives is a "guess" and cannot be used directly; use `extend_full.py` below instead |
| IK local minima | Symptoms: chest heading clearly disagrees with the video, one arm stuck or circling, one shoulder hunched up. The default `--init_from_human --restart_tol 0.1` (the `RETARGET_ARGS` default) fixes most cases. To judge whether a large body turn is real, check whether GVHMR's in-camera pelvis heading is continuous: a single-frame heading flip is usually an estimation error |
| Grounding | Use `csv_drift` for training. Airborne phases lasting longer than about 1 s get "flattened" by drift mode. For such motions, compare with `csv_constant`, or process the clip in segments |

**`extend_full.py`: completing motion cut off by close-ups (experimental)**

Use case: a fixed-camera dance video whose shot order is A wide → B close-up (legs not visible) → C wide. You already have a CSV for segment A and want the complete motion all the way until the dancer settles into a standing pose in segment C. Steps:

1. Cut the whole A+B+C span into one video. **Its start frame must be the same as the start frame of the A-segment CSV**, i.e. CSV row i = frame i of both videos.
2. Run the pipeline on the whole span with per-shot person boxes: `SHOTS=1 tools/video2motion/run_pipeline.sh clip_full.mp4`. This gives `csv_drift/clip_full.csv` (called N below) and `gvhmr/clip_full/hmr4d_results.pt`.
3. Determine the CSV row numbers of the shot cuts: `C0` is the first row of B, `C1` is the first row of C. `shot_bbx.py` prints the cuts it detects.
4. Run:

```bash
$GMR_PY tools/video2motion/extend_full.py --old_csv csv_drift/clip.csv --whole_csv csv_drift/clip_full.csv \
    --gvhmr_pred gvhmr/clip_full/hmr4d_results.pt --cuts C0 C1 --feet steps --place cam --human_height 1.70 \
    --out clip_extended.csv
```

How each part is handled:

| Part | Handling |
|---|---|
| Segment A | Existing rows are kept value for value. The rest of segment A uses N rigidly aligned to the existing rows, with the offset faded out within 12 frames |
| Segment C | Uses N's whole-body motion, placed under the static-camera assumption. Heading = in-camera body heading + the camera→world yaw offset fitted on segment A. Position can be placed in two ways: `--place cam` (default `feet`) uses the camera-frame displacement × GMR's root scale 0.9·height/1.8 and needs `--human_height` (take the value printed by `gvhmr_to_supermimic.py`); `--place feet` aligns the grounded foot. Segment C gets its own drift grounding, and support feet floating 2–12 cm above the floor are put back on the floor with leg IK |
| Segment B | Waist, arms, head and pelvis tilt come from N. Root xy, height and heading transition smoothly between the end of A and the start of C, keeping part of N's sway. Legs use IK: `--feet fixed` keeps both feet in place (suits horse-stance-type moves); `--feet steps` steps in time with N's footwork, locks feet flat when grounded, keeps the swing foot's average speed at most 2.5 m/s, and lifts the foot at most 20 cm |
| Finish | Joint spikes > 12 rad/s in the added rows are re-interpolated; after the closing stand is detected, it is held for 15 frames before ending; finally everything is clipped to the joint limits |

The log `<out>_log.json` (path changeable with `--log`) records joint velocity and root position jumps at the splice points, foot IK error and slip in segment B, the settle frame, etc.; use it together with `render_full_preview.py` to cross-check.
