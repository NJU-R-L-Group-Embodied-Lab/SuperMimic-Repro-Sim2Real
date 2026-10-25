# Third-party notices / 第三方声明

This repository's own code and the retargeted motions in `motions/custom_dance/` are released under the MIT License (see [`LICENSE`](../LICENSE); Copyright (c) 2026 Yuxi Liu — upstream [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) — and Copyright (c) 2026 Hongyu Ding). It depends on, or interoperates with, the third-party software and data listed below. **None of the third-party code, robot model assets or datasets below is vendored in this repository** unless stated otherwise; they are installed by `uv sync` (Python packages), cloned/installed separately by the user (Unitree tools), or downloaded by the user under their own licenses (datasets). License names below are given for convenience; the upstream license files are authoritative.

本仓库自身的代码和 `motions/custom_dance/` 中的重定向动作按 MIT 许可证发布（见 [`LICENSE`](../LICENSE)；Copyright (c) 2026 Yuxi Liu，即上游 CuboiLeo/SuperMimic-Reproduced，以及 Copyright (c) 2026 Hongyu Ding）。下列第三方软件和数据**没有被复制进本仓库**（除非特别说明），由 `uv sync` 安装、由用户另行获取，或由用户按其各自许可证下载。许可证名称仅供参考，以上游的许可证文件为准。

## Software / 软件

| Project | Used for | How it is used | License (upstream) |
|---|---|---|---|
| [SuperMimic](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) | the method reproduced here | re-implementation from the paper (via the upstream reproduction below); **no code from the paper authors** | paper © its authors |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (Yuxi Liu) | **upstream reproduction this repository builds on**: world model, BPTT actor-critic, reset sampler, training loop, benchmark | code base of this repository (forked and extended, not written from scratch); copyright notice kept in [`LICENSE`](../LICENSE) | MIT, Copyright (c) 2026 Yuxi Liu |
| [mjlab](https://github.com/mujocolab/mjlab) (commit `8bb2bc9`) | simulation framework, G1 robot configuration and MJCF, motion conventions | Python dependency (git source in `pyproject.toml`); `motion_convert.py` re-implements the conventions of mjlab's `csv_to_npz.py` | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | CPU physics (data conversion, FK, sim2sim) | Python dependency | Apache-2.0 |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | GPU physics for training | Python dependency (via mjlab) | Apache-2.0 |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) (`warp-lang`) | GPU kernels used by MuJoCo Warp | Python dependency | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | networks, BPTT, `torch.compile`, ONNX export | Python dependency | BSD-3-Clause |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | policy inference on the deploy side | installed by the user in the deploy environment | MIT |
| NumPy, SciPy, pandas, pyarrow, tyro, tqdm, TensorBoard, huggingface-hub | utilities | Python dependencies | BSD-3-Clause / MIT / Apache-2.0 (see each package) |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (C++) | DDS I/O of the C++ bridge `scripts/deploy/bridge/g1_bridge.cpp` | headers/libraries of the user's checkout, linked by `bridge/build.sh`; the CRC32 routine `Crc32Core` in `g1_bridge.cpp` is copied from the SDK's `example/g1/low_level/g1_ankle_swing_example.cpp` (see *Included third-party code* below) | BSD-3-Clause |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) (`unitree_sdk2py`) | DDS I/O of the Python runner, read-only checks, MotionSwitcher `CheckMode` | installed by the user in the deploy environment | BSD-3-Clause |
| [Eclipse Cyclone DDS](https://github.com/eclipse-cyclonedds/cyclonedds) (+ `cyclonedds-python`) | DDS transport used by both Unitree SDKs | installed with the SDKs | EPL-2.0 OR EDL-1.0 |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | official G1 29-DoF MJCF for sim2sim (`scene_29dof.xml`) and its `simulate_python/unitree_sdk2py_bridge.py`, imported unchanged by `scripts/deploy/sim_g1_mujoco.py` | cloned by the user under `$G1_DEPLOY_HOME/unitree_mujoco` | BSD-3-Clause |
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | reference only: `mode_machine` table and the waist-locked G1 model description (see [docs/DEPLOY_G1.md](../docs/DEPLOY_G1.md#g1-model-ai-mode-and-debug-mode)) | not used by the code | BSD-3-Clause |
| SONIC data-processing keyword list ([NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), `gear_sonic/data_process/filter_and_copy_bones_data.py`) | BONES-SEED name filter (`scripts/data/filter_bones_seed.py`) | a short factual list of dataset clip-name keywords taken from that file (`SONIC_KEYWORDS`), plus our own additions | Apache-2.0 (source code of that repository; see its `LICENSE`) |

## Included third-party code / 包含的第三方代码

| File | Origin | License |
|---|---|---|
| `scripts/deploy/bridge/g1_bridge.cpp`, function `Crc32Core` | copied from [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) `example/g1/low_level/g1_ankle_swing_example.cpp`, Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"). All rights reserved. | BSD-3-Clause; full text in [`LICENSES/unitree_sdk2-BSD-3-Clause.txt`](unitree_sdk2-BSD-3-Clause.txt) |
| `scripts/data/filter_bones_seed.py`, list `SONIC_KEYWORDS` | keyword list from [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) `gear_sonic/data_process/filter_and_copy_bones_data.py` | Apache-2.0 |

## Video-to-motion pipeline / 视频到动作流程（`tools/video2motion/`）

The scripts in `tools/video2motion/` are our own code (MIT). They drive or import the external projects below, which are **not included** in this repository and must be installed / downloaded by the user under their own licenses. See [docs/VIDEO_TO_MOTION.md](../docs/VIDEO_TO_MOTION.md) (installation: [Environment setup](../docs/VIDEO_TO_MOTION.md#environment-setup); licenses: [License notes](../docs/VIDEO_TO_MOTION.md#license-notes-read-first)).

`tools/video2motion/` 中的脚本是我们自己的代码（MIT），它们调用或导入下列外部项目；这些项目**不包含**在本仓库中，需要用户按各自的许可证自行安装或下载。

| Project | Used for | How it is used | License (upstream) |
|---|---|---|---|
| [GVHMR](https://github.com/zju3dv/GVHMR) (commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`, plus a one-line local patch removing `from turtle import forward`) | monocular video → world-frame SMPL-X motion | **external, not included**; `run_pipeline.sh` runs its `tools/demo/demo.py` as a subprocess, `shot_bbx.py` imports two helpers from the user's checkout at run time | **non-commercial**: educational, research and non-profit use only (ZJU 3DV group license, see its `LICENSE`); commercial use requires permission from the authors |
| GVHMR's pretrained weights (GVHMR, HMR2, ViTPose, YOLOv8) | inside GVHMR | downloaded by the user following GVHMR's instructions | their own licenses (e.g. ultralytics / YOLOv8: AGPL-3.0) |
| [GMR](https://github.com/YanjieZe/GMR) (commit `25f56db79ac411258b377b6d4ccba0c8ff3372ac`, unmodified) | SMPL-X → G1 IK retargeting | **external, not included**; imported as a Python package by `gvhmr_to_supermimic.py` | MIT (© 2025 Yanjie Ze) |
| [smplx](https://github.com/vchoutas/smplx) (commit `1265df7ba545e8b00f72e7c557c766e15c71632f`, installed as a GMR dependency) | SMPL-X forward kinematics | external Python dependency of GMR | see its `LICENSE` (tied to the SMPL-X model license) |
| [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models (MPI-IS) | human body model for GVHMR and GMR | **not included**; the user registers and downloads them | MPI model license, **non-commercial research only** |
| Unitree G1 description (URDF / MJCF, Unitree Robotics) | G1 model used for joint limits, foot contact and preview rendering (read from the user's mjlab install, or another G1 MJCF given by `G1_XML`) | not included | BSD-3-Clause (as shipped in GMR's `assets/unitree_g1/LICENSE`; mjlab's copy: see mjlab) |

**Because GVHMR and SMPL-X are non-commercial, running this pipeline, and its intermediate results (SMPL-X parameters), are subject to those terms even though our glue code is MIT.** Source videos remain the property of their owners.

**由于 GVHMR 与 SMPL-X 仅限非商业用途，即使本仓库的胶水代码是 MIT，运行这条流程及其中间结果（SMPL-X 参数）仍受这些条款约束。**源视频的版权属于其作者。

## Robot model assets / 机器人模型

- **Unitree G1 29-DoF MJCF and meshes.** This repository contains **no** robot model files. Training uses the G1 model shipped with mjlab (`mjlab/asset_zoo/robots/unitree_g1/`, originally from Unitree Robotics; see mjlab's license and notices for its provenance). Sim2sim and the end-to-end tests use Unitree's official model from `unitree_mujoco` (BSD-3-Clause).
- `python -m supermimic.deploy.export --cmd mjcf` writes a copy of the training scene (MJCF + meshes) to `deploy_artifacts/models/`. That output is derived from the mjlab / Unitree assets and remains under their licenses; do not redistribute it as MIT-licensed material.
- "Unitree" and "G1" are trademarks of Unitree Robotics. This project is not affiliated with or endorsed by Unitree Robotics.

## Datasets / 数据集（均不随仓库分发）

| Dataset | Used for | Notes |
|---|---|---|
| **LAFAN1** (Ubisoft La Forge, Harvey et al., SIGGRAPH 2020) via the G1 retargeting on Hugging Face (`lvhaidong/LAFAN1_Retargeting_Dataset`) | training / evaluation (40 clips) | LAFAN1 is licensed **CC BY-NC-ND 4.0** (non-commercial, no derivatives). **Not redistributed here.** Download it yourself and comply with the licenses of LAFAN1 and of the retargeted release. |
| **BONES-SEED** (Hugging Face `bones-studio/seed`, gated) | large-scale training / evaluation (~130k clips) | Access requires accepting the dataset's terms on Hugging Face. **Not redistributed here**; no BONES-SEED-derived clip files are included. Checkpoints trained on it are released for research use; check the dataset terms before any other use. |
| Dance videos (source of `motions/custom_dance/*.csv`) | custom motions | Only retargeted robot joint trajectories are distributed. The rights to the source videos belong to their owners. See [docs/CUSTOM_MOTIONS.md](../docs/CUSTOM_MOTIONS.md#custom-motion-data-motionscustom_dance). |

## Released checkpoints / 发布的模型

The checkpoints and ONNX deploy bundles published at <https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases> were trained on BONES-SEED (and, for fine-tuned models, the custom dance motions above). Using them may be subject to the BONES-SEED terms in addition to this repository's MIT License.
