# SuperMimic-Repro-Sim2Real

中文 | [English](README_EN.md)

这个仓库教 Unitree G1 人形机器人模仿人的动作。先在仿真里训练，让机器人学会跟着一段参考动作做；训练好的策略再导出，放到真机上跑。我们用它让一台 G1 完整跳完了一段约 46 秒的舞蹈。

方法来自论文 [SuperMimic](https://arxiv.org/abs/2610.28480)（Liao 等，CoRL 2026）：它先训练一个神经网络代替物理仿真器，预测机器人下一步怎么动（论文叫"世界模型"）。这个网络能求导，所以可以像训练普通网络一样，直接用反向传播优化控制策略，不用靠强化学习反复试错。

实机演示视频和预训练模型在 [GitHub Release](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases)。

> **安全警告**：真机代码会直接控制 G1 的电机。模型、参数或操作出错，机器人可能摔倒、损坏或伤人。请只在吊架保护下运行，并且始终有人拿着遥控器，随时准备按 L2+B 让机器人进入阻尼（软急停）。上机前完整读一遍 [docs/DEPLOY_G1.md](docs/DEPLOY_G1.md)。本项目按原样提供，使用风险自负。

## 基于谁的工作

- 代码不是从零写的。训练算法部分基于 Yuxi Liu 的开源复现 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（MIT 许可证）。
- 我们在它上面加了：大数据集训练、自定义舞蹈动作、从视频提取动作的工具，以及整套上真机的部署代码。
- 这是非官方复现，与论文作者和 Unitree 都没有关系。论文没写清楚的细节是我们自己定的，所以数字不一定和论文一致。

## 效果

得分 1.0 表示完全跟上参考动作，越高越好。

| 训练数据 | 我们的得分 | 论文的得分 |
|---|---|---|
| LAFAN1（40 段动作），严格按论文设置 | 0.56 | 0.82 |
| LAFAN1，额外输入机器人相对参考动作的位置（`--env.anchor-obs`） | 0.90 | – |
| BONES-SEED（约 13 万段动作），严格按论文设置 | 0.93，约 13 万段里 46 段失败 | 0.90 以上，25 段失败 |

- 严格按论文设置，LAFAN1 得分明显低于论文。加上这个位置输入就能追上，但真机测不到自己在房间里的位置，所以这种模型上不了真机。
- 真机：G1 完整跳完了 4 段舞蹈连起来的约 46 秒动作，身体倾斜和参考动作最多差 16.3°（超过 20° 程序会自动停止）。
- 详细数字和原因见 [docs/METHOD.md](docs/METHOD.md#results)。

## 快速开始

### 安装

训练需要 Linux 和 NVIDIA 显卡（驱动 ≥ 580）；真机那一侧只要 numpy、onnxruntime 和 Unitree SDK。依赖用 [uv](https://docs.astral.sh/uv/) 管理。

```bash
git clone https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real.git && cd SuperMimic-Repro-Sim2Real
uv sync --group dev
uv run pytest tests -q          # 部分测试需要 GPU 或已转换的数据
```

### 准备数据

数据集不随仓库分发，请自己下载并遵守它们的许可证。最小的例子是 LAFAN1，下载后转成训练用的格式：

```bash
uv run hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --local-dir data/raw/lafan1
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 16
```

大数据集 BONES-SEED 的下载、筛选和打包见 [docs/METHOD.md](docs/METHOD.md#data-and-filtering)。

### 训练

```bash
uv run sm-train --run-name lafan1_paper --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 3
```

- 默认参数按论文设置。所有选项用 `uv run sm-train --help` 查看。
- 模型每隔一段时间存到 `logs/<run>/ckpt_XXXXXXX.pt`，训练曲线用 TensorBoard 看。
- 用 BONES-SEED 或多卡训练，见 [docs/METHOD.md](docs/METHOD.md#data-and-filtering)。

### 评测和看效果

```bash
# 对每段动作从头到尾跑一遍，算得分和失败数
uv run python scripts/eval_benchmark.py --ckpt logs/lafan1_paper/ckpt_XXXXXXX.pt --npz-dir data/npz/lafan1
# 打开窗口回放（没有显示器时会改用浏览器 http://localhost:8080）
uv run python scripts/play.py --ckpt logs/lafan1_paper/ckpt_XXXXXXX.pt --clips dance1_subject1
# 不开窗口直接渲染 mp4（实体 = 策略，半透明蓝 = 参考动作）
MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg python scripts/render_video.py \
  --ckpt <ckpt> --clips dance1_subject1 --seconds 30 --tag demo
```

## 真机部署

流程分三步：

1. 把训练好的模型导出成 ONNX 文件。
2. 在仿真里用 Unitree 官方的机器人模型反复测试，加上延迟、负重、摩擦变化等干扰，全部通过才算合格（见 [docs/SIM2SIM.md](docs/SIM2SIM.md)）。
3. 在机器人自带的电脑上运行控制程序。先让机器人吊在空中，再逐步放到地面上。

```bash
# 仿真测试（Release 模型）
python scripts/deploy/sim2sim.py --cmd g2,g3,report --clips combo_c4_full_vc_b15f20 --tag combo --ckpt e13i_0291000 --workers 16
# 真机第一步：吊在空中
python scripts/deploy/bridge/g1_runner_bridge.py --iface <nic> --domain 0 --stage air \
  --artifacts deploy_artifacts/<ckpt> \
  --clip deploy_artifacts/clips/<clip>.npz \
  --log-dir deploy_logs/real
```

这些命令在部署环境里跑，环境配置见 [docs/SIM2SIM.md](docs/SIM2SIM.md) 和 [docs/DEPLOY_G1.md](docs/DEPLOY_G1.md)。程序在倾斜过大、关节太快或电机过热时会自己停下，但吊架和遥控器仍然必须有（见上面的安全警告）。

## 文档

| 文件 | 讲什么 |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | 算法怎么实现的，哪些地方和论文不同，为什么 |
| [docs/CUSTOM_MOTIONS.md](docs/CUSTOM_MOTIONS.md) | 怎么用自己的舞蹈动作训练和上机 |
| [docs/VIDEO_TO_MOTION.md](docs/VIDEO_TO_MOTION.md) | 怎么从一段跳舞视频里提取出机器人能做的动作 |
| [docs/SIM2SIM.md](docs/SIM2SIM.md) | 怎么导出模型，以及上真机前在仿真里要过哪些测试 |
| [docs/DEPLOY_G1.md](docs/DEPLOY_G1.md) | 真机操作手册：安全措施、遥控器按键、上机步骤、出问题怎么办 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 所有引用的 BibTeX |

## 预训练模型

在 [GitHub Release](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases) 下载。每个模型都带训练存档和可直接部署的 ONNX 文件。

- `e11c_0220000`：在 BONES-SEED 上训练的基础模型。
- `e13i_0291000`：在基础模型上继续训练，训练数据是 BONES-SEED 加 4 段自定义舞蹈（不是只有这 4 段），Release 里有它在真机上跳舞的视频。

用法：解压到 `deploy_artifacts/`，然后按 [docs/SIM2SIM.md](docs/SIM2SIM.md) 重新跑一遍仿真测试，再按 [docs/DEPLOY_G1.md](docs/DEPLOY_G1.md) 上机。脚本默认用的是没发布的旧模型，所以要手动加 `--ckpt e13i_0291000`（仿真测试）或 `--artifacts deploy_artifacts/e13i_0291000`（真机程序）。两个模型都用 BONES-SEED 训练，使用前请确认它的数据条款。

Release 里没有动作目标表 `deploy_artifacts/clips/<clip>.npz`，要按 [docs/SIM2SIM.md](docs/SIM2SIM.md#exporting-the-deploy-bundle) 用 `--cmd clips` 自己生成。机器人自带电脑（PC2）上的 onnxruntime 较旧，要把包里的 `policy_ir9.onnx` 复制成 `policy.onnx`。

```bash
mkdir -p deploy_artifacts
tar -xzf /path/to/e13i_0291000.tar.gz -C deploy_artifacts/   # -> deploy_artifacts/e13i_0291000/
```

## 许可证

本项目采用 MIT 许可证，见 [LICENSE](LICENSE)，其中同时保留了上游作者 Yuxi Liu 和本仓库作者 Hongyu Ding 的版权声明。第三方软件和数据集的声明见 [LICENSES/](LICENSES/)。

## 引用

如果这个仓库对你有帮助，请先引用原论文，也请引用上游复现代码：

```bibtex
@inproceedings{liao2026supermimic,
  title         = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author        = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  booktitle     = {Conference on Robot Learning (CoRL)},
  year          = {2026},
  eprint        = {2610.28480},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.28480}
}

@misc{liu2026supermimicreproduced,
  title        = {SuperMimic-Reproduced: Reproduction of SuperMimic (arXiv:2610.28480) on mjlab/MuJoCo Warp},
  author       = {Liu, Yuxi},
  year         = {2026},
  howpublished = {\url{https://github.com/CuboiLeo/SuperMimic-Reproduced}},
  note         = {GitHub repository, MIT License}
}
```

## 致谢

感谢以下项目，所有 BibTeX 见 [docs/REFERENCES.md](docs/REFERENCES.md)：

- [mjlab](https://github.com/mujocolab/mjlab)：仿真训练框架
- [BeyondMimic](https://arxiv.org/abs/2508.08241)：动作跟踪任务的设计
- [MuJoCo](https://github.com/google-deepmind/mujoco) / [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)：物理引擎
- [Unitree SDK](https://github.com/unitreerobotics/unitree_sdk2) / [Unitree MuJoCo 模型](https://github.com/unitreerobotics/unitree_mujoco)：真机通信和官方机器人模型
- [GVHMR](https://github.com/zju3dv/GVHMR) / [GMR](https://github.com/YanjieZe/GMR) / [SMPL-X](https://smpl-x.is.tue.mpg.de/)：从视频提取动作（GVHMR 和 SMPL-X 仅限非商业用途）
- [LAFAN1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) / [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)：动作数据集

"Unitree" 和 "G1" 是 Unitree Robotics 的商标。
