# Expressive Motion

![Platform](https://img.shields.io/badge/Platform-Ubuntu%20x86__64-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![License](https://img.shields.io/badge/Code-MIT-lightgrey)

Expressive Motion converts a monocular human-motion video into a Booster K1 reference motion. It connects pinned GVHMR, GMR and Booster checkouts into one pipeline; training and deployment remain manual upstream operations.

## Overview

Give the pipeline a video of a person moving and it returns a grounded motion clip a Booster K1 can be trained to imitate.

```
video → 30 fps → GVHMR → ground alignment → GMR retarget → CSV → Booster NPZ → task
        ffmpeg   (GPU)    (this repo)        (CPU)                 (Isaac Lab)
```

- **GVHMR** lifts the video into SMPL-X body parameters in world space.
- **This repository** detects foot contacts, levels the floor and removes root drift.
- **GMR** solves an optimisation mapping the human skeleton onto the robot's joints.
- **Booster Train** receives a generated BeyondMimic task.

GVHMR runs in static-camera mode, so DPVO is skipped. The complete path supports `booster_k1`.

Two conda environments are created because GVHMR and GMR have incompatible dependency pins. The split is load-bearing, not stylistic.

## Prerequisites

- **OS**: x86_64 Linux. The pinned `pytorch3d` wheel is `linux_x86_64` only.
- **Conda**: 4.9 or newer.
- **GPU**: NVIDIA GPU with a driver exposing CUDA 12.1+, for GVHMR only.
- **Disk**: about 25 GB for the environments, plus 5.2 GB of weights.

```bash
sudo apt update && sudo apt install -y ffmpeg git
```

Isaac Lab, the SMPL/SMPL-X body models, the GVHMR checkpoints and the Booster SDK are licensed by third parties and are not installed automatically.

## Quick Start

```bash
git clone <this-repo> expressive-motion
cd expressive-motion
git submodule update --init --recursive

# Asks which components to install, then puts `em` on your PATH
bash install.sh

# Add the licensed body models and checkpoints — see below
em doctor

export EM_ISAAC_ENV=your_isaac_conda_env
em process inputs/videos/ai1.mp4
```

## Installation

The installer builds three independent components. Run it with no arguments for an interactive prompt, or choose explicitly:

```bash
bash install.sh --list                    # what is available
bash install.sh --components gmr          # retargeting only, no GPU needed
bash install.sh --components gmr,gvhmr    # both processing environments
bash install.sh --yes --isaac-env env_isaaclab
bash install.sh --skip gvhmr
```

| Component | Creates | Purpose |
| --- | --- | --- |
| `gmr` | env `expressive-motion-gmr` | Retargeting, CSV conversion, pipeline driver |
| `gvhmr` | env `expressive-motion-gvhmr` | GVHMR inference. Needs a CUDA GPU |
| `isaac` | nothing new | Adds Booster packages to **your existing** Isaac Lab env |

The `isaac` component never installs Isaac Lab itself — name your existing environment with `--isaac-env NAME` or `--isaac-python /path/to/python`.

```bash
bash install.sh --check          # validate, install nothing (same as `em doctor`)
bash install.sh --clean          # remove the envs, after confirming
bash install.sh --no-submodules  # use checkouts you populated yourself
bash install.sh --no-link        # do not put `em` on PATH
bash install.sh --help
```

Re-running skips components whose inputs have not changed and repairs any whose previous run died partway. Each run logs to `.install/`.

### The `em` command

The installer symlinks `em` into `~/.local/bin` (override with `EM_BIN_DIR`), so it works from any directory. If that directory is not on your PATH the installer tells you how to add it. Without the symlink, `./em` from the repository root behaves identically.

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `EM_ISAAC_ENV` / `EM_ISAAC_PYTHON` | Where Isaac Lab lives. Required by the last stage of `em process` |
| `EM_GMR_ENV` / `EM_GVHMR_ENV` | Override the conda environment names |
| `EM_BOOSTER_ROOT` | Override the Booster checkout root |
| `EM_BIN_DIR` | Where to symlink `em` |

## Body Models and Weights

This is the step that usually blocks a fresh install. About **5.2 GB** total. The installer reports what is missing but cannot download any of it.

### 1. Register for the body models

Separate registrations, each requiring a signed licence:

- **SMPL-X**: https://smpl-x.is.tue.mpg.de/ → `SMPLX_NEUTRAL.npz` (104 MB)
- **SMPL**: https://smpl.is.tue.mpg.de/ → `SMPL_NEUTRAL.pkl` (236 MB). Some archives name it `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`; rename it.

### 2. Download the GVHMR checkpoints

From the upstream Google Drive folder — downloading means accepting the upstream licences:

https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD

| File | Size | Needed for |
| --- | --- | --- |
| `gvhmr/gvhmr_siga24_release.ckpt` | 156 MB | GVHMR |
| `hmr2/epoch=10-step=25000.ckpt` | 2.6 GB | HMR2.0a features |
| `vitpose/vitpose-h-multi-coco.pth` | 2.4 GB | 2D pose |
| `yolo/yolov8x.pt` | 131 MB | Person detection |
| `dpvo/dpvo.pth` | 14 MB | Optional, moving-camera only |

`dpvo.pth` is not needed: this repository passes `-s` to GVHMR, which skips DPVO.

### 3. Place the files

```bash
mkdir -p external/GVHMR/inputs/checkpoints/{body_models/smpl,body_models/smplx,gvhmr,hmr2,vitpose,yolo}
```

The result must match:

```text
external/GVHMR/inputs/checkpoints/
├── body_models/smpl/SMPL_NEUTRAL.pkl
├── body_models/smplx/SMPLX_NEUTRAL.npz
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

GMR needs the same SMPL-X model at a second path. Symlink rather than copy:

```bash
mkdir -p external/GMR/assets/body_models/smplx
ln -s "$(pwd)/external/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
      external/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

### 4. Verify

```bash
em doctor
```

Each required checkpoint is listed `OK` or `missing`. Missing weights are a warning rather than a failure, so read the list instead of relying on the exit code.

**⚠️ Never commit these files.** They are non-commercial licensed. Three separate `.gitignore` files cover them, and because `external/GMR` and `external/GVHMR` are submodules the parent repository structurally cannot contain them. The real risk is `git add` *inside* a submodule.

## Usage

`em` picks the right conda environment for each entry point. Flags after the command pass straight through, so `em process --help` lists that stage's options.

```bash
em process inputs/videos/ai1.mp4 --dry-run   # probe only, runs nothing
em process inputs/videos/ai1.mp4             # full pipeline
em process inputs/videos --recursive         # a whole tree
em process inputs/videos/ai1.mp4 --force     # recompute existing stages

em task --clip ai1                           # generate and register a task
em task --clip ai1 --dry-run

em doctor      # validate the installation
em versions    # environments and key package versions
em robots      # robots GMR can retarget onto
em shell gmr   # a shell inside an environment
```

`ai1.mp4` produces motion name `ai1`. Completed stages are reused unless `--force` is given, logs land in `outputs/<clip>/logs/`, and one failing clip does not abort a batch.

Outputs:

```text
outputs/ai1/gvhmr/ai1/hmr4d_results.pt                     # SMPL-X world params
outputs/ai1/robot_data/booster_k1/ai1_booster_k1.pkl       # retargeted trajectory
outputs/ai1/robot_data/booster_k1/csv/ai1_booster_k1.csv   # joint CSV
external/booster/booster_assets/motions/K1/ai1.{csv,npz}   # installed motion
```

`em task` renders the template into Booster Train, registers it, and **prints but does not run** the training command. The default task id is `Booster-K1-Ai1-v0`.

### Training and deployment

Manual upstream operations:

```bash
cd external/booster/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-Ai1-v0 --headless --device cuda:0
python scripts/rsl_rl/play.py --task=Booster-K1-Ai1-v0 --checkpoint=/path/to/checkpoint.pt

cd ../booster_deploy
python scripts/deploy.py --list
python scripts/deploy.py --task TASK_NAME --mujoco
```

**⚠️ Hardware execution additionally requires the upstream SDK, firmware, ROS 2, networking and safety procedures. This repository never starts it automatically.**

## Configuration

Configuration lives in `configs/` and resolves relative to the repository, not the working directory. Any leaf can be overridden per run with `--set key=value`.

| File | Contents |
| --- | --- |
| `configs/pipeline.json` | Contact detection, ground alignment, retargeting, training defaults |
| `configs/paths.json` | Booster checkouts, pinned commits, upstream subpaths |
| `configs/robots/booster_k1.json` | Robot control rate and joints |

```bash
em process inputs/videos/ai1.mp4 --set ground.enabled=false
em task --clip ai1 --set train.max_iterations=15000
```

## Troubleshooting

**`conda environment "expressive-motion-gmr" is missing`**

```bash
bash install.sh --components gmr
```

**`em: command not found`**

```bash
# ~/.local/bin is not on your PATH
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc && source ~/.bashrc
```

**`ModuleNotFoundError: No module named 'pkg_resources'`**

```bash
# lightning==2.3.0 needs pkg_resources, which setuptools >= 81 removed
conda run -n expressive-motion-gvhmr python -m pip install 'setuptools<81'
```

The installer pins this correctly; a later `pip install --upgrade setuptools` reintroduces it. `em doctor` detects it.

**`Booster CSV-to-NPZ conversion requires an Isaac-capable Python`**

```bash
export EM_ISAAC_ENV=your_isaac_conda_env
```

**GVHMR fails with a CUDA or missing-file error**

```bash
em versions   # gvhmr should report: cuda 12.1 | available True
em doctor     # lists any absent checkpoint
```

**Pinned Booster commit mismatch**

```bash
git submodule update --init --recursive
```

⚠️ A forced submodule update discards generated tasks and motions written into `booster_train` and `booster_assets`.

## Known Limitations

1. **Static camera only.** Moving-camera clips need the DPVO path, which is not wired up.
2. **One robot.** The complete path supports `booster_k1`; GMR itself supports more (`em robots`).
3. **Booster checkouts are modified in place.** Generated tasks live inside the submodules, so they always show as dirty in `git status`.
4. **The GVHMR environment is frozen.** Python 3.10, torch 2.3.0, cu121 and numpy 1.23.5 are mutually load-bearing, and the pinned `pytorch3d` wheel installs without complaint when they no longer match.
5. **`--dry-run` skips the Booster and Isaac preflight checks**, so it can pass where a full run would not.

## Repository Structure

```text
expressive-motion/
├── em                          # launcher: em <command>
├── install.sh                  # component-based installer
├── configs/                    # paths, pipeline and robot configuration
├── scripts/
│   ├── batch_process.py        # video to installed Booster motion
│   ├── make_task.py            # task generation
│   ├── retarget_gvhmr.py       # single-clip retargeting
│   └── expressive_motion/      # internal helpers
├── overlay/train/task_template # rendered into booster_train
├── external/                   # pinned upstream submodules
│   ├── GVHMR/                  # monocular motion recovery
│   ├── GMR/                    # motion retargeting
│   └── booster/                # booster_train, booster_assets, booster_deploy
├── inputs/videos/              # your source videos (git-ignored)
├── outputs/                    # per-clip stage outputs (git-ignored)
└── README.md                   # This file
```

Do not commit inputs, outputs, checkpoints, licensed models, trained policies or robot credentials.

## Poster QR target

`docs/` is published with GitHub Pages (source: branch `main`, folder `/docs`).
The printed poster QR code encodes a stable URL that never changes:

```text
https://kyk0.github.io/expressive-motion/video/
```

`docs/video/index.html` is a redirect page, so the destination can be changed
after the poster is printed. To retarget it, edit the `url=` value in:

```html
<meta http-equiv="refresh" content="0; url=https://github.com/Kyk0/expressive-motion">
```

Update the visible fallback `<a href>` in the same file to match, so browsers
that block meta-refresh land in the same place.

## License

The code in this repository is MIT licensed — see [LICENSE](LICENSE).

**⚠️ The pipeline as a whole is research and non-profit use only.** The MIT grant covers this repository's own code and nothing else. Two dependencies independently forbid commercial use:

- **GVHMR** (ZJU 3D Vision Group) — *"educational, research and non-profit purposes only. Any modification based on this work must be open-source and prohibited for commercial use."* Commercial enquiries: xwzhou@zju.edu.cn
- **SMPL / SMPL-X** (Max Planck Institute) — non-commercial research licence, registration required, redistribution prohibited.

| Component | License | Commercial |
| --- | --- | --- |
| This repository | MIT | Yes |
| `external/GVHMR` | Research / non-profit only | **No** |
| `external/GMR` | MIT | Yes |
| `booster_train`, `booster_deploy` | Apache-2.0 | Yes |
| `booster_assets` | BSD-3-Clause | Yes |
| SMPL / SMPL-X | MPI non-commercial | **No** |

GMR ships robot assets under mixed licences; `fourier_n1` is LGPL-3.0 and `external/GMR/third_party/poselib` carries no licence file. Neither affects the `booster_k1` path.

If you use this work, please cite GVHMR and GMR as their authors request.
