# Expressive Motion

![Platform](https://img.shields.io/badge/Platform-Ubuntu%20x86__64-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![License](https://img.shields.io/badge/Code-MIT-lightgrey)

Expressive Motion converts a monocular human-motion video into a Booster K1 reference motion. It connects pinned GVHMR, GMR and Booster checkouts into one reproducible pipeline; training, policy export and deployment remain manual upstream operations.

## Overview

Give the pipeline an ordinary video of a person moving and it returns a physically grounded motion clip that a Booster K1 humanoid can be trained to imitate. The work splits into four stages, each owned by a different upstream project:

- **Motion recovery** — GVHMR lifts the video into SMPL-X body parameters in world space.
- **Grounding** — this repository detects foot contacts, levels the floor plane and removes root drift.
- **Retargeting** — GMR solves an optimisation that maps the human skeleton onto the robot's joint limits.
- **Task generation** — a BeyondMimic training task is rendered and registered in Booster Train.

GVHMR runs in static-camera mode, so the DPVO visual-odometry path is skipped. The complete preparation path supports `booster_k1`.

**Key characteristics:**

- **Two isolated conda environments** — GVHMR and GMR have mutually incompatible dependency pins. The split is load-bearing, not stylistic.
- **Stage caching** — every stage writes to disk and is reused on the next run; `--force` recomputes.
- **Pinned upstreams** — the three Booster checkouts are verified against commit hashes in `configs/paths.json` before a task is generated.
- **No automatic hardware execution** — the repository prints deployment commands but never starts them.

## Architecture

```
  inputs/videos/clip.mp4
          │
          ▼  ffmpeg, resample to exactly 30 fps
  ┌───────────────────────────────────────────────────────────────┐
  │  expressive-motion-gvhmr        (CUDA GPU required)           │
  │  YOLOv8 → ViTPose → HMR2.0a → GVHMR                           │
  │  outputs/<clip>/gvhmr/<clip>/hmr4d_results.pt                 │
  └───────────────────────────────────────────────────────────────┘
          │  SMPL-X params in world space
          ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  expressive-motion-gmr          (CPU)                         │
  │  ground alignment → foot IK offsets → GMR retargeting         │
  │  outputs/<clip>/robot_data/booster_k1/<clip>_booster_k1.pkl   │
  │  outputs/<clip>/robot_data/booster_k1/csv/…csv                │
  └───────────────────────────────────────────────────────────────┘
          │  joint trajectory CSV
          ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  your Isaac Lab environment                                   │
  │  booster_train/scripts/csv_to_npz.py                          │
  │  external/booster/booster_assets/motions/K1/<clip>.npz        │
  └───────────────────────────────────────────────────────────────┘
          │
          ▼  scripts/make_task.py
  Booster-K1-<Clip>-v0  →  manual train  →  manual deploy
```

## Prerequisites

### System Requirements

- **OS**: x86_64 Linux. Enforced by the installer — the pinned `pytorch3d` wheel is `linux_x86_64` only.
- **Python**: 3.10, created automatically inside each conda environment.
- **Conda**: 4.9 or newer (`conda run --no-capture-output` is required).
- **GPU**: an NVIDIA CUDA GPU with a driver exposing CUDA 12.1 or newer, for GVHMR only. Retargeting is CPU-bound.
- **Disk**: roughly 25 GB for the environments, plus 5.2 GB of GVHMR weights and body models.

### System Dependencies

```bash
# ffmpeg provides both ffmpeg and ffprobe, used for frame-rate probing and resampling
sudo apt update
sudo apt install -y ffmpeg git
```

### Supplied Separately

The installer deliberately does **not** fetch these. Each is licensed by a third party and cannot be redistributed:

- **Isaac Lab / Isaac Sim** — install upstream, then point the installer at it.
- **SMPL and SMPL-X body models** — require individual registration. See [Body Models and Weights](#body-models-and-weights).
- **GVHMR pretrained checkpoints** — see [Body Models and Weights](#body-models-and-weights).
- **Booster SDK, firmware, ROS 2** — needed only for hardware execution.

## Quick Start

```bash
# 1. Clone with upstream checkouts
git clone <this-repo> expressive-motion
cd expressive-motion
git submodule update --init --recursive

# 2. Install. With no flags it asks which components you want.
bash install.sh

# 3. Add the licensed body models and GVHMR checkpoints (see below).
#    Nothing will run end to end until this is done.

# 4. Confirm the installation
./em doctor

# 5. Probe a video without running anything expensive
./em process inputs/videos/ai1.mp4 --dry-run

# 6. Run the pipeline
export EM_ISAAC_ENV=your_isaac_conda_env
./em process inputs/videos/ai1.mp4 --robot booster_k1
```

## Installation

### Component Selection

The installer builds three independent components. Run it with no arguments for an interactive prompt, or select them explicitly:

```bash
# See what is available
bash install.sh --list

# Interactive: asks about each component in turn
bash install.sh

# Retargeting only, no GPU needed on this machine
bash install.sh --components gmr

# Both processing environments, skip Isaac Lab wiring
bash install.sh --components gmr,gvhmr

# Everything, unattended, wiring Booster Train into an existing Isaac Lab env
bash install.sh --yes --isaac-env your_isaac_conda_env

# Everything except GVHMR
bash install.sh --skip gvhmr
```

| Component | Creates | Purpose |
| --- | --- | --- |
| `gmr` | conda env `expressive-motion-gmr` | Retargeting, CSV conversion, pipeline driver, `booster_assets` |
| `gvhmr` | conda env `expressive-motion-gvhmr` | GVHMR inference. Needs a CUDA GPU |
| `isaac` | nothing new | Installs `booster_assets` + `booster_train` into **your existing** Isaac Lab environment |

**Note:** the `isaac` component never installs Isaac Lab or Isaac Sim. It only adds the two Booster packages to an environment you already built, named with `--isaac-env NAME` or `--isaac-python /path/to/python`.

### Other Installer Options

```bash
# Validate an existing installation, change nothing (same as ./em doctor)
bash install.sh --check

# Remove the conda environments this project created. Lists what it will delete
# and asks for confirmation; refuses to run non-interactively without --yes.
# Never touches Isaac Lab, upstream checkouts or downloaded weights.
bash install.sh --clean
bash install.sh --clean gvhmr

# Use checkouts you populated yourself instead of running git submodule update
bash install.sh --no-submodules

# Full option list
bash install.sh --help
```

Each component records a stamp in `.install/`. Re-running the installer skips a component whose inputs have not changed, and **repairs** one whose previous run died partway through. Every install writes a full transcript to `.install/install-<timestamp>.log`.

### Environment Variables

| Variable | Read by | Purpose |
| --- | --- | --- |
| `EM_ISAAC_ENV` | installer, `process` | Conda environment containing Isaac Lab |
| `EM_ISAAC_PYTHON` | installer, `process` | Absolute path to an Isaac-capable Python; takes precedence over `EM_ISAAC_ENV` |
| `EM_GMR_ENV` | installer, launcher, `process` | Override the GMR conda environment name |
| `EM_GVHMR_ENV` | installer, launcher, `process` | Override the GVHMR conda environment name |
| `EM_BOOSTER_ROOT` | installer, `process`, `task` | Override the Booster checkout root |

`EM_ISAAC_ENV` or `EM_ISAAC_PYTHON` must be set before the final CSV-to-NPZ stage of `./em process`. A `--dry-run` does not need it.

## Body Models and Weights

This is the step that most often blocks a fresh installation. The installer reports which files are absent but cannot download any of them — SMPL/SMPL-X require a signed licence agreement, and the GVHMR checkpoints are distributed through Google Drive under the upstream licences.

Total download is about **5.2 GB**.

### Step 1: Register for the Body Models

Create an account on each site and accept the licence. They are separate registrations:

- **SMPL-X**: https://smpl-x.is.tue.mpg.de/ — download the *SMPL-X v1.1* archive, take `SMPLX_NEUTRAL.npz` (104 MB).
- **SMPL**: https://smpl.is.tue.mpg.de/ — download *SMPL for Python users*, take `SMPL_NEUTRAL.pkl` (236 MB). Some archives name it `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`; rename it.

**⚠️ These files are non-commercial research licensed and must never be committed or redistributed.**

### Step 2: Download the GVHMR Checkpoints

All four pretrained models come from the upstream Google Drive folder. By downloading them you accept the corresponding licences:

https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD

| File | Size | Needed for |
| --- | --- | --- |
| `gvhmr/gvhmr_siga24_release.ckpt` | 156 MB | GVHMR itself |
| `hmr2/epoch=10-step=25000.ckpt` | 2.6 GB | HMR2.0a feature extraction |
| `vitpose/vitpose-h-multi-coco.pth` | 2.4 GB | 2D pose estimation |
| `yolo/yolov8x.pt` | 131 MB | Person detection |
| `dpvo/dpvo.pth` | 14 MB | **Optional.** Moving-camera odometry only |

**Note:** `dpvo.pth` is not required. This repository passes `-s` to GVHMR, selecting the static-camera pipeline, which skips DPVO entirely.

### Step 3: Place the Files

```bash
cd /path/to/expressive-motion

# Create the tree GVHMR expects
mkdir -p external/GVHMR/inputs/checkpoints/{body_models/smpl,body_models/smplx,gvhmr,hmr2,vitpose,yolo}

# Copy in what you downloaded, so the result matches:
#   external/GVHMR/inputs/checkpoints/
#   ├── body_models/smpl/SMPL_NEUTRAL.pkl
#   ├── body_models/smplx/SMPLX_NEUTRAL.npz
#   ├── gvhmr/gvhmr_siga24_release.ckpt
#   ├── hmr2/epoch=10-step=25000.ckpt
#   ├── vitpose/vitpose-h-multi-coco.pth
#   └── yolo/yolov8x.pt
```

GMR needs the **same** neutral SMPL-X model at a second path. Symlink it rather than keeping two 104 MB copies:

```bash
mkdir -p external/GMR/assets/body_models/smplx
ln -s "$(pwd)/external/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
      external/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

### Step 4: Verify

```bash
./em doctor
```

Each required checkpoint is listed as `OK` or `missing`. The optional `dpvo.pth` is not checked. Missing weights are reported as a warning rather than a failure, so `doctor` can still exit 0 while reporting them — read the list, do not rely on the exit code alone.

**⚠️ Never commit these files.** They are covered by three separate `.gitignore` files — the root one, GMR's own, and GVHMR's own. Because `external/GMR` and `external/GVHMR` are git submodules, the parent repository stores only a commit hash and structurally cannot contain them. The real risk is running `git add` *inside* a submodule.

## Usage

`./em` selects the correct conda environment for each entry point, so `conda run -n …` never needs to be typed. Every flag after the subcommand is passed through unchanged.

### Preparing a Motion

```bash
# Probe only: report frame rate and duration, run nothing
./em process inputs/videos/ai1.mp4 --dry-run

# Full pipeline for one clip
export EM_ISAAC_ENV=your_isaac_conda_env
./em process inputs/videos/ai1.mp4 --robot booster_k1

# A whole directory, descending into subdirectories
./em process inputs/videos --recursive

# Recompute stages that already exist
./em process inputs/videos/ai1.mp4 --force

# Full flag list for this stage
./em process --help
```

`ai1.mp4` produces motion name `ai1`. Completed stages are reused unless `--force` is given. Per-clip logs land in `outputs/<clip>/logs/<stage>.log`, and one failing clip does not abort the batch.

**Outputs:**

```text
outputs/ai1/gvhmr/ai1/hmr4d_results.pt                     # SMPL-X world params
outputs/ai1/robot_data/booster_k1/ai1_booster_k1.pkl       # retargeted trajectory
outputs/ai1/robot_data/booster_k1/csv/ai1_booster_k1.csv   # joint CSV
external/booster/booster_assets/motions/K1/ai1.{csv,npz}   # installed motion
```

### Generating a Training Task

```bash
# Write and register the BeyondMimic task
./em task --clip ai1 --robot booster_k1

# Preview without writing anything
./em task --clip ai1 --dry-run

# Override a training hyper-parameter
./em task --clip ai1 --set train.max_iterations=15000
```

This renders `overlay/train/task_template` into Booster Train, appends the import to the task registry, and **prints but does not run** the training command. The default task id is `Booster-K1-Ai1-v0`.

### Training and Deployment

Both are manual upstream operations, run from the Booster checkouts:

```bash
cd external/booster/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-Ai1-v0 --headless --device cuda:0
python scripts/rsl_rl/play.py --task=Booster-K1-Ai1-v0 --checkpoint=/path/to/checkpoint.pt

# After configuring the exported policy as an upstream Booster Deploy task
cd ../booster_deploy
python scripts/deploy.py --list
python scripts/deploy.py --task TASK_NAME --mujoco
```

**⚠️ Hardware execution additionally requires the upstream SDK, firmware, ROS 2, networking and safety procedures. This repository never starts it automatically.**

### Utility Commands

```bash
./em doctor            # validate the installation
./em versions          # print resolved environments and key package versions
./em robots            # list the robots GMR can retarget onto
./em shell gmr         # open a shell inside the GMR environment
./em help              # command list
```

## Configuration

Configuration lives in `configs/` and is resolved relative to the repository, not the working directory. Any leaf can be overridden per run with `--set key=value`.

| File | Contents |
| --- | --- |
| `configs/pipeline.json` | Contact detection, ground alignment, retargeting and training defaults |
| `configs/paths.json` | Booster checkout locations, pinned commits, upstream subpaths |
| `configs/robots/booster_k1.json` | Robot control rate and joint configuration |

Frequently adjusted keys:

| Key | Default | Effect |
| --- | --- | --- |
| `ground.enabled` | `true` | Contact-based ground correction |
| `ground.max_level_deg` | `12.0` | Cap on floor-plane tilt correction |
| `contacts.source` | `auto` | `auto`, GVHMR probabilities, or geometric fallback |
| `retarget.auto_foot_offset` | `true` | Derive foot IK offsets from robot geometry |
| `train.max_iterations` | `30000` | Iterations in the printed training command |
| `batch.gvhmr_workers` | `1` | Parallel GVHMR clips |

```bash
# Example: disable ground correction for one run
./em process inputs/videos/ai1.mp4 --set ground.enabled=false
```

## Troubleshooting

### Common Issues

**`conda environment "expressive-motion-gmr" is missing`**

```bash
# The launcher could not find the environment. Build it:
bash install.sh --components gmr
```

**`Conda not found. Run bash install.sh first.`**

```bash
# The launcher reads .install/conda_path, written by the installer.
# If conda is installed but that file is stale, re-run:
bash install.sh --check
```

**`ModuleNotFoundError: No module named 'pkg_resources'` during GVHMR inference**

```bash
# lightning==2.3.0 imports pkg_resources, which setuptools >= 81 removed.
conda run -n expressive-motion-gvhmr python -m pip install 'setuptools<81'
```

The installer pins this correctly, but a later `pip install --upgrade setuptools` in that environment reintroduces the break. `./em doctor` detects it.

**`Booster CSV-to-NPZ conversion requires an Isaac-capable Python`**

```bash
# The last pipeline stage runs inside Isaac Lab. Point at it:
export EM_ISAAC_ENV=your_isaac_conda_env
# or
export EM_ISAAC_PYTHON=/absolute/path/to/isaac/python
```

**GVHMR stage fails with a CUDA or driver error**

```bash
# Confirm torch sees the GPU in the GVHMR environment
./em versions
# gvhmr: torch 2.3.0+cu121 | cuda 12.1 | available True

# torch is built for CUDA 12.1 and needs a driver exposing 12.1 or newer
nvidia-smi
```

**GVHMR stage fails immediately with a missing-file traceback**

The checkpoints are absent or misplaced. `./em doctor` lists exactly which ones. See [Body Models and Weights](#body-models-and-weights).

**Pinned Booster commit mismatch**

```bash
# make_task.py verifies the three checkouts against configs/paths.json
git submodule update --init --recursive

# ⚠️ Note: a forced submodule update discards generated tasks and motions
# that make_task.py wrote into booster_train and booster_assets.
```

### Environment Validation

```bash
# Full installation report: tools, GPU, environments, weights, checkout pins
./em doctor

# Which environments and package versions are actually in use
./em versions
```

## Known Limitations

1. **Static camera only.** GVHMR runs with `-s`. Moving-camera clips need the DPVO path, which is not wired up.
2. **One robot.** The complete preparation path supports `booster_k1`. GMR itself can retarget onto more, listed by `./em robots`.
3. **`booster_train` and `booster_deploy` are modified in place.** Generated tasks are written into the submodule checkouts, so they will always show as dirty in `git status`, and a forced submodule update discards them.
4. **The GVHMR environment is frozen.** Python 3.10 + torch 2.3.0 + cu121 + numpy 1.23.5 are mutually load-bearing; the pinned `pytorch3d` wheel encodes all four, and pip installs it without complaint even when they no longer match.
5. **The GMR environment resolves torch transitively.** `external/GMR/setup.py` pins no versions and pulls torch in through `smplx`, so an unpinned install can land on a build newer than the local driver. The installer now pins torch before GMR resolves it; retargeting is CPU-only regardless.
6. **`--dry-run` skips the Booster and Isaac preflight checks**, so it can succeed on a machine where the full run would not.

## Repository Structure

```text
expressive-motion/
├── em                          # launcher: ./em <command>
├── install.sh                  # component-based installer
├── configs/
│   ├── paths.json              # Booster checkouts and pinned commits
│   ├── pipeline.json           # pipeline and training defaults
│   └── robots/booster_k1.json  # robot control configuration
├── scripts/
│   ├── batch_process.py        # video to installed Booster motion
│   ├── make_task.py            # BeyondMimic task generation
│   ├── retarget_gvhmr.py       # single-clip retargeting
│   └── expressive_motion/      # internal helpers
├── overlay/train/task_template # rendered into booster_train
├── external/                   # pinned upstream submodules
│   ├── GVHMR/                  # monocular motion recovery
│   ├── GMR/                    # general motion retargeting
│   └── booster/{booster_train,booster_assets,booster_deploy}
├── inputs/videos/              # your source videos (git-ignored)
├── outputs/                    # per-clip stage outputs (git-ignored)
└── README.md                   # This file
```

Use `--help` on any entry point for the verified option list. Do not commit inputs, outputs, checkpoints, licensed models, trained policies or robot credentials.

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

**⚠️ The pipeline as a whole is research and non-profit use only.** The MIT grant covers this repository's own code and nothing else. Two dependencies independently forbid commercial use, and their terms govern any pipeline that includes them:

- **GVHMR** (ZJU 3D Vision Group) — *"Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. Any modification based on this work must be open-source and prohibited for commercial use."* Commercial enquiries: xwzhou@zju.edu.cn
- **SMPL / SMPL-X** (Max Planck Institute for Intelligent Systems) — non-commercial research licence, individual registration required, redistribution prohibited.

### Upstream Licenses

| Component | License | Commercial use |
| --- | --- | --- |
| This repository | MIT | Yes |
| `external/GVHMR` | Research / non-profit only | **No** |
| `external/GMR` | MIT | Yes |
| `external/booster/booster_train` | Apache-2.0 | Yes |
| `external/booster/booster_deploy` | Apache-2.0 | Yes |
| `external/booster/booster_assets` | BSD-3-Clause | Yes |
| SMPL / SMPL-X body models | MPI non-commercial | **No** |

GMR ships robot assets under mixed licences, mostly Apache-2.0, BSD-3-Clause and MIT; `fourier_n1` is LGPL-3.0. `external/GMR/third_party/poselib` carries no licence file in-tree. Neither affects the `booster_k1` path, which uses Booster's own assets.

If you use this work, please cite GVHMR and GMR as requested by their authors.
