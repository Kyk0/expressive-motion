# Expressive Motion

Expressive Motion converts a monocular human-motion video into a Booster K1
reference motion. It connects pinned GVHMR, GMR, and Booster repositories;
training, policy export, and deployment remain manual upstream operations.

## Pipeline

```text
video -> 30 fps -> GVHMR -> grounded GMR pickle -> GMR CSV
      -> Booster NPZ -> generated BeyondMimic task -> manual train/deploy
```

GVHMR is currently run in static-camera mode. The complete preparation path
supports `booster_k1`.

## Installation

Use x86_64 Linux with Git, Conda, `ffmpeg`, `ffprobe`, an NVIDIA CUDA setup
for GVHMR, and an existing Isaac Lab/Isaac Sim environment for Booster:

```bash
git submodule update --init --recursive
EM_ISAAC_ENV=your_isaac_conda_env bash install.sh
```

`EM_ISAAC_PYTHON=/absolute/path/to/python` may be used instead. The installer
creates `expressive-motion-gvhmr` and `expressive-motion-gmr`; it does not
install Isaac Lab.

### Models and weights

The installer installs Python packages, not licensed body models or pretrained
weights. Follow `external/GVHMR/docs/INSTALL.md` and populate:

```text
external/GVHMR/inputs/checkpoints/
├── body_models/smpl/SMPL_NEUTRAL.pkl
├── body_models/smplx/SMPLX_NEUTRAL.npz
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

`dpvo/dpvo.pth` is optional for upstream moving-camera DPVO processing. This
repository passes `-s` to GVHMR, so its static-camera pipeline skips DPVO.

GMR has no learned checkpoint: its retargeting is optimization based and its
robot models/configuration are included. It still needs the licensed neutral
SMPL-X model at:

```text
external/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

The same `SMPLX_NEUTRAL.npz` can be placed at both required paths. Download
SMPL/SMPL-X from their official licensed websites; never commit or redistribute
those files.

## Prepare a video

Run from the repository root. A dry run only probes the input:

```bash
conda run --no-capture-output -n expressive-motion-gmr \
  python scripts/batch_process.py inputs/videos/ai1.mp4 --dry-run
```

Run the preparation pipeline with the Isaac environment selected:

```bash
export EM_ISAAC_ENV=your_isaac_conda_env
conda run --no-capture-output -n expressive-motion-gmr \
  python scripts/batch_process.py inputs/videos/ai1.mp4 --robot booster_k1
```

`ai1.mp4` produces motion name `ai1`. Existing stages are reused; `--force`
recomputes them. Main outputs are:

```text
outputs/ai1/gvhmr/ai1/hmr4d_results.pt
outputs/ai1/robot_data/booster_k1/ai1_booster_k1.pkl
outputs/ai1/robot_data/booster_k1/csv/ai1_booster_k1.csv
external/booster/booster_assets/motions/K1/ai1.{csv,npz}
```

## Generate and train a task

```bash
conda run --no-capture-output -n expressive-motion-gmr \
  python scripts/make_task.py --clip ai1 --robot booster_k1
```

This writes and registers the Booster Train task, then prints—but does not
run—the training command. The default task id is `Booster-K1-Ai1-v0`:

```bash
cd external/booster/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-Ai1-v0 --headless --device cuda:0
python scripts/rsl_rl/play.py \
  --task=Booster-K1-Ai1-v0 --checkpoint=/path/to/checkpoint.pt
```

After configuring the exported policy as an upstream Booster Deploy task:

```bash
cd ../booster_deploy
python scripts/deploy.py --list
python scripts/deploy.py --task TASK_NAME --mujoco
```

Hardware execution additionally requires the upstream SDK, firmware, ROS 2,
networking, and safety procedures. This repository never starts it
automatically.

The command-line entry points are in `scripts/`;
`scripts/expressive_motion/` contains internal helpers. Configuration is
under `configs/`. Use `--help` for verified options.

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

Do not commit inputs, outputs, checkpoints, licensed models, trained policies,
or robot credentials.
