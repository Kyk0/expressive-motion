#!/usr/bin/env bash
# Usage:
#   bash install.sh                  both environments
#   bash install.sh main             expressive-motion only (retargeting, no GPU)
#   bash install.sh gvhmr            expressive-motion-gvhmr only
#   bash install.sh isaac ENV        add Booster packages to an existing Isaac Lab env
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ENV="expressive-motion"
GVHMR_ENV="expressive-motion-gvhmr"
GVHMR="$ROOT/external/GVHMR"
GMR="$ROOT/external/GMR"
BOOSTER="$ROOT/external/booster"

# GVHMR's pinned pytorch3d wheel is built for exactly py310 + torch 2.3.0 + cu121.
TORCH="torch==2.3.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
# lightning==2.3.0 imports pkg_resources, which setuptools 81 removed.
SETUPTOOLS="setuptools>=68,<81"

die() { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pip_in() { conda run --no-capture-output -n "$1" python -m pip install "${@:2}"; }
make_env() {
    conda env list | awk '{print $1}' | grep -qx "$1" || conda create -y -n "$1" python=3.10
}

[[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]] || die "x86_64 Linux only."
command -v conda >/dev/null || die "conda not found."
command -v ffmpeg >/dev/null || die "ffmpeg not found: sudo apt install ffmpeg"

target="${1:-all}"

if [[ "$target" == "isaac" ]]; then
    [[ -n "${2:-}" ]] || die "usage: bash install.sh isaac ISAAC_LAB_ENV"
    step "Booster packages -> $2"
    pip_in "$2" -e "$BOOSTER/booster_assets" -e "$BOOSTER/booster_train/source/booster_train"
    exit 0
fi
[[ "$target" =~ ^(all|main|gvhmr)$ ]] || die "unknown target: $target"

step "Submodules"
git -C "$ROOT" submodule update --init --recursive

if [[ "$target" != "gvhmr" ]]; then
    step "Environment: $MAIN_ENV"
    make_env "$MAIN_ENV"
    conda install -y -n "$MAIN_ENV" -c conda-forge libstdcxx-ng
    pip_in "$MAIN_ENV" --upgrade pip wheel "$SETUPTOOLS"
    # GMR pulls torch in unpinned; pin it so it matches the driver.
    pip_in "$MAIN_ENV" --index-url "$TORCH_INDEX" "$TORCH"
    pip_in "$MAIN_ENV" -e "$GMR" -r "$BOOSTER/booster_deploy/requirements.txt" -e "$BOOSTER/booster_assets"
fi

if [[ "$target" != "main" ]]; then
    step "Environment: $GVHMR_ENV"
    make_env "$GVHMR_ENV"
    pip_in "$GVHMR_ENV" --upgrade pip wheel "$SETUPTOOLS"
    pip_in "$GVHMR_ENV" -r "$GVHMR/requirements.txt" -e "$GVHMR"
    # requirements.txt can upgrade setuptools past the limit; put it back.
    pip_in "$GVHMR_ENV" "$SETUPTOOLS"
fi

step "Licensed weights (download manually, see README)"
missing=0
for f in \
    "$GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
    "$GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
    "$GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" \
    "$GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
    "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
    "$GVHMR/inputs/checkpoints/yolo/yolov8x.pt" \
    "$GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz"; do
    if [[ -f "$f" ]]; then printf '   ok       %s\n' "${f#"$ROOT/"}"
    else printf '   \033[33mmissing\033[0m  %s\n' "${f#"$ROOT/"}"; missing=1; fi
done

printf '\n\033[32mDone.\033[0m\n\n  conda activate %s\n  python scripts/process.py inputs/videos/clip.mp4 --isaac-env YOUR_ISAAC_ENV\n\n' "$MAIN_ENV"
(( missing )) && printf 'Add the missing weights before running GVHMR. scripts/process.py --dry-run re-checks them.\n'
exit 0
