#!/usr/bin/env bash
set -Eeuo pipefail

# Install the Expressive Motion environments and upstream Python packages.
#
#   EM_SKIP_SUBMODULES=1   do not update git submodules (supply pinned checkouts)
#   EM_SKIP_CONDA=1        do not create or update the conda environments
#   EM_ISAAC_PYTHON=/path  install Booster Train with this Isaac Lab Python
#   EM_ISAAC_ENV=name      or install Booster Train in this Conda environment

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$ROOT/external/GVHMR"
GMR="$ROOT/external/GMR"
BOOSTER_ROOT="${EM_BOOSTER_ROOT:-$ROOT/external/booster}"
BOOSTER_TRAIN="$BOOSTER_ROOT/booster_train"
BOOSTER_ASSETS="$BOOSTER_ROOT/booster_assets"
BOOSTER_DEPLOY="$BOOSTER_ROOT/booster_deploy"
GVHMR_ENV="expressive-motion-gvhmr"
GMR_ENV="expressive-motion-gmr"

step() { printf '\n== %s\n' "$1"; }
warn() { printf 'WARNING: %s\n' "$1" >&2; }

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "Expressive Motion currently supports x86_64 Linux only."
    exit 1
fi

command -v conda >/dev/null 2>&1 || {
    echo "Missing command: conda"
    echo "The launchers use conda run; install Conda and re-run this installer."
    exit 1
}
CONDA="$(command -v conda)"

# ---------------------------------------------------------------- submodules
step "Upstream checkouts"
if [[ "${EM_SKIP_SUBMODULES:-0}" == "1" ]]; then
    echo "EM_SKIP_SUBMODULES=1, using already-populated upstream checkouts."
elif [[ ! -e "$ROOT/.git" ]]; then
    echo "Not a git checkout, so upstream submodules cannot be initialised."
    echo "Populate all upstream paths first, then set EM_SKIP_SUBMODULES=1."
    exit 1
elif ! command -v git >/dev/null 2>&1; then
    echo "Missing command: git (required to initialise upstream submodules)."
    exit 1
elif ! git -C "$ROOT" submodule update --init --recursive; then
    echo "Upstream submodule initialisation failed."
    echo "Re-run when online, or pre-populate all upstream paths and set EM_SKIP_SUBMODULES=1."
    exit 1
fi

# ------------------------------------------------------------------ upstream
if [[ ! -f "$GVHMR/setup.py" || ! -f "$GVHMR/requirements.txt" ]]; then
    echo "Missing or incomplete GVHMR checkout: $GVHMR"
    exit 1
fi
if [[ ! -f "$GMR/setup.py" ]]; then
    echo "Missing GMR checkout: $GMR"
    exit 1
fi

# ------------------------------------------------------------- external tools
step "External tools"
missing_tools=0
for command in ffmpeg ffprobe; do
    if command -v "$command" >/dev/null 2>&1; then
        echo "  $command: $(command -v "$command")"
    else
        echo "Missing command: $command (required for frame-rate probing/resampling)."
        missing_tools=1
    fi
done
if [[ "$missing_tools" != "0" ]]; then
    exit 1
fi

have_cuda=0
if command -v nvidia-smi >/dev/null 2>&1; then
    have_cuda=1
    echo "  CUDA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    warn "nvidia-smi not found. GVHMR inference needs a CUDA GPU; retargeting does not."
fi

# ------------------------------------------------------------------ conda env
mkdir -p "$ROOT/.install"
printf '%s\n' "$CONDA" > "$ROOT/.install/conda_path"

ensure_env() {
    local name="$1"
    if "$CONDA" run -n "$name" python --version >/dev/null 2>&1; then
        echo "Using existing Conda environment: $name"
    else
        "$CONDA" create -y -n "$name" python=3.10
    fi
}

if [[ "${EM_SKIP_CONDA:-0}" == "1" ]]; then
    step "Conda environments"
    echo "EM_SKIP_CONDA=1, skipping creation and package installation."
else
    step "Conda environment: $GMR_ENV"
    ensure_env "$GMR_ENV"
    "$CONDA" install -y -n "$GMR_ENV" -c conda-forge libstdcxx-ng
    "$CONDA" run --no-capture-output -n "$GMR_ENV" \
        python -m pip install --upgrade pip setuptools wheel
    "$CONDA" run --no-capture-output -n "$GMR_ENV" \
        python -m pip install -e "$GMR"

    if [[ -f "$BOOSTER_DEPLOY/requirements.txt" ]]; then
        "$CONDA" run --no-capture-output -n "$GMR_ENV" \
            python -m pip install -r "$BOOSTER_DEPLOY/requirements.txt"
    fi
    if [[ -f "$BOOSTER_ASSETS/pyproject.toml" || -f "$BOOSTER_ASSETS/setup.py" ]]; then
        "$CONDA" run --no-capture-output -n "$GMR_ENV" \
            python -m pip install -e "$BOOSTER_ASSETS"
    fi

    step "Conda environment: $GVHMR_ENV"
    ensure_env "$GVHMR_ENV"
    "$CONDA" run --no-capture-output -n "$GVHMR_ENV" \
        python -m pip install --upgrade pip setuptools wheel
    "$CONDA" run --no-capture-output -n "$GVHMR_ENV" \
        python -m pip install -r "$GVHMR/requirements.txt"
    "$CONDA" run --no-capture-output -n "$GVHMR_ENV" \
        python -m pip install -e "$GVHMR"

    step "Optional Booster Train package"
    if [[ -n "${EM_ISAAC_PYTHON:-}" ]]; then
        if [[ ! -x "$EM_ISAAC_PYTHON" ]]; then
            echo "EM_ISAAC_PYTHON is not executable: $EM_ISAAC_PYTHON"
            exit 1
        fi
        "$EM_ISAAC_PYTHON" -m pip install -e "$BOOSTER_ASSETS"
        "$EM_ISAAC_PYTHON" -m pip install -e "$BOOSTER_TRAIN/source/booster_train"
    elif [[ -n "${EM_ISAAC_ENV:-}" ]]; then
        "$CONDA" run --no-capture-output -n "$EM_ISAAC_ENV" \
            python -m pip install -e "$BOOSTER_ASSETS"
        "$CONDA" run --no-capture-output -n "$EM_ISAAC_ENV" \
            python -m pip install -e "$BOOSTER_TRAIN/source/booster_train"
    else
        warn "Booster Train requires an existing Isaac Lab installation. Set EM_ISAAC_PYTHON or EM_ISAAC_ENV to run its documented editable install commands."
    fi
fi

step "Environment validation"
"$CONDA" run --no-capture-output -n "$GMR_ENV" \
    python -c "import mujoco, torch, general_motion_retargeting; print('GMR environment OK')"
"$CONDA" run --no-capture-output -n "$GVHMR_ENV" \
    python -c "import hmr4d, torch; print('GVHMR environment OK; torch CUDA:', torch.version.cuda)"
if [[ "$have_cuda" == "1" ]]; then
    "$CONDA" run --no-capture-output -n "$GVHMR_ENV" \
        python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is visible to nvidia-smi but not to GVHMR torch'; print('GVHMR CUDA runtime OK')"
fi

# ------------------------------------------------------- verify the checkouts
step "Verifying Booster checkouts"
"$CONDA" run --no-capture-output -n "$GMR_ENV" \
    python "$ROOT/scripts/expressive_motion/booster_paths.py"

# ------------------------------------------------------------------ next step
cat <<EOF

Installation complete.

Created/validated:
  $GVHMR_ENV   GVHMR inference
  $GMR_ENV     GMR retargeting and CSV conversion

From the project directory:
  conda run -n $GMR_ENV python scripts/batch_process.py inputs/videos --dry-run
  python scripts/make_task.py --clip <motion-name>

The installer cannot supply licensed SMPL/SMPL-X body models, GVHMR checkpoints,
Isaac Lab, robot firmware, ROS 2, or the Booster SDK. See README.md.
EOF
