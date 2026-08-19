#!/usr/bin/env bash
set -Eeuo pipefail

# Install the Expressive Motion environments and upstream Python packages.
# Run `bash install.sh --help` for options.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$ROOT/external/GVHMR"
GMR="$ROOT/external/GMR"
BOOSTER_ROOT="${EM_BOOSTER_ROOT:-$ROOT/external/booster}"
BOOSTER_TRAIN="$BOOSTER_ROOT/booster_train"
BOOSTER_ASSETS="$BOOSTER_ROOT/booster_assets"
BOOSTER_DEPLOY="$BOOSTER_ROOT/booster_deploy"
GVHMR_ENV="${EM_GVHMR_ENV:-expressive-motion-gvhmr}"
GMR_ENV="${EM_GMR_ENV:-expressive-motion-gmr}"
STATE="$ROOT/.install"

# The GVHMR environment is a rigid lock: the pinned pytorch3d wheel in
# external/GVHMR/requirements.txt is built for exactly py310 + cu121 + torch
# 2.3.0. Keep these three in step or the wheel installs and fails at import.
PY_VERSION="3.10"
TORCH_VERSION="2.3.0"
TORCH_CUDA="cu121"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
# lightning==2.3.0 imports pkg_resources, removed in setuptools 81.
SETUPTOOLS_MAX="81"
MIN_CONDA_MAJOR=4
MIN_CONDA_MINOR=9
MIN_FREE_GB=25

DO_SUBMODULES=1
DO_GMR=0
DO_GVHMR=0
DO_ISAAC=0
MODE="install"
ASSUME_YES=0
SELECTED=""

# ------------------------------------------------------------------ utilities
step()  { printf '\n\033[1m== %s\033[0m\n' "$1"; }
info()  { printf '   %s\n' "$1"; }
ok()    { printf '   \033[32mOK\033[0m  %s\n' "$1"; }
warn()  { printf '\033[33mWARNING:\033[0m %s\n' "$1" >&2; }
die()   { printf '\033[31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

usage() {
    cat <<'EOF'
Expressive Motion installer

USAGE
    bash install.sh [OPTIONS]

    With no options and an interactive terminal, the installer asks which
    components to install. Pass --components (or --yes) to run unattended.

OPTIONS
    -h, --help              Show this help and exit.
    -l, --list              List installable components and exit.
    -c, --components LIST   Install only LIST (comma separated). See COMPONENTS.
        --skip LIST         Install everything except LIST.
        --check             Validate an existing install; install nothing.
        --clean [LIST]      Remove the conda environments for LIST (default:
                            gmr,gvhmr), then exit. Never touches Isaac Lab.
        --no-submodules     Do not run `git submodule update`. Use when you
                            supplied the upstream checkouts yourself.
    -y, --yes               Non-interactive; install every component.
        --isaac-env NAME    Conda env holding Isaac Lab (same as EM_ISAAC_ENV).
        --isaac-python PATH Python holding Isaac Lab (same as EM_ISAAC_PYTHON).

COMPONENTS
    gmr      Conda env for retargeting, CSV conversion and the pipeline driver.
             Also installs booster_assets and the booster_deploy requirements.
    gvhmr    Conda env for GVHMR monocular motion recovery. Needs a CUDA GPU.
    isaac    Installs booster_assets + booster_train into YOUR EXISTING Isaac
             Lab environment. Requires --isaac-env or --isaac-python.
             The installer never installs Isaac Lab or Isaac Sim itself.

ENVIRONMENT
    EM_GMR_ENV / EM_GVHMR_ENV   Override the created conda env names.
    EM_ISAAC_ENV / EM_ISAAC_PYTHON   Where Isaac Lab lives.
    EM_BOOSTER_ROOT             Override the Booster checkout root.

EXAMPLES
    bash install.sh --list
    bash install.sh --components gmr
    bash install.sh --components gmr,gvhmr
    bash install.sh --yes --isaac-env env_isaaclab
    bash install.sh --check
    bash install.sh --clean gvhmr

The installer cannot supply licensed SMPL/SMPL-X body models, GVHMR
checkpoints, Isaac Lab, robot firmware, ROS 2 or the Booster SDK.
See README.md.
EOF
}

list_components() {
    cat <<EOF
Installable components:

  gmr      conda env "$GMR_ENV"
           GMR retargeting, CSV conversion, pipeline driver, booster_assets
  gvhmr    conda env "$GVHMR_ENV"
           GVHMR inference (needs an NVIDIA CUDA GPU)
  isaac    booster_assets + booster_train into an existing Isaac Lab env
           (set --isaac-env or --isaac-python)
EOF
}

select_components() {
    local list="$1" name
    IFS=',' read -ra _parts <<<"$list"
    for name in "${_parts[@]}"; do
        case "$(echo "$name" | tr -d '[:space:]')" in
            gmr)    DO_GMR=1 ;;
            gvhmr)  DO_GVHMR=1 ;;
            isaac)  DO_ISAAC=1 ;;
            all)    DO_GMR=1; DO_GVHMR=1; DO_ISAAC=1 ;;
            "")     ;;
            *)      die "Unknown component: $name (see --list)" ;;
        esac
    done
}

# ------------------------------------------------------------------- arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage; exit 0 ;;
        -l|--list)    list_components; exit 0 ;;
        -c|--components) [[ $# -ge 2 ]] || die "--components needs a value"
                      SELECTED="$2"; shift 2 ;;
        --skip)       [[ $# -ge 2 ]] || die "--skip needs a value"
                      DO_GMR=1; DO_GVHMR=1; DO_ISAAC=1
                      IFS=',' read -ra _skip <<<"$2"
                      for _s in "${_skip[@]}"; do
                          case "$(echo "$_s" | tr -d '[:space:]')" in
                              gmr) DO_GMR=0 ;; gvhmr) DO_GVHMR=0 ;;
                              isaac) DO_ISAAC=0 ;; "") ;;
                              *) die "Unknown component: $_s (see --list)" ;;
                          esac
                      done
                      SELECTED="explicit"; shift 2 ;;
        --check)      MODE="check"; shift ;;
        --clean)      MODE="clean"
                      if [[ $# -ge 2 && "$2" != -* ]]; then SELECTED="$2"; shift 2
                      else SELECTED="gmr,gvhmr"; shift; fi ;;
        --no-submodules) DO_SUBMODULES=0; shift ;;
        -y|--yes)     ASSUME_YES=1; shift ;;
        --isaac-env)  [[ $# -ge 2 ]] || die "--isaac-env needs a value"
                      export EM_ISAAC_ENV="$2"; shift 2 ;;
        --isaac-python) [[ $# -ge 2 ]] || die "--isaac-python needs a value"
                      export EM_ISAAC_PYTHON="$2"; shift 2 ;;
        *)            die "Unknown option: $1 (see --help)" ;;
    esac
done

# ------------------------------------------------------------------- platform
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    die "Expressive Motion supports x86_64 Linux only (the pinned pytorch3d wheel is linux_x86_64)."
fi

command -v conda >/dev/null 2>&1 || die \
    "Missing command: conda. The launchers use \`conda run\`; install Conda and re-run."
CONDA="$(command -v conda)"

# `conda run --no-capture-output` requires conda >= 4.9.
conda_version="$("$CONDA" --version 2>/dev/null | awk '{print $2}')"
conda_major="${conda_version%%.*}"
conda_rest="${conda_version#*.}"
conda_minor="${conda_rest%%.*}"
if [[ -n "$conda_major" && -n "$conda_minor" ]]; then
    if (( conda_major < MIN_CONDA_MAJOR )) ||
       (( conda_major == MIN_CONDA_MAJOR && conda_minor < MIN_CONDA_MINOR )); then
        die "conda $conda_version is too old; need >= $MIN_CONDA_MAJOR.$MIN_CONDA_MINOR for 'conda run --no-capture-output'."
    fi
fi

env_exists() { "$CONDA" env list | awk '{print $1}' | grep -qx "$1"; }

# ---------------------------------------------------------------------- clean
if [[ "$MODE" == "clean" ]]; then
    select_components "$SELECTED"
    step "Removing conda environments"
    (( DO_ISAAC )) && warn "Refusing to clean the Isaac Lab environment; remove those packages yourself."

    targets=()
    (( DO_GMR ))   && env_exists "$GMR_ENV"   && targets+=("$GMR_ENV")
    (( DO_GVHMR )) && env_exists "$GVHMR_ENV" && targets+=("$GVHMR_ENV")
    if (( ${#targets[@]} == 0 )); then
        info "Nothing to remove."
        exit 0
    fi
    printf '   This permanently deletes:\n'
    printf '     %s\n' "${targets[@]}"
    if (( ! ASSUME_YES )); then
        if [[ -t 0 ]]; then
            read -r -p "   Proceed? [y/N] " reply
            [[ "${reply,,}" == y* ]] || die "Aborted; nothing was removed."
        else
            die "Refusing to remove environments non-interactively. Re-run with --yes to confirm."
        fi
    fi

    for pair in "$DO_GMR:$GMR_ENV" "$DO_GVHMR:$GVHMR_ENV"; do
        flag="${pair%%:*}"; name="${pair#*:}"
        if (( flag )); then
            if env_exists "$name"; then
                info "conda env remove -n $name"
                "$CONDA" env remove -y -n "$name"
            else
                info "not present: $name"
            fi
            rm -f "$STATE/${name}.stamp"
        fi
    done
    ok "Clean complete. Upstream checkouts and downloaded weights were left alone."
    exit 0
fi

# ------------------------------------------------------------ component choice
if [[ "$MODE" == "check" ]]; then
    DO_GMR=1; DO_GVHMR=1; DO_ISAAC=1; DO_SUBMODULES=0
elif [[ -n "$SELECTED" && "$SELECTED" != "explicit" ]]; then
    select_components "$SELECTED"
elif [[ "$SELECTED" == "explicit" ]]; then
    :
elif (( ASSUME_YES )); then
    DO_GMR=1; DO_GVHMR=1; DO_ISAAC=1
elif [[ -t 0 ]]; then
    step "Choose components"
    list_components
    printf '\n'
    read -r -p "Install GMR retargeting env \"$GMR_ENV\"? [Y/n] " a
    [[ "${a,,}" == n* ]] || DO_GMR=1
    read -r -p "Install GVHMR inference env \"$GVHMR_ENV\" (needs CUDA GPU)? [Y/n] " a
    [[ "${a,,}" == n* ]] || DO_GVHMR=1
    read -r -p "Install Booster Train into an existing Isaac Lab env? [y/N] " a
    if [[ "${a,,}" == y* ]]; then
        DO_ISAAC=1
        if [[ -z "${EM_ISAAC_ENV:-}${EM_ISAAC_PYTHON:-}" ]]; then
            read -r -p "  Isaac Lab conda env name: " a
            [[ -n "$a" ]] && export EM_ISAAC_ENV="$a"
        fi
    fi
else
    DO_GMR=1; DO_GVHMR=1; DO_ISAAC=1
fi

if (( ! DO_GMR && ! DO_GVHMR && ! DO_ISAAC )); then
    die "No components selected. See --list."
fi

# --------------------------------------------------------------------- logging
mkdir -p "$STATE"
if [[ "$MODE" == "install" ]]; then
    LOG="$STATE/install-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
    info "Logging to $LOG"
fi

printf '%s\n' "$CONDA" > "$STATE/conda_path"
printf '%s\n' "$GMR_ENV" > "$STATE/gmr_env"
printf '%s\n' "$GVHMR_ENV" > "$STATE/gvhmr_env"

# ------------------------------------------------------------------ submodules
step "Upstream checkouts"
if (( ! DO_SUBMODULES )); then
    info "Skipping submodule update; using the checkouts already on disk."
elif [[ ! -e "$ROOT/.git" ]]; then
    die "Not a git checkout, so upstream submodules cannot be initialised.
Populate external/{GVHMR,GMR,booster/*} yourself, then re-run with --no-submodules."
elif ! command -v git >/dev/null 2>&1; then
    die "Missing command: git (required to initialise upstream submodules)."
elif ! git -C "$ROOT" submodule update --init --recursive; then
    die "Upstream submodule initialisation failed. Expected checkouts:
  external/GVHMR                    external/booster/booster_train
  external/GMR                      external/booster/booster_assets
                                    external/booster/booster_deploy
Re-run when online, or populate them yourself and pass --no-submodules."
fi

if (( DO_GVHMR )) && [[ ! -f "$GVHMR/setup.py" || ! -f "$GVHMR/requirements.txt" ]]; then
    die "Missing or incomplete GVHMR checkout: $GVHMR"
fi
if (( DO_GMR )) && [[ ! -f "$GMR/setup.py" ]]; then
    die "Missing GMR checkout: $GMR"
fi

# ------------------------------------------------------------- external tools
step "External tools"
missing_tools=0
for command in ffmpeg ffprobe; do
    if command -v "$command" >/dev/null 2>&1; then
        ok "$command: $(command -v "$command")"
    else
        warn "Missing command: $command (required for frame-rate probing/resampling)."
        missing_tools=1
    fi
done
(( missing_tools == 0 )) || die "Install ffmpeg (which provides ffprobe) and re-run."

have_cuda=0
driver_cuda=""
if command -v nvidia-smi >/dev/null 2>&1; then
    have_cuda=1
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    driver_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
    info "driver $driver_version, supports CUDA up to ${driver_cuda:-unknown}"
    # torch cu121 runs on any driver exposing CUDA >= 12.1 (minor-version compat).
    if [[ -n "$driver_cuda" ]]; then
        awk -v have="$driver_cuda" 'BEGIN{ if (have+0 < 12.1) exit 1 }' || warn \
            "Driver exposes CUDA $driver_cuda but torch is built for 12.1; GVHMR inference will fall back to CPU or fail."
    fi
elif (( DO_GVHMR )); then
    warn "nvidia-smi not found. GVHMR inference needs a CUDA GPU; GMR retargeting does not."
fi

if (( DO_GMR || DO_GVHMR )) && [[ "$MODE" == "install" ]]; then
    conda_base="$("$CONDA" info --base 2>/dev/null || echo "$HOME")"
    free_gb="$(df -BG --output=avail "$conda_base" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")"
    if [[ -n "$free_gb" ]]; then
        if (( free_gb < MIN_FREE_GB )); then
            warn "Only ${free_gb}G free on $conda_base; the environments need roughly ${MIN_FREE_GB}G."
        else
            ok "disk: ${free_gb}G free on $conda_base"
        fi
    fi
fi

# ------------------------------------------------------------------ conda envs
ensure_env() {
    local name="$1"
    if env_exists "$name"; then
        info "Using existing conda environment: $name"
    else
        info "Creating conda environment: $name (python $PY_VERSION)"
        "$CONDA" create -y -n "$name" "python=$PY_VERSION"
    fi
}

# A component is skipped only when its stamp matches the inputs that built it,
# so a run interrupted mid-pip is repaired rather than silently reused.
stamp_of() {
    { printf '%s\n' "$PY_VERSION $TORCH_VERSION $TORCH_CUDA"
      cat "$@" 2>/dev/null; } | sha1sum | awk '{print $1}'
}

run_in() {  # run_in <env> <args...>
    "$CONDA" run --no-capture-output -n "$1" "${@:2}"
}

if [[ "$MODE" == "install" ]] && (( DO_GMR )); then
    step "Component: gmr  (conda env $GMR_ENV)"
    want="$(stamp_of "$GMR/setup.py" "$BOOSTER_DEPLOY/requirements.txt")"
    if [[ -f "$STATE/$GMR_ENV.stamp" ]] && [[ "$(cat "$STATE/$GMR_ENV.stamp")" == "$want" ]] && env_exists "$GMR_ENV"; then
        ok "already installed and up to date (delete $STATE/$GMR_ENV.stamp to force)"
    else
        ensure_env "$GMR_ENV"
        "$CONDA" install -y -n "$GMR_ENV" -c conda-forge libstdcxx-ng
        run_in "$GMR_ENV" python -m pip install --upgrade pip wheel
        run_in "$GMR_ENV" python -m pip install "setuptools<$SETUPTOOLS_MAX"

        # GMR's setup.py pins nothing and pulls torch transitively via smplx,
        # so an unconstrained install grabs whatever build is newest that day —
        # frequently one that outruns the local driver. Pin it first.
        info "Pinning torch $TORCH_VERSION+$TORCH_CUDA before GMR resolves it transitively"
        run_in "$GMR_ENV" python -m pip install --index-url "$TORCH_INDEX" \
            "torch==$TORCH_VERSION"

        run_in "$GMR_ENV" python -m pip install -e "$GMR"

        if [[ -f "$BOOSTER_DEPLOY/requirements.txt" ]]; then
            run_in "$GMR_ENV" python -m pip install -r "$BOOSTER_DEPLOY/requirements.txt"
        fi
        if [[ -f "$BOOSTER_ASSETS/pyproject.toml" || -f "$BOOSTER_ASSETS/setup.py" ]]; then
            run_in "$GMR_ENV" python -m pip install -e "$BOOSTER_ASSETS"
        fi
        printf '%s\n' "$want" > "$STATE/$GMR_ENV.stamp"
    fi
fi

if [[ "$MODE" == "install" ]] && (( DO_GVHMR )); then
    step "Component: gvhmr  (conda env $GVHMR_ENV)"
    want="$(stamp_of "$GVHMR/requirements.txt" "$GVHMR/setup.py")"
    if [[ -f "$STATE/$GVHMR_ENV.stamp" ]] && [[ "$(cat "$STATE/$GVHMR_ENV.stamp")" == "$want" ]] && env_exists "$GVHMR_ENV"; then
        ok "already installed and up to date (delete $STATE/$GVHMR_ENV.stamp to force)"
    else
        ensure_env "$GVHMR_ENV"
        run_in "$GVHMR_ENV" python -m pip install --upgrade pip wheel
        # Not `--upgrade setuptools`: lightning==2.3.0 imports pkg_resources,
        # which setuptools >= 81 removed.
        run_in "$GVHMR_ENV" python -m pip install "setuptools>=68,<$SETUPTOOLS_MAX"
        run_in "$GVHMR_ENV" python -m pip install -r "$GVHMR/requirements.txt"
        run_in "$GVHMR_ENV" python -m pip install -e "$GVHMR"
        # requirements.txt lets pip re-resolve setuptools upward; put it back.
        run_in "$GVHMR_ENV" python -m pip install "setuptools>=68,<$SETUPTOOLS_MAX"
        printf '%s\n' "$want" > "$STATE/$GVHMR_ENV.stamp"
    fi
fi

if [[ "$MODE" == "install" ]] && (( DO_ISAAC )); then
    step "Component: isaac  (Booster Train into your Isaac Lab environment)"
    if [[ -n "${EM_ISAAC_PYTHON:-}" ]]; then
        [[ -x "$EM_ISAAC_PYTHON" ]] || die "EM_ISAAC_PYTHON is not executable: $EM_ISAAC_PYTHON"
        "$EM_ISAAC_PYTHON" -m pip install -e "$BOOSTER_ASSETS"
        "$EM_ISAAC_PYTHON" -m pip install -e "$BOOSTER_TRAIN/source/booster_train"
    elif [[ -n "${EM_ISAAC_ENV:-}" ]]; then
        env_exists "$EM_ISAAC_ENV" || die "No such conda environment: $EM_ISAAC_ENV"
        run_in "$EM_ISAAC_ENV" python -m pip install -e "$BOOSTER_ASSETS"
        run_in "$EM_ISAAC_ENV" python -m pip install -e "$BOOSTER_TRAIN/source/booster_train"
    else
        warn "Skipping: Booster Train needs an existing Isaac Lab installation.
   Re-run with --isaac-env NAME or --isaac-python /path/to/python."
        DO_ISAAC=0
    fi
fi

# ------------------------------------------------------------------ validation
step "Validation"
validation_failed=0

if (( DO_GMR )); then
    if env_exists "$GMR_ENV"; then
        run_in "$GMR_ENV" python - <<'PY' || validation_failed=1
import sys
import mujoco, torch, general_motion_retargeting  # noqa: F401
import numpy
print(f"   OK  gmr env: python {sys.version.split()[0]}, "
      f"torch {torch.__version__}, numpy {numpy.__version__}, mujoco {mujoco.__version__}")
if not torch.cuda.is_available():
    print("   NOTE  torch reports no usable CUDA device in this environment.")
    print(f"         torch was built for CUDA {torch.version.cuda}. Retargeting is "
          "CPU-only and unaffected; this matters only if you add GPU work here.")
PY
    else
        warn "conda env $GMR_ENV does not exist"; validation_failed=1
    fi
fi

if (( DO_GVHMR )); then
    if env_exists "$GVHMR_ENV"; then
        run_in "$GVHMR_ENV" python - <<'PY' || validation_failed=1
import sys
import torch, hmr4d  # noqa: F401
import numpy, setuptools
print(f"   OK  gvhmr env: python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}, numpy {numpy.__version__}, setuptools {setuptools.__version__}")
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    raise SystemExit(
        "   FAIL  pkg_resources is missing, so `import lightning` will fail.\n"
        "         setuptools is too new. Fix: pip install 'setuptools<81'")
import pytorch_lightning  # noqa: F401
print(f"   OK  lightning stack importable (pytorch_lightning {pytorch_lightning.__version__})")
PY
        if (( have_cuda )); then
            run_in "$GVHMR_ENV" python -c "
import torch
assert torch.cuda.is_available(), (
    'A GPU is visible to nvidia-smi but not to this torch build. '
    'Check that the driver is new enough for the cu121 wheels.')
print('   OK  gvhmr CUDA runtime:', torch.cuda.get_device_name(0))
" || validation_failed=1
        fi
    else
        warn "conda env $GVHMR_ENV does not exist"; validation_failed=1
    fi
fi

if (( DO_ISAAC )); then
    isaac_check() {
        "$@" -c "
import importlib.util as u
missing = [m for m in ('isaaclab', 'booster_assets', 'booster_train') if u.find_spec(m) is None]
raise SystemExit('   FAIL  not importable: ' + ', '.join(missing) if missing
                 else print('   OK  isaac env: isaaclab, booster_assets, booster_train'))
"
    }
    if [[ -n "${EM_ISAAC_PYTHON:-}" ]]; then
        isaac_check "$EM_ISAAC_PYTHON" || validation_failed=1
    elif [[ -n "${EM_ISAAC_ENV:-}" ]]; then
        isaac_check "$CONDA" run --no-capture-output -n "$EM_ISAAC_ENV" python || validation_failed=1
    fi
fi

# Body models and checkpoints are user-supplied; report rather than fail.
step "User-supplied assets"
assets_missing=0
check_file() {
    if [[ -f "$1" ]]; then ok "${1#"$ROOT/"}"
    else printf '   \033[33mmissing\033[0m %s\n' "${1#"$ROOT/"}"; assets_missing=1; fi
}
if (( DO_GVHMR )); then
    for f in \
        "$GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
        "$GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
        "$GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" \
        "$GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
        "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
        "$GVHMR/inputs/checkpoints/yolo/yolov8x.pt"; do
        check_file "$f"
    done
fi
(( DO_GMR )) && check_file "$GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz"
if (( assets_missing )); then
    warn "Some weights or body models are absent. They are licensed separately and
   cannot be installed automatically. See \"Models and weights\" in README.md."
fi

if (( DO_GMR )) && env_exists "$GMR_ENV"; then
    step "Booster checkouts"
    run_in "$GMR_ENV" python "$ROOT/scripts/expressive_motion/booster_paths.py" || validation_failed=1
fi

# ------------------------------------------------------------------ next steps
if (( validation_failed )); then
    printf '\n'
    die "Validation failed; see the output above."
fi

if [[ "$MODE" == "check" ]]; then
    printf '\n'
    ok "Check complete."
    exit 0
fi

cat <<EOF

$(printf '\033[32mInstallation complete.\033[0m')

Installed:$( (( DO_GMR )) && printf '\n  %-28s GMR retargeting and CSV conversion' "$GMR_ENV" )$( (( DO_GVHMR )) && printf '\n  %-28s GVHMR inference' "$GVHMR_ENV" )$( (( DO_ISAAC )) && printf '\n  %-28s Booster Train + assets' "${EM_ISAAC_ENV:-$EM_ISAAC_PYTHON}" )

Next steps:
  ./em doctor                          re-check this installation
  ./em process inputs/videos --dry-run probe videos without running the pipeline
  ./em process inputs/videos/clip.mp4  run the full preparation pipeline
  ./em task --clip clip                generate and register a BeyondMimic task

EOF
