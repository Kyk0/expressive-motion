#!/usr/bin/env bash
set -Eeuo pipefail

# Convenience launcher: picks the right conda environment for each entry point
# so you do not have to spell out `conda run --no-capture-output -n ... python`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/.install"

die() { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }

read_state() {  # read_state <file> <default>
    if [[ -f "$STATE/$1" ]]; then
        local value; value="$(tr -d '[:space:]' <"$STATE/$1")"
        [[ -n "$value" ]] && { printf '%s' "$value"; return; }
    fi
    printf '%s' "$2"
}

GMR_ENV="${EM_GMR_ENV:-$(read_state gmr_env expressive-motion-gmr)}"
GVHMR_ENV="${EM_GVHMR_ENV:-$(read_state gvhmr_env expressive-motion-gvhmr)}"

conda_bin() {
    local from_state; from_state="$(read_state conda_path "")"
    if [[ -n "$from_state" && -x "$from_state" ]]; then printf '%s' "$from_state"; return; fi
    command -v conda 2>/dev/null || die "conda not found. Run: bash install.sh"
}

in_env() {  # in_env <component> <env> <args...>
    local conda; conda="$(conda_bin)"
    "$conda" env list | awk '{print $1}' | grep -qx "$2" ||
        die "conda environment \"$2\" is missing. Run: bash install.sh --components $1"
    exec "$conda" run --no-capture-output -n "$2" "${@:3}"
}

usage() {
    cat <<EOF
Expressive Motion — video to Booster K1 reference motion

USAGE
    ./em <command> [args...]

COMMANDS
    process <video|dir>...   Run the preparation pipeline (video to Booster NPZ).
                             Accepts every scripts/batch_process.py flag.
    task --clip NAME         Generate and register a BeyondMimic training task.
    retarget [args]          Retarget one GVHMR result directly (advanced).
    robots                   List the robots GMR can retarget onto.
    doctor                   Validate the installation without changing it.
    install [args]           Run the installer (bash install.sh).
    shell gmr|gvhmr          Open a shell inside one of the conda environments.
    versions                 Print the resolved environments and key versions.
    help                     Show this message.

    Anything after the command is passed straight through, so
    \`./em process --help\` shows the full flag list for that stage.

EXAMPLES
    ./em process inputs/videos/ai1.mp4 --dry-run
    ./em process inputs/videos --recursive --robot booster_k1
    ./em task --clip ai1
    ./em doctor

ENVIRONMENT
    EM_ISAAC_ENV / EM_ISAAC_PYTHON   Isaac Lab environment, required by the
                                     final CSV-to-NPZ stage of \`process\`.
    EM_GMR_ENV / EM_GVHMR_ENV        Override the conda environment names.
    EM_BOOSTER_ROOT                  Override the Booster checkout root.

Environments in use:
    gmr    $GMR_ENV
    gvhmr  $GVHMR_ENV
EOF
}

[[ $# -gt 0 ]] || { usage; exit 0; }
cmd="$1"; shift

case "$cmd" in
    process|batch)
        # batch_process.py re-invokes the GVHMR env itself for the inference stage.
        in_env gmr "$GMR_ENV" python "$ROOT/scripts/batch_process.py" "$@" ;;
    task|make-task)
        in_env gmr "$GMR_ENV" python "$ROOT/scripts/make_task.py" "$@" ;;
    retarget)
        in_env gmr "$GMR_ENV" python "$ROOT/scripts/retarget_gvhmr.py" "$@" ;;
    robots)
        in_env gmr "$GMR_ENV" python "$ROOT/scripts/retarget_gvhmr.py" --list-robots ;;
    doctor|check)
        exec bash "$ROOT/install.sh" --check "$@" ;;
    install|setup)
        exec bash "$ROOT/install.sh" "$@" ;;
    shell)
        case "${1:-}" in
            gmr)   in_env gmr "$GMR_ENV" bash ;;
            gvhmr) in_env gvhmr "$GVHMR_ENV" bash ;;
            *)     die "usage: ./em shell gmr|gvhmr" ;;
        esac ;;
    versions)
        conda="$(conda_bin)"
        printf 'conda      %s\n' "$conda"
        printf 'gmr env    %s\n' "$GMR_ENV"
        printf 'gvhmr env  %s\n' "$GVHMR_ENV"
        printf 'isaac      %s\n' "${EM_ISAAC_PYTHON:-${EM_ISAAC_ENV:-<unset: set EM_ISAAC_ENV>}}"
        for pair in "gmr:$GMR_ENV" "gvhmr:$GVHMR_ENV"; do
            name="${pair#*:}"
            "$conda" env list | awk '{print $1}' | grep -qx "$name" || continue
            printf '\n[%s]\n' "${pair%%:*}"
            "$conda" run -n "$name" python -c "
import sys, torch
print('  python', sys.version.split()[0])
print('  torch ', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())
" 2>/dev/null || printf '  (could not query)\n'
        done ;;
    help|-h|--help)
        usage ;;
    *)
        die "unknown command: $cmd (try ./em help)" ;;
esac
