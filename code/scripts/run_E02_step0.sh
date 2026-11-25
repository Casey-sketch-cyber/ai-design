#!/usr/bin/env bash
# Reproduce the independent Step 0 diagnostics against E01 and E01b archives.
set -euo pipefail
repo="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
python="$AD_ROOT/envs/e01/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
action="${1:-}"
case "$action" in
    d3)
        shift
        exec flock "$AD_ROOT/tmp/e01-gpu.lock" "$python" -m dielinediff.e02_d3 "$@" ;;
    physical|m4)
        shift
        exec "$python" -m dielinediff.e02_history "$action" "$@" ;;
    diagnose)
        shift
        exec "$python" code/scripts/diagnose_E02_physical.py "$@" ;;
    *)
        printf 'Usage: %s {d3|physical|m4|diagnose} [options]\n' "$0" >&2
        exit 2 ;;
esac
