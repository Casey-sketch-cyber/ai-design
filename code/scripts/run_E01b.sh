#!/usr/bin/env bash
# Run v2 only. Pilot evidence must be reviewed before the tuning or full phases.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
run_python="$AD_ROOT/envs/e01/bin/python"
export E01_OCR_PYTHON="${E01_OCR_PYTHON:-$AD_ROOT/envs/e01-ocr/bin/python}"
if [[ ! -x "$run_python" ]]; then
    printf 'Prepare the E01 environment with bash code/scripts/setup_E01.sh\n' >&2
    exit 2
fi
phase=pilot
case "${1:-}" in
    pilot|tune|full) phase="$1"; shift ;;
    paths|review|baseline) command="$1"; shift; exec "$run_python" -m dielinediff.e01b "$command" "$@" ;;
    backfill)
        shift
        "$run_python" -m dielinediff.metrics.prepare_image_reward
        export CUDA_VISIBLE_DEVICES=
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
        exec "$run_python" -m dielinediff.e01b_backfill all "$@" ;;
    -h|--help)
        printf 'Usage: %s [pilot|tune|full|paths|review|baseline|backfill] [E01b options]\n' "$0"
        "$run_python" -m dielinediff.e01b --help
        exit 0 ;;
    ""|-*) ;;
    *) printf 'Unrecognized E01b action: %s\n' "$1" >&2; exit 2 ;;
esac
"$run_python" -m dielinediff.e01b prepare --phase "$phase" "$@"
"$run_python" -m dielinediff.e01b check --phase "$phase" "$@"
"$run_python" -m dielinediff.metrics.prepare_image_reward
runtime_dir="$(mktemp -d "$AD_ROOT/tmp/e01b-queue-XXXXXX")"
generation_done="$runtime_dir/generation-finished"
trap 'touch "$generation_done"' EXIT

baseline_worker=
if [[ "$phase" == pilot ]]; then
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
        "$run_python" -m dielinediff.e01b baseline "$@" --phase full &
    baseline_worker="$!"
fi

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
    "$run_python" -m dielinediff.e01b measure "$@" --phase "$phase" --device cpu \
        --follow-until "$generation_done" &
scorer="$!"
status=0
if ! flock "$AD_ROOT/tmp/e01-gpu.lock" "$run_python" -m dielinediff.e01b generate \
        "$@" --phase "$phase" --device cuda; then
    status=1
fi
touch "$generation_done"
if ! wait "$scorer"; then
    status=1
fi
if [[ -n "$baseline_worker" ]] && ! wait "$baseline_worker"; then
    status=1
fi
if ! "$run_python" -m dielinediff.e01b summarize --phase "$phase" "$@"; then
    status=1
fi
exit "$status"
