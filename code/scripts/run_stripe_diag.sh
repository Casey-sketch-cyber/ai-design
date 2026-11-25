#!/usr/bin/env bash
# Task 12 uses existing image files and CPU-only stripe and LPIPS calculations.
set -euo pipefail
repo="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec 8>"$AD_ROOT/tmp/stripe-diagnostic.lock"
flock 8
"$AD_ROOT/envs/e01/bin/python" -u -m dielinediff.stripe_diag_run "$@"
