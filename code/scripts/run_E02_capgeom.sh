#!/usr/bin/env bash
# Task 10. Approaches are selected on seed 42; seed 1234 only checks the selection.
set -euo pipefail
repo="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
python="$AD_ROOT/envs/e01/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export E01_OCR_PYTHON="${E01_OCR_PYTHON:-$AD_ROOT/envs/e01-ocr/bin/python}"
exec 8>"$AD_ROOT/tmp/e01-gpu.lock"
flock 8
"$python" -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes -r code/configs/E02_capgeom-requirements.txt
"$python" -m dielinediff.metrics.prepare_image_reward
"$python" -u -m dielinediff.e02_capgeom run "$@"
