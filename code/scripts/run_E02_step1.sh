#!/usr/bin/env bash
# Generate and measure the ten registered E02 Step 1 cases on the shared RTX 3090.
set -euo pipefail
repo="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
python="$AD_ROOT/envs/e01/bin/python"
config="${1:-code/configs/E02_step1.yaml}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export E01_OCR_PYTHON="${E01_OCR_PYTHON:-$AD_ROOT/envs/e01-ocr/bin/python}"
run() { "$python" -u -m dielinediff.e02 "$1" --config "$config" "${@:2}"; }
exec 8>"$AD_ROOT/tmp/e01-gpu.lock"
flock 8
run paths
"$python" -m dielinediff.metrics.prepare_image_reward
run generate
run measure
run select-resolution
shared="$("$python" -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["shared_attention_candidate"])' "$config")"
if [[ "$shared" == True ]]; then
    if run generate --methods v3_shared; then
        run measure --methods v3_shared
    else
        printf 'Optional shared-attention candidate did not complete. Keeping the selected main run.\n' >&2
    fi
fi
run select-final
run summarize
