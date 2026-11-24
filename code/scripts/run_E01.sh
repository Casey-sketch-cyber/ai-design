#!/usr/bin/env bash
# Run the E01 pilot or frozen full study on one GPU. Failed jobs remain resumable.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh

python="$AD_ROOT/envs/e01/bin/python"
if [[ ! -x "$python" ]]; then
    printf 'Prepare the E01 environment with bash code/scripts/setup_E01.sh\n' >&2
    exit 2
fi
phase=""
case "${1:-}" in
    pilot|tune|full) phase="$1"; shift ;;
    ""|-*) ;;
    *) printf 'Usage: %s [pilot|tune|full] [E01 settings]\n' "$0" >&2; exit 2 ;;
esac
config=code/configs/E01.yaml
if [[ "$phase" == full || ( -z "$phase" && -f code/configs/E01.frozen.yaml ) ]]; then
    config=code/configs/E01.frozen.yaml
fi
geometry="$AD_ROOT/outputs/geometry"
args=("$@")
for ((n=0; n<${#args[@]}; n++)); do
    case "${args[n]}" in
        --config) config="${args[n+1]}" ;;
        --geometry) geometry="${args[n+1]}" ;;
    esac
done
read -r short_side planned < <("$python" - "$config" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
print(config["short_side"], "full" if config["frozen"] else "pilot")
PY
)
phase="${phase:-$planned}"
export E01_OCR_PYTHON="${E01_OCR_PYTHON:-$AD_ROOT/envs/e01-ocr/bin/python}"
compat="$AD_ROOT/third_party/envs/syncmvd-compat"
if [[ -x "$compat/bin/python" ]]; then
    export E01_SYNCMVD_ENV="${E01_SYNCMVD_ENV:-$compat}"
fi
if [[ "$geometry" == "$AD_ROOT/outputs/geometry" ]]; then
    "$python" -m dielinediff.dieline --output "$geometry" --short-side "$short_side"
fi
status=0
method_list=(b2 ours b3 b4)
if [[ "$phase" == full ]]; then
    method_list=(b2 ours b3 no_sync no_projection b4)
    score_dir="$(mktemp -d "$AD_ROOT/tmp/e01-queue-XXXXXX")"
    score_done="$score_dir/generation-finished"
    trap 'touch "$score_done"' EXIT
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
        "$python" -m dielinediff.e01 measure --phase "$phase" --config "$config" --device cpu \
        --follow-until "$score_done" "$@" &
    scorer="$!"
fi
for method in "${method_list[@]}"; do
    if ! flock "$AD_ROOT/tmp/e01-gpu.lock" "$python" -m dielinediff.e01 generate \
        --phase "$phase" --config "$config" --methods "$method" "$@"; then
        printf 'Generation failed for %s. Inspect the recorded error and resume.\n' "$method" >&2
        if [[ "$method" != b4 ]]; then
            status=1
        fi
    fi
done
if [[ "$phase" == full ]]; then
    touch "$score_done"
    if ! wait "$scorer"; then
        status=1
    fi
elif ! flock "$AD_ROOT/tmp/e01-gpu.lock" "$python" -m dielinediff.e01 measure --phase "$phase" --config "$config" "$@"; then
    status=1
fi
exit "$status"
