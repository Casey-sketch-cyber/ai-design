#!/usr/bin/env bash
# Host prerequisite: CUDA-capable PyTorch and matching torchvision.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
base="${E01_BASE_PYTHON:-python}"
python="$AD_ROOT/envs/e01/bin/python"
ocr="$AD_ROOT/envs/e01-ocr/bin/python"
if [[ ! -x "$python" ]]; then
    "$base" -m venv --system-site-packages "$AD_ROOT/envs/e01"
fi
if [[ ! -x "$ocr" ]]; then
    "$base" -m venv "$AD_ROOT/envs/e01-ocr"
fi
"$python" -m pip install -r code/requirements-e01.txt
"$ocr" -m pip install 'paddlepaddle==3.2.2' 'paddleocr==3.3.2' 'paddlex==3.3.13' pillow
"$python" -m dielinediff.color build --output "$AD_ROOT/cache/icc/iso-coated-v2-lut.npy"
"$python" code/scripts/fetch_E01_models.py
"$python" code/scripts/fetch_E01_metrics.py
