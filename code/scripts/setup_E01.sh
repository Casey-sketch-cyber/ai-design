#!/usr/bin/env bash
# Host prerequisite: CUDA-capable PyTorch and matching torchvision.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source code/scripts/e01_env.sh
base="${E01_BASE_PYTHON:-python}"
python="$AD_ROOT/envs/e01/bin/python"
if [[ ! -x "$python" ]]; then
    "$base" -m venv --system-site-packages "$AD_ROOT/envs/e01"
fi
"$python" -m pip install -r code/requirements-e01.txt
"$python" code/scripts/fetch_E01_models.py
