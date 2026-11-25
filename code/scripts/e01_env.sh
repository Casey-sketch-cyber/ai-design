#!/usr/bin/env bash
# Source this before running E01 tooling.
AD_ROOT="${DIELINEDIFF_ROOT:-/root/autodl-tmp/ai-design}"
export DIELINEDIFF_ROOT="$AD_ROOT"
export TMPDIR="$AD_ROOT/tmp" TMP="$AD_ROOT/tmp" TEMP="$AD_ROOT/tmp"
export XDG_CACHE_HOME="$AD_ROOT/cache" PIP_CACHE_DIR="$AD_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$AD_ROOT/cache/conda" TORCH_HOME="$AD_ROOT/cache/torch"
export HF_HOME="$AD_ROOT/cache/huggingface" MPLCONFIGDIR="$AD_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$AD_ROOT/cache/cuda" TRITON_CACHE_DIR="$AD_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AD_ROOT/cache/torchinductor"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=90 PYTHONDONTWRITEBYTECODE=1
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$AD_ROOT/logs" "$AD_ROOT/outputs"
