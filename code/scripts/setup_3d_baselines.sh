#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
workspace="${E01_WORKSPACE:-$(dirname "$repo")}"
third_party="$workspace/third_party"
profile="${1:-syncmvd}"
mkdir -p "$workspace"/{tmp,cache,logs} "$third_party/envs"
export TMPDIR="$workspace/tmp" TMP="$workspace/tmp" TEMP="$workspace/tmp"
export CONDA_PKGS_DIRS="$workspace/cache/conda-pkgs"
export CONDA_ENVS_PATH="$third_party/envs" CONDA_REGISTER_ENVS=false
export XDG_CACHE_HOME="$workspace/cache" PIP_CACHE_DIR="$workspace/cache/pip"
export PYTHONPYCACHEPREFIX="$workspace/cache/pycache"
export TORCH_HOME="$workspace/cache/torch" HF_HOME="$workspace/cache/huggingface"
export CUPY_CACHE_DIR="$workspace/cache/cupy" CUDA_CACHE_PATH="$workspace/cache/cuda"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export GIT_TERMINAL_PROMPT=0

checkout() {
    local name="$1" url="$2" revision="$3" dst="$third_party/$1"
    if [[ ! -d "$dst/.git" ]]; then
        git init -q "$dst"
        git -C "$dst" remote add origin "$url"
    fi
    if [[ "$(git -C "$dst" rev-parse --verify HEAD 2>/dev/null || true)" != "$revision" ]]; then
        git -C "$dst" fetch --depth 1 "$url" "$revision"
        git -C "$dst" checkout --detach FETCH_HEAD
    fi
}

copy_cu124_torch() {
    local env_dir="$1" base source_site target_site package
    base="${E01_TORCH_CONDA_PREFIX:-$(conda info --base)}"
    "$base/bin/python" -c 'import sys, torch; assert sys.version_info[:2] == (3, 12) and torch.__version__.startswith("2.5.1") and torch.version.cuda == "12.4"'
    if [[ ! -x "$env_dir/bin/python" ]]; then
        conda create -y --prefix "$env_dir" 'python=3.12' pip
    fi
    source_site="$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    target_site="$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    shopt -s nullglob
    for package in "$source_site"/{torch,torch-*.dist-info,torchvision,torchvision-*.dist-info,torchvision.libs,nvidia,nvidia*.dist-info,triton,triton*.dist-info,functorch,torchgen}; do
        if [[ ! -e "$target_site/${package##*/}" ]]; then
            cp -a "$package" "$target_site/"
        fi
    done
    shopt -u nullglob
}

install_syncmvd_packages() {
    local python="$1" cupy="$2"
    "$python" -m pip install 'pillow==10.4.0' 'diffusers==0.19.3' 'transformers==4.36.0' \
        'accelerate==0.25.0' 'huggingface_hub==0.25.2' "$cupy" 'xatlas==0.0.9' \
        'fvcore==0.1.5.post20221221' 'iopath==0.1.9' 'configargparse==1.7' \
        'safetensors==0.4.5'
}

download_syncmvd_models() {
    "$1" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    allow_patterns=["model_index.json", "*/*.json", "tokenizer/*.txt", "*/*.safetensors"],
    ignore_patterns=["*.fp16.safetensors", "*.non_ema.safetensors", "safety_checker/*", "feature_extractor/*"],
)
snapshot_download(
    "lllyasviel/control_v11f1p_sd15_depth",
    revision="539f99181d33db39cf1af2e517cd8056785f0a87",
    allow_patterns=["config.json", "diffusion_pytorch_model.fp16.safetensors"],
)
PY
}

download_paint3d_models() {
    PYTHONPATH="$repo/code" "$1" - <<'PY'
from huggingface_hub import snapshot_download
from dielinediff.methods.b4_paint3d import BASE_MODEL, MODEL_REVISIONS

for model, revision in MODEL_REVISIONS.items():
    if model in (BASE_MODEL, "GeorgeQi/realisticVisionV13_v13"):
        patterns = ["model_index.json", "scheduler/*", "tokenizer/*",
                    "text_encoder/config.json", "unet/config.json", "vae/config.json"]
        if model == BASE_MODEL:
            patterns += ["text_encoder/model.safetensors", "*/diffusion_pytorch_model.safetensors"]
        else:
            patterns += ["text_encoder/pytorch_model.bin", "*/diffusion_pytorch_model.bin"]
    else:
        suffix = "safetensors" if model.endswith(("_depth", "_inpaint")) else "bin"
        patterns = ["config.json", f"diffusion_pytorch_model.{suffix}"]
    snapshot_download(model, revision=revision, allow_patterns=patterns)
PY
}

case "$profile" in
    syncmvd|syncmvd-cu124)
        checkout SyncMVD https://github.com/LIU-Yuxin/SyncMVD.git faa95621d220d7327847a994cd380608c04af5ab
        if [[ "$profile" == syncmvd ]]; then
            env_dir="$third_party/envs/syncmvd"
            if [[ ! -x "$env_dir/bin/python" ]]; then
                conda create -y --prefix "$env_dir" python=3.8 pip
            fi
            python="$env_dir/bin/python"
            "$python" -m pip install 'torch==2.0.1' 'torchvision==0.15.2' 'numpy==1.24.4' 'ipython==8.12.3'
            install_syncmvd_packages "$python" 'cupy-cuda11x==12.2.0'
            "$python" -m pip install --no-deps \
                'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl'
        else
            env_dir="$third_party/envs/syncmvd-compat"
            copy_cu124_torch "$env_dir"
            python="$env_dir/bin/python"
            "$python" -m pip install 'numpy==1.26.4' 'ipython==8.32.0'
            install_syncmvd_packages "$python" 'cupy-cuda12x==13.4.1'
            "$python" -m pip install --no-deps \
                'https://wheels.astral.sh/artifacts/5b43e87e070480d818677c4457c01186d97d2315358799baf147b828da3b0f6a/pytorch3d-0.7.9%2Bcu.12.4.torch.2.5-cp312-cp312-manylinux_2_27_x86_64.whl#sha256=5b43e87e070480d818677c4457c01186d97d2315358799baf147b828da3b0f6a'
        fi
        "$python" -c 'import cupy, torch; from pytorch3d import _C; print(torch.__version__)'
        touch "$env_dir/.ready"
        if [[ "${E01_SKIP_WEIGHT_DOWNLOAD:-0}" != 1 ]]; then
            download_syncmvd_models "$python"
        fi
        ;;
    paint3d|paint3d-cu124)
        checkout Paint3D https://github.com/OpenTexture/Paint3D.git d5545d38db6aaf78efc563269f70177688a4218e
        if [[ "$profile" == paint3d ]]; then
            env_dir="$third_party/envs/paint3d"
            if [[ ! -x "$env_dir/bin/python" ]]; then
                conda create -y --prefix "$env_dir" python=3.8 pip
            fi
            python="$env_dir/bin/python"
            "$python" -m pip install --index-url https://download.pytorch.org/whl/cu113 \
                'torch==1.12.1+cu113' 'torchvision==0.13.1+cu113'
            numpy='numpy==1.23.5'
            transformers='transformers==4.27.1'
            opencv='opencv-python-headless==4.6.0.66'
            pillow='pillow==10.4.0'
            kaolin=('kaolin==0.13.0')
            kaolin_index='https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html'
        else
            env_dir="$third_party/envs/paint3d-compat"
            copy_cu124_torch "$env_dir"
            python="$env_dir/bin/python"
            numpy='numpy==1.26.4'
            transformers='transformers==4.36.2'
            opencv='opencv-python-headless==4.10.0.84'
            pillow='pillow==11.3.0'
            kaolin=('kaolin==0.17.0' 'warp-lang==1.19.0')
            kaolin_index='https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html'
        fi
        "$python" -m pip install "$numpy" "$opencv" \
            'imageio==2.9.0' 'imageio-ffmpeg==0.4.2' 'omegaconf==2.1.1' "$transformers" \
            'diffusers==0.25.0' 'accelerate==0.29.2' 'huggingface_hub==0.25.2' \
            'loguru==0.7.2' 'trimesh==3.20.2' 'xatlas==0.0.9' "$pillow"
        "$python" -m pip install "${kaolin[@]}" -f "$kaolin_index"
        (cd "$third_party/Paint3D"; "$python" -c 'import pipeline_paint3d_stage1, pipeline_paint3d_stage2, torch; print(torch.__version__)')
        touch "$env_dir/.ready"
        if [[ "${E01_SKIP_WEIGHT_DOWNLOAD:-0}" != 1 ]]; then
            download_paint3d_models "$python"
        fi
        ;;
    *)
        printf 'Usage: %s [syncmvd|syncmvd-cu124|paint3d|paint3d-cu124]\n' "$0" >&2
        exit 2
        ;;
esac
