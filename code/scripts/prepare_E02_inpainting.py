"""Fetch and verify the exact public FP16 UNet used by the E02 C2 candidate."""

import argparse
import json
import os
from pathlib import Path
from subprocess import CalledProcessError

from filelock import FileLock

from dielinediff.e01 import write_json
from dielinediff.methods.v3_inpainting import (
    MODEL_DIRECTORY, MODEL_ID, MODEL_REVISION, UPSTREAM, WEIGHT_FILE,
    file_sha256, verify_inpaint_model,
)
from dielinediff.metrics.prepare_image_reward import download_file

WEIGHT_MIRROR = (
    "https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1"
    f"/resolve/master/{WEIGHT_FILE}"
)


def prepare(models, endpoint, check_only):
    directory = models / MODEL_DIRECTORY
    if check_only:
        return verify_inpaint_model(models)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / ".prepare.lock"):
        for name in sorted(UPSTREAM, key=lambda item: item == WEIGHT_FILE):
            upstream, checksum = UPSTREAM[name]
            target = directory / name
            if target.is_file() and file_sha256(target) == checksum:
                continue
            source = f"{endpoint.rstrip('/')}/{MODEL_ID}/resolve/{MODEL_REVISION}/{upstream}"
            if name == WEIGHT_FILE:
                try:
                    download_file(WEIGHT_MIRROR, target, checksum)
                except CalledProcessError:
                    download_file(source, target, checksum)
            else:
                download_file(source, target, checksum)
        record = verify_inpaint_model(models)
        write_json(directory / "model-evidence.json", record)
        return record


def main():
    root = os.environ.get("DIELINEDIFF_ROOT", os.environ.get("AD_ROOT"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path(root) / "cache/models" if root else None)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.models_dir is None:
        parser.error("specify --models-dir or source code/scripts/e01_env.sh")
    print(json.dumps(prepare(args.models_dir, args.endpoint, args.check_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
