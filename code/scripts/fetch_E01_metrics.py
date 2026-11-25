"""Prepare the fixed print and perceptual resources for E01 scoring."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from dielinediff.metrics.learned import LearnedMetrics

PADDLE_MODELS = {
    "PP-OCRv5_mobile_det": "0d63e78",
    "PP-OCRv5_mobile_rec": "682f205",
    "PP-LCNet_x1_0_textline_ori": "cd237a4",
}
FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def main():
    cache = Path(os.environ["XDG_CACHE_HOME"])
    paddle_home = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(cache / "paddle")))
    for name, revision in PADDLE_MODELS.items():
        directory = paddle_home / "official_models" / name
        if not all((directory / part).is_file() for part in FILES):
            snapshot_download(
                f"PaddlePaddle/{name}", revision=revision, local_dir=directory,
                allow_patterns=list(FILES),
            )
        if not all((directory / part).is_file() for part in FILES):
            raise FileNotFoundError(f"Missing fixed OCR model resources in {directory}")

    # Exercise each model's normal load path, including the checkpoint hash checks.
    models = LearnedMetrics("cpu", 0.5)
    for name in ("_lpips", "_clip", "_hps"):
        getattr(models, name)
        print(f"Ready: {name.removeprefix('_')}", flush=True)


if __name__ == "__main__":
    main()
