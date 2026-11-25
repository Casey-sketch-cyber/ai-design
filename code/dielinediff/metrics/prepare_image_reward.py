"""Prepare and verify the pinned ImageReward resources used by E01b."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download

from .image_reward import (
    BERT_FILES, BERT_REPO, BERT_REVISION, CHECKPOINT_SHA256, CLIP_CODE_REVISION, CONFIG_SHA256,
    IMAGE_REWARD_CODE_REVISION, IMAGE_REWARD_REPO, IMAGE_REWARD_REVISION,
    cache_directory, verify_asset,
)

ARCHIVES = (
    ("image-reward", "zai-org/ImageReward", IMAGE_REWARD_CODE_REVISION,
     "a8380757d807e966ab37219753ffee0aaef04dc0a7bf1c3c90fb441fe0ccb205"),
    ("clip", "openai/CLIP", CLIP_CODE_REVISION,
     "553c80a6364ac7350dd7800f535531175f22c11fa742bda1b50b62121eb5f40f"),
)
FAIRSCALE_VERSION = "0.4.13"
CHECKPOINT_MIRROR = (
    "https://modelscope.cn/api/v1/models/ZhipuAI/ImageReward/repo"
    "?Revision=master&FilePath=ImageReward.pt"
)


def download_file(url, path, sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    subprocess.run([
        "curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3",
        "--connect-timeout", "15", "--continue-at", "-", "--output", str(pending), url,
    ], check=True)
    verify_asset(pending, sha256).replace(path)


def prepare_assets(cache, check_only):
    checkpoint = cache / "ImageReward.pt"
    if not check_only and not checkpoint.exists():
        try:
            download_file(CHECKPOINT_MIRROR, checkpoint, CHECKPOINT_SHA256)
        except subprocess.CalledProcessError:
            hf_hub_download(IMAGE_REWARD_REPO, checkpoint.name,
                            revision=IMAGE_REWARD_REVISION, local_dir=cache)
    verify_asset(checkpoint, CHECKPOINT_SHA256)
    assets = [(cache, IMAGE_REWARD_REPO, IMAGE_REWARD_REVISION, "med_config.json", CONFIG_SHA256)]
    assets.extend(
        (cache / "bert-base-uncased", BERT_REPO, BERT_REVISION, name, digest)
        for name, digest in BERT_FILES.items()
    )
    for directory, repo, revision, filename, digest in assets:
        path = directory / filename
        if not check_only and not path.exists():
            hf_hub_download(repo, filename, revision=revision, local_dir=directory)
        verify_asset(path, digest)


def distribution(name):
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def ensure_dependencies(cache, check_only):
    missing = []
    for name, repo, revision, sha256 in ARCHIVES:
        installed = distribution(name)
        origin = json.loads(installed.read_text("direct_url.json") or "{}") if installed else {}
        if origin.get("archive_info", {}).get("hashes", {}).get("sha256") == sha256:
            continue
        if check_only:
            raise RuntimeError(f"Install the pinned {name} archive with this module first.")
        path = cache / "code" / f"{repo.split('/')[-1]}-{revision}.tar.gz"
        if not path.exists():
            download_file(f"https://codeload.github.com/{repo}/tar.gz/{revision}", path, sha256)
        missing.append(str(verify_asset(path, sha256)))
    installed = distribution("fairscale")
    if installed is None or installed.version != FAIRSCALE_VERSION:
        if check_only:
            raise RuntimeError(f"Install fairscale=={FAIRSCALE_VERSION} with this module first.")
        missing.append(f"fairscale=={FAIRSCALE_VERSION}")
    if missing:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--force-reinstall", *missing,
        ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=cache_directory())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    prepare_assets(args.cache_dir, args.check_only)
    ensure_dependencies(args.cache_dir, args.check_only)
    print(f"Verified ImageReward v1.0 assets and dependencies in {args.cache_dir}")


if __name__ == "__main__":
    main()
