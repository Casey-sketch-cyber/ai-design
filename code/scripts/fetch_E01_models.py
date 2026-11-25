"""Fetch the immutable E01 SDXL, VAE and ControlNet weights and verify SHA-256."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time

import requests
from huggingface_hub import snapshot_download

ROOT = Path(os.environ["DIELINEDIFF_ROOT"]) / "cache" / "models"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MS = "https://www.modelscope.cn/models/AI-ModelScope"
REPOS = {
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b"),
    "vae": ("madebyollin/sdxl-vae-fp16-fix", "207b116dae70ace3637169f1ddd2434b91b3a8cd"),
    "controlnet": ("diffusers/controlnet-canny-sdxl-1.0", "eb115a19a10d14909256db740ed109532ab1483c"),
}
WEIGHTS = (
    ("sdxl", "unet/diffusion_pytorch_model.fp16.safetensors", 5135149760,
     "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139"),
    ("sdxl", "text_encoder/model.fp16.safetensors", 246144152,
     "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
    ("sdxl", "text_encoder_2/model.fp16.safetensors", 1389382176,
     "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4"),
    ("vae", "diffusion_pytorch_model.safetensors", 334643238,
     "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"),
    ("controlnet", "diffusion_pytorch_model.fp16.safetensors", 2502139136,
     "b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25"),
)
CHUNK = 128 * 1024 ** 2


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pull_part(task):
    path, urls, size, idx = task
    start, end = idx * CHUNK, min((idx + 1) * CHUNK, size) - 1
    expected = end - start + 1
    part = path.with_name(f"{path.name}.part{idx:03}")
    tmp = part.with_suffix(part.suffix + ".tmp")
    if part.exists() and part.stat().st_size == expected:
        return part
    for attempt in range(6):
        try:
            previous = tmp.stat().st_size if tmp.exists() else 0
            if previous > expected:
                tmp.unlink()
                previous = 0
            if previous == expected:
                tmp.replace(part)
                return part
            request_start = start + previous
            with requests.get(
                urls[attempt % len(urls)], headers={"Range": f"bytes={request_start}-{end}"},
                timeout=(12, 25), stream=True,
            ) as response:
                response.raise_for_status()
                if response.headers.get("Content-Range") != f"bytes {request_start}-{end}/{size}":
                    raise ValueError("Unexpected Content-Range from model host")
                with tmp.open("ab") as output:
                    for content in response.iter_content(2 ** 20):
                        output.write(content)
            if tmp.stat().st_size != expected:
                raise ValueError("Incomplete model part")
            tmp.replace(part)
            return part
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Unexpected download state")


def fetch_metadata():
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(
            snapshot_download, repo, revision=revision, local_dir=ROOT / name,
            allow_patterns=["model_index.json", "scheduler/*.json", "tokenizer/*",
                            "tokenizer_2/*", "*/config.json", "config.json"],
            ignore_patterns=["vae/*", "vae_1_0/*"], max_workers=6,
        ) for name, (repo, revision) in REPOS.items()]
        for task in futures:
            task.result()


def fetch_large():
    pending = []
    for name, filename, size, digest in WEIGHTS:
        path = ROOT / name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == size and checksum(path) == digest:
            continue
        repo, revision = REPOS[name]
        urls = (
            f"{MS}/{repo.rsplit('/', 1)[1]}/resolve/master/{filename}",
            f"{HF}/{repo}/resolve/{revision}/{filename}",
        )
        pending.append((path, urls, size, digest))
    jobs = []
    # Cycle through models to avoid starving the small component downloads.
    for idx in range(max(((size + CHUNK - 1) // CHUNK for _, _, size, _ in pending), default=0)):
        jobs.extend((p, u, s, idx) for p, u, s, _ in pending if idx * CHUNK < s)
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {p: [] for p, _, _, _ in pending}
        for path, urls, size, idx in jobs:
            futures[path].append(pool.submit(pull_part, (path, urls, size, idx)))
        for path, _, size, digest in pending:
            parts = [task.result() for task in futures[path]]
            combined = path.with_suffix(".complete")
            with combined.open("wb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output, 8 * 2 ** 20)
            if combined.stat().st_size != size or checksum(combined) != digest:
                raise ValueError(f"SHA-256 verification failed: {path}")
            combined.replace(path)
            for part in parts:
                part.unlink()
            print(f"Verified {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata = pool.submit(fetch_metadata)
        fetch_large()
        metadata.result()
    for name, (_, revision) in REPOS.items():
        (ROOT / f"{name}.revision").write_text(revision + "\n")
