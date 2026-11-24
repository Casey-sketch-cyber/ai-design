"""Learned E01 metrics with reusable model instances."""

import atexit
import base64
from functools import cached_property, partial
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
HPS_SHA256 = "07cb0d20626232204ca3a6ec53ff5984e7e53212a8a7b0e0ec2bac1bdaf125a1"
HPS_REVISION = "697403c78157020a1ae59d23f111aa58ced35b0a"
ORIENTATION_PROMPTS = (
    "an upright graphic design with text and imagery in the correct orientation",
    "a graphic design turned upside down with inverted text and imagery",
    "a sideways graphic design with text and imagery rotated by ninety degrees",
)


def _square(image):
    size = max(image.size)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _shutdown_worker(worker):
    try:
        worker.stdin.close()
    except BrokenPipeError:
        pass
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
    finally:
        worker.stdout.close()


class LearnedMetrics:
    def __init__(self, device, ocr_confidence):
        self.device = str(device)
        self.ocr_confidence = ocr_confidence
        self.cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    @cached_property
    def _lpips(self):
        import lpips
        return lpips.LPIPS(net="alex", version="0.1", verbose=False).to(self.device).eval()

    def seam_lpips(self, pairs):
        import torch
        from torchvision.transforms.functional import to_tensor

        def strip_tensor(strip):
            image = Image.fromarray(np.ascontiguousarray(strip))
            width = max(64, image.width)
            image = image.resize((width, 224), Image.Resampling.BILINEAR)
            return to_tensor(image).mul(2).sub(1)

        distances = []
        with torch.inference_mode():
            for patch, first, second in pairs:
                reference = torch.stack([strip_tensor(first), strip_tensor(second)]).to(self.device)
                candidate = strip_tensor(patch).expand_as(reference).to(self.device)
                distances.append(self._lpips(candidate, reference).mean().item())
        return float(np.mean(distances))

    @cached_property
    def _clip(self):
        import open_clip
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform

        cache = self.cache / "clip"
        path = cache / "ViT-B-32.pt"
        if path.exists() and _digest(path) != CLIP_SHA256:
            raise ValueError(f"OpenAI CLIP checkpoint failed SHA-256 verification: {path}")
        source = str(path) if path.exists() else "ViT-B-32"
        model = load_openai_model(source, precision="fp32", device=self.device, cache_dir=str(cache))
        preprocess = image_transform(
            model.visual.image_size, is_train=False,
            mean=model.visual.image_mean, std=model.visual.image_std,
        )
        return model.eval(), preprocess, open_clip.get_tokenizer("ViT-B-32-quickgelu")

    def clip_scores(self, images, prompts):
        import torch
        from torch.nn.functional import normalize

        model, preprocess, tokenizer = self._clip
        pixels = torch.stack([preprocess(_square(image)) for image in images]).to(self.device)
        with torch.inference_mode():
            image_features = normalize(model.encode_image(pixels).float())
            orientation = normalize(
                model.encode_text(tokenizer(list(ORIENTATION_PROMPTS)).to(self.device)).float()
            )
            upright = ((image_features @ orientation.T).argmax(dim=-1) == 0).float().mean().item()
            if all(prompts):
                descriptions = tokenizer(["A photo depicts " + prompt for prompt in prompts])
                text_features = normalize(model.encode_text(descriptions.to(self.device)).float())
                score = (image_features * text_features).sum(dim=-1).clamp_min(0).mean().item() * 2.5
                return upright, score
        return upright, None

    @cached_property
    def _hps(self):
        import torch
        import open_clip
        from huggingface_hub import hf_hub_download

        checkpoint_path = os.environ.get("E01_HPS_CHECKPOINT") or hf_hub_download(
            "xswu/HPSv2", "HPS_v2_compressed.pt", revision=HPS_REVISION
        )
        checkpoint_path = Path(checkpoint_path)
        if _digest(checkpoint_path) != HPS_SHA256:
            raise ValueError(f"HPS v2 checkpoint failed SHA-256 verification: {checkpoint_path}")
        # The released fine-tuned state dict contains all weights.
        # OpenCLIP's ViT-H-14 configuration matches the released HPS architecture.
        base = open_clip.get_pretrained_cfg("ViT-H-14", "laion2b_s32b_b79k")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp32", device="cpu", output_dict=True,
            image_mean=base.get("mean"), image_std=base.get("std"),
            cache_dir=str(self.cache / "hps"),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(self.device).eval(), preprocess, open_clip.get_tokenizer("ViT-H-14")

    def hps_score(self, images, prompts):
        import torch
        model, preprocess, tokenizer = self._hps
        pixels = torch.stack([preprocess(_square(image)) for image in images]).to(self.device)
        text = tokenizer(prompts).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            result = model(pixels, text)
            return (result["image_features"] * result["text_features"]).sum(dim=-1).mean().item()

    @cached_property
    def _ocr(self):
        python = os.environ.get("E01_OCR_PYTHON", sys.executable)
        environment = os.environ.copy()
        environment.setdefault("PADDLE_PDX_CACHE_HOME", str(self.cache / "paddle"))
        environment.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        worker = subprocess.Popen(
            [python, "-u", str(Path(__file__).with_name("ocr_worker.py")), str(self.ocr_confidence)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8", env=environment,
        )
        self._ocr_cleanup = partial(_shutdown_worker, worker)
        atexit.register(self._ocr_cleanup)
        return worker

    def close(self):
        cleanup = self.__dict__.pop("_ocr_cleanup", None)
        if cleanup is not None:
            atexit.unregister(cleanup)
            self.__dict__.pop("_ocr", None)
            cleanup()

    def ocr_lines(self, image):
        if min(image.size) < 12:
            return []
        samples = np.asarray(image)
        if (samples.max(axis=(0, 1)).astype(int) - samples.min(axis=(0, 1))).max() < 20:
            return []
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        worker = self._ocr
        try:
            worker.stdin.write(base64.b64encode(buffer.getvalue()).decode("ascii") + "\n")
            worker.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError("The E01 PaddleOCR worker exited.") from error
        for line in worker.stdout:
            if line.startswith("E01_OCR:"):
                return json.loads(line.removeprefix("E01_OCR:"))
        raise RuntimeError("The E01 PaddleOCR worker exited without a response.")
