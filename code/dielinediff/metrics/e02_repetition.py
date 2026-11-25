"""Frozen DINOv2 image-level cosine comparison for E02 repeat detection."""

from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


DINO_REPOSITORY = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DEFAULT_M4 = {
    "model": DINO_REPOSITORY,
    "revision": DINO_REVISION,
    "size_px": 224,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "face_mean_weight": 0.5,
}
SIDE_ORDER = ("L", "F", "R", "B")
CROP_COUNT = len(SIDE_ORDER) + 2
FACE_PAIRS = tuple(combinations(range(len(SIDE_ORDER)), 2))


def crop_regions(atlas, topology):
    """Crop whole side panels plus two front halves at registered coordinates."""
    bounds = topology["faces"]
    sides = [atlas.crop(bounds[face]["rect"]) for face in SIDE_ORDER]
    front = sides[1]
    middle = front.width // 2
    return (*sides, front.crop((0, 0, middle, front.height)),
            front.crop((middle, 0, front.width, front.height)))


def summarize_embeddings(features, face_weight):
    """Return side-to-side, front half-to-half, and combined cosine similarities."""
    if features.ndim != 3 or features.shape[1] != CROP_COUNT:
        raise ValueError(f"Expected an atlas batch with {CROP_COUNT} embeddings each")
    values = F.normalize(features.float(), dim=-1)
    pairs = torch.stack([(values[:, a] * values[:, b]).sum(dim=-1) for a, b in FACE_PAIRS], dim=-1)
    faces = pairs.mean(dim=-1)
    halves = (values[:, 4] * values[:, 5]).sum(dim=-1)
    combined = face_weight * faces + (1 - face_weight) * halves
    return torch.stack((faces, halves, combined, pairs.amax(dim=-1)), dim=-1).clamp(-1, 1)


class DinoRepetition:
    """Score atlas crops with the pinned, local DINOv2-S weights."""

    def __init__(self, model_dir, config=None, *, device="cpu"):
        from transformers import AutoModel

        self.config = DEFAULT_M4 if config is None else config
        self.device = torch.device(device)
        self.side = self.config["size_px"]
        self.mean = torch.tensor(self.config["image_mean"], device=self.device)[None, :, None, None]
        self.std = torch.tensor(self.config["image_std"], device=self.device)[None, :, None, None]
        self.model = AutoModel.from_pretrained(Path(model_dir), local_files_only=True, use_safetensors=True)
        self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.inference_mode()
    def measure_batch(self, samples):
        """Measure (PIL atlas, registered topology) pairs without cropping away panel content."""
        if not samples:
            return []
        tiles = [
            np.asarray(view.resize((self.side, self.side), Image.Resampling.BICUBIC))
            for atlas, topology in samples for view in crop_regions(atlas.convert("RGB"), topology)
        ]
        pixels = torch.from_numpy(np.stack(tiles)).to(self.device, dtype=torch.float32)
        pixels = pixels.permute(0, 3, 1, 2).div_(255).sub_(self.mean).div_(self.std)
        cls = self.model(pixel_values=pixels).last_hidden_state[:, 0].unflatten(0, (-1, CROP_COUNT))
        values = summarize_embeddings(cls, self.config["face_mean_weight"]).cpu().tolist()
        fields = ("m4_side_mean_cosine", "m4_front_halves_cosine", "m4_repeat_score", "m4_side_max_cosine")
        return [dict(zip(fields, row, strict=True)) for row in values]
