"""Evaluate native E01/E01b atlases before and after a common preflight reset."""

from functools import cached_property

import numpy as np
from PIL import ImageColor

from . import Evaluator, face_prompts, rgb_atlas
from .image_reward import ImageRewardScorer

WHITE = (255, 255, 255)
RESET_LABELS = (0, 7, 8, 9)
FACE_SCORES = ("seam_lpips", "upright_rate", "clip_score", "hps_v2")


def preflight_atlas(texture, labels, main_color):
    """Return independent RGB pixels with identical fixed-region rules for every method."""
    labels = np.asarray(labels)
    atlas = rgb_atlas(texture)
    if labels.ndim != 2 or atlas.shape != (*labels.shape, 3) or atlas.dtype != np.uint8:
        raise ValueError(f"Expected native uint8 RGB atlas for label map {labels.shape}, got {atlas.shape}.")
    color = ImageColor.getrgb(main_color)
    if len(color) != 3:
        raise ValueError("The brief main color must specify opaque RGB.")
    output = atlas.copy()
    output[(labels == 0) | (labels == 7)] = WHITE
    output[(labels == 8) | (labels == 9)] = color
    return output


class E01bEvaluator:
    """Reuse metric models for two geometries and measure both output variants."""

    def __init__(self, legacy_geometry, new_geometry, lut_path, device="cpu",
                 learned=("lpips", "clip", "hps", "image_reward")):
        self.device = device
        self.learned = set(learned)
        shared_backends = self.learned - {"image_reward"}
        self.evaluators = {
            key: Evaluator(path, lut_path, device=device, learned=shared_backends)
            for key, path in (("legacy", legacy_geometry), ("new", new_geometry))
        }
        self.evaluators["new"].learned = self.evaluators["legacy"].learned
        for key, evaluator in self.evaluators.items():
            fixed = np.isin(evaluator.labels, RESET_LABELS)
            if any(face.any() for face in evaluator._crop_faces(fixed).values()):
                raise ValueError(f"Preflight labels overlap a visible face in {key} geometry.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.evaluators["legacy"].close()

    @cached_property
    def image_reward(self):
        return ImageRewardScorer(device=self.device)

    def measure(self, source, brief, *, geometry, prompt, elapsed_seconds=None, peak_memory_mib=None):
        evaluator = self.evaluators[geometry]
        original = rgb_atlas(source)
        prepared = preflight_atlas(original, evaluator.labels, brief["main_color"])
        costs = {"elapsed_seconds": elapsed_seconds, "peak_memory_mib": peak_memory_mib}
        raw = evaluator.measure(original, prompt, **costs)
        # Preflight changes pixels outside the visible face rectangles. Reuse
        # learned scores based on those rectangles and rescore flap OCR if enabled.
        preflight = evaluator.measure(prepared, prompt, learned=self.learned & {"ocr"}, **costs)
        preflight.update({name: raw[name] for name in FACE_SCORES})
        preflight.pop("glue_ink_pct")
        raw["image_reward"] = None
        if "image_reward" in self.learned:
            prompts = face_prompts(prompt)
            if not all(prompts):
                raise ValueError("ImageReward requires a scoring prompt for each face.")
            images = evaluator._upright_images(evaluator._crop_faces(original))
            raw["image_reward"] = self.image_reward.score(images, prompts)
        preflight["image_reward"] = raw["image_reward"]
        return {"raw": raw, "preflight": preflight}
