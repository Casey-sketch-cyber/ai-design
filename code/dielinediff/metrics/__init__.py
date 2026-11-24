"""Geometry-indexed E01 metrics for flat sRGB print atlases."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab

from dielinediff.color import ICCRoundtrip

from .learned import LearnedMetrics

DEFAULT_THRESHOLDS = {
    "sdr_width": 2,
    "lpips_width": 32,
    "sdr_floor_delta_e": 1.0,
    "glue_white_rgb": 250,
    "flap_edge_threshold": 0.10,
    "bleed_white_rgb": 250,
    "bleed_match_delta_e": 9.0,
    "bleed_change_delta_e": 0.8,
    "lighting_sigma_fraction": 0.08,
    "oog_delta_e": 2.0,
    "ocr_confidence": 0.5,
}
FACES = ("F", "B", "L", "R", "T", "Btm")
MODEL_METRICS = {
    "lpips": ("seam_lpips",),
    "clip": ("upright_rate", "clip_score"),
    "hps": ("hps_v2",),
    "ocr": ("flap_ocr_count", "ocr_upright_rate"),
}


def _oriented_strip(face, edge, width):
    """Return length by depth by channel with depth 0 on the edge."""
    if edge == "top":
        return face[:width].transpose(1, 0, 2)
    if edge == "bottom":
        return face[-width:][::-1].transpose(1, 0, 2)
    if edge == "left":
        return face[:, :width]
    if edge == "right":
        return face[:, -width:][:, ::-1]
    raise ValueError(f"Unknown topology edge: {edge}")


def _aligned_strips(faces, seam, width):
    a, b = seam["a"], seam["b"]
    first = _oriented_strip(faces[a["face"]], a["edge"], width)
    second = _oriented_strip(faces[b["face"]], b["edge"], width)
    if seam["reverse"]:
        second = second[::-1]
    length = min(len(first), len(second))
    first_indices = np.linspace(0, len(first) - 1, length).round().astype(int)
    second_indices = np.linspace(0, len(second) - 1, length).round().astype(int)
    return first[first_indices], second[second_indices]


def _sdr(faces, seams, width, floor):
    inside = []
    for face in faces.values():
        inside.append(deltaE_ciede2000(face[:, 1:], face[:, :-1]).ravel())
        inside.append(deltaE_ciede2000(face[1:], face[:-1]).ravel())
    adjacent_mean = np.concatenate(inside).mean()
    means = []
    for seam in seams:
        first, second = _aligned_strips(faces, seam, width)
        depth = min(first.shape[1], second.shape[1])
        means.append(deltaE_ciede2000(first[:, :depth], second[:, :depth]).mean())
    return float(np.mean(means) / max(adjacent_mean, floor))


def _lpips_pairs(faces, seams, width):
    for seam in seams:
        first, second = _aligned_strips(faces, seam, 2 * width)
        depth = min(width, first.shape[1] // 2, second.shape[1] // 2)
        if depth == 0:
            raise ValueError("Faces are too small to compare seam strips.")
        # Use two face-interior strips as reference patches, leaving the
        # seam at the center of the test patch.
        patch = np.concatenate([first[:, :depth][:, ::-1], second[:, :depth]], axis=1)
        yield patch, first[:, :2 * depth][:, ::-1], second[:, :2 * depth]


def _flap_edges(luminance, labels, threshold):
    width = np.abs(np.diff(luminance, axis=1))
    height = np.abs(np.diff(luminance, axis=0))
    same_horizontal = (labels[:, :-1] == labels[:, 1:]) & np.isin(labels[:, 1:], [8, 9])
    same_vertical = (labels[:-1] == labels[1:]) & np.isin(labels[1:], [8, 9])
    hits = np.count_nonzero(width[same_horizontal] > threshold)
    hits += np.count_nonzero(height[same_vertical] > threshold)
    total = np.count_nonzero(same_horizontal) + np.count_nonzero(same_vertical)
    return 100 * hits / total if total else None


def _flap_crops(rgb, labels):
    for flap in (8, 9):
        components, _ = ndimage.label(labels == flap)
        for component, bounds in enumerate(ndimage.find_objects(components), start=1):
            image = np.where((components[bounds] == component)[..., None], rgb[bounds], 255)
            yield Image.fromarray(image)


def _low_frequency_gradient(face, fraction):
    sigma = max(1, fraction * min(face.shape))
    smooth = ndimage.gaussian_filter(face, sigma=sigma)
    dy, dx = np.gradient(smooth)
    return float(np.hypot(dy, dx).mean())


class Evaluator:
    """Reuse the same ICC profile and metric models for an E01 batch."""

    def __init__(self, geometry_dir, lut_path=None, device="cpu", thresholds=None,
                 *, icc_path=None, learned=("lpips", "clip", "hps", "ocr")):
        geometry_dir = Path(geometry_dir)
        self.topology = json.loads((geometry_dir / "topology.json").read_text())
        self.labels = np.array(Image.open(geometry_dir / "region_map.png"))
        if self.labels.ndim != 2:
            raise ValueError("region_map.png must be a single-channel label image.")
        self.thresholds = DEFAULT_THRESHOLDS | dict(thresholds or {})
        expected_hash = None
        if lut_path is not None:
            metadata = json.loads(Path(lut_path).with_suffix(".json").read_text())
            expected_hash = metadata["profile_sha256"]
            if icc_path is None:
                icc_path = metadata["profile_path"]
        if icc_path is None:
            raise ValueError("Pass an ICC profile or a LUT with its ICC metadata JSON.")
        if expected_hash is not None:
            digest = hashlib.sha256(Path(icc_path).read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("The evaluation profile is different from the LUT profile.")
        self.icc = ICCRoundtrip(icc_path)
        self.enabled = set(learned)
        if unknown := self.enabled.difference(MODEL_METRICS):
            raise ValueError(f"Unknown learned metric groups: {sorted(unknown)}")
        self.learned = LearnedMetrics(device=device, ocr_confidence=self.thresholds["ocr_confidence"])
        self.visible = np.isin(self.labels, [1, 2, 3, 4, 5, 6])
        millimeters = self.topology["image"]["pixels_per_mm"]
        distance, nearest = ndimage.distance_transform_edt(~self.visible, return_indices=True)
        distance_limit = millimeters * self.topology["box_mm"]["bleed"] + 1
        self.bleed_y, self.bleed_x = np.where((self.labels == 10) & (distance <= distance_limit))
        self.near_y = nearest[0, self.bleed_y, self.bleed_x]
        self.near_x = nearest[1, self.bleed_y, self.bleed_x]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release the optional persistent OCR subprocess."""
        self.learned.close()

    def _crop_faces(self, image):
        faces = {}
        for name in FACES:
            left, top, right, bottom = self.topology["faces"][name]["rect"]
            faces[name] = image[top:bottom, left:right]
        return faces

    def _bleed_fill(self, rgb, lab):
        if not len(self.bleed_y):
            return None
        bleed = lab[self.bleed_y, self.bleed_x]
        reference = lab[self.near_y, self.near_x]
        delta = deltaE_ciede2000(bleed, reference)
        nonwhite = (rgb[self.bleed_y, self.bleed_x] < self.thresholds["bleed_white_rgb"]).any(axis=1)
        continuous = delta <= self.thresholds["bleed_match_delta_e"]
        not_copied = delta >= self.thresholds["bleed_change_delta_e"]
        return float(100 * np.mean(nonwhite & continuous & not_copied))

    def measure(self, texture, prompt=None, *, elapsed_seconds=None, peak_memory_mib=None):
        if isinstance(texture, (str, Path)):
            texture = Image.open(texture)
        if isinstance(texture, Image.Image):
            white = Image.new("RGBA", texture.size, "white")
            texture = np.array(Image.alpha_composite(white, texture.convert("RGBA")).convert("RGB"))
        rgb = np.asarray(texture)
        if rgb.shape != (*self.labels.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB texture for label map {self.labels.shape}, got {rgb.shape}.")
        lab = rgb2lab(rgb.astype(np.float32) / 255)
        rgb_faces = self._crop_faces(rgb)
        lab_faces = self._crop_faces(lab)
        seams = self.topology["adjacencies"]
        t = self.thresholds
        glue = self.labels == 7
        luminance = lab[..., 0] / 100
        printable = self.visible | np.isin(self.labels, [8, 9, 10])
        colors = rgb[printable]
        mapped = rgb2lab(self.icc(colors).astype(np.float32) / 255)
        delta = deltaE_ciede2000(lab[printable], mapped)
        result = dict.fromkeys(field for group in MODEL_METRICS.values() for field in group)
        result.update(
            seam_sdr=_sdr(lab_faces, seams, t["sdr_width"], t["sdr_floor_delta_e"]),
            glue_ink_pct=float(100 * np.mean((rgb[glue] < t["glue_white_rgb"]).any(axis=1))) if glue.any() else None,
            flap_edge_pct=_flap_edges(luminance, self.labels, t["flap_edge_threshold"]),
            bleed_fill_pct=self._bleed_fill(rgb, lab),
            lighting_residual=float(np.mean([
                _low_frequency_gradient(face[..., 0], t["lighting_sigma_fraction"])
                for face in lab_faces.values()
            ])),
            oog_pct=float(100 * np.mean(delta > t["oog_delta_e"])),
            roundtrip_delta_e00=float(delta.mean()),
            elapsed_seconds=elapsed_seconds,
            peak_memory_mib=peak_memory_mib,
        )
        upright = [
            Image.fromarray(np.ascontiguousarray(np.rot90(
                rgb_faces[name], self.topology["faces"][name]["upright_rotation"] // 90
            )))
            for name in FACES
        ]
        prompts = [prompt.get(name, prompt.get("F")) for name in FACES] if isinstance(prompt, dict) else [prompt] * len(FACES)
        if "lpips" in self.enabled:
            result["seam_lpips"] = self.learned.seam_lpips(_lpips_pairs(rgb_faces, seams, t["lpips_width"]))
        if "clip" in self.enabled:
            result["upright_rate"], result["clip_score"] = self.learned.clip_scores(upright, prompts)
        if "hps" in self.enabled and all(prompts):
            result["hps_v2"] = self.learned.hps_score(upright, prompts)
        if "ocr" in self.enabled:
            result["flap_ocr_count"] = sum(
                len(self.learned.ocr_lines(crop)) for crop in _flap_crops(rgb, self.labels)
            )
            orientations = [angle for crop in upright for _, angle in self.learned.ocr_lines(crop)]
            valid_angles = [angle for angle in orientations if angle in (0, 180)]
            if valid_angles:
                result["ocr_upright_rate"] = sum(angle == 0 for angle in valid_angles) / len(valid_angles)
        return result
