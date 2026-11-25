"""E01b print metrics on a shared physical UV grid."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely import contains_xy
from skimage.color import deltaE_ciede2000, rgb2lab

from dielinediff.dieline.geometry import BoxSpec, build_geometry
from . import DEFAULT_THRESHOLDS, FACES, _lpips_pairs, _oriented_strip, _sdr_components, rgb_atlas
from .learned import LearnedMetrics

PIXELS_PER_MM = 4.0
SDR_MM = 0.5
LPIPS_MM = 5.0
STRIPE_MM = 3.0
STRIPE_REFERENCE_OFFSET_MM = 9.0
STRIPE_ENERGY_FLOOR = (3 / 255) ** 2


def _edges_on_sheet(rect, edge):
    x0, y0, x1, y1 = rect
    return {
        "top": ((x0, y0), (x1, y0)), "bottom": ((x0, y1), (x1, y1)),
        "left": ((x0, y0), (x0, y1)), "right": ((x1, y0), (x1, y1)),
    }[edge]


def _press_creases(geometry):
    for seam in geometry.topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        first = _edges_on_sheet(geometry.faces_mm[a["face"]], a["edge"])
        second = _edges_on_sheet(geometry.faces_mm[b["face"]], b["edge"])
        if frozenset(first) == frozenset(second):
            yield seam


def _stripe_side(face, edge):
    width = round(STRIPE_MM * PIXELS_PER_MM)
    offset = round(STRIPE_REFERENCE_OFFSET_MM * PIXELS_PER_MM)
    strip = _oriented_strip(face, edge, offset + width).astype(np.float32) / 255
    if strip.shape[0] < 2 or strip.shape[1] < offset + width:
        raise ValueError("A crease face is too small for the physical stripe reference.")
    crease_energy = np.square(np.diff(strip[:, :width], axis=0)).mean()
    reference_energy = np.square(np.diff(strip[:, offset:offset + width], axis=0)).mean()
    return 1 + max(0.0, float(crease_energy - reference_energy)) / (
        float(reference_energy) + STRIPE_ENERGY_FLOOR
    )


def stripe_energy(faces, creases):
    """Return the highest mean crease-side energy ratio and the B–Btm ratio."""
    values = {}
    for seam in creases:
        a, b = seam["a"], seam["b"]
        values[frozenset([a["face"], b["face"]])] = (
            _stripe_side(faces[a["face"]], a["edge"]) + _stripe_side(faces[b["face"]], b["edge"])
        ) / 2
    return max(values.values()), values.get(frozenset(["B", "Btm"]))


class PhysicalEvaluator:
    """Apply consistent physical registration while keeping native quality crops."""

    def __init__(self, geometry_dir, device="cpu", learned=None):
        geometry_dir = Path(geometry_dir)
        self.native_topology = json.loads((geometry_dir / "topology.json").read_text())
        with Image.open(geometry_dir / "region_map.png") as region_map:
            labels = np.array(region_map)
        native = self.native_topology["image"]
        if labels.shape != (native["height"], native["width"]):
            raise ValueError(f"Region map disagrees with the native topology: {geometry_dir}")
        for name, face in self.native_topology["faces"].items():
            x0, y0, x1, y1 = face["rect"]
            if np.isin(labels[y0:y1, x0:x1], [0, 7, 8, 9]).any():
                raise ValueError(f"Preflight labels overlap native face {name}: {geometry_dir}")
        self.native_shape = labels.shape
        self.native_bleed = (labels == 10).astype(np.float32)
        self.bleed_mm = self.native_topology["box_mm"]["bleed"]
        self.geometry = build_geometry(BoxSpec(**self.native_topology["box_mm"]), pixels_per_mm=PIXELS_PER_MM)
        self.topology = self.geometry.topology
        self.creases = tuple(_press_creases(self.geometry))
        self.learned = learned if learned is not None else LearnedMetrics(device, DEFAULT_THRESHOLDS["ocr_confidence"])
        self.seams = self.topology["adjacencies"]
        self.caps = {
            name: np.array([
                name in (seam["a"]["face"], seam["b"]["face"]) for seam in self.seams
            ])
            for name in ("T", "Btm")
        }
        self._prepare_bleed()

    def _prepare_bleed(self):
        target = self.topology["image"]
        origin_x, origin_y = target["origin_mm"]
        width, height = target["width"], target["height"]
        xs = origin_x + (np.arange(width)[None, :] + 0.5) / PIXELS_PER_MM
        ys = origin_y + (np.arange(height)[:, None] + 0.5) / PIXELS_PER_MM
        outside_bleed = contains_xy(self.geometry.bleed_outline, xs, ys)
        outside_bleed &= ~contains_xy(self.geometry.outline, xs, ys)
        distance = np.full((height, width), np.inf)
        face_labels = np.zeros((height, width), dtype=np.uint8)
        for index, name in enumerate(FACES, start=1):
            x0, y0, x1, y1 = self.geometry.faces_mm[name]
            dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0)
            dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0)
            np.minimum(distance, np.hypot(dx, dy), out=distance)
            x0, y0, x1, y1 = self.topology["faces"][name]["rect"]
            face_labels[y0:y1, x0:x1] = index
        rows, columns = np.where(outside_bleed & (face_labels == 0) & (distance <= self.bleed_mm))
        nearest = ndimage.distance_transform_edt(face_labels == 0, return_distances=False, return_indices=True)
        y_indices, x_indices = nearest[:, rows, columns]
        owner = face_labels[y_indices, x_indices]
        self.bleed_refs = {}
        for index, name in enumerate(FACES, start=1):
            mask = owner == index
            x0, y0, _, _ = self.topology["faces"][name]["rect"]
            self.bleed_refs[name] = (mask, y_indices[mask] - y0, x_indices[mask] - x0)
        native = self.native_topology["image"]
        scale = native["pixels_per_mm"]
        ox, oy = native["origin_mm"]
        self.bleed_coordinates = np.stack([
            (ys[rows, 0] - oy) * scale - 0.5,
            (xs[0, columns] - ox) * scale - 0.5,
        ])
        self.bleed_sigma = 0.5 * np.sqrt(max((scale / PIXELS_PER_MM) ** 2 - 1, 0))
        self.bleed_support = self._sample_bleed(self.native_bleed)

    def _sample_bleed(self, channel):
        if self.bleed_sigma > 0:
            channel = ndimage.gaussian_filter(channel, sigma=self.bleed_sigma, mode="constant")
        return ndimage.map_coordinates(channel, self.bleed_coordinates, order=1, mode="constant")

    def _native_atlas(self, texture):
        rgb = rgb_atlas(texture)
        if rgb.shape != (*self.native_shape, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected native RGB image for {self.native_shape}, got {rgb.shape}.")
        return rgb

    def _physical_faces(self, rgb):
        faces = {}
        for name in FACES:
            x0, y0, x1, y1 = self.native_topology["faces"][name]["rect"]
            face = Image.fromarray(rgb[y0:y1, x0:x1])
            x0, y0, x1, y1 = self.topology["faces"][name]["rect"]
            shape = (x1 - x0, y1 - y0)
            faces[name] = np.asarray(face.resize(shape, Image.Resampling.BILINEAR))
        return faces

    def _bleed_fill(self, rgb, lab_faces):
        if not self.bleed_support.size:
            return None
        support = self.bleed_support
        valid = support > 1e-5
        sampled = np.full((len(support), 3), 255.0, dtype=np.float32)
        for channel in range(3):
            weighted = self._sample_bleed(rgb[:, :, channel].astype(np.float32) * self.native_bleed)
            sampled[valid, channel] = weighted[valid] / support[valid]
        reference = np.empty_like(sampled)
        for name, (mask, y, x) in self.bleed_refs.items():
            reference[mask] = lab_faces[name][y, x]
        change = deltaE_ciede2000(rgb2lab(sampled.clip(0, 255) / 255), reference)
        t = DEFAULT_THRESHOLDS
        passes = (sampled < t["bleed_white_rgb"]).any(axis=1)
        passes &= valid & (change <= t["bleed_match_delta_e"]) & (change >= t["bleed_change_delta_e"])
        return float(100 * passes.mean())

    def measure(self, texture, *, lpips=True):
        rgb = self._native_atlas(texture)
        faces = self._physical_faces(rgb)
        lab_faces = {name: rgb2lab(face.astype(np.float32) / 255) for name, face in faces.items()}
        widths = (round(SDR_MM * PIXELS_PER_MM), round(LPIPS_MM * PIXELS_PER_MM))
        sdr, normalization = _sdr_components(
            lab_faces, self.seams, widths[0], DEFAULT_THRESHOLDS["sdr_floor_delta_e"]
        )
        distances = (
            self.learned.seam_lpips(_lpips_pairs(faces, self.seams, widths[1]), per_seam=True)
            if lpips else None
        )
        stripe, back_bottom_stripe = stripe_energy(faces, self.creases)
        return {
            "seam_sdr": float(sdr.mean() / normalization),
            "seam_lpips": float(distances.mean()) if distances is not None else None,
            "top_seam_sdr": float(sdr[self.caps["T"]].mean() / normalization),
            "top_seam_lpips": float(distances[self.caps["T"]].mean()) if distances is not None else None,
            "bottom_seam_sdr": float(sdr[self.caps["Btm"]].mean() / normalization),
            "bottom_seam_lpips": float(distances[self.caps["Btm"]].mean()) if distances is not None else None,
            "bleed_fill_pct": self._bleed_fill(rgb, lab_faces),
            "stripe_index": stripe,
            "back_bottom_stripe_index": back_bottom_stripe,
        }

    def top_bottom_orientation(self, texture):
        rgb = self._native_atlas(texture)
        scores = {}
        for name, prefix in (("T", "top"), ("Btm", "bottom")):
            face = self.native_topology["faces"][name]
            x0, y0, x1, y1 = face["rect"]
            upright = np.rot90(rgb[y0:y1, x0:x1], face["upright_rotation"] // 90)
            image = Image.fromarray(np.ascontiguousarray(upright))
            scores[f"{prefix}_upright_rate"] = self.learned.clip_scores([image], [None])[0]
        return scores
