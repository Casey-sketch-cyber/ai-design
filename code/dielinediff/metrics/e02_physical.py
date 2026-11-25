"""E02 seam and bleed diagnostics on the existing 4 px/mm physical grid."""

import numpy as np
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab

from . import DEFAULT_THRESHOLDS, _aligned_strips, _lpips_pairs, _oriented_strip, _sdr_components
from .physical import LPIPS_MM, PIXELS_PER_MM, SDR_MM, PhysicalEvaluator, stripe_energy

DEFAULT_CONFIG = {
    "m2_band_mm": 4.0,
    "m2_blur_mm": 0.25,
    "m2_salience_quantile": 0.75,
    "m2_min_gradient_rgb": 2.0,
    "m3_flat_std_rgb": 1.0,
    "m3_guard_mm": 0.5,
    "m3_min_normal_delta_rgb": 0.65,
    "m3_min_normal_to_tangent": 0.12,
    "m3_zero_delta_rgb": 0.35,
    "m3_max_zero_fraction": 0.7,
}

_WRAP = frozenset(("L", "B"))
_EDGES = ("top", "bottom", "left", "right")
_D2_BUCKETS = ("pass", "white_halo", "flat_bg_trap", "color_jump", "invalid")


def _seam_key(seam):
    return frozenset((seam["a"]["face"], seam["b"]["face"]))


def _edge_gradient(strip, sigma):
    # One-sided normal at the crease; reflect padding halves its derivative.
    luminance = strip @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = ndimage.gaussian_filter(luminance, sigma=sigma, mode="nearest")
    tangent = ndimage.sobel(luminance, axis=0, mode="nearest")[:, 0] / 8
    normal = ndimage.sobel(luminance, axis=1, mode="nearest")[:, 0] / 4
    return np.stack((tangent, normal), axis=-1)


def edge_continuity(first, second, config=DEFAULT_CONFIG):
    """Strength agreement times positive gradient cosine at salient crease locations."""
    sigma = config["m2_blur_mm"] * PIXELS_PER_MM
    grad_a = _edge_gradient(first, sigma)
    grad_b = _edge_gradient(second, sigma)
    grad_b[:, 1] *= -1  # Both strips have depth increasing inward from the fold.
    mag_a = np.linalg.norm(grad_a, axis=1)
    mag_b = np.linalg.norm(grad_b, axis=1)
    strongest = np.maximum(mag_a, mag_b)
    strongest_positive = strongest[strongest >= config["m2_min_gradient_rgb"]]
    if not len(strongest_positive):
        return None
    cutoff = np.quantile(strongest_positive, config["m2_salience_quantile"])
    salient = strongest >= cutoff
    # min(a,b)/max(a,b) times cosine is dot(a,b)/max(a,b)^2.
    agreement = np.einsum("ij,ij->i", grad_a[salient], grad_b[salient])
    return float(np.clip(agreement / np.square(strongest[salient]), 0, 1).mean())


def _mean_supported(values, valid):
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0).sum(axis=1)
    return np.divide(total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0)


def _rgb_rms(first, second):
    return np.sqrt(np.square(first - second).mean(axis=-1))


class E02PhysicalEvaluator(PhysicalEvaluator):
    """Measure historical physical metrics and their E02 decompositions in one pass."""

    def __init__(self, geometry_dir, *, device="cpu", learned=None, config=None):
        self.config = dict(DEFAULT_CONFIG if config is None else config)
        if self.config.keys() != DEFAULT_CONFIG.keys():
            raise ValueError("E02 physical thresholds must match DEFAULT_CONFIG keys.")
        super().__init__(geometry_dir, device=device, learned=learned)
        native = {_seam_key(seam) for seam in self.creases}
        self.groups = {
            "native_5": np.array([_seam_key(seam) in native for seam in self.seams]),
            "non_native_7": np.array([_seam_key(seam) not in native for seam in self.seams]),
            "wrap_lb": np.array([_seam_key(seam) == _WRAP for seam in self.seams]),
        }
        if tuple(mask.sum() for mask in self.groups.values()) != (5, 7, 1):
            raise ValueError("E02 metrics require five planar folds and seven folded-only seams.")

    def _prepare_bleed(self):
        super()._prepare_bleed()
        target = self.topology["image"]
        native = self.native_topology["image"]
        native_origin = np.array(native["origin_mm"][::-1])
        target_origin = np.array(target["origin_mm"][::-1])
        world = (self.bleed_coordinates.T + 0.5) / native["pixels_per_mm"] + native_origin
        px = np.rint((world - target_origin) * PIXELS_PER_MM - 0.5).astype(int)
        self.ray_depth = round(self.bleed_mm * PIXELS_PER_MM)
        ray_groups = []
        pixel_rays = np.empty(len(px), dtype=np.intp)
        self.edge_slices = []
        offset = 0
        for name, (mask, y, x) in self.bleed_refs.items():
            x0, y0, x1, y1 = self.topology["faces"][name]["rect"]
            y_delta = px[mask, 0] - (y + y0)
            x_delta = px[mask, 1] - (x + x0)
            # A diagonal corner inherits the dominant adjacent cardinal ray;
            # only cardinal pixels contribute to that ray's profile.
            vertical = np.abs(y_delta) >= np.abs(x_delta)
            profile_edge = np.where(vertical, np.where(y_delta < 0, 0, 1), np.where(x_delta < 0, 2, 3))
            for side, edge in enumerate(_EDGES):
                along = x if side < 2 else y
                perpendicular = y_delta if side < 2 else x_delta
                lateral = x_delta if side < 2 else y_delta
                length = x1 - x0 if side < 2 else y1 - y0
                indices = np.flatnonzero(mask)[profile_edge == side]
                tangent = along[profile_edge == side]
                depth = np.abs(perpendicular[profile_edge == side])
                cardinal = (lateral[profile_edge == side] == 0) & (depth > 0) & (depth <= self.ray_depth)
                table = np.full((length, self.ray_depth), len(px), dtype=np.intp)
                table[tangent[cardinal], depth[cardinal] - 1] = indices[cardinal]
                ray_groups.append(table)
                pixel_rays[indices] = offset + tangent
                self.edge_slices.append((name, edge, slice(offset, offset + length)))
                offset += length
        self.rays = np.concatenate(ray_groups)
        self.pixel_rays = pixel_rays
        self.tangent_neighbors = np.arange(offset)
        for _, _, edge_slice in self.edge_slices:
            neighbors = self.tangent_neighbors[edge_slice]
            neighbors[:-1] += 1
            neighbors[-1] -= 1

    def _sample_bleed_lab(self, rgb, lab_faces):
        valid = self.bleed_support > 1e-5
        sampled = np.full((len(valid), 3), 255.0, dtype=np.float32)
        for channel in range(3):
            masked = rgb[:, :, channel].astype(np.float32) * self.native_bleed
            numerator = self._sample_bleed(masked)
            sampled[valid, channel] = numerator[valid] / self.bleed_support[valid]
        reference = np.empty_like(sampled)
        for name, (mask, y, x) in self.bleed_refs.items():
            reference[mask] = lab_faces[name][y, x]
        delta = deltaE_ciede2000(rgb2lab(sampled.clip(0, 255) / 255), reference)
        return sampled, valid, delta

    def _face_is_flat(self, faces):
        flat = np.empty(len(self.rays), dtype=bool)
        for name, edge, edge_slice in self.edge_slices:
            band = _oriented_strip(faces[name], edge, self.ray_depth).astype(np.float32)
            radius = self.ray_depth // 2
            mean = ndimage.uniform_filter1d(band.mean(axis=1), 2 * radius + 1, axis=0, mode="nearest")
            square_mean = ndimage.uniform_filter1d(
                np.square(band).mean(axis=1), 2 * radius + 1, axis=0, mode="nearest",
            )
            local_std = np.sqrt(np.maximum(0, square_mean - np.square(mean)).mean(axis=1))
            # Keep the complete texture neighborhood inside the face at corners.
            center = np.arange(len(band)).clip(radius, len(band) - radius - 1)
            flat[edge_slice] = local_std[center] <= self.config["m3_flat_std_rgb"]
        return flat[self.pixel_rays]

    def _bleed_normal_has_structure(self, sampled, valid):
        config = self.config
        guard = round(config["m3_guard_mm"] * PIXELS_PER_MM)
        table = self.rays[:, guard:self.ray_depth - guard]
        ray_rgb = np.concatenate((sampled, np.zeros((1, 3), dtype=sampled.dtype)))[table]
        ray_valid = np.append(valid, False)[table]
        normal_valid = ray_valid[:, 1:] & ray_valid[:, :-1]
        normal = _rgb_rms(ray_rgb[:, 1:], ray_rgb[:, :-1])
        strength = _mean_supported(normal, normal_valid)
        zero_fraction = _mean_supported(normal <= config["m3_zero_delta_rgb"], normal_valid)
        neighbors = self.tangent_neighbors
        tangent_valid = ray_valid & ray_valid[neighbors]
        tangent = _mean_supported(_rgb_rms(ray_rgb, ray_rgb[neighbors]), tangent_valid)
        has_structure = (
            (normal_valid.sum(axis=1) >= 2)
            & (strength >= config["m3_min_normal_delta_rgb"])
            & (strength >= config["m3_min_normal_to_tangent"] * tangent)
            & (zero_fraction <= config["m3_max_zero_fraction"])
        )
        return has_structure[self.pixel_rays]

    def _measure_bleed(self, rgb, faces, lab_faces):
        if not len(self.bleed_support):
            return {
                "bleed_fill_pct": None, "bleed_fill_v2_pct": None,
                **{f"d2_{name}_pct": None for name in _D2_BUCKETS},
            }
        sampled, valid, delta = self._sample_bleed_lab(rgb, lab_faces)
        thresholds = DEFAULT_THRESHOLDS
        nonwhite = (sampled < thresholds["bleed_white_rgb"]).any(axis=1)
        close = delta <= thresholds["bleed_match_delta_e"]
        changed = delta >= thresholds["bleed_change_delta_e"]
        legacy_pass = valid & nonwhite & close & changed
        m3_pass = valid & nonwhite & close & (
            self._face_is_flat(faces) | (changed & self._bleed_normal_has_structure(sampled, valid))
        )
        # Disjoint legacy categories; a sample outside native bleed support is
        # invalid even when the white fill sentinel reads RGB 255.
        buckets = (legacy_pass, valid & ~nonwhite, valid & nonwhite & close & ~changed,
                   valid & nonwhite & ~close, ~valid)
        return {
            "bleed_fill_pct": float(100 * legacy_pass.mean()),
            "bleed_fill_v2_pct": float(100 * m3_pass.mean()),
            **{f"d2_{name}_pct": float(100 * bucket.mean()) for name, bucket in zip(_D2_BUCKETS, buckets)},
        }

    def measure(self, texture, *, lpips=True):
        rgb = self._native_atlas(texture)
        faces = self._physical_faces(rgb)
        lab_faces = {name: rgb2lab(face.astype(np.float32) / 255) for name, face in faces.items()}
        distances, inside = _sdr_components(
            lab_faces, self.seams, round(SDR_MM * PIXELS_PER_MM), float("-inf"),
        )
        denominator = max(inside, DEFAULT_THRESHOLDS["sdr_floor_delta_e"])
        perceptual = (
            self.learned.seam_lpips(_lpips_pairs(faces, self.seams, round(LPIPS_MM * PIXELS_PER_MM)), per_seam=True)
            if lpips else None
        )
        stripe, back_bottom = stripe_energy(faces, self.creases)
        result = {
            "seam_sdr": float(distances.mean() / denominator),
            "seam_lpips": float(perceptual.mean()) if perceptual is not None else None,
            "inside_dE00": float(inside),
            "sdr_denominator": float(denominator),
            "stripe_index": stripe,
            "back_bottom_stripe_index": back_bottom,
        }
        for name, mask in self.caps.items():
            label = "top" if name == "T" else "bottom"
            result[f"{label}_seam_sdr"] = float(distances[mask].mean() / denominator)
            result[f"{label}_seam_lpips"] = float(perceptual[mask].mean()) if perceptual is not None else None
        for name, mask in self.groups.items():
            absolute = distances[mask].mean()
            result[f"{name}_delta_e00"] = float(absolute)
            result[f"{name}_sdr"] = float(absolute / denominator)
            result[f"{name}_lpips"] = float(perceptual[mask].mean()) if perceptual is not None else None
        width = round(self.config["m2_band_mm"] * PIXELS_PER_MM)
        edges = [edge_continuity(*_aligned_strips(faces, seam, width), self.config)
                 for seam, selected in zip(self.seams, self.groups["non_native_7"]) if selected]
        evidence = [score for score in edges if score is not None]
        result["cross_seam_edge_cont"] = float(sum(evidence) / len(edges)) if evidence else None
        result["m2_salient_seam_count"] = len(evidence)
        result.update(self._measure_bleed(rgb, faces, lab_faces))
        return result
