"""Inject physically registered frozen side texture into a harmonic cap prior."""

import numpy as np
from scipy.ndimage import gaussian_filter

from .v3_capfix_geometry import _pixel_rect, _span, _strip
from .v3_geometry import EDGES, LATENT_SCALE


_SIGMA_AT_SCALE_ONE_PX = 4.0


def _float_rgb(rgb):
    pixels = np.asarray(rgb)
    if pixels.dtype == np.uint8:
        return pixels.astype(np.float32) / 255
    if np.issubdtype(pixels.dtype, np.floating):
        return pixels.astype(np.float32, copy=False)
    raise ValueError("Texture injection expects uint8 or floating RGB")


def _inward_view(array, edge, depth):
    """Count inward from a cap edge while keeping its tangential order."""
    if edge == "top":
        return array[:depth]
    if edge == "right":
        return array[:, -depth:][:, ::-1].swapaxes(0, 1)
    if edge == "bottom":
        return array[-depth:][::-1]
    return array[:, :depth].swapaxes(0, 1)


def inject_texture(harmonic_rgb, frozen_rgb, layout):
    """Mirror each G1 neighbor's high-pass residual once into the cap core."""
    harmonic, frozen = _float_rgb(harmonic_rgb), _float_rgb(frozen_rgb)
    expected = (layout.height_px, layout.width_px, 3)
    if harmonic.shape != expected or frozen.shape != expected:
        raise ValueError("Texture and harmonic RGB must match the G1 cap canvas")
    if not np.isfinite(harmonic).all() or harmonic.min() < 0 or harmonic.max() > 1:
        raise ValueError("Harmonic RGB must be finite and normalized to [0, 1]")
    if any(layout.bleed_cells.values()):
        raise ValueError("Texture injection requires G1's attached frozen neighbors")
    mask = np.repeat(np.repeat(layout.known_mask, LATENT_SCALE, axis=0), LATENT_SCALE, axis=1)
    if mask.shape != expected[:2] or not np.array_equal(harmonic[mask], frozen[mask]):
        raise ValueError("Harmonic RGB must already preserve the frozen boundary exactly")

    rect = _pixel_rect(layout.center_rect)
    x0, y0, x1, y1 = rect
    band_px = int(layout.band * layout.scale * LATENT_SCALE)
    if (band_px < 2 or not (band_px <= x0 < x1 <= layout.width_px - band_px)
            or not (band_px <= y0 < y1 <= layout.height_px - band_px)
            or mask[y0:y1, x0:x1].any()):
        raise ValueError("Invalid core or frozen strip extent for texture injection")

    sigma = _SIGMA_AT_SCALE_ONE_PX * layout.scale
    attenuation = (1 + np.cos(np.linspace(0, np.pi, band_px, dtype=np.float32))) * 0.5
    core = harmonic[y0:y1, x0:x1]
    residual_sum = np.zeros_like(core)
    weights = np.zeros(core.shape[:2], dtype=np.float32)
    uncovered = np.ones_like(weights)
    edge_rms = {}

    for edge in EDGES:
        length = _span(rect, edge)
        coordinates = _strip(rect, edge, length, band_px, 0)
        if not mask[coordinates].all():
            raise ValueError(f"The {edge} strip includes unfrozen pixels")
        source = frozen[coordinates]
        highpass = source - gaussian_filter(source, sigma=(sigma, sigma, 0), mode="reflect")
        normal_extent = core.shape[0] if edge in ("top", "bottom") else core.shape[1]
        depth = min(band_px, normal_extent)
        weight = attenuation[:depth, None]
        _inward_view(residual_sum, edge, depth)[:] += weight[..., None] * highpass[:depth]
        _inward_view(weights, edge, depth)[:] += weight
        _inward_view(uncovered, edge, depth)[:] *= 1 - weight
        edge_rms[edge] = float(np.sqrt(np.square(highpass).mean()))

    # A convex mix keeps overlaps from amplifying an edge's residual. The smooth
    # union envelope retains decay where only one reflected strip contributes.
    envelope = 1 - uncovered
    ratio = np.divide(envelope, weights, out=np.zeros_like(weights), where=weights > 0)
    injected = core + residual_sum * ratio[..., None]
    result = harmonic.copy()
    result[y0:y1, x0:x1] = np.clip(injected, 0, 1)
    applied = result[y0:y1, x0:x1] - core
    return result, {
        "filter": "rgb_minus_gaussian",
        "gaussian_sigma_px": float(sigma),
        "mirrored_band_px": band_px,
        "decay": "raised_cosine",
        "overlap": "distance_weighted_mean_with_union_envelope",
        "edge_highpass_rms": edge_rms,
        "applied_core_residual_rms": float(np.sqrt(np.square(applied).mean())),
        "supported_core_pixels": int(np.count_nonzero(weights)),
        "clipped_core_pixels": int(np.any((injected < 0) | (injected > 1), axis=2).sum()),
        "frozen_pixels": int(mask.sum()),
    }
