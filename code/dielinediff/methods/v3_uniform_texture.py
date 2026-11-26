"""Build a full-core cap texture prior from the four frozen G1 neighbor strips."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .v3_capfix_geometry import _pixel_rect, _span, _strip
from .v3_captexture_inject import _float_rgb, _inward_view
from .v3_geometry import EDGES, LATENT_SCALE


UNIFORM_TEXTURE_PARAMETERS = {
    "highpass_sigma_px_at_scale_1": 4.0,
    "distance_offset_px": 1.0,
    "center_envelope_floor": 0.60,
    "phase_lock_px_at_scale_1": 8,
    "phase_seed": 15,
    "phase_shift_fraction_min": 0.18,
    "phase_shift_fraction_max": 0.32,
    "coarse_sigma_multiplier": 2.0,
    "normal_extension": "reflect",
    "tangential_extension": "reflect_bilinear",
    "blend": "inverse_distance_l2",
    "per_pixel_bound": "max_abs_sample",
    "envelope": "floor_plus_cosine_squared",
}


def _smoothstep(value):
    unit = np.clip(value, 0, 1)
    return unit * unit * (3 - 2 * unit)


def _phase_fractions():
    p = UNIFORM_TEXTURE_PARAMETERS
    magnitude = np.random.default_rng(p["phase_seed"]).uniform(
        p["phase_shift_fraction_min"], p["phase_shift_fraction_max"], len(EDGES),
    )
    return magnitude * (1, -1, 1, -1)


def _extend(texture, depth, fraction, lock):
    """Reflect depth and gradually shift the tangential sampling phase."""
    along = np.arange(texture.shape[1], dtype=np.float32)[None, :]
    inward = np.arange(depth, dtype=np.float32)[:, None]
    phase = _smoothstep((inward - lock + 1) / max(1, depth - lock))
    offset = phase * float(fraction * (texture.shape[1] - 1))
    tangential = along + offset
    coordinates = np.stack((np.broadcast_to(inward, tangential.shape), tangential))
    return np.stack([
        map_coordinates(texture[..., c], coordinates, order=1, mode="reflect", prefilter=False)
        for c in range(texture.shape[2])
    ], axis=-1)


def _gradient_energy(rgb, samples):
    values = []
    for axis in (0, 1):
        ordered = np.moveaxis(samples, axis, 0)
        pairs = ordered[:-1] & ordered[1:]
        values.append(float(np.square(np.diff(rgb, axis=axis)[np.moveaxis(pairs, 0, axis)]).mean()))
    return sum(values) / len(values)


def _rms(values):
    return float(np.sqrt(np.square(values).mean()))


def inject_uniform_texture(harmonic_rgb, frozen_rgb, layout):
    """Return a float32 SDEdit prior with a distance-weighted full-core residual."""
    p = UNIFORM_TEXTURE_PARAMETERS
    harmonic, frozen = _float_rgb(harmonic_rgb), _float_rgb(frozen_rgb)
    shape = (layout.height_px, layout.width_px, 3)
    if harmonic.shape != shape or frozen.shape != shape:
        raise ValueError("Uniform texture inputs must match the RGB cap canvas")
    if not np.isfinite(harmonic).all() or harmonic.min() < 0 or harmonic.max() > 1:
        raise ValueError("Harmonic RGB must be finite and normalized to [0, 1]")
    if any(layout.bleed_cells.values()):
        raise ValueError("Uniform texture requires the attached G1 neighbors")
    mask = layout.known_mask.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)
    if mask.shape != shape[:2] or not np.array_equal(harmonic[mask], frozen[mask]):
        raise ValueError("Harmonic RGB must already preserve the frozen boundary exactly")

    rect = _pixel_rect(layout.center_rect)
    x0, y0, x1, y1 = rect
    band = int(layout.band * layout.scale * LATENT_SCALE)
    height, width = y1 - y0, x1 - x0
    if (band < 2 or min(height, width) < 8 or not (band <= x0 < x1 <= layout.width_px - band)
            or not (band <= y0 < y1 <= layout.height_px - band) or mask[y0:y1, x0:x1].any()):
        raise ValueError("Invalid cap core or frozen strip bounds")

    sigma = float(p["highpass_sigma_px_at_scale_1"] * layout.scale)
    coarse_sigma = sigma * p["coarse_sigma_multiplier"]
    lock_px = int(p["phase_lock_px_at_scale_1"] * layout.scale)
    core = harmonic[y0:y1, x0:x1]
    accumulated = np.zeros_like(core)
    peak = np.zeros_like(core)
    norm_squares = np.zeros(core.shape[:2], dtype=np.float32)
    source_rms = {}
    phases = {}
    for edge, fraction in zip(EDGES, _phase_fractions(), strict=True):
        coordinates = _strip(rect, edge, _span(rect, edge), band, 0)
        if not mask[coordinates].all():
            raise ValueError(f"The {edge} source strip includes unfrozen pixels")
        source = frozen[coordinates]
        highpass = source - gaussian_filter(source, sigma=(sigma, sigma, 0), mode="reflect")
        depth = height if edge in ("top", "bottom") else width
        extended = _extend(highpass, depth, fraction, lock_px)
        distances = np.arange(depth, dtype=np.float32)[:, None]
        weight = 1 / (distances + p["distance_offset_px"])
        _inward_view(accumulated, edge, depth)[:] += weight[..., None] * extended
        _inward_view(norm_squares, edge, depth)[:] += np.square(weight)
        cap = _inward_view(peak, edge, depth)
        np.maximum(cap, np.abs(extended), out=cap)
        phases[edge] = float(fraction)
        source_rms[edge] = _rms(highpass)

    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    nearest = np.minimum(np.minimum(y, height - 1 - y)[:, None], np.minimum(x, width - 1 - x)[None, :])
    floor = p["center_envelope_floor"]
    envelope = floor + (1 - floor) * np.square(np.cos((np.pi / 2) * nearest / nearest.max()))
    blended = accumulated / np.sqrt(norm_squares)[..., None]
    # Mixing and shifting can leak coarse energy. Keep the near-fold phase,
    # then remove the blurred component gradually outside the phase-lock region.
    coarse = gaussian_filter(blended, sigma=(coarse_sigma, coarse_sigma, 0), mode="reflect")
    coarse_gate = _smoothstep((nearest - lock_px + 1) / lock_px)
    blended = np.clip(blended - coarse_gate[..., None] * coarse, -peak, peak)
    injected = core + envelope[..., None] * blended
    result = harmonic.copy()
    actual = result[y0:y1, x0:x1]
    np.clip(injected, 0, 1, out=actual)
    residual = actual - core
    low_residual = gaussian_filter(residual, sigma=(coarse_sigma, coarse_sigma, 0), mode="reflect")
    edge_zone = nearest < min(lock_px, max(2, min(height, width) // 3))
    center = np.zeros_like(edge_zone)
    center[
        height // 3:height - height // 3,
        width // 3:width - width // 3,
    ] = True
    edge_rms, center_rms = _rms(residual[edge_zone]), _rms(residual[center])
    return result, {
        "parameters": dict(p),
        "gaussian_sigma_px": sigma,
        "phase_lock_px": lock_px,
        "source_band_px": band,
        "source_phase_fractions": phases,
        "edge_source_residual_rms": source_rms,
        "center_envelope_min": float(envelope.min()),
        "edge_envelope_max": float(envelope.max()),
        "supported_core_pixels": height * width,
        "frozen_pixels": int(mask.sum()),
        "clipped_core_pixels": int(np.any((injected < 0) | (injected > 1), axis=-1).sum()),
        "applied_core_residual_rms": _rms(residual),
        "applied_edge_residual_rms": edge_rms,
        "applied_center_residual_rms": center_rms,
        "applied_center_to_edge_rms": center_rms / edge_rms if edge_rms > 0 else None,
        "applied_coarse_residual_rms": _rms(low_residual),
        "applied_center_coarse_residual_rms": _rms(low_residual[center]),
        "center_rgb_gradient_energy_before": _gradient_energy(core, center),
        "center_rgb_gradient_energy_after": _gradient_energy(actual, center),
        "edge_rgb_gradient_energy_before": _gradient_energy(core, edge_zone),
        "edge_rgb_gradient_energy_after": _gradient_energy(actual, edge_zone),
    }
