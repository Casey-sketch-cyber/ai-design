"""Match interior cap texture energy using the four registered frozen sides."""

import numpy as np
from scipy.ndimage import gaussian_filter

from .v3_capfix_geometry import _pixel_rect, _span, _strip
from .v3_captexture_inject import _float_rgb, _inward_view
from .v3_geometry import EDGES, LATENT_SCALE
from .v3_ring_midscale_cap import inject_texture_at_scale
from .v3_uniform_texture import _gradient_energy, _rms, _smoothstep


ENERGY_MATCH_PARAMETERS = {
    "sigma_px_at_scale_1": 14.0,
    "source_band_mm": 3.0,
    "preserve_edge_mm": 3.0,
    "transition_mm": 6.0,
    "local_rms_sigma_mm": 3.0,
    "distance_offset_mm": 1.0,
    "max_gain": 4.0,
    "rms_floor": 1.0e-6,
}


def _edge_distances(shape, core_mm):
    height, width = shape
    mm_width, mm_height = core_mm
    y = (np.arange(height, dtype=np.float32) + 0.5) * (mm_height / height)
    x = (np.arange(width, dtype=np.float32) + 0.5) * (mm_width / width)
    distances = (y, x[::-1], y[::-1], x)
    nearest = np.minimum(np.minimum(y, y[::-1])[:, None], np.minimum(x, x[::-1])[None, :])
    return dict(zip(EDGES, distances, strict=True)), nearest


def _target_energy(layout, frozen, rect, distances, sigma, parameters):
    shape = (rect[3] - rect[1], rect[2] - rect[0])
    target = np.zeros(shape, dtype=np.float32)
    weights = np.zeros_like(target)
    source_rms = {}
    source_rows = {}
    band = layout.band * layout.scale * LATENT_SCALE
    for edge, distance in distances.items():
        coordinates = _strip(rect, edge, _span(rect, edge), band, 0)
        source = frozen[coordinates]
        step_mm = 2 * distance.min()
        depth = np.count_nonzero((np.arange(band) + 0.5) * step_mm < parameters["source_band_mm"])
        if depth == 0:
            raise ValueError(f"No pixel centers inside the physical {edge} reference band")
        residual = source - gaussian_filter(source, sigma=(sigma, sigma, 0), mode="reflect")
        source_rms[edge] = _rms(residual[:depth])
        source_rows[edge] = int(depth)
        weight = (distance[:, None] + parameters["distance_offset_mm"]) ** -2
        _inward_view(target, edge, len(distance))[:] += weight * source_rms[edge] ** 2
        _inward_view(weights, edge, len(distance))[:] += weight
    return target / weights, source_rms, source_rows


def _interior_gain(residual, sigma, target, parameters):
    # Only add band-limited detail. Keep the slow color/form component of E0.
    carrier = residual - gaussian_filter(residual, sigma=(*sigma, 0), mode="reflect")
    moments = np.stack((
        np.square(residual).mean(axis=-1),
        (residual * carrier).mean(axis=-1),
        np.square(carrier).mean(axis=-1),
    ), axis=-1)
    current, covariance, variance = np.moveaxis(
        gaussian_filter(moments, sigma=(*sigma, 0), mode="reflect"), -1, 0,
    )
    gap = np.maximum(0, target - current)
    # Solve local E[(residual + gain*carrier)^2] = target with gain >= 0.
    # A flat source has no carrier and is kept flat.
    numerator = np.sqrt(np.square(covariance) + variance * gap) - covariance
    gain = np.divide(numerator, variance, out=np.zeros_like(variance),
                     where=(gap > 0) & (variance > parameters["rms_floor"] ** 2))
    return carrier, np.clip(gain, 0, parameters["max_gain"] - 1)


def inject_energy_matched(harmonic_rgb, frozen_rgb, layout, *, settings):
    """Return the E1 float32 prior and observed physical-band energy diagnostics."""
    parameters = {**ENERGY_MATCH_PARAMETERS, **settings}
    core_mm = np.asarray(parameters.pop("core_mm"), dtype=np.float64)
    values = np.asarray(list(parameters.values()), dtype=np.float64)
    if (set(parameters) != set(ENERGY_MATCH_PARAMETERS) or not np.isfinite(values).all()
            or np.any(values <= 0) or parameters["max_gain"] < 1 or core_mm.shape != (2,)
            or not np.isfinite(core_mm).all() or np.any(core_mm <= 2 * parameters["preserve_edge_mm"])):
        raise ValueError("Energy matching needs positive finite settings and cap width/height in mm")

    harmonic, frozen = _float_rgb(harmonic_rgb), _float_rgb(frozen_rgb)
    baseline, base_diagnostics = inject_texture_at_scale(
        harmonic, frozen, layout, sigma_px_at_scale_1=parameters["sigma_px_at_scale_1"],
    )
    rect = _pixel_rect(layout.center_rect)
    x0, y0, x1, y1 = rect
    core_slice = np.s_[y0:y1, x0:x1]
    core = harmonic[core_slice]
    initial = baseline[core_slice]
    distances, nearest = _edge_distances(core.shape[:2], core_mm)
    highpass_sigma = base_diagnostics["gaussian_sigma_px"]
    targets, source_rms, source_rows = _target_energy(
        layout, frozen, rect, distances, highpass_sigma, parameters,
    )
    local_sigma = tuple(parameters["local_rms_sigma_mm"] * np.array(core.shape[:2]) / core_mm[::-1])
    carrier, gain = _interior_gain(initial - core, local_sigma, targets, parameters)
    protection = nearest < parameters["preserve_edge_mm"]
    ramp = _smoothstep((nearest - parameters["preserve_edge_mm"]) / parameters["transition_mm"])
    unbounded = initial + (gain * ramp)[..., None] * carrier
    result = baseline.copy()
    # Start from the literal E0 RGB and write only interior pixels.
    result[core_slice][~protection] = np.clip(unbounded[~protection], 0, 1)
    actual = result[core_slice]
    residual = actual - core
    new_detail = actual - initial
    h, w = core.shape[:2]
    center = np.zeros((h, w), dtype=bool)
    center[h // 3:h - h // 3, w // 3:w - w // 3] = True
    edge_rms, center_rms = _rms(residual[protection]), _rms(residual[center])
    base_edge_rms, base_center_rms = _rms((initial - core)[protection]), _rms((initial - core)[center])
    normal_means = [np.abs(new_detail.mean(axis=axis)).max() for axis in (0, 1)]
    diagnostics = {
        "parameters": {**parameters, "core_mm": core_mm.tolist()},
        "gaussian_sigma_px": highpass_sigma,
        "local_rms_sigma_yx_px": list(local_sigma),
        "edge_source_residual_rms": source_rms,
        "edge_source_rows": source_rows,
        "target_center_residual_rms": float(np.sqrt(targets[center].mean())),
        "applied_edge_residual_rms": edge_rms,
        "applied_center_residual_rms": center_rms,
        "applied_center_to_edge_rms": center_rms / edge_rms if edge_rms else None,
        "e0_center_to_edge_rms": base_center_rms / base_edge_rms if base_edge_rms else None,
        "protected_edge_pixels": int(protection.sum()),
        "protected_edge_pixels_changed": int(np.any(new_detail[protection] != 0, axis=-1).sum()),
        "frozen_pixels": base_diagnostics["frozen_pixels"],
        "gain_capped_core_pixels": int((gain >= parameters["max_gain"] - 1).sum()),
        "newly_clipped_core_pixels": int(np.any((unbounded < 0) | (unbounded > 1), axis=-1).sum()),
        "additional_coarse_residual_rms": _rms(gaussian_filter(new_detail, sigma=(*local_sigma, 0))),
        "additional_normal_mean_change_max_8bit": float(max(normal_means) * 255),
        "center_rgb_gradient_energy_e0": _gradient_energy(initial, center),
        "center_rgb_gradient_energy_e1": _gradient_energy(actual, center),
    }
    return result, diagnostics
