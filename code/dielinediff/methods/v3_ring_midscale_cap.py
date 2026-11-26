"""Task 17's scale ablation alongside the hash-bound Task 15 implementation."""

import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from .v3_capfix_geometry import _pixel_rect, _span, _strip, cap_at_original_scale
from .v3_capgeom_init import harmonic_fill
from .v3_captexture_inject import _float_rgb, _inward_view
from .v3_circular import full_vae_canvas
from .v3_geometry import EDGES, LATENT_SCALE
from .v3_inpainting import prepare_inpaint_condition
from .v3_uniform_cap import UniformTextureEngine
from .v3_uniform_texture import (
    UNIFORM_TEXTURE_PARAMETERS,
    _extend,
    _gradient_energy,
    _phase_fractions,
    _rms,
    _smoothstep,
)


def inject_texture_at_scale(harmonic_rgb, frozen_rgb, layout, *, sigma_px_at_scale_1):
    """Return a float32 SDEdit prior with a distance-weighted full-core residual."""
    if not np.isfinite(sigma_px_at_scale_1) or sigma_px_at_scale_1 <= 0:
        raise ValueError("Texture high-pass sigma must be positive and finite")
    p = {**UNIFORM_TEXTURE_PARAMETERS, "highpass_sigma_px_at_scale_1": float(sigma_px_at_scale_1)}
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


class MidscaleTextureEngine(UniformTextureEngine):
    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        if "texture_sigma_px_at_scale_1" not in candidate:
            return super()._candidate_cap(layout, prepared, generator, network, text, pooled, routing, candidate)
        if (not candidate["tight"] or not candidate["inpaint"] or candidate["repaint"]
                or candidate.get("inject_texture") or candidate.get("ilvr_factor") or prepared.style_bank):
            raise ValueError("Uniform texture requires the registered G1 inpainting geometry")
        rgb = np.asarray(layout.read_frozen_ring_rgb(prepared.image)).copy()
        mask = layout.known_mask.repeat(8, 0).repeat(8, 1)
        try:
            with full_vae_canvas(self.pipe.vae), routing:
                pixels = torch.as_tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1)[None] / 255
                paint, known = prepare_inpaint_condition(pixels, layout.known_mask, self.encode)
                self._active_inpaint = network, paint, known
                started = time.perf_counter()
                harmonic, proof = harmonic_fill(rgb, mask)
                proof["harmonic_seconds"] = time.perf_counter() - started
                initial, diagnostics = inject_texture_at_scale(
                    harmonic, rgb, layout, sigma_px_at_scale_1=candidate["texture_sigma_px_at_scale_1"])
                latent, schedule = self._injected_sdedit(
                    layout, initial, candidate["strength"], generator, text, pooled)
                image = cap_at_original_scale(layout, self._rgb(latent))
                return image, {**proof, "mode": "uniform_texture_sdedit", "texture_injection": diagnostics, **schedule}
        finally:
            self._active_inpaint = None
