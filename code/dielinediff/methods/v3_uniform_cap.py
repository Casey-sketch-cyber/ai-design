"""Uniform texture initialization inside the existing tightly registered cap generator."""

import time

import numpy as np
import torch

from .v3_capfix_geometry import cap_at_original_scale
from .v3_capgeom_init import harmonic_fill
from .v3_captexture import CapTextureEngine
from .v3_circular import full_vae_canvas
from .v3_inpainting import prepare_inpaint_condition
from .v3_uniform_texture import inject_uniform_texture


class UniformTextureEngine(CapTextureEngine):
    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        if not candidate.get("uniform_texture"):
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
                initial, diagnostics = inject_uniform_texture(harmonic, rgb, layout)
                latent, schedule = self._injected_sdedit(
                    layout, initial, candidate["strength"], generator, text, pooled)
                image = cap_at_original_scale(layout, self._rgb(latent))
                return image, {**proof, "mode": "uniform_texture_sdedit", "texture_injection": diagnostics, **schedule}
        finally:
            self._active_inpaint = None
