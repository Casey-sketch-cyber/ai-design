"""Task 18 cap initialization and the exact registered SDEdit suffix."""

import os
import time

import numpy as np
import torch

from .v3_cap_energy import inject_energy_matched
from .v3_capfix_geometry import cap_at_original_scale
from .v3_capgeom_init import harmonic_fill
from .v3_circular import full_vae_canvas
from .v3_inpainting import prepare_inpaint_condition
from .v3_ring_midscale_cap import MidscaleTextureEngine, inject_texture_at_scale
from ..e01b import file_hash
from ..e02_cap_energy_sources import initial_paths


class EnergyCapEngine(MidscaleTextureEngine):
    def generate_caps(self, brief, prepared, candidate):
        self.initial_caps = {}
        return super().generate_caps(brief, prepared, candidate)

    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        if (not candidate["tight"] or not candidate["inpaint"] or candidate["repaint"]
                or candidate["aligned_attention"] or prepared.style_bank):
            raise ValueError("Task 18 uses the reviewed tight two-cap inpainting geometry")
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
                started = time.perf_counter()
                if candidate["match_energy"]:
                    dimensions = self.topology["box_mm"]
                    settings = {**self.config["energy_match"], "core_mm": (dimensions["length"], dimensions["width"])}
                    initial, diagnostics = inject_energy_matched(harmonic, rgb, layout, settings=settings)
                else:
                    initial, diagnostics = inject_texture_at_scale(
                        harmonic, rgb, layout, sigma_px_at_scale_1=candidate["texture_sigma_px_at_scale_1"])
                proof["texture_initialization_seconds"] = time.perf_counter() - started
                self.initial_caps[layout.face] = initial
                if candidate["differential"]:
                    from .v3_capenergy_differential import differential_sdedit
                    latent, steps = differential_sdedit(
                        self, layout, initial, generator, text, pooled, settings=self.config["e4"])
                    mode = "differential_sdedit"
                else:
                    latent, steps = self._injected_sdedit(
                        layout, initial, candidate["strength"], generator, text, pooled)
                    mode = "energy_matched_sdedit" if candidate["match_energy"] else "e0_sdedit"
                cap = cap_at_original_scale(layout, self._rgb(latent))
                return cap, {**proof, "mode": mode, "texture_injection": diagnostics, **steps}
        finally:
            self._active_inpaint = None


def capture_initial_caps(engine, atlas):
    """Write the exact pre-VAE float images separately from the timed generation."""
    evidence = {}
    for face, path in initial_paths(atlas).items():
        pixels = engine.initial_caps[face]
        if pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Task 18 needs the exact float32 RGB input of each cap")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as stream:
            np.save(stream, pixels, allow_pickle=False)
        tmp.replace(path)
        evidence[face] = {"filename": path.name, "sha256": file_hash(path),
                          "shape": list(pixels.shape), "dtype": str(pixels.dtype)}
    return evidence
