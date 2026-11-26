"""Task 13 ILVR and texture injection over the reviewed G1 cap geometry."""

import time

import numpy as np
import torch

from .v3_capfix_geometry import cap_at_original_scale
from .v3_capgeom import CapGeomEngine
from .v3_capgeom_init import harmonic_fill, sdedit_latents, sdedit_schedule
from .v3_captexture_ilvr import ILVRCapMixin
from .v3_circular import full_vae_canvas
from .v3_inpainting import prepare_inpaint_condition


class CapTextureEngine(ILVRCapMixin, CapGeomEngine):
    def _injected_sdedit(self, layout, initial_rgb, strength, generator, text, pooled):
        pixels = torch.as_tensor(initial_rgb, device=self.device).permute(2, 0, 1)[None]
        initial = self.encode(pixels)
        scheduler = self.pipe.scheduler
        steps = int(self.config["steps"])
        start, timesteps = sdedit_schedule(scheduler, steps, strength, self.device)
        latent, _ = sdedit_latents(scheduler, initial, timesteps[0], generator)
        height, width = layout.height_px, layout.width_px
        condition = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                (height, width, 0, 0, height, width), dtype=self.dtype, device=self.device
            )[None].expand(2, -1),
        }
        first_projection = round(steps * (1 - float(self.config.get("projection_tail", 0.30))))
        interval = int(self.config.get("projection_interval", 3))
        project = self.config.get("projection_strength", 0) > 0
        projection_steps = []
        for index, timestep in enumerate(timesteps, start):
            epsilon = self._predict(latent, timestep, text, condition, None)
            if project and index >= first_projection and (index - first_projection) % interval == 0:
                epsilon = self.color_constraint(latent, epsilon, timestep)
                projection_steps.append(index)
            latent = scheduler.step(epsilon, timestep, latent, return_dict=False)[0]
        return latent, {
            "strength": strength, "full_steps": steps, "start_index": start,
            "executed_steps": len(timesteps), "first_timestep": int(timesteps[0]),
            "projection_steps": projection_steps,
        }

    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        factor = candidate.get("ilvr_factor")
        inject = candidate.get("inject_texture")
        if not (factor or inject):
            return super()._candidate_cap(layout, prepared, generator, network, text, pooled, routing, candidate)
        if (bool(factor) == bool(inject) or not candidate.get("tight") or not candidate.get("inpaint")
                or prepared.style_bank or candidate.get("repaint")):
            raise ValueError("A texture variant can change only the tightly registered G1 inpainting cap.")
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
                if factor:
                    latent, extra = self._ilvr_denoise(
                        (layout.width_px, layout.height_px), generator, text, pooled,
                        harmonic, factor, candidate["ilvr_fraction"], core_rect=layout.center_rect,
                    )
                    proof.update({"mode": "ilvr", **extra})
                else:
                    from .v3_captexture_inject import inject_texture

                    initial, injection = inject_texture(harmonic, rgb, layout)
                    latent, extra = self._injected_sdedit(
                        layout, initial, candidate["strength"], generator, text, pooled
                    )
                    proof.update({"mode": "injected_sdedit", "texture_injection": injection, **extra})
                return cap_at_original_scale(layout, self._rgb(latent)), proof
        finally:
            self._active_inpaint = None
