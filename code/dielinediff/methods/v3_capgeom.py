"""Cap-only geometry and SDEdit candidates using the immutable Task 07 ring."""

from contextlib import nullcontext
import time

import numpy as np
import torch

from .v3 import conditioned_latent
from .v3_attention import region_attention, ring_style_attention
from .v3_capfix import CapFixEngine
from .v3_capfix_geometry import build_capfix_layout, cap_at_original_scale
from .v3_capgeom_geometry import build_capgeom_layout, compose_capgeom_atlas
from .v3_capgeom_init import harmonic_fill, sdedit_latents, sdedit_schedule
from .v3_circular import full_vae_canvas
from .v3_inpainting import prepare_inpaint_condition


class CapGeomEngine(CapFixEngine):
    """Extend the approved C3 cap stage with tight registration and an optional SDEdit prior."""

    def _sdedit(self, layout, rgb, strength, generator, text, pooled, frozen, mask):
        started = time.perf_counter()
        known_pixels = np.repeat(np.repeat(layout.known_mask, 8, axis=0), 8, axis=1)
        initial_rgb, proof = harmonic_fill(rgb, known_pixels)
        elapsed = time.perf_counter() - started
        pixels = torch.from_numpy(initial_rgb).to(device=self.device).permute(2, 0, 1)[None]
        initial = self.encode(pixels)
        scheduler = self.pipe.scheduler
        steps = int(self.config["steps"])
        start, timesteps = sdedit_schedule(scheduler, steps, strength, self.device)
        latent, noise = sdedit_latents(scheduler, initial, timesteps[0], generator)
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
            if frozen is not None:
                latent = conditioned_latent(latent, frozen, noise, mask, scheduler, timestep)
            epsilon = self._predict(latent, timestep, text, condition, None)
            if project and index >= first_projection and (index - first_projection) % interval == 0:
                epsilon = self.color_constraint(latent, epsilon, timestep)
                projection_steps.append(index)
            latent = scheduler.step(epsilon, timestep, latent, return_dict=False)[0]
        if frozen is not None:
            latent = conditioned_latent(latent, frozen, noise, mask, scheduler, None)
        details = {**proof, "harmonic_seconds": elapsed, "strength": strength,
                   "full_steps": steps, "start_index": start, "executed_steps": len(timesteps),
                   "first_timestep": int(timesteps[0]), "projection_steps": projection_steps}
        return latent, details

    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        inpaint = bool(candidate.get("inpaint", False))
        strength = candidate.get("strength")
        repaint = bool(candidate.get("repaint", False))
        rgb = np.array(layout.read_frozen_ring_rgb(prepared.image)) if inpaint or strength is not None else None
        frozen, mask = None, None
        shape = (layout.width_px, layout.height_px)
        old_bank = prepared.style_bank
        style = ring_style_attention(network, old_bank, capture=False) if old_bank else nullcontext()
        try:
            with full_vae_canvas(self.pipe.vae), routing, style:
                if inpaint:
                    pixels = torch.as_tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1)[None] / 255
                    paint, known = prepare_inpaint_condition(pixels, layout.known_mask, self.encode)
                    self._active_inpaint = network, paint, known
                else:
                    ring = prepared.frozen_latent.to(device=self.device, dtype=self.dtype)
                    frozen = layout.read_frozen_ring(ring)
                    mask = torch.as_tensor(np.array(layout.known_mask), device=self.device)[None, None]
                diagnostics = None
                if strength is None:
                    latent = self._cap_denoise(shape, generator, text, pooled, frozen, mask, repaint)
                else:
                    context = self._repaint_once(shape, generator, text, pooled, mask) if repaint else nullcontext()
                    with context:
                        latent, diagnostics = self._sdedit(
                            layout, rgb, float(strength), generator, text, pooled, frozen, mask
                        )
                return cap_at_original_scale(layout, self._rgb(latent)), diagnostics
        finally:
            self._active_inpaint = None

    @torch.inference_mode()
    def generate_caps(self, brief, prepared, candidate):
        tight = bool(candidate.get("tight", False))
        if not tight and candidate.get("strength") is None:
            return super().generate_caps(brief, prepared, candidate)

        mode = candidate.get("prompt_mode", "original")
        scale = int(candidate.get("cap_scale", 1))
        band = int(self.config.get("cap_band", 16))
        inpaint = bool(candidate.get("inpaint", False))
        ring = prepared.layout
        ring_size = ring.width_px, ring.height_px
        _, _, original = self._ring_layout(ring_size)
        generator = torch.Generator(device=self.device)
        generator.set_state(prepared.rng_state.cpu())
        network = self._open_inpaint() if inpaint else self._base_unet()
        if inpaint:
            scheduler = self.pipe.scheduler
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        if mode == "original":
            original_text, original_pooled = self._conditions(brief)

        caps = {}
        statistics = {}
        build_layout = build_capgeom_layout if tight else build_capfix_layout
        for label, face in ((5, "T"), (6, "Btm")):
            layout = build_layout(original[face], ring, band, scale)
            if mode == "original":
                text, pooled = original_text[label], original_pooled
                routing, prompts = nullcontext(), []
            else:
                contexts, pooled, router, prompts = self._cap_conditions(brief, layout, mode)
                text = contexts[0]
                routing = region_attention((network,), router, contexts)
            caps[face], proof = self._candidate_cap(
                layout, prepared, generator, network, text, pooled, routing, candidate
            )
            x0, y0, x1, y1 = layout.center_rect
            statistics[face] = {
                "size_px": [layout.width_px, layout.height_px],
                "core_size_px": [(x1 - x0) * 8, (y1 - y0) * 8],
                "known_latents": int(layout.known_mask.sum()), "prompts": prompts,
                "sdedit": proof,
            }

        compose = compose_capgeom_atlas if tight else self.geometry.compose_atlas
        image, support = compose(self.topology, self.region_map, self.bleed_owner, ring,
                                 prepared.image, caps, return_support=True)
        result = {
            "candidate": dict(candidate), "atlas_size_px": list(image.size),
            "ring_size_px": list(ring_size), "caps": statistics,
            "bleed_mode": "frozen_ring_3mm" if tight else "free_cap_legacy",
            "atlas_support": self.geometry.compose_support_counts(self.region_map, self.bleed_owner, support),
        }
        self.last_run = result
        return image, result
