"""Task 08 cap-only candidates on the unchanged v3 ring sampler."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch

from .sdxl import NEGATIVE, STYLE
from .v3 import DieLineDiffV3
from .v3_attention import RegionRouting, region_attention, ring_style_attention
from .v3_capfix_geometry import build_capfix_layout, cap_at_original_scale
from .v3_circular import full_vae_canvas


@dataclass(frozen=True)
class PreparedRing:
    """Reusable result at the exact boundary between the ring and cap stages."""

    layout: object
    frozen_latent: torch.Tensor
    image: Image.Image
    rng_state: torch.Tensor
    style_bank: dict | None


def cap_prompts(brief, templates):
    """Render the same continuation and emblem templates for each brief."""
    product = brief.get("product_name_zh")
    side = brief["side_prompt"]
    motif = brief["top_prompt"].split(",", 1)[0].replace("upright ", "")
    if product:
        side = side.replace(product, "")
        motif = motif.replace(product, "")
    style = STYLE + (", wraparound artwork" if brief["continuous"] else "")
    values = {"style": style, "side_prompt": side, "top_prompt": motif,
              "main_color": brief["main_color"]}
    continuation = templates["continuation"].format_map(values).strip()
    emblem = templates["emblem"].format_map({**values, "continuation": continuation}).strip()
    if not continuation or not emblem:
        raise ValueError("Both cap prompts must expand to nonempty text")
    return continuation, emblem


def cap_region_ids(layout, prompt_mode, outer_fraction):
    """Label the narrow centered emblem region; all edge and free bleed slots continue the ring."""
    if prompt_mode not in ("plain", "emblem"):
        raise ValueError(f"Unsupported regional cap prompt mode {prompt_mode}")
    if not 0.25 <= outer_fraction < 0.5:
        raise ValueError("Cap continuation must occupy at least 25% of the short side inward from each edge")
    mask = np.zeros(layout.latent_shape, dtype=np.uint8)
    if prompt_mode == "plain":
        return mask
    x0, y0, x1, y1 = layout.center_rect
    border = int(np.ceil(min(x1 - x0, y1 - y0) * outer_fraction))
    if 2 * border >= min(x1 - x0, y1 - y0):
        raise ValueError("Selected cap edge margin leaves no center region")
    mask[y0 + border:y1 - border, x0 + border:x1 - border] = 1
    return mask


class CapFixEngine(DieLineDiffV3):
    """Run cap candidates while treating the prepared ring and its random state as read-only."""

    _active_inpaint = None
    _inpaint_model = None

    def _base_unet(self):
        if self._inpaint_model is not None:
            self._inpaint_model.load("cpu")
        return self.pipe.unet.to(self.device)

    @torch.inference_mode()
    def prepare_ring(self, brief, seed, ring_size=(1280, 1024)):
        self._base_unet()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        contexts, pooled = self._conditions(brief)
        ring, _, bank, frozen, pixels = self._prepare_ring(contexts, pooled, generator, ring_size)
        return PreparedRing(ring, frozen.detach().cpu(), pixels, generator.get_state().cpu(), bank)

    def _cap_conditions(self, brief, layout, prompt_mode):
        templates = self.config["capfix_prompts"]
        labels = cap_region_ids(layout, prompt_mode, float(templates.get("outer_fraction", 0.25)))
        continuation, emblem = cap_prompts(brief, templates)
        prompts = [continuation] if prompt_mode == "plain" else [continuation, emblem]
        positive, negative, pooled, negative_pooled = self.pipe.encode_prompt(
            prompt=prompts, negative_prompt=[NEGATIVE] * len(prompts), device=self.device,
            do_classifier_free_guidance=True,
        )
        contexts = {index: torch.stack((negative[0], positive[index])) for index in range(len(prompts))}
        return contexts, torch.stack((negative_pooled[0], pooled[0])), RegionRouting(labels), prompts

    def _open_inpaint(self):
        from .v3_inpainting import InpaintUNet

        self.pipe.unet.to("cpu")
        if self._inpaint_model is None:
            self._inpaint_model = InpaintUNet(self.models, self.dtype)
        return self._inpaint_model.load(self.device)

    def _predict(self, latent, timestep, text, condition, control):
        if self._active_inpaint is None:
            return super()._predict(latent, timestep, text, condition, control)
        from .v3_inpainting import predict_inpaint

        network, paint_mask, masked = self._active_inpaint
        scheduler = self.pipe.scheduler
        scaled = scheduler.scale_model_input(latent.expand(2, -1, -1, -1), timestep)
        output = predict_inpaint(network, scaled, timestep, text, condition, paint_mask, masked)
        unconditioned, conditioned = output.chunk(2)
        return unconditioned + self.config["guidance"] * (conditioned - unconditioned)

    @contextmanager
    def _repaint_once(self, shape, generator, text, pooled, mask):
        """Revisit one middle DDIM step in a cap without changing the shared scheduler code."""
        scheduler = self.pipe.scheduler
        original = scheduler.step
        repaint_index = int(self.config.get("repaint_index", 14))
        steps = int(self.config["steps"])
        if not 0 <= repaint_index < steps - 1:
            raise ValueError("The cap repaint step must have a following DDIM timestep")
        width, height = shape
        condition = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                (height, width, 0, 0, height, width), dtype=self.dtype, device=self.device
            )[None].expand(2, -1),
        }
        index = -1

        def step(prediction, timestep, current, *args, **kwargs):
            nonlocal index
            index += 1
            result = original(prediction, timestep, current, *args, **kwargs)
            if index != repaint_index:
                return result
            after = result[0]
            previous = scheduler.timesteps[index + 1]
            alpha = scheduler.alphas_cumprod
            ratio = (alpha[int(timestep)] / alpha[int(previous)]).to(after.device).clamp(0, 1)
            noise = torch.randn(after.shape, generator=generator, device=after.device, dtype=after.dtype)
            retry = ratio.sqrt() * after + (1 - ratio).sqrt() * noise
            if mask is not None:
                # current already contains the exact frozen strip at this timestep.
                retry = torch.where(mask, current, retry)
            repeated = self._predict(retry, timestep, text, condition, None)
            start = round(steps * (1 - float(self.config.get("projection_tail", 0.30))))
            interval = int(self.config.get("projection_interval", 3))
            if self.config.get("projection_strength", 0) > 0 and index >= start and (index - start) % interval == 0:
                repeated = self.color_constraint(retry, repeated, timestep)
            return original(repeated, timestep, retry, *args, **kwargs)

        scheduler.step = step
        try:
            yield
        finally:
            scheduler.step = original

    def _cap_denoise(self, shape, generator, text, pooled, frozen, mask, repaint):
        context = self._repaint_once(shape, generator, text, pooled, mask) if repaint else nullcontext()
        with context:
            return self._denoise(shape, generator, text, pooled, frozen=frozen, mask=mask)

    @torch.inference_mode()
    def generate_caps(self, brief, prepared, candidate):
        mode = candidate.get("prompt_mode", "original")
        cap_scale = candidate.get("cap_scale", 1)
        inpaint = bool(candidate.get("inpaint", False))
        repaint = bool(candidate.get("repaint", False))
        ring = prepared.layout
        ring_size = (ring.width_px, ring.height_px)
        _, _, originals = self._ring_layout(ring_size)
        generator = torch.Generator(device=self.device)
        generator.set_state(prepared.rng_state.cpu())
        frozen_ring = prepared.frozen_latent.to(device=self.device, dtype=self.dtype)
        cap_info = {}
        generated_caps = {}
        network = self._open_inpaint() if inpaint else self._base_unet()
        if inpaint:
            # Blended Latent moves the DDIM coefficients through add_noise before its first step.
            scheduler = self.pipe.scheduler
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        if mode == "original":
            original_text, original_pooled = self._conditions(brief)
        for label, face in ((5, "T"), (6, "Btm")):
            layout = build_capfix_layout(originals[face], ring, int(self.config.get("cap_band", 16)), cap_scale)
            if mode == "original":
                text, pooled = original_text[label], original_pooled
                routing_context, prompts = nullcontext(), []
            else:
                contexts, pooled, routing, prompts = self._cap_conditions(brief, layout, mode)
                text = contexts[0]
                routing_context = region_attention((network,), routing, contexts)
            anchor, mask = None, None
            if inpaint:
                from .v3_inpainting import prepare_inpaint_condition

                rgb = np.array(layout.read_frozen_ring_rgb(prepared.image))
                pixels = torch.as_tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1)[None] / 255
                with full_vae_canvas(self.pipe.vae):
                    paint_mask, masked = prepare_inpaint_condition(pixels, layout.known_mask, self.encode)
                self._active_inpaint = network, paint_mask, masked
            else:
                anchor = layout.read_frozen_ring(frozen_ring)
                mask = torch.as_tensor(np.array(layout.known_mask), device=self.device)[None, None]
            bank = prepared.style_bank
            style_context = ring_style_attention(network, bank, capture=False) if bank else nullcontext()
            try:
                with full_vae_canvas(self.pipe.vae), routing_context, style_context:
                    latent = self._cap_denoise(
                        (layout.width_px, layout.height_px), generator, text, pooled, anchor, mask, repaint,
                    )
                    generated_caps[face] = cap_at_original_scale(layout, self._rgb(latent))
            finally:
                self._active_inpaint = None
            x0, y0, x1, y1 = layout.center_rect
            cap_info[face] = {
                "size_px": [layout.width_px, layout.height_px],
                "core_size_px": [(x1 - x0) * 8, (y1 - y0) * 8],
                "known_latents": int(layout.known_mask.sum()),
                "prompts": prompts,
            }

        image, support = self.geometry.compose_atlas(
            self.topology, self.region_map, self.bleed_owner, ring, prepared.image, generated_caps,
            return_support=True,
        )
        metadata = {
            "candidate": dict(candidate),
            "atlas_size_px": list(image.size),
            "ring_size_px": list(ring_size),
            "caps": cap_info,
            "repaint_index": int(self.config.get("repaint_index", 14)) if repaint else None,
            "atlas_support": self.geometry.compose_support_counts(self.region_map, self.bleed_owner, support),
        }
        self.last_run = metadata
        return image, metadata
