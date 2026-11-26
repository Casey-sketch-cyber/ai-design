"""Generate the unchanged J0 prior with the J2/J3 denoising variants."""

import time

import numpy as np
import torch

from .v3_capfix_geometry import cap_at_original_scale
from .v3_capgeom_init import harmonic_fill, sdedit_latents, sdedit_schedule
from .v3_circular import full_vae_canvas
from .v3_inpainting import prepare_inpaint_condition
from .v3_joint_cap_blend import blended_patch, denoise_blended, feather_known_mask
from .v3_ring_midscale_cap import inject_texture_at_scale


def generate_refined_cap(engine, layout, prepared, generator, network, text, pooled, routing, candidate):
    use_base = candidate.get("cap_backend", "inpaint") == "base_blend"
    gamma = float(candidate.get("guidance_gamma", 0))
    if (not candidate["tight"] or candidate["repaint"] or candidate["aligned_attention"]
            or prepared.style_bank):
        raise ValueError("Joint-cap refinements require the original tight cap layout")
    blend = engine.config["joint_cap"]["blended"]
    if use_base and (blend["feather_latents"] != 1 or candidate["strength"] != blend["strength"]):
        raise ValueError("J2 must use the registered SDEdit strength and one-cell closing feather")
    rgb = np.asarray(layout.read_frozen_ring_rgb(prepared.image)).copy()
    pixel_mask = layout.known_mask.repeat(8, 0).repeat(8, 1)
    known = {}
    schedule = engine.pipe.scheduler
    try:
        with full_vae_canvas(engine.pipe.vae), routing:
            if not use_base:
                pixels = torch.as_tensor(rgb, dtype=engine.dtype, device=engine.device).permute(2, 0, 1)[None] / 255
                paint, masked = prepare_inpaint_condition(pixels, layout.known_mask, engine.encode)
                engine._active_inpaint = network, paint, masked
            started = time.perf_counter()
            harmonic, proof = harmonic_fill(rgb, pixel_mask)
            proof["harmonic_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            initial, initial_proof = inject_texture_at_scale(
                harmonic, rgb, layout, sigma_px_at_scale_1=candidate["texture_sigma_px_at_scale_1"]
            )
            proof["texture_initialization_seconds"] = time.perf_counter() - started
            engine.initial_caps[layout.face] = initial
            pixels = torch.as_tensor(initial, device=engine.device).permute(2, 0, 1)[None]
            start, timesteps = sdedit_schedule(schedule, int(engine.config["steps"]),
                                               candidate["strength"], engine.device)
            latent, noise = sdedit_latents(schedule, engine.encode(pixels), timesteps[0], generator)
            if use_base:
                frozen = prepared.frozen_latent.to(device=engine.device, dtype=engine.dtype)
                keep = torch.as_tensor(layout.known_mask, device=engine.device)[None, None]
                known = {"known_clean": layout.read_frozen_ring(frozen), "noise": noise,
                         "known_mask": keep, "feather_last": int(blend["feather_steps"])}
            guidance = None
            if gamma:
                from .v3_joint_cap_guidance import texture_guidance

                guidance = texture_guidance(engine, layout, prepared.frozen_latent, gamma,
                                            engine.config["joint_cap"]["guidance"])
            latent, steps = denoise_blended(engine, latent, timesteps, text, pooled,
                                            size=(layout.width_px, layout.height_px), start_index=start,
                                            epsilon_transform=guidance, **known)
            if use_base:
                alpha = feather_known_mask(keep) if known["feather_last"] else keep
                latent = blended_patch(latent, known["known_clean"], noise, alpha, schedule, None)
            final = cap_at_original_scale(layout, engine._rgb(latent))
            return final, {
                **proof, **steps, "strength": candidate["strength"], "texture_injection": initial_proof,
                "mode": "base_blended_sdedit" if use_base else "inpaint_sdedit",
                "known_latents": int(layout.known_mask.sum()), "noise_draws": 1,
                "closing_feather_latents": int(blend["feather_latents"]) if use_base else 0,
                "guidance": guidance.evidence() if guidance is not None else None,
            }
    finally:
        engine._active_inpaint = None
