"""Refine only the non-native edges of joint caps while preserving native RGB."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .v3_capfix_geometry import _pixel_rect
from .v3_capgeom_geometry import build_capgeom_layout
from .v3_capgeom_init import sdedit_latents, sdedit_schedule
from .v3_circular import full_vae_canvas
from .v3_geometry import CAPS, LATENT_SCALE, build_cap_layout
from .v3_inpainting import prepare_inpaint_condition
from .v3_joint_cap_geometry import NATIVE, compose_joint_stage1
from .v3_joint_cap_ring import _image_tensor, _set_timesteps, _temporary_config
from .v3_uniform_texture import _smoothstep


STAGE2_OPTIONS = {
    "base": True,
    "steps": 30,
    "cap_scale": 1,
    "native_lock_mm": 15.0,
    "edge_transition_mm": 15.0,
    "edge_strength": 0.70,
    "interior_strength": 0.30,
    "native_unlock_mm": 3.0,
    "feather_last": 3,
}


def _native_y(face, height, width_mm):
    if face not in NATIVE or height < 1 or width_mm <= 0:
        raise ValueError("Native cap depth needs an oriented cap and a positive physical length")
    y = (np.arange(height, dtype=np.float64) + 0.5) * (width_mm / height)
    return y[::-1] if NATIVE[face][2] == "bottom" else y


def stage2_strength_map(layout, box_mm, *, settings):
    """Render the registered native-lock, three-edge edit and interior strength."""
    p = {**STAGE2_OPTIONS, **settings}
    mm_w, mm_h = float(box_mm["length"]), float(box_mm["width"])
    if (not 0 < p["native_lock_mm"] < mm_h or not 0 < p["native_unlock_mm"] < mm_h
            or not 0 < p["edge_transition_mm"] <= min(mm_w, mm_h) / 2
            or not 0 < p["interior_strength"] <= p["edge_strength"] <= 1):
        raise ValueError("Invalid native lock or Stage 2 edit strengths")
    h, w = layout.latent_shape
    left, top, right, bottom = layout.center_rect
    if (not 0 <= left < right <= w or not 0 <= top < bottom <= h
            or np.any(layout.known_mask[top:bottom, left:right])):
        raise ValueError("Stage 2 edit core must be inside the unknown canvas")
    nh, nw = bottom - top, right - left
    native = _native_y(layout.face, nh, mm_h)
    x = np.minimum(np.arange(nw), np.arange(nw)[::-1]) * (mm_w / nw)
    # The outermost center gets the nominal edge strength; its pixel footprint
    # touches the cut. Falloff still reaches interior strength at 15 mm.
    away = np.maximum(0, mm_h - native - mm_h / (2 * nh))
    nonnative = np.minimum(x[None, :], away[:, None])
    near_edge = _smoothstep(1 - nonnative / p["edge_transition_mm"])
    native_gate = _smoothstep((native - p["native_lock_mm"]) / p["native_unlock_mm"])
    core = (p["interior_strength"] + near_edge * (p["edge_strength"] - p["interior_strength"]))
    field = np.zeros((h, w), dtype=np.float64)
    field[top:bottom, left:right] = core * native_gate[:, None]
    return field


def _layout(topology, stage_one, face, band, scale):
    original = build_cap_layout(topology, stage_one.ring, face, band_latent=band)
    return build_capgeom_layout(original, stage_one.ring, band, scale)


def save_stage2_strength_maps(topology, stage_one, options, destination, *, band=16):
    """Save actual model-grid strengths (gray value round(255*strength))."""
    p = {**STAGE2_OPTIONS, **options}
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for face in CAPS:
        layout = _layout(topology, stage_one, face, int(band), int(p["cap_scale"]))
        field = stage2_strength_map(layout, topology["box_mm"], settings=p)
        output = folder / f"{face}-stage2-strength.png"
        Image.fromarray(np.rint(field * 255).astype(np.uint8)).resize(
            (layout.width_px, layout.height_px), Image.Resampling.NEAREST,
        ).save(output, optimize=True)
        paths[face] = output
    return paths


def _initial_pixels(stage_one, layout):
    core = stage_one.caps[layout.face]
    left, top, right, bottom = _pixel_rect(layout.center_rect)
    expected = (right - left, bottom - top)
    if layout.scale == 1 and core.size != expected:
        raise ValueError("Native Stage 1 cap must fit Stage 2 without resampling")
    if core.size != expected:
        core = core.resize(expected, Image.Resampling.BICUBIC)
    frozen = layout.read_frozen_ring_rgb(stage_one.ring_image)
    mask = layout.known_mask.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)
    # Replicate finite edge values only into virtual corners of the prior.
    canvas = np.pad(np.asarray(core), ((top, layout.height_px - bottom),
                                      (left, layout.width_px - right), (0, 0)), mode="edge")
    canvas[mask] = np.asarray(frozen)[mask]
    return Image.fromarray(canvas), frozen


def _preserve_native(original, edited, *, face, box_mm, settings):
    """Reattach the exact native strip after the VAE, leaving a smooth unlock."""
    if edited.size != original.size or original.mode != "RGB" or edited.mode != "RGB":
        raise ValueError("Native Stage 2 restoration expects aligned RGB cap cores")
    mm_h = float(box_mm["width"])
    native = _native_y(face, original.height, mm_h)
    locked = native <= settings["native_lock_mm"]
    # One interpolation footprint beyond the physical lock keeps atlas
    # projection byte-identical at pixel centers inside the 0-15 mm zone.
    guard = settings["native_lock_mm"] + mm_h / original.height
    ramp = _smoothstep((native - guard) / settings["native_unlock_mm"])[:, None, None]
    prior, next_image = np.asarray(original), np.asarray(edited)
    pixels = np.rint(prior * (1 - ramp) + next_image * ramp).clip(0, 255).astype(np.uint8)
    if not np.array_equal(pixels[locked], prior[locked]):
        raise RuntimeError("Stage 2 changed its native locked RGB")
    return Image.fromarray(pixels), {"locked_native_pixels": int(locked.sum()) * original.width,
                                   "locked_native_pixels_changed": 0,
                                   "native_interpolation_guard_mm": guard - settings["native_lock_mm"]}


class JointStageTwoMixin:
    @torch.inference_mode()
    def refine_joint_caps(self, brief, seed, stage_one, *, options, guidance=None):
        from .v3_joint_cap_blend import denoise_blended
        from .v3_joint_cap_prompts import joint_cap_negative, joint_cap_prompt

        p = {**STAGE2_OPTIONS, **options}
        box = self.topology["box_mm"]
        band = int(self.config.get("cap_band", 16))
        caps, reports = {}, {}
        rng = torch.Generator(device=self.device).manual_seed(seed)
        positive = joint_cap_prompt(brief)
        negative = joint_cap_negative()
        pos, neg, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=positive, negative_prompt=negative, device=self.device,
            do_classifier_free_guidance=True,
        )
        context = torch.stack((neg[0], pos[0]))
        pooled = torch.stack((neg_pooled[0], pooled[0]))
        base = bool(p["base"])
        network = self._base_unet() if base else self._open_inpaint()
        if base and stage_one.ring_latent is None:
            raise ValueError("Stage 2 base blending requires the archived native ring latents")
        ring_latents = stage_one.ring_latent.to(self.device, self.dtype) if base else None
        steps = int(p["steps"])
        with _temporary_config(self, steps=steps), full_vae_canvas(self.pipe.vae):
            for face in CAPS:
                layout = _layout(self.topology, stage_one, face, band, int(p["cap_scale"]))
                initial_image, frozen = _initial_pixels(stage_one, layout)
                initial = self.encode(_image_tensor(self, initial_image))
                field = stage2_strength_map(layout, box, settings=p)
                size = layout.width_px, layout.height_px
                if tuple(initial.shape[-2:]) != field.shape:
                    raise RuntimeError("Stage 2 strength must align with cap latent cells")
                counts = np.floor(field * steps + 1.0e-8).astype(np.int64)
                scheduler = _set_timesteps(self, steps)
                start, timesteps = sdedit_schedule(scheduler, steps, p["edge_strength"], self.device)
                if int(counts.max()) != len(timesteps):
                    raise RuntimeError("Stage 2 strength levels disagree with the native DDIM suffix")
                latent, noise = sdedit_latents(scheduler, initial, timesteps[0], rng)
                start_at = torch.as_tensor(steps - counts, device=self.device)[None, None]
                known = torch.as_tensor(layout.known_mask, device=self.device)
                known_clean = layout.read_frozen_ring(ring_latents) if base else None
                restore_select = ~known[None, None] if base else torch.ones_like(start_at, dtype=torch.bool)

                def restore_prior(index, timestep, current):
                    reference = scheduler.add_noise(initial, noise, timestep.reshape(1))
                    return torch.where((start_at >= index) & restore_select, reference, current)

                try:
                    if not base:
                        paint, masked = prepare_inpaint_condition(
                            _image_tensor(self, frozen), layout.known_mask, self.encode,
                        )
                        self._active_inpaint = network, paint, masked
                    modified = guidance(face, layout, stage_one) if guidance is not None else None
                    latent, schedule = denoise_blended(
                        self, latent, timesteps, context, pooled, size=size, start_index=start,
                        known_clean=known_clean, noise=noise if base else None,
                        known_mask=known if base else None, feather_last=int(p["feather_last"]),
                        before_step=restore_prior, epsilon_transform=modified,
                    )
                    edited = self._rgb(latent).crop(_pixel_rect(layout.center_rect))
                finally:
                    self._active_inpaint = None
                original = stage_one.caps[face]
                if edited.size != original.size:
                    edited = edited.resize(original.size, Image.Resampling.LANCZOS)
                caps[face], lock = _preserve_native(original, edited, face=face, box_mm=box, settings=p)
                x0, y0, x1, y1 = layout.center_rect
                levels, n = np.unique(counts[y0:y1, x0:x1], return_counts=True)
                reports[face] = {
                    **schedule, **lock, "strength_shape_latent": list(field.shape),
                    "strength_map_sha256": sha256(field.astype("<f8", copy=False).tobytes()).hexdigest(),
                    "core_latent_step_histogram": {str(level): int(count) for level, count in
                                                  zip(levels, n, strict=True)},
                    "ring_method": "base_blended_latent" if base else "sdxl_inpainting",
                    "guidance_active": modified is not None,
                }
                if modified is not None and hasattr(modified, "evidence"):
                    reports[face]["guidance"] = modified.evidence()
        return replace(stage_one, caps=caps, proof={**stage_one.proof, "stage2": reports,
                                                   "stage2_options": p})

    def joint_atlas(self, brief, seed, stage_one, *, options, guidance=None):
        result = self.refine_joint_caps(brief, seed, stage_one, options=options, guidance=guidance)
        atlas = compose_joint_stage1(self.topology, self.region_map, self.bleed_owner, result)
        return atlas, result.proof
