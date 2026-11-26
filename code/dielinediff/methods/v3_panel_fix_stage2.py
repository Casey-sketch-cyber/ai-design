"""Seam repair that keeps the native outpaint and mixes in the registered U1 prior."""

import hashlib
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .v3_capgeom_init import harmonic_fill, sdedit_latents, sdedit_schedule
from .v3_circular import full_vae_canvas
from .v3_geometry import CAPS, LATENT_SCALE
from .v3_inpainting import prepare_inpaint_condition
from .v3_joint_cap_blend import denoise_blended
from .v3_joint_cap_geometry import compose_joint_stage1
from .v3_joint_cap_guidance import texture_guidance
from .v3_joint_cap_ring import _temporary_config
from .v3_joint_cap_stage2 import _layout, _native_y, _preserve_native, _smoothstep
from .v3_ring_midscale_cap import inject_texture_at_scale


def _fields(face, shape, box, options):
    """Return the protected native distance, seam strength and U1 blend weight."""
    h, w = shape
    length, depth = float(box["length"]), float(box["width"])
    native = _native_y(face, h, depth)
    # Distances use the first cell touching a cut, so the cut itself reaches 0.85.
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) * length / w
    opposite = np.maximum(0.0, depth - native - depth / (2 * h))
    nonnative = np.minimum(x[None, :], opposite[:, None])
    native_unlock = _smoothstep((native - options["native_lock_mm"]) / options["native_unlock_mm"])[:, None]
    seam = _smoothstep(1 - nonnative / options["edge_transition_mm"])
    strength = options["interior_strength"] + seam * (options["edge_strength"] - options["interior_strength"])
    mix = _smoothstep(1 - nonnative / options["mix_edge_transition_mm"])
    return native, strength * native_unlock, mix * native_unlock


def panel_fields(layout, box, settings):
    left, top, right, bottom = layout.center_rect
    _, core, _ = _fields(layout.face, (bottom - top, right - left), box, settings)
    strength = np.zeros(layout.latent_shape, dtype=np.float64)
    strength[top:bottom, left:right] = core
    if np.any(strength[layout.known_mask] != 0):
        raise ValueError("Panel repair would denoise frozen ring context")
    return strength


def _core_pixels(layout):
    return tuple(v * LATENT_SCALE for v in layout.center_rect)


def _area_axis(pixels, axis, target):
    """Average over the exact fractional footprints of the registered target grid."""
    ordered = np.moveaxis(pixels, axis, 0)
    source = len(ordered)
    if not 0 < target <= source:
        raise ValueError("U1 resampling must fit inside the higher-resolution source")
    if target == source:
        return pixels
    limits = np.linspace(0, source, target + 1)
    cell = np.minimum(limits.astype(np.int64), source - 1)
    offset = (limits - cell)[:, None, None]
    zeros = np.zeros((1, *ordered.shape[1:]), dtype=np.float64)
    integral = np.concatenate((zeros, np.cumsum(ordered, axis=0, dtype=np.float64)), axis=0)
    at_edge = integral[cell] + ordered[cell] * offset
    averaged = np.diff(at_edge, axis=0) / (source / target)
    return np.moveaxis(averaged, 0, axis)


def _mixed_paint(topology, stage, layout, options, band):
    """Build the actual float32 prior and keep known ring RGB byte-exact."""
    hi = _layout(topology, stage, layout.face, band, int(options["harmonic_scale"]))
    hi_rgb = np.array(hi.read_frozen_ring_rgb(stage.ring_image))
    hi_mask = hi.known_mask.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)
    harmonic, solver = harmonic_fill(hi_rgb, hi_mask)
    prior, texture = inject_texture_at_scale(harmonic, hi_rgb, hi, sigma_px_at_scale_1=14.0)
    x0, y0, x1, y1 = _core_pixels(hi)
    high_core = prior[y0:y1, x0:x1]
    left, top, right, bottom = _core_pixels(layout)
    shape = bottom - top, right - left
    h, w = shape
    high_core = _area_axis(_area_axis(high_core, 0, h), 1, w).astype(np.float32)
    outpaint = np.array(stage.caps[layout.face], dtype=np.float32) / 255
    if outpaint.shape != (h, w, 3):
        raise ValueError("The repaired native cap does not fit the seam canvas")
    native, _, weights = _fields(layout.face, shape, topology["box_mm"], options)
    alpha = weights.astype(np.float32)[..., None]
    mixed = np.where(alpha == 0, outpaint, outpaint * (1 - alpha) + high_core * alpha)
    np.clip(mixed, 0, 1, out=mixed)
    locked = native <= options["native_lock_mm"]
    if not np.array_equal(mixed[locked], outpaint[locked]):
        raise RuntimeError("An initial-image blend changed the native fold")
    pixels = np.pad(mixed, ((top, layout.height_px - bottom), (left, layout.width_px - right),
                            (0, 0)), mode="edge")
    frozen = np.array(layout.read_frozen_ring_rgb(stage.ring_image))
    mask = layout.known_mask.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)
    pixels[mask] = frozen[mask].astype(np.float32) / 255
    return pixels, frozen, {
        "harmonic_scale": hi.scale, "target_scale": layout.scale,
        "u1_core_sha256": hashlib.sha256(high_core.tobytes()).hexdigest(),
        "blend_weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "native_prior_pixels_changed": 0, "u1_solver": solver, "u1_texture": texture,
    }


def export_panel_maps(topology, stage, options, destination, *, band=16):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    for name in CAPS:
        layout = _layout(topology, stage, name, band, int(options["cap_scale"]))
        left, top, right, bottom = _core_pixels(layout)
        shape = bottom - top, right - left
        _, _, mix = _fields(name, shape, topology["box_mm"], options)
        strength = panel_fields(layout, topology["box_mm"], options)
        gain = strength.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)[top:bottom, left:right]
        tiles = (("U1 mixing weight (0 to 1)", mix), ("DDIM strength (0 to 0.85)", gain))
        h, w = shape
        fig = Image.new("RGB", (2 * w + 30, h + 76), "white")
        pen = ImageDraw.Draw(fig)
        font = ImageFont.load_default(size=15)
        fold = "lower edge" if name == "T" else "upper edge"
        pen.text((10, 10), f"{name}: native fold at the {fold}; 0–15 mm locked",
                 fill="black", font=font)
        for index, (label, values) in enumerate(tiles):
            x = 10 + index * (w + 10)
            pen.text((x, 38), label, fill="black", font=font)
            fig.paste(Image.fromarray(np.rint(values * 255).astype(np.uint8)).convert("RGB"), (x, 62))
        fig.save(folder / f"{name}-panel-fix-maps.png", optimize=True)


class PanelFixStageTwoMixin:
    """Run 9-channel nonnative seam repainting with optional masked J3 guidance."""

    @torch.inference_mode()
    def panel_atlas(self, brief, seed, stage, *, options, gamma):
        from .v3_panel_fix_engine import panel_cap_conditions

        if (options["native_lock_mm"] != 15 or options["edge_transition_mm"] != 20
                or options["edge_strength"] != 0.85 or options["interior_strength"] != 0.30
                or options["harmonic_scale"] != 2 or options["cap_scale"] != 1 or gamma not in (0.0, 0.30)):
            raise ValueError("Unsupported Task 22 Stage 2 parameters")
        _, _, contexts, pooled, prompt_proof = panel_cap_conditions(self, brief)
        text = contexts[0]
        steps = int(options["steps"])
        band = int(self.config.get("cap_band", 16))
        box = self.topology["box_mm"]
        network = self._open_inpaint()
        rng = torch.Generator(device=self.device).manual_seed(seed)
        generated, reports = {}, {}
        self.initial_caps = {}
        with _temporary_config(self, steps=steps), full_vae_canvas(self.pipe.vae):
            for face in CAPS:
                layout = _layout(self.topology, stage, face, band, int(options["cap_scale"]))
                start = time.perf_counter()
                pixels, context, proof = _mixed_paint(self.topology, stage, layout, options, band)
                self.initial_caps[face] = pixels
                proof["mixed_initial_seconds"] = time.perf_counter() - start
                image = torch.as_tensor(pixels, device=self.device).permute(2, 0, 1)[None]
                context = torch.as_tensor(context, dtype=self.dtype, device=self.device).permute(2, 0, 1)[None] / 255
                paint, masked = prepare_inpaint_condition(context, layout.known_mask, self.encode)
                field = panel_fields(layout, box, options)
                counts = np.floor(field * steps + 1e-8).astype(np.int64)
                scheduler = self.pipe.scheduler
                index, timesteps = sdedit_schedule(scheduler, steps, options["edge_strength"], self.device)
                if len(timesteps) != int(counts.max()):
                    raise ValueError("The spatial strength maximum differs from its DDIM suffix")
                clean = self.encode(image)
                latent, noise = sdedit_latents(scheduler, clean, timesteps[0], rng)
                release = torch.as_tensor(steps - counts, device=self.device)[None, None]

                def restore(step, timestep, current):
                    prior = scheduler.add_noise(clean, noise, timestep.reshape(1))
                    return torch.where(release >= step, prior, current)

                guide = None
                self._active_inpaint = network, paint, masked
                try:
                    transform = None
                    if gamma:
                        guide = texture_guidance(self, layout, stage.ring_latent, gamma,
                                                 self.config["joint_cap"]["guidance"], editable_mask=field > 0)
                        base_editable = guide.editable

                        def transform(step, timestep, current, epsilon):
                            # Earlier updates would be replaced by the scheduled prior on the next step.
                            guide.editable = base_editable & (release <= step)
                            try:
                                return guide(step, timestep, current, epsilon)
                            finally:
                                guide.editable = base_editable

                    latent, schedule = denoise_blended(self, latent, timesteps, text, pooled,
                                                        size=(layout.width_px, layout.height_px),
                                                        start_index=index, before_step=restore,
                                                        epsilon_transform=transform)
                    latent = torch.where(release == steps, clean, latent)
                    edited = self._rgb(latent).crop(_core_pixels(layout))
                finally:
                    self._active_inpaint = None
                cap, lock = _preserve_native(stage.caps[face], edited, face=face, box_mm=box, settings=options)
                generated[face] = cap
                reports[face] = {"size_px": [layout.width_px, layout.height_px],
                                 "sdedit": {**proof, **schedule, **lock,
                                            "strength_map_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
                                            "zero_step_latents_restored": int((counts == 0).sum()),
                                            "guidance": guide.evidence() if guide is not None else None}}
        result = replace(stage, caps=generated)
        atlas = compose_joint_stage1(self.topology, self.region_map, self.bleed_owner, result)
        return atlas, {"caps": reports, "stage2_prompts": prompt_proof, "stage2_options": dict(options)}
