"""Native circular joint ring/cap sampling and frozen-ring outpainting."""

from contextlib import contextmanager
from types import MethodType
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from .sdxl import STYLE, prompt_for
from .v3_capgeom_init import sdedit_latents, sdedit_schedule
from .v3_circular import circular_ring_codec
from .v3_geometry import CAPS, LATENT_SCALE, SIDES
from .v3_joint_cap_geometry import (
    NATIVE, JointStageOne, build_joint_ring_layout, build_outpaint_layout, coarse_joint_layout,
    compose_joint_stage1, ring_layout_from_metadata, ring_registration,
)
from .v3_ring_m2 import M2Routing, m2_attention


J4_OPTIONS = {
    "mode": "coarse_to_fine",
    "target_width_px": 1280,
    "cap_buffer_px": 64,
    "stage1a_scale": 0.70,
    "stage1a_steps": 28,
    "stage1a_guidance": 5.0,
    "stage1b_total_steps": 30,
    "stage1b_strength": 0.47,
    "stage1b_guidance": 5.0,
    "stage1b_projection_tail": 0.30,
    "stage1b_projection_interval": 3,
    "subject_center_fraction": 0.60,
    "feather_latent_rows": 6,
}
J4B_OPTIONS = {"strip_height_px": 320, "cap_buffer_px": 64, "steps": 22,
              "guidance": 5.0, "feather_last": 3}
REDILATION_BLOCKS = ("down_blocks.2", "mid_block", "up_blocks.0")
SIDE_LABEL = {side: index for index, side in enumerate(SIDES, 1)}
CONTINUATION_LABEL = len(SIDES) + 1


@contextmanager
def _temporary_config(engine, **derived):
    prior = engine.config
    engine.config = {**prior, **derived}
    try:
        yield
    finally:
        engine.config = prior


def joint_weights(ring, *, subject_center_fraction=0.60, feather_latent_rows=6):
    if not 0 < subject_center_fraction < 1 or feather_latent_rows <= 0:
        raise ValueError("Joint prompt routing needs a central fraction and positive feather")
    h, w = ring.latent_shape
    lo, hi = ring.body_rows
    collar = (hi - lo) * (1 - subject_center_fraction) / 2
    if feather_latent_rows > collar:
        raise ValueError("The feather must not reach the native fold")
    y = np.arange(h, dtype=np.float32) + 0.5
    activation = (np.minimum(y - lo, hi - y) - collar + feather_latent_rows) / (2 * feather_latent_rows)
    activation = np.clip(activation, 0, 1)
    result = np.zeros((CONTINUATION_LABEL + 1, h, w), dtype=np.float32)
    for face, (left, _, right, _) in ring.face_rects.items():
        if face in ("F", "B"):
            result[SIDE_LABEL[face], :, left:right] = activation[:, None]
            result[CONTINUATION_LABEL, :, left:right] = (1 - activation)[:, None]
        else:
            result[SIDE_LABEL[face], :, left:right] = 1
    return result


def _outpaint_weights(ring, face, height):
    weights = np.zeros((CONTINUATION_LABEL + 1, height, ring.latent_shape[1]), dtype=np.float32)
    for side, (left, _, right, _) in ring.face_rects.items():
        label = CONTINUATION_LABEL if side == NATIVE[face][0] else SIDE_LABEL[side]
        weights[label, :, left:right] = 1
    return weights


def periodic_bicubic(image, size):
    """Resize with RGB bicubic support crossing only the cyclic horizontal boundary."""
    width, height = image.size
    pad = min(16, width)
    if image.mode != "RGB" or min(size) < 1:
        raise ValueError("Cyclic resampling expects RGB and positive output dimensions")
    rgb = np.asarray(image)
    extended = Image.fromarray(np.pad(rgb, ((0, 0), (pad, pad), (0, 0)), mode="wrap"))
    return extended.resize(size, Image.Resampling.BICUBIC, box=(pad, 0, pad + width, height))


def _tangential_energy(image, circular):
    rgb = np.asarray(image, dtype=np.float32)
    luma = rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    horizontal = np.roll(luma, 1, axis=1) - luma if circular else np.diff(luma, axis=1)
    vertical = np.diff(luma, axis=0)
    return float((np.square(horizontal).mean() + np.square(vertical).mean()) / 2)


def downsample_energy_retention(image, *, circular=False):
    """Ratio of common-grid luma difference energy after 2x down/up to original."""
    reference = _tangential_energy(image, circular)
    half = tuple(max(1, n // 2) for n in image.size)
    if circular:
        reconstructed = periodic_bicubic(periodic_bicubic(image, half), image.size)
    else:
        small = image.resize(half, Image.Resampling.BICUBIC)
        reconstructed = small.resize(image.size, Image.Resampling.BICUBIC)
    restored = _tangential_energy(reconstructed, circular)
    return {"original_energy": reference, "roundtrip_energy": restored,
            "retention_ratio": restored / reference if reference else None}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _image_tensor(engine, image):
    return torch.from_numpy(np.array(image)).to(engine.device, engine.dtype).permute(2, 0, 1)[None] / 255


def _pure_noise(engine, generator, size):
    return torch.randn((1, 4, size[1] // LATENT_SCALE, size[0] // LATENT_SCALE),
                       generator=generator, device=engine.device, dtype=engine.dtype)


def _set_timesteps(engine, steps):
    scheduler = engine.pipe.scheduler
    scheduler.set_timesteps(steps, device=engine.device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(engine.device)
    return scheduler


@contextmanager
def _early_redilation(model, active, prefixes=REDILATION_BLOCKS):
    patched = []
    missing = object()

    def bind(layer, original):
        def dilated(_layer, *args, **kwargs):
            if not active[0]:
                return original(*args, **kwargs)
            previous = layer.dilation, layer.padding
            layer.dilation, layer.padding = (2, 1), (2, 1)
            try:
                return original(*args, **kwargs)
            finally:
                layer.dilation, layer.padding = previous
        return MethodType(dilated, layer)

    try:
        for name, layer in model.named_modules():
            if (isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3) and layer.stride == (1, 1)
                    and layer.padding == (1, 1) and layer.dilation == (1, 1)
                    and any(name.startswith(prefix + ".") for prefix in prefixes)):
                patched.append((name, layer, layer.__dict__.get("forward", missing)))
                layer.forward = bind(layer, layer.forward)
        if not patched:
            raise ValueError("ScaleCrafter did not find the registered SDXL convolutions")
        yield tuple(name for name, _, _ in patched)
    finally:
        for _, layer, previous in reversed(patched):
            if previous is missing:
                del layer.forward
            else:
                layer.forward = previous


class JointRingMixin:
    def _joint_conditions(self, brief, *, outpaint=False):
        from .v3_joint_cap_prompts import joint_cap_negative, joint_cap_prompt

        continuity = ", wraparound artwork" if brief["continuous"] else ""
        base = f"{STYLE}{continuity}, {brief['main_color']}"
        continuation = joint_cap_prompt(brief)
        regional = [prompt_for(brief, side) for side in SIDES]
        if outpaint:
            directions = ("left", "front", "right", "back")
            regional = [joint_cap_prompt(
                brief, template="{style}, {side_prompt}, {main_color}, "
                                f"continue the adjoining {direction} side artwork beyond its edge",
            ) for direction in directions]
        prompts = [base, *regional, continuation]
        negative = joint_cap_negative()
        if not outpaint:
            negative += ", duplicate subject, second sun, repeated emblem, picture frame, rectangular border"
        token_lengths = []
        for tokenizer in (self.pipe.tokenizer, self.pipe.tokenizer_2):
            lengths = [len(ids) for ids in tokenizer([*prompts, negative], truncation=False)["input_ids"]]
            if max(lengths) > tokenizer.model_max_length:
                raise ValueError(f"Joint prompt exceeds {tokenizer.model_max_length} tokens: {lengths}")
            token_lengths.append(lengths)
        pos, neg, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=prompts, negative_prompt=[negative] * len(prompts),
            device=self.device, do_classifier_free_guidance=True,
        )
        contexts = {index: torch.stack((neg[0], text)) for index, text in enumerate(pos)}
        proof = {"pooled_prompt": base, "regional_prompts": dict(zip(SIDES, regional, strict=True)),
                 "continuation_prompt": continuation, "negative_prompt": negative,
                 "token_lengths": token_lengths}
        return contexts, torch.stack((neg_pooled[0], pooled[0])), proof

    @torch.inference_mode()
    def generate_joint_stage1(self, brief, seed, *, options):
        from .v3_joint_cap_blend import denoise_blended

        p = {**J4_OPTIONS, **options}
        full = build_joint_ring_layout(self.topology, int(p["target_width_px"]), int(p["cap_buffer_px"]))
        ring = full.ring
        size = ring.width_px, ring.height_px
        network = self._base_unet()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        contexts, pooled, prompt_proof = self._joint_conditions(brief)
        subject = p["subject_center_fraction"]
        feather = p["feather_latent_rows"]
        full_routing = M2Routing(joint_weights(ring, subject_center_fraction=subject, feather_latent_rows=feather))
        proof = {"method": "j4", "mode": p["mode"], "options": p, "prompts": prompt_proof}
        proof.update(ring_registration(ring))
        with circular_ring_codec(network, self.pipe.vae):
            if p["mode"] == "coarse_to_fine":
                low = coarse_joint_layout(self.topology, full, p["stage1a_scale"]).ring
                low_size = low.width_px, low.height_px
                low_feather = max(1, round(feather * low.height_px / ring.height_px))
                low_routes = M2Routing(joint_weights(low, subject_center_fraction=subject,
                                                     feather_latent_rows=low_feather))
                low_steps = int(p["stage1a_steps"])
                _sync(self.device)
                started = time.perf_counter()
                with _temporary_config(self, steps=low_steps, guidance=p["stage1a_guidance"],
                                       projection_strength=0), m2_attention((network,), low_routes, contexts, (1,)):
                    coarse = self._denoise(low_size, generator, contexts[0], pooled)
                enlarged = periodic_bicubic(self._rgb(coarse), size)
                initial = self.encode(_image_tensor(self, enlarged))
                _sync(self.device)
                proof["stage1a"] = {"size_px": list(low_size), "body_rows_latent": list(low.body_rows),
                                    "steps": low_steps, "seconds": time.perf_counter() - started,
                                    "projection": False}
                steps = int(p["stage1b_total_steps"])
                scheduler = _set_timesteps(self, steps)
                start, timesteps = sdedit_schedule(scheduler, steps, p["stage1b_strength"], self.device)
                latent, _ = sdedit_latents(scheduler, initial, timesteps[0], generator)
                started = time.perf_counter()
                with _temporary_config(self, steps=steps, guidance=p["stage1b_guidance"],
                                       projection_tail=p["stage1b_projection_tail"],
                                       projection_interval=p["stage1b_projection_interval"]),                         m2_attention((network,), full_routing, contexts, (1,)):
                    frozen, schedule = denoise_blended(self, latent, timesteps, contexts[0], pooled,
                                                        size=size, start_index=start)
                image = self._rgb(frozen)
                _sync(self.device)
                proof["stage1b"] = {**schedule, "size_px": list(size), "strength": p["stage1b_strength"],
                                    "seconds": time.perf_counter() - started}
            elif p["mode"] == "single_pass_redilation":
                steps = int(p.get("steps", p["stage1b_total_steps"]))
                scheduler = _set_timesteps(self, steps)
                latent = _pure_noise(self, generator, size) * scheduler.init_noise_sigma
                active = [False]
                _sync(self.device)
                started = time.perf_counter()
                with _temporary_config(self, steps=steps, guidance=p.get("guidance", p["stage1b_guidance"])),                         m2_attention((network,), full_routing, contexts, (1,)), _early_redilation(network, active) as layers:
                    def set_phase(index, _timestep, value):
                        active[0] = index <= 9
                        return value
                    frozen, schedule = denoise_blended(self, latent, scheduler.timesteps, contexts[0], pooled,
                                                        size=size, before_step=set_phase)
                image = self._rgb(frozen)
                _sync(self.device)
                proof["stage1b"] = {**schedule, "size_px": list(size), "seconds": time.perf_counter() - started,
                                    "dilation_hw": [2, 1], "dilation_steps": [0, min(9, steps - 1)],
                                    "dilation_layers": list(layers)}
            else:
                raise ValueError(f"Unknown native joint ring stage {p['mode']}")
        caps = {face: image.crop(rect) for face, rect in full.cap_rects_px.items()}
        lo, hi = ring.body_rows
        body = image.crop((0, lo * LATENT_SCALE, ring.width_px, hi * LATENT_SCALE))
        proof["cap_rects_px"] = {face: list(rect) for face, rect in full.cap_rects_px.items()}
        proof["cap_rect_source"] = "ring_image"
        proof["retained_side_buffer_px"] = int(p["cap_buffer_px"])
        proof["downsample_energy_2x"] = {"ring_body": downsample_energy_retention(body, circular=True),
                                        **{face: downsample_energy_retention(cap) for face, cap in caps.items()}}
        return JointStageOne(ring, image, caps, proof, frozen.detach().cpu())

    @torch.inference_mode()
    def generate_frozen_outpaint(self, brief, seed, prepared, *, options):
        from .v3_joint_cap_blend import denoise_blended

        p = {**J4B_OPTIONS, **options}
        network = self._base_unet()
        ring = prepared.layout
        original = prepared.frozen_latent.to(self.device, self.dtype)
        if tuple(original.shape[-2:]) != ring.latent_shape:
            raise ValueError("Frozen R0 latents do not match the registered ring")
        rng = torch.Generator(device=self.device).manual_seed(seed)
        rng.set_state(prepared.rng_state.cpu())
        contexts, pooled, prompts = self._joint_conditions(brief, outpaint=True)
        steps = int(p["steps"])
        caps, reports, rectangles = {}, {}, {}
        with _temporary_config(self, steps=steps, guidance=p["guidance"]), circular_ring_codec(network, self.pipe.vae):
            for face in CAPS:
                canvas = build_outpaint_layout(self.topology, ring, face, int(p["strip_height_px"]),
                                               int(p["cap_buffer_px"]))
                size = canvas.size_px
                known_start, known_end = canvas.known_rows_latent
                source_start, source_end = canvas.source_rows_latent
                noise = _pure_noise(self, rng, size)
                known = torch.zeros_like(noise)
                known[..., known_start:known_end, :] = original[..., source_start:source_end, :]
                mask = torch.zeros(known.shape[-2:], device=self.device, dtype=torch.bool)
                mask[known_start:known_end] = True
                routes = M2Routing(_outpaint_weights(ring, face, mask.shape[0]))
                scheduler = _set_timesteps(self, steps)
                _sync(self.device)
                started = time.perf_counter()
                with m2_attention((network,), routes, contexts, (1,)):
                    sample, schedule = denoise_blended(
                        self, noise * scheduler.init_noise_sigma, scheduler.timesteps, contexts[0], pooled,
                        size=size, known_clean=known, noise=noise, known_mask=mask, periodic_x=True,
                        feather_last=int(p["feather_last"]),
                    )
                caps[face] = self._rgb(sample).crop(canvas.cap_rect_px)
                _sync(self.device)
                rectangles[face] = list(canvas.cap_rect_px)
                reports[face] = {**schedule, "size_px": list(size),
                                 "known_rows_latent": list(canvas.known_rows_latent),
                                 "active_cap_column": NATIVE[face][0],
                                 "source_rows_latent": list(canvas.source_rows_latent),
                                 "seconds": time.perf_counter() - started}
        proof = {**ring_registration(ring), "method": "j4b", "options": p, "prompts": prompts,
                 "outpaint": reports, "cap_rects_px": rectangles, "cap_rect_source": "outpaint_canvas",
                 "ring_source": "frozen_r0", "has_harmonic_initialization": False,
                 "downsample_energy_2x": {face: downsample_energy_retention(cap) for face, cap in caps.items()}}
        return JointStageOne(ring, prepared.image, caps, proof, prepared.frozen_latent)
