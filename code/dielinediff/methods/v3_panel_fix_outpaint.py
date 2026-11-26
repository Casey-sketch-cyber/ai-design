"""J4b repair with fixed, independently keyed noise for every brief and cap."""

from hashlib import sha256
import json
import time

import numpy as np
import torch

from .v3_circular import circular_ring_codec
from .v3_geometry import CAPS, SIDES
from .v3_joint_cap_blend import denoise_blended
from .v3_joint_cap_geometry import JointStageOne, NATIVE, build_outpaint_layout, ring_registration
from .v3_joint_cap_prompts import cap_local_side
from .v3_joint_cap_ring import (
    J4B_OPTIONS, _outpaint_weights, _pure_noise, _set_timesteps, _sync, _temporary_config,
    downsample_energy_retention,
)
from .v3_panel_fix_engine import panel_cap_negative, panel_cap_prompt
from .v3_ring_m2 import M2Routing, m2_attention


NOISE_DOMAIN = "e02-panel-fix-outpaint-v1"
REPAIR = {"repair": "per_brief_face_noise", "noise_domain": NOISE_DOMAIN}


def panel_noise_seed(seed, brief_id, face, *, domain=NOISE_DOMAIN):
    """Use a stable, order-independent uint64 seed for one cap's single noise draw."""
    if (not isinstance(seed, int) or seed < 0 or not isinstance(brief_id, str) or not brief_id
            or face not in CAPS or domain != NOISE_DOMAIN):
        raise ValueError("Panel outpainting needs the registered noise domain, seed, brief and cap")
    key = json.dumps([domain, seed, brief_id, face], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(key).digest()[:8], "little")


def _hash_tensor(tensor):
    return sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def registered_outpaint_noise(engine, seed, brief_id, face, size, *, domain=NOISE_DOMAIN):
    actual = panel_noise_seed(seed, brief_id, face, domain=domain)
    generator = torch.Generator(device=engine.device).manual_seed(actual)
    initial_state = _hash_tensor(generator.get_state())
    noise = _pure_noise(engine, generator, size)
    return noise, {
        "domain": domain,
        "derived_seed_uint64": actual,
        "rng_state_before_sha256": initial_state,
        "rng_state_after_sha256": _hash_tensor(generator.get_state()),
        "noise_sha256": _hash_tensor(noise),
        "noise_shape": list(noise.shape),
        "noise_dtype": str(noise.dtype),
        "draw_count": 1,
    }


class PanelFixOutpaintMixin:
    def _panel_outpaint_conditions(self, brief):
        options = self.config["panel_fix"]["cap_prompt"]
        positive = panel_cap_prompt(brief, options)
        negative = panel_cap_negative(options)
        style = options["style"] + (", wraparound artwork" if brief.get("continuous", False) else "")
        base = f"{style}, {brief['main_color']}"
        motif = brief["side_prompt"]
        product = brief.get("product_name_zh")
        motif = cap_local_side(motif.replace(product, "") if product else motif)
        direction_names = ("left", "front", "right", "back")
        regional = [
            f"{base}, {motif}, continue the adjoining {direction} side artwork beyond its edge"
            for direction in direction_names
        ]
        prompts = [base, *regional, positive]
        proof = {}
        for i, tokenizer in enumerate((self.pipe.tokenizer, self.pipe.tokenizer_2), 1):
            ids = tokenizer([*prompts, negative], add_special_tokens=True, truncation=False)["input_ids"]
            lens = [len(item) for item in ids]
            if max(lens) > tokenizer.model_max_length:
                raise ValueError(f"Panel outpaint text exceeds CLIP stream {i} capacity: {lens}")
            proof[str(i)] = {"lengths": lens, "capacity": int(tokenizer.model_max_length),
                             "token_ids_sha256": sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
        pos, neg, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=prompts, negative_prompt=[negative] * len(prompts),
            device=self.device, do_classifier_free_guidance=True,
        )
        contexts = {index: torch.stack((neg[0], text)) for index, text in enumerate(pos)}
        return contexts, torch.stack((neg_pooled[0], pooled[0])), {
            "pooled_prompt": base, "regional_prompts": dict(zip(SIDES, regional, strict=True)),
            "continuation_prompt": positive, "negative_prompt": negative, "tokenizers": proof,
        }

    @torch.inference_mode()
    def generate_panel_outpaint(self, brief, seed, prepared, *, options):
        supplied = {**J4B_OPTIONS, **REPAIR, **options}
        if any(supplied[key] != value for key, value in REPAIR.items()):
            raise ValueError("Use the diagnosed and registered independent-noise repair")
        ring = prepared.layout
        frozen = prepared.frozen_latent.to(self.device, self.dtype)
        if tuple(frozen.shape[-2:]) != ring.latent_shape:
            raise ValueError("Original R0 latents and ring geometry disagree")
        contexts, pooled, prompts = self._panel_outpaint_conditions(brief)
        network = self._base_unet()
        steps = int(supplied["steps"])
        caps, reports, rectangles = {}, {}, {}
        with (_temporary_config(self, steps=steps, guidance=supplied["guidance"]),
              circular_ring_codec(network, self.pipe.vae)):
            for face in CAPS:
                view = build_outpaint_layout(self.topology, ring, face, int(supplied["strip_height_px"]),
                                             int(supplied["cap_buffer_px"]))
                size = view.size_px
                noise, noise_proof = registered_outpaint_noise(self, seed, brief["id"], face, size)
                start, stop = view.known_rows_latent
                source_start, source_stop = view.source_rows_latent
                aligned = torch.zeros_like(noise)
                aligned[..., start:stop, :] = frozen[..., source_start:source_stop, :]
                mask = torch.zeros(noise.shape[-2:], device=self.device, dtype=torch.bool)
                mask[start:stop] = True
                if not torch.equal(aligned[..., start:stop, :], frozen[..., source_start:source_stop, :]):
                    raise RuntimeError("K3/K4 must keep original latent registration in the frozen patch")
                routes = M2Routing(_outpaint_weights(ring, face, mask.shape[0]))
                scheduler = _set_timesteps(self, steps)
                _sync(self.device)
                started = time.perf_counter()
                with m2_attention((network,), routes, contexts, (1,)):
                    sample, detail = denoise_blended(
                        self, noise * scheduler.init_noise_sigma, scheduler.timesteps, contexts[0], pooled,
                        size=size, known_clean=aligned, noise=noise, known_mask=mask, periodic_x=True,
                        feather_last=int(supplied["feather_last"]),
                    )
                caps[face] = self._rgb(sample).crop(view.cap_rect_px)
                _sync(self.device)
                rectangles[face] = list(view.cap_rect_px)
                reports[face] = {
                    **detail, "size_px": list(size), "active_cap_column": NATIVE[face][0],
                    "known_rows_latent": list(view.known_rows_latent),
                    "source_rows_latent": list(view.source_rows_latent),
                    "registered_context_unchanged": True,
                    "seconds": time.perf_counter() - started,
                    "noise": noise_proof,
                }
        if reports["T"]["noise"]["noise_sha256"] == reports["Btm"]["noise"]["noise_sha256"]:
            raise RuntimeError("Both caps unexpectedly used the same noise")
        proof = {
            **ring_registration(ring), "method": "j4b_v2", "options": supplied, "prompts": prompts,
            "outpaint": reports, "cap_rects_px": rectangles, "cap_rect_source": "outpaint_canvas",
            "ring_source": "frozen_r0", "vertical_transform_used": False,
            "r0_archived_rng_sha256": _hash_tensor(prepared.rng_state),
            "has_harmonic_initialization": False,
            "downsample_energy_2x": {face: downsample_energy_retention(cap) for face, cap in caps.items()},
        }
        return JointStageOne(ring, prepared.image, caps, proof, prepared.frozen_latent)
