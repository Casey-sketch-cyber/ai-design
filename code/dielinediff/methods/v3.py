"""Run the circular-ring and frozen-ring cap stages, or the flat regional B2 control."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel
from PIL import Image

from ..e01 import read_json
from .b2_controlnet import panel_edges
from .ours_tf import DieLineDiff
from .sdxl import NEGATIVE, STYLE, load_pipeline, prompt_for
from .v3_attention import RegionRouting, region_attention, ring_style_attention
from .v3_circular import circular_ring_codec, full_vae_canvas


REGIONAL_FACES = {1: "F", 2: "B", 3: "L", 5: "T", 6: "Btm"}


def regional_routing(labels):
    labels = np.asarray(labels).copy()
    labels[labels == 4] = 3
    if not np.isin(labels, (0, *REGIONAL_FACES)).all():
        raise ValueError("Regional labels must select the shared style or a known face")
    return RegionRouting(labels)


def conditioned_latent(latent, frozen, noise, mask, scheduler, timestep):
    """Replace known cap context with the corresponding ring at the current noise level."""
    if timestep is not None:
        frozen = scheduler.add_noise(frozen, noise, timestep.reshape(1))
    return torch.where(mask, frozen, latent)


class DieLineDiffV3(DieLineDiff):
    """Hold one frozen SDXL pipeline and load the original B2 ControlNet only on demand."""

    def __init__(self, models_dir, geometry_dir, config, device="cuda", lut_path=None):
        from . import v3_geometry

        self.geometry = v3_geometry
        self.models = Path(models_dir)
        self.geometry_dir = Path(geometry_dir)
        self.config = config
        self.device = torch.device(device)
        self.topology = read_json(self.geometry_dir / "topology.json")
        self.region_map = self._labels("region_map.png")
        self.bleed_owner = self._labels("bleed_owner.png")
        self.atlas_size = self.geometry.atlas_canvas_size(self.region_map)
        self.atlas_routing = regional_routing(self.geometry.atlas_region_ids(self.region_map))
        self.layouts = {}
        self.pipe = load_pipeline(self.models, device=self.device)
        if self.pipe.scheduler.config.prediction_type != "epsilon":
            raise ValueError("V3 uses an epsilon-prediction DDIM scheduler")
        self.dtype = self.pipe.unet.dtype
        strength = self.config.get("projection_strength", 0)
        if not 0 <= strength <= 1:
            raise ValueError("projection_strength must be between 0 and 1")
        if strength > 0 and lut_path is None:
            raise ValueError("LUT projection requires the shared LUT path")
        self.lut = torch.as_tensor(np.load(lut_path), device=self.device) if lut_path is not None else None
        self._controlnet = None
        self._control_image = None
        self.last_run = None

    def _labels(self, name):
        with Image.open(self.geometry_dir / name) as image:
            return np.array(image)

    def _ring_layout(self, size):
        size = tuple(size)
        band = self.config.get("cap_band", 16)
        key = (*size, band)
        if key not in self.layouts:
            ring = self.geometry.build_ring_layout(self.topology, *size)
            caps = {face: self.geometry.build_cap_layout(
                self.topology, ring, face, band_latent=band
            ) for face in ("T", "Btm")}
            self.layouts[key] = ring, regional_routing(ring.region_ids), caps
        return self.layouts[key]

    def _control(self):
        if self._controlnet is None:
            self._controlnet = ControlNetModel.from_pretrained(
                self.models / "controlnet", torch_dtype=self.dtype, variant="fp16",
                local_files_only=True, use_safetensors=True,
            )
            height, width = self.region_map.shape
            native = panel_edges(self.geometry_dir / "region_map.png", width, height)
            padded = Image.new("RGB", self.atlas_size)
            padded.paste(native)
            pixels = torch.from_numpy(np.array(padded)).permute(2, 0, 1)[None]
            pixels = pixels.to(device=self.device, dtype=self.dtype).div_(255)
            self._control_image = pixels.expand(2, -1, -1, -1)
        return self._controlnet.to(self.device), self._control_image

    def _conditions(self, brief):
        weight = float(self.config.get("regional_weight", 1.0))
        if not 0 <= weight <= 1:
            raise ValueError("regional_weight must be between 0 and 1")
        continuity = ", wraparound artwork" if brief["continuous"] else ""
        style = f"{STYLE}{continuity}, {brief['main_color']}"
        faces = list(REGIONAL_FACES.items())
        prompts = [style, *(prompt_for(brief, face) for _, face in faces)]
        positive, negative, pooled, negative_pooled = self.pipe.encode_prompt(
            prompt=prompts, negative_prompt=[NEGATIVE] * len(prompts), device=self.device,
            do_classifier_free_guidance=True,
        )
        contexts = {0: torch.stack((negative[0], positive[0]))}
        for index, (label, _) in enumerate(faces, start=1):
            contexts[label] = torch.stack((negative[0], positive[0].lerp(positive[index], weight)))
        return contexts, torch.stack((negative_pooled[0], pooled[0]))

    def _predict(self, latent, timestep, text, condition, control):
        scheduler = self.pipe.scheduler
        model_input = scheduler.scale_model_input(latent.expand(2, -1, -1, -1), timestep)
        residuals = {}
        if control is not None:
            network, image = control
            down, mid = network(
                model_input, timestep, encoder_hidden_states=text, controlnet_cond=image,
                conditioning_scale=float(self.config.get("controlnet_scale", 0.65)),
                added_cond_kwargs=condition, return_dict=False,
            )
            residuals = {"down_block_additional_residuals": down, "mid_block_additional_residual": mid}
        output = self.pipe.unet(model_input, timestep, encoder_hidden_states=text,
                                added_cond_kwargs=condition, return_dict=False, **residuals)[0]
        unconditioned, conditioned = output.chunk(2)
        return unconditioned + self.config["guidance"] * (conditioned - unconditioned)

    def _denoise(self, size, generator, text, pooled, *,
                 control=None, frozen=None, mask=None, capture_style=None):
        width, height = size
        if min(size) < 512 or width % 64 or height % 64:
            raise ValueError(f"V3 expects SDXL canvases of at least 512 px aligned to 64 px: {size}")
        if (frozen is None) != (mask is None):
            raise ValueError("Cap blending requires both frozen ring values and a known-region mask")

        condition = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                (height, width, 0, 0, height, width), dtype=self.dtype, device=self.device
            )[None].expand(2, -1),
        }
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.config["steps"], device=self.device)
        timesteps = scheduler.timesteps
        noise = torch.randn((1, 4, height // 8, width // 8), device=self.device,
                            dtype=self.dtype, generator=generator)
        latent = noise * scheduler.init_noise_sigma
        tail = float(self.config.get("projection_tail", 0.30))
        interval = int(self.config.get("projection_interval", 3))
        if not 0 <= tail <= 1 or interval < 1:
            raise ValueError("Invalid LUT projection schedule")
        start = round(len(timesteps) * (1 - tail))
        project = self.config.get("projection_strength", 0) > 0
        for index, timestep in enumerate(timesteps):
            if frozen is not None:
                latent = conditioned_latent(latent, frozen, noise, mask, scheduler, timestep)
            if capture_style is not None and index == len(timesteps) - 1:
                with ring_style_attention(self.pipe.unet, capture_style, capture=True):
                    prediction = self._predict(latent, timestep, text, condition, control)
            else:
                prediction = self._predict(latent, timestep, text, condition, control)
            if project and index >= start and (index - start) % interval == 0:
                prediction = self.color_constraint(latent, prediction, timestep)
            latent = scheduler.step(prediction, timestep, latent, return_dict=False)[0]
        if frozen is not None:
            latent = conditioned_latent(latent, frozen, noise, mask, scheduler, None)
        return latent

    def _rgb(self, latent):
        decoded = self.decode(latent)[0].permute(1, 2, 0)
        if not torch.isfinite(decoded).all():
            raise RuntimeError("SDXL VAE produced nonfinite pixels")
        return Image.fromarray(decoded.mul(255).round().to(torch.uint8).cpu().numpy())

    def _generate_flat(self, contexts, pooled, generator):
        control = self._control()
        with region_attention((self.pipe.unet, control[0]), self.atlas_routing, contexts):
            latent = self._denoise(self.atlas_size, generator, contexts[0], pooled, control=control)
        height, width = self.region_map.shape
        self.last_run["flat_size_px"] = list(self.atlas_size)
        return self._rgb(latent).crop((0, 0, width, height))

    def _generate_ring(self, contexts, pooled, generator, size):
        if self._controlnet is not None:
            self._controlnet.to("cpu")
        ring, routing, layouts = self._ring_layout(size)
        bank = {} if self.config.get("share_attention", False) else None
        unet = self.pipe.unet
        with circular_ring_codec(unet, self.pipe.vae):
            with region_attention((unet,), routing, contexts):
                frozen_ring = self._denoise((ring.width_px, ring.height_px), generator,
                                             contexts[0], pooled, capture_style=bank)
            ring_image = self._rgb(frozen_ring)

        generated_caps = {}
        cap_info = {}
        for label, face in ((5, "T"), (6, "Btm")):
            cap = layouts[face]
            anchor = cap.read_frozen_ring(frozen_ring)
            mask = torch.as_tensor(np.array(cap.known_mask), device=self.device)[None, None]
            context = ring_style_attention(unet, bank, capture=False) if bank is not None else nullcontext()
            with full_vae_canvas(self.pipe.vae), context:
                latent = self._denoise((cap.width_px, cap.height_px), generator, contexts[label], pooled,
                                       frozen=anchor, mask=mask)
                generated_caps[face] = (cap, self._rgb(latent))
            cap_info[face] = {"size_px": [cap.width_px, cap.height_px],
                              "known_latents": int(np.count_nonzero(cap.known_mask))}
        self.last_run.update(
            ring_size_px=[ring.width_px, ring.height_px], caps=cap_info,
            cap_band_latent=int(self.config.get("cap_band", 16)),
            shared_self_attention_layers=len(bank) // 2 if bank is not None else 0,
        )
        image, support = self.geometry.compose_atlas(
            self.topology, self.region_map, self.bleed_owner, ring, ring_image, generated_caps,
            return_support=True,
        )
        self.last_run["atlas_support"] = self.geometry.compose_support_counts(
            self.region_map, self.bleed_owner, support
        )
        return image

    @torch.inference_mode()
    def generate(self, brief, seed, *, method="v3", ring_size=(1280, 1024)):
        if method not in ("v3", "b2_regional"):
            raise ValueError(f"Unknown E02 method: {method}")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        contexts, pooled = self._conditions(brief)
        self.last_run = {
            "method": method,
            "steps": int(self.config["steps"]),
            "atlas_size_px": [self.region_map.shape[1], self.region_map.shape[0]],
            "regional_weight": float(self.config.get("regional_weight", 1.0)),
            "projection_strength": float(self.config.get("projection_strength", 0)),
            "projection_tail": float(self.config.get("projection_tail", 0.30)),
            "projection_interval": int(self.config.get("projection_interval", 3)),
            "share_attention": method == "v3" and bool(self.config.get("share_attention", False)),
        }
        if method == "b2_regional":
            return self._generate_flat(contexts, pooled, generator)
        return self._generate_ring(contexts, pooled, generator, ring_size)
