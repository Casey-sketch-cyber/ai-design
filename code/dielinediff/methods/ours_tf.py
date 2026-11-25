"""Training-free shared-atlas denoising with print constraints."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dielinediff.color import project_rgb

from .sdxl import NEGATIVE, colors, load_pipeline, padded_size, prompt_for
from .windows import build_windows


class DieLineDiff:
    def __init__(self, models, geometry, config, device, lut):
        self.config = config
        self.device = device
        self.pipe = load_pipeline(models, device=device)
        self.dtype = self.pipe.unet.dtype
        topology = geometry["topology"]
        self.width = topology["image"]["width"]
        self.height = topology["image"]["height"]
        self.padded_width, self.padded_height = padded_size(self.width, self.height)
        self.shape = self.padded_height // 8, self.padded_width // 8
        with Image.open(geometry["directory"] / "region_map.png") as image:
            self.labels = np.array(image)
        with Image.open(geometry["directory"] / "bleed_owner.png") as image:
            owner = np.array(image)
        owner = self.downsample(owner)
        self.lut = torch.as_tensor(np.load(Path(lut)), device=device)
        self.windows = {
            synchronized: build_windows(
                topology, owner, self.shape, config["margin_latent"], synchronized, device
            )
            for synchronized in (True, False)
        }
        covered = np.zeros(self.shape, dtype=bool)
        for window in self.windows[True]:
            covered.flat[window.bleed_destinations.cpu().numpy()] = True
        covered_pixels = covered.repeat(8, axis=0).repeat(8, axis=1)
        self.base_bleed = (self.labels == 10) & ~covered_pixels[:self.height, :self.width]
        self.flaps_and_base_bleed = np.isin(self.labels, (8, 9)) | self.base_bleed
        self.fixed_output = np.isin(self.labels, (0, 7)) | self.flaps_and_base_bleed
        self.replace_mask = torch.as_tensor(
            self.downsample(self.fixed_output).astype(bool), device=device
        )[None, None]

    def downsample(self, labels):
        canvas = np.zeros((self.padded_height, self.padded_width), dtype=np.uint8)
        canvas[:self.height, :self.width] = labels
        return canvas[4::8, 4::8]

    def encode(self, pixels):
        normalized = pixels.mul(2).sub(1).to(dtype=self.dtype)
        vae = self.pipe.vae
        return vae.encode(normalized).latent_dist.mode() * vae.config.scaling_factor

    def decode(self, latent):
        vae = self.pipe.vae
        decoded = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        return decoded.float().div(2).add(0.5).clamp(0, 1)

    def reference(self, brief):
        # VAE sees the same substrate and base swatch at every step.
        rgb = np.full((self.padded_height, self.padded_width, 3), 255, dtype=np.uint8)
        rgb[:self.height, :self.width][self.flaps_and_base_bleed] = colors(brief)
        image = torch.as_tensor(rgb, device=self.device).float().div(255)
        return self.encode(image.permute(2, 0, 1)[None]), rgb

    def condition(self, brief, window):
        positive, negative, pooled, pooled_negative = self.pipe.encode_prompt(
            prompt=prompt_for(brief, window.name), negative_prompt=NEGATIVE,
            device=self.device, do_classifier_free_guidance=True,
        )
        height, width = window.shape
        if window.rotation % 2:
            height, width = width, height
        size = [8 * height, 8 * width]
        time_ids = torch.tensor(
            size + [0, 0] + size, dtype=self.dtype, device=self.device
        )[None].repeat(2, 1)
        return torch.cat((negative, positive)), {
            "text_embeds": torch.cat((pooled_negative, pooled)),
            "time_ids": time_ids,
        }

    def predict(self, latent, t, windows, conditions):
        flat = latent.flatten(-2)
        total = torch.zeros_like(flat, dtype=torch.float32)
        count = torch.zeros(flat.shape[-1], device=self.device)
        for window, (tokens, kwargs) in zip(windows, conditions, strict=True):
            tile = window.gather(flat)
            model_input = self.pipe.scheduler.scale_model_input(tile.repeat(2, 1, 1, 1), t)
            prediction = self.pipe.unet(
                model_input, t, encoder_hidden_states=tokens, added_cond_kwargs=kwargs,
                return_dict=False,
            )[0]
            unconditioned, conditioned = prediction.chunk(2)
            guided = unconditioned + self.config["guidance"] * (conditioned - unconditioned)
            window.merge(guided, total, count)
        return (total / count.clamp_min(1e-6)).unflatten(-1, self.shape).to(self.dtype)

    def color_constraint(self, latent, noise_prediction, t):
        alpha = self.pipe.scheduler.alphas_cumprod[int(t)].to(self.device).float()
        alpha_root, sigma = alpha.sqrt(), (1 - alpha).sqrt()
        x0 = (latent.float() - sigma * noise_prediction.float()) / alpha_root
        rgb = self.decode(x0.to(self.dtype))
        mapped = project_rgb(rgb.permute(0, 2, 3, 1), self.lut)
        corrected = self.encode(mapped.permute(0, 3, 1, 2))
        x0 = x0.lerp(corrected.float(), self.config["projection_strength"])
        return ((latent.float() - alpha_root * x0) / sigma.clamp_min(1e-6)).to(self.dtype)

    @torch.inference_mode()
    def generate(self, brief, seed, *, synchronized=True, project=True):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.config["steps"], device=self.device)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("The color projection expects an epsilon prediction scheduler")
        noise = torch.randn((1, 4, *self.shape), generator=generator, device=self.device,
                            dtype=self.dtype)
        latent = noise * scheduler.init_noise_sigma
        anchor, paper = self.reference(brief)
        windows = self.windows[synchronized]
        conditions = [self.condition(brief, window) for window in windows]
        first_projection = round(len(scheduler.timesteps) * (1 - self.config["projection_tail"]))
        for index, t in enumerate(scheduler.timesteps):
            anchor_at_t = scheduler.add_noise(anchor, noise, t[None])
            latent = torch.where(self.replace_mask, anchor_at_t, latent)
            prediction = self.predict(latent, t, windows, conditions)
            use_projection = (
                project and index >= first_projection
                and (index - first_projection) % self.config["projection_interval"] == 0
            )
            if use_projection:
                prediction = self.color_constraint(latent, prediction, t)
            latent = scheduler.step(prediction, t, latent, return_dict=False)[0]
        latent = torch.where(self.replace_mask, anchor, latent)
        decoded = self.decode(latent)[0].permute(1, 2, 0)[:self.height, :self.width]
        rgb = decoded.mul(255).round().to(torch.uint8).cpu().numpy()
        # Remove VAE spillover across region boundaries at output resolution.
        rgb[self.fixed_output] = paper[:self.height, :self.width][self.fixed_output]
        return Image.fromarray(rgb)
