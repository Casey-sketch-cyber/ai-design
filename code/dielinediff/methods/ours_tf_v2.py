"""DieLineDiff-TF v2 with full-height art and direct bleed predictions."""

import numpy as np
import torch
from PIL import Image

from .ours_tf import DieLineDiff
from .sdxl import NEGATIVE, colors, load_pipeline, prompt_for
from .windows_v2 import build_windows, save_coverage_heatmap


def window_prompt(brief, face):
    """Reuse B2's global template with the brief's face-specific art direction."""
    if face == "F":
        return prompt_for(brief)
    category = brief["category"].replace("_", " ")
    description = brief["top_prompt"] if face == "T" else brief["side_prompt"]
    if face == "Btm":
        product_name = brief.get("product_name_zh")
        if product_name:
            description = description.replace(product_name, "")
    return prompt_for({**brief, "front_prompt": f"{category} packaging motifs, {description}"})


class DieLineDiffV2(DieLineDiff):
    """Reuse the SDXL latent codec and the E01 ICC projection."""

    def __init__(self, models, geometry, config, device, lut):
        self.config = config
        self.device = device
        self.topology = geometry["topology"]
        with Image.open(geometry["directory"] / "region_map.png") as image:
            self.labels = np.array(image)
        self.height, self.width = self.labels.shape
        if self.height % 8 or self.width % 8:
            raise ValueError("E01b requires an atlas aligned to the VAE grid")
        self.plans = {}
        plan = self.window_plan(True)
        self.shape = plan.shape
        fixed = np.isin(plan.labels, (0, 7, 8, 9))
        self.replace_mask = torch.as_tensor(fixed, device=device)[None, None]
        self.active = ~self.replace_mask.flatten()
        self.required = int(self.active.sum().item())
        self.pipe = load_pipeline(models, device=device)
        self.dtype = self.pipe.unet.dtype
        self.lut = torch.as_tensor(np.load(lut), device=device)
        self.last_coverage = None

    def window_plan(self, synchronized):
        if synchronized not in self.plans:
            self.plans[synchronized] = build_windows(
                self.topology, self.labels, stride_px=self.config["window_stride"],
                lid_band_latent=self.config["top_band"],
                synchronized=synchronized, device=self.device,
                transition_mm=self.config.get("transition_mm", 2.0),
            )
        return self.plans[synchronized]

    def save_coverage(self, path, *, synchronized=True):
        save_coverage_heatmap(self.window_plan(synchronized), path)

    def reference(self, brief):
        rgb = np.full((*self.labels.shape, 3), 255, dtype=np.uint8)
        rgb[np.isin(self.labels, (8, 9))] = colors(brief)
        image = torch.as_tensor(rgb, device=self.device).float().div(255)
        return self.encode(image.permute(2, 0, 1)[None])

    def condition(self, brief, window):
        pos, neg, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=window_prompt(brief, window.primary_face),
            negative_prompt=NEGATIVE, device=self.device, do_classifier_free_guidance=True,
        )
        pixel_height, pixel_width = (value * 8 for value in window.shape)
        if min(pixel_height, pixel_width) < 512:
            raise ValueError(f"{window.name} is smaller than 512 px on its short side")
        size = (pixel_height, pixel_width)
        time_ids = torch.tensor(
            (*size, 0, 0, *size), dtype=self.dtype, device=self.device
        )[None].expand(2, -1)
        return torch.cat((neg, pos)), {
            "text_embeds": torch.cat((neg_pooled, pooled)),
            "time_ids": time_ids,
        }

    def predict(self, latent, t, plan, conditions):
        flat = latent.flatten(-2)
        total = torch.zeros_like(flat, dtype=torch.float32)
        weight = torch.zeros(flat.shape[-1], dtype=torch.float32, device=self.device)
        for window in plan.windows:
            tokens, kwargs = conditions[window.name]
            tile = window.gather(flat)
            if tuple(tile.shape[-2:]) != tuple(window.shape):
                raise ValueError(f"{window.name} does not match its SDXL conditioning size")
            model_input = self.pipe.scheduler.scale_model_input(tile.repeat(2, 1, 1, 1), t)
            prediction = self.pipe.unet(
                model_input, t, encoder_hidden_states=tokens, added_cond_kwargs=kwargs,
                return_dict=False,
            )[0]
            unconditioned, conditioned = prediction.chunk(2)
            guided = unconditioned + self.config["guidance"] * (conditioned - unconditioned)
            window.merge(guided, total, weight)

        # Fixed pixels are restored in latent space. Every other pixel must be
        # represented by positive window weights at this denoising step.
        plan.assert_covered(weight)
        minimum_weight = weight[self.active].min().item()
        if not np.isfinite(minimum_weight) or minimum_weight <= 0:
            raise RuntimeError("E01b denoising coverage contains a nonpositive or invalid weight")
        divisor = torch.where(weight > 0, weight, 1)
        blended = (total / divisor).unflatten(-1, self.shape).to(self.dtype)
        return blended, minimum_weight

    @torch.inference_mode()
    def generate(self, brief, seed, *, synchronized=True, project=True, anchor_outside=True):
        self.last_coverage = None
        plan = self.window_plan(synchronized)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.config["steps"], device=self.device)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("The color projection expects an epsilon prediction scheduler")

        noise = torch.randn((1, 4, *self.shape), generator=generator, device=self.device,
                            dtype=self.dtype)
        latent = noise * scheduler.init_noise_sigma
        anchor = self.reference(brief)
        # The D3 control keeps the original window coverage and releases only Outside.
        replace_mask = self.replace_mask
        if not anchor_outside:
            nonoutside = torch.as_tensor(plan.labels != 0, device=self.device)[None, None]
            replace_mask = replace_mask & nonoutside
        conditions = {window.name: self.condition(brief, window) for window in plan.windows}
        first_projection = round(len(scheduler.timesteps) * (1 - self.config["projection_tail"]))
        minimum_weight = float("inf")
        steps_checked = 0
        for index, t in enumerate(scheduler.timesteps):
            anchor_at_t = scheduler.add_noise(anchor, noise, t[None])
            latent = torch.where(replace_mask, anchor_at_t, latent)
            prediction, step_weight = self.predict(latent, t, plan, conditions)
            minimum_weight = min(minimum_weight, step_weight)
            steps_checked += 1
            if (project and index >= first_projection
                    and (index - first_projection) % self.config["projection_interval"] == 0):
                prediction = self.color_constraint(latent, prediction, t)
            latent = scheduler.step(prediction, t, latent, return_dict=False)[0]

        latent = torch.where(replace_mask, anchor, latent)
        decoded = self.decode(latent)[0].permute(1, 2, 0)
        if not torch.isfinite(decoded).all():
            raise RuntimeError("E01b VAE decoded a nonfinite image")
        rgb = decoded.mul(255).round().to(torch.uint8).cpu().numpy()
        self.last_coverage = {
            "covered": self.required,
            "required": self.required,
            "minimum_weight": minimum_weight,
            "steps_checked": steps_checked,
            "windows": len(plan.windows),
            "window_sizes_px": [[window.shape[1] * 8, window.shape[0] * 8] for window in plan.windows],
        }
        return Image.fromarray(rgb)
