"""ILVR refinement of cap latents on the unchanged inpainting DDIM schedule."""

import torch
from torch.nn import functional as F


def latent_lowpass(latent, factor):
    """Downsample by exact area averaging and return to the original latent grid."""
    height, width = latent.shape[-2:]
    if type(factor) is not int or factor < 1 or height % factor or width % factor:
        raise ValueError(f"ILVR factor {factor!r} must divide the latent grid {(height, width)}")
    reduced = F.avg_pool2d(latent.float(), kernel_size=factor, stride=factor)
    return F.interpolate(reduced, size=(height, width), mode="bilinear", align_corners=False)


def refine_cap_latent(latent, reference, factor, core_rect):
    """Apply the ILVR residual on the cap core, reading the complete latent canvas."""
    if reference.shape != latent.shape:
        raise ValueError("ILVR proposal and reference must share the same latent canvas")
    height, width = latent.shape[-2:]
    x0, y0, x1, y1 = core_rect
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid ILVR core rectangle: {core_rect}")
    correction = latent_lowpass(reference, factor) - latent_lowpass(latent, factor)
    area = (..., slice(y0, y1), slice(x0, x1))
    result = latent.clone()
    result[area] = (latent[area].float() + correction[area]).to(latent.dtype)
    return result


class ILVRCapMixin:
    """Extend the native cap sampler with a low-pass harmonic reference."""

    def _ilvr_denoise(self, shape, generator, text, pooled, harmonic_rgb, factor,
                     guided_fraction, *, core_rect):
        width, height = shape
        if width % 64 or height % 64 or min(shape) < 512:
            raise ValueError(f"ILVR expects an SDXL canvas of at least 512 px, aligned to 64 px: {shape}")
        if not 0 < guided_fraction <= 1:
            raise ValueError("ILVR guidance fraction must be in (0, 1]")
        pixels = torch.as_tensor(harmonic_rgb, device=self.device)
        if pixels.shape != (height, width, 3) or not pixels.is_floating_point():
            raise ValueError("Expected the normalized H×W×3 harmonic RGB prior")
        harmonic = self.encode(pixels.permute(2, 0, 1)[None])
        latent_shape = (1, 4, height // 8, width // 8)
        if harmonic.shape != latent_shape:
            raise ValueError(f"The harmonic prior did not encode to {latent_shape}")

        scheduler = self.pipe.scheduler
        steps = int(self.config["steps"])
        scheduler.set_timesteps(steps, device=self.device)
        timesteps = scheduler.timesteps
        if scheduler.order != 1 or len(timesteps) != steps:
            raise ValueError("ILVR requires the full, first-order DDIM schedule")
        guided_steps = round(steps * guided_fraction)
        if guided_steps < 1:
            raise ValueError("The ILVR guidance fraction must select at least one step")

        noise = torch.randn(latent_shape, generator=generator, device=self.device, dtype=self.dtype)
        latent = noise * scheduler.init_noise_sigma
        harmonic = harmonic.float()
        reference_noise = noise.float()
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
        for index, timestep in enumerate(timesteps):
            epsilon = self._predict(latent, timestep, text, condition, None)
            if project and index >= first_projection and (index - first_projection) % interval == 0:
                epsilon = self.color_constraint(latent, epsilon, timestep)
                projection_steps.append(index)
            latent = scheduler.step(epsilon, timestep, latent, return_dict=False)[0]
            if index < guided_steps:
                # Algorithm 1 refines the proposal at its destination time.
                # Reuse the initial draw for the forward path throughout this run.
                reference = harmonic
                if index + 1 < steps:
                    reference = scheduler.add_noise(harmonic, reference_noise, timesteps[index + 1:index + 2])
                latent = refine_cap_latent(latent, reference, factor, core_rect)

        targets = timesteps[1:guided_steps + 1].detach().cpu().tolist()
        if guided_steps == steps:
            targets.append(None)
        return latent, {
            "factor": factor, "guided_fraction": guided_fraction, "strength": 1.0,
            "full_steps": steps, "guided_steps": guided_steps, "guided_target_timesteps": targets,
            "projection_steps": projection_steps, "write_scope": "cap_core",
            "core_rect_latent": list(core_rect), "lowpass": "area_mean_then_bilinear",
            "reference_noise": "shared_initial_draw",
        }
