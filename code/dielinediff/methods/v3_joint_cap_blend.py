"""Time-aligned blending for the original SDXL-base latent denoiser."""

import torch


def blended_patch(latent, known_clean, noise, mask, scheduler, timestep):
    """Blend the frozen reference at the current scheduler timestep.

    Blended Latent Diffusion, Avrahami et al. (SIGGRAPH 2023):
    https://arxiv.org/abs/2206.02779
    """
    reference = known_clean if timestep is None else scheduler.add_noise(
        known_clean, noise, timestep.reshape(1)
    )
    if mask.dtype == torch.bool:
        return torch.where(mask, reference, latent)
    return latent.lerp(reference, mask.to(dtype=latent.dtype))


def feather_known_mask(mask, *, periodic_x=False):
    """Give the inner one-cell ring boundary 0.5 alpha and keep all free cells at zero."""
    if mask.dtype != torch.bool or mask.ndim < 2 or min(mask.shape[-2:]) < 2:
        raise ValueError("Feathering needs a Boolean spatial keep mask")
    north = torch.cat((mask[..., :1, :], mask[..., :-1, :]), dim=-2)
    south = torch.cat((mask[..., 1:, :], mask[..., -1:, :]), dim=-2)
    if periodic_x:
        west = torch.roll(mask, 1, dims=-1)
        east = torch.roll(mask, -1, dims=-1)
    else:
        west = torch.cat((mask[..., :1], mask[..., :-1]), dim=-1)
        east = torch.cat((mask[..., 1:], mask[..., -1:]), dim=-1)
    boundary = mask & ~(north & south & west & east)
    return mask.float() - 0.5 * boundary.float()


def denoise_blended(engine, latent, timesteps, text, pooled, *, size, start_index=0,
                    known_clean=None, noise=None, known_mask=None, periodic_x=False,
                    feather_last=3, before_step=None, epsilon_transform=None):
    """Traverse a prepared DDIM suffix with optional context replacements and residual guidance."""
    width, height = size
    if tuple(latent.shape[-2:]) != (height // 8, width // 8) or not len(timesteps):
        raise ValueError("DDIM timesteps or the SDXL latent canvas is inconsistent")
    supplied = tuple(value is not None for value in (known_clean, noise, known_mask))
    if any(supplied) and not all(supplied):
        raise ValueError("Frozen BLD context requires clean latents, the original noise and a keep mask")
    if not 0 <= feather_last <= len(timesteps):
        raise ValueError("Invalid number of closing feather steps")
    blend = all(supplied)
    feather = None
    if blend:
        if (known_clean.shape != latent.shape or noise.shape != latent.shape
                or known_mask.dtype != torch.bool or known_mask.shape[-2:] != latent.shape[-2:]):
            raise ValueError("BLD context shape or dtype is not aligned with the latent canvas")
        if feather_last:
            feather = feather_known_mask(known_mask, periodic_x=periodic_x)

    condition = {
        "text_embeds": pooled,
        "time_ids": torch.tensor((height, width, 0, 0, height, width), dtype=engine.dtype,
                                 device=engine.device)[None].expand(2, -1),
    }
    scheduler = engine.pipe.scheduler
    steps = int(engine.config["steps"])
    projection_start = round(steps * (1 - float(engine.config.get("projection_tail", 0.30))))
    interval = int(engine.config.get("projection_interval", 3))
    project = engine.config.get("projection_strength", 0) > 0
    feather_from = start_index + len(timesteps) - feather_last
    projected, feathered = [], []
    for index, source_time in enumerate(timesteps, start_index):
        if before_step is not None:
            latent = before_step(index, source_time, latent)
        if blend:
            mask = known_mask
            if feather is not None and index >= feather_from:
                mask = feather
                feathered.append(index)
            latent = blended_patch(latent, known_clean, noise, mask, scheduler, source_time)
        epsilon = engine._predict(latent, source_time, text, condition, None)
        if epsilon_transform is not None:
            epsilon = epsilon_transform(index, source_time, latent, epsilon)
        if project and index >= projection_start and (index - projection_start) % interval == 0:
            epsilon = engine.color_constraint(latent, epsilon, source_time)
            projected.append(index)
        latent = scheduler.step(epsilon, source_time, latent, return_dict=False)[0]

    return latent, {"full_steps": steps, "start_index": start_index, "executed_steps": len(timesteps),
                    "first_timestep": int(timesteps[0]), "projection_steps": projected,
                    "feather_steps": feathered}
