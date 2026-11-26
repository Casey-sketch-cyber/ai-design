"""Differential Diffusion over the core of an already registered SDEdit cap."""

from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .v3_capgeom_init import sdedit_latents, sdedit_schedule


E4_SETTINGS = {
    "edge_mm": 3.0,
    "interior_mm": 15.0,
    "edge_strength": 0.85,
    "interior_strength": 0.70,
    "transition": "smoothstep",
}


def cap_strength_map(layout, box_mm, settings):
    """Sample the change map at latent-cell centers in physical cap coordinates."""
    edge_mm = float(settings["edge_mm"])
    inner_mm = float(settings["interior_mm"])
    edge_strength = float(settings["edge_strength"])
    inner_strength = float(settings["interior_strength"])
    face_mm = float(box_mm["length"]), float(box_mm["width"])
    if (not np.isfinite((*face_mm, edge_mm, inner_mm, edge_strength, inner_strength)).all()
            or not 0 <= edge_mm < inner_mm <= min(face_mm) / 2
            or not 0 < inner_strength <= edge_strength <= 1):
        raise ValueError("Invalid E4 cap distances or editing strengths")
    if settings["transition"] != "smoothstep":
        raise ValueError("E4 expects the registered smoothstep transition")
    height, width = layout.latent_shape
    x0, y0, x1, y1 = layout.center_rect
    known = np.asarray(layout.known_mask, dtype=bool)
    if (not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
            or known.shape != (height, width) or known[y0:y1, x0:x1].any()):
        raise ValueError("The E4 core must be disjoint from the frozen ring context")
    core_width, core_height = x1 - x0, y1 - y0
    x = np.arange(core_width)
    y = np.arange(core_height)
    from_x_fold = (np.minimum(x, core_width - 1 - x) + 0.5) * face_mm[0] / core_width
    from_y_fold = (np.minimum(y, core_height - 1 - y) + 0.5) * face_mm[1] / core_height
    distance = np.minimum(from_x_fold[None, :], from_y_fold[:, None])
    progress = np.clip((distance - edge_mm) / (inner_mm - edge_mm), 0.0, 1.0)
    blend = progress * progress * (3 - 2 * progress)
    plane = np.full((height, width), edge_strength, dtype=np.float64)
    plane[y0:y1, x0:x1] = edge_strength + (inner_strength - edge_strength) * blend
    return plane


def save_strength_map(layout, box_mm, settings, path):
    """Export the exact latent-cell map as grayscale with code round(255 × strength)."""
    strength = cap_strength_map(layout, box_mm, settings)
    image = Image.fromarray(np.rint(strength * 255).astype(np.uint8))
    expanded = image.resize((layout.width_px, layout.height_px), resample=Image.Resampling.NEAREST)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expanded.save(destination)
    return destination


def differential_sdedit(engine, layout, initial_rgb, generator, text, pooled, *, settings):
    """Run nested strength masks before the existing inpainting predictions.

    Levin and Fried, Differential Diffusion, Algorithm 1 and §3.3:
    https://doi.org/10.1111/cgf.70040
    """
    strength = cap_strength_map(layout, engine.topology["box_mm"], settings)
    edge_strength = float(settings["edge_strength"])
    height, width = layout.height_px, layout.width_px
    rgb = np.asarray(initial_rgb)
    if (rgb.shape != (height, width, 3) or not np.issubdtype(rgb.dtype, np.floating)
            or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1):
        raise ValueError("E4 expects a normalized floating-point full-canvas HWC RGB prior")
    pixels = torch.as_tensor(rgb, device=engine.device).permute(2, 0, 1)[None]
    initial = engine.encode(pixels)
    if tuple(initial.shape[-2:]) != strength.shape:
        raise ValueError("E4 strength map must align with the cap VAE latents")
    scheduler = engine.pipe.scheduler
    steps = int(engine.config["steps"])
    start, timesteps = sdedit_schedule(scheduler, steps, edge_strength, engine.device)
    counts = np.floor(strength * steps).astype(np.int64)
    minimum = int(counts.min())
    if minimum < 1 or int(counts.max()) != len(timesteps):
        raise ValueError("The E4 strength map must fit the original DDIM suffix")
    starts = torch.as_tensor(steps - counts, device=initial.device)[None, None]
    last_start = steps - minimum
    latent, noise = sdedit_latents(scheduler, initial, timesteps[0], generator)
    condition = {
        "text_embeds": pooled,
        "time_ids": torch.tensor((height, width, 0, 0, height, width), dtype=engine.dtype,
                                 device=engine.device)[None].expand(2, -1),
    }
    projection_start = round(steps * (1 - float(engine.config.get("projection_tail", 0.30))))
    interval = int(engine.config.get("projection_interval", 3))
    project = engine.config.get("projection_strength", 0) > 0
    projected = []
    restored = []
    for index, timestep in enumerate(timesteps, start):
        # The initialization already supplies every cell's exact noised reference at start.
        if start < index <= last_start:
            reference = scheduler.add_noise(initial, noise, timestep.reshape(1))
            latent = torch.where(starts >= index, reference, latent)
            restored.append(index)
        epsilon = engine._predict(latent, timestep, text, condition, None)
        if project and index >= projection_start and (index - projection_start) % interval == 0:
            epsilon = engine.color_constraint(latent, epsilon, timestep)
            projected.append(index)
        latent = scheduler.step(epsilon, timestep, latent, return_dict=False)[0]

    x0, y0, x1, y1 = layout.center_rect
    levels, frequencies = np.unique(counts[y0:y1, x0:x1], return_counts=True)
    return latent, {
        "mode": "differential_sdedit",
        "strength": edge_strength,
        "strength_settings": dict(settings),
        "strength_shape_latent": list(strength.shape),
        "strength_map_sha256": sha256(strength.astype("<f8", copy=False).tobytes()).hexdigest(),
        "core_latent_step_histogram": {str(level): int(n) for level, n in zip(levels, frequencies, strict=True)},
        "noise_draws": 1,
        "full_steps": steps,
        "start_index": start,
        "executed_steps": len(timesteps),
        "first_timestep": int(timesteps[0]),
        "restoration_steps": restored,
        "projection_steps": projected,
    }
