"""Single-sided texture statistics in the clean-latent estimate of SDXL."""

import numpy as np
import torch
from torch.nn import functional as F


HIGHPASS_RADIUS_MM = 1.5
LOCAL_STD_RADIUS_MM = 3.0


def _box_mean(tensor, radius_y, radius_x, *, cyclic_x=False):
    if radius_x:
        tensor = F.pad(tensor, (radius_x, radius_x, 0, 0),
                       mode="circular" if cyclic_x else "replicate")
    if radius_y:
        tensor = F.pad(tensor, (0, 0, radius_y, radius_y), mode="replicate")
    return F.avg_pool2d(tensor, (2 * radius_y + 1, 2 * radius_x + 1), stride=1)


def _radii(radius_mm, cell_mm, direction):
    y_mm, x_mm = cell_mm
    return (0 if direction == "tangent" else max(1, round(radius_mm / y_mm)),
            max(1, round(radius_mm / x_mm)))


def cap_interior(layout, box_mm, protected_mm, editable_mask=None):
    """Select editable latent-cell centers strictly more than the margin from a cap edge."""
    height, width = layout.latent_shape
    x0, y0, x1, y1 = layout.center_rect
    core_w, core_h = x1 - x0, y1 - y0
    dx, dy = float(box_mm["length"]) / core_w, float(box_mm["width"]) / core_h
    ix, iy = np.arange(core_w), np.arange(core_h)
    x_dist = (np.minimum(ix, core_w - ix - 1) + 0.5) * dx
    y_dist = (np.minimum(iy, core_h - iy - 1) + 0.5) * dy
    support = np.zeros((height, width), dtype=bool)
    support[y0:y1, x0:x1] = np.minimum(y_dist[:, None], x_dist[None, :]) > protected_mm
    known = np.asarray(layout.known_mask, dtype=bool)
    if np.any(support & known):
        raise ValueError("Texture guidance cannot edit frozen ring context")
    if editable_mask is not None:
        edit = np.asarray(editable_mask, dtype=bool)
        if edit.shape != support.shape:
            raise ValueError("Texture guidance requires an aligned edit mask")
        support &= edit
    return support, (dy, dx)


def ring_reference_std(ring_latent, layout, box_mm, reference_mm, direction):
    """Take raw high-pass statistics from the physically registered frozen side interiors."""
    body0, body1 = layout.ring.body_rows
    height_mm = float(box_mm["height"])
    cell_y = height_mm / (body1 - body0)
    cell_x = (2 * (float(box_mm["length"]) + float(box_mm["width"]))) / layout.ring.latent_shape[1]
    depth_mm = (np.arange(body1 - body0) + 0.5) * cell_y
    depths = np.flatnonzero((depth_mm >= reference_mm[0]) & (depth_mm < reference_mm[1]))
    if not len(depths):
        raise ValueError("The physical 9–12 mm ring reference has no latent cells")
    if tuple(ring_latent.shape[-2:]) != layout.ring.latent_shape:
        raise ValueError("The frozen reference and ring geometry must match")
    clean = ring_latent.detach().float()
    radius = _radii(HIGHPASS_RADIUS_MM, (cell_y, cell_x), direction)
    detail = clean - _box_mean(clean, *radius, cyclic_x=True)
    source_bands, named = [], []
    native = "F" if layout.face == "T" else "B"
    for side, edge, *_ in layout.original.edge_sources.values():
        if direction == "tangent" and side != native:
            continue
        x0, y0, x1, y1 = layout.ring.face_rects[side]
        if edge == "top":
            rows = y0 + depths
        elif edge == "bottom":
            rows = y1 - 1 - depths
        else:
            raise ValueError("Cap reference requires a horizontal edge of the frozen side")
        indices = torch.as_tensor(rows, device=clean.device)
        source_bands.append(detail[..., x0:x1].index_select(-2, indices).flatten(-2))
        named.append(side)
    if not source_bands:
        raise ValueError("No matching frozen side reference for the specified guidance")
    values = torch.cat(source_bands, dim=-1)
    return values.std(dim=-1, unbiased=False)[..., None, None], {
        "reference_sides": named, "reference_latent_cells": values.shape[-1],
        "reference_latent_rows_mm": depth_mm[depths].tolist(),
        "reference_highpass_radius_latents": list(radius),
    }


def uplift_clean_latent(clean, target, editable, *, gamma, highpass_radius, local_radius, direction):
    """Apply a zero-mean increment only where local detail falls below the frozen target."""
    highpass = clean - _box_mean(clean, *highpass_radius)
    local_mean = _box_mean(highpass, *local_radius)
    local_second = _box_mean(highpass.square(), *local_radius)
    sigma = (local_second - local_mean.square()).clamp_min(0).sqrt()
    cutoff = 0.9 if direction == "tangent" else 1.0
    active = editable & (sigma < cutoff * target)
    amplitude = gamma * (target - sigma).clamp_min(0)
    proposed = amplitude * (highpass - local_mean) / sigma.clamp_min(1e-6)
    delta = torch.where(active, proposed, 0.0)
    dimensions = (-1,) if direction == "tangent" else (-2, -1)
    count = active.sum(dim=dimensions, keepdim=True).clamp_min(1)
    delta = torch.where(active, delta - delta.sum(dim=dimensions, keepdim=True) / count, 0.0)
    return delta, active


class TextureGuidance:
    """Convert a guided clean estimate back to epsilon on the original DDIM schedule."""

    def __init__(self, engine, layout, frozen_ring, gamma, settings, *, editable_mask, direction):
        if direction not in ("isotropic", "tangent") or not 0 < gamma <= 1:
            raise ValueError("Invalid joint-cap texture guidance configuration")
        if engine.pipe.scheduler.config.prediction_type != "epsilon":
            raise ValueError("Texture guidance expects epsilon prediction")
        self.gamma = float(gamma)
        self.direction = direction
        self.first = int(settings["first_step"])
        self.last = int(settings["last_step"])
        self.stride = int(settings["stride"])
        self.protected_mm = float(settings["protected_mm"])
        self.reference_mm = tuple(float(v) for v in settings["reference_mm"])
        if (not 0 <= self.first <= self.last or self.stride < 1 or self.protected_mm < 0
                or not 0 <= self.reference_mm[0] < self.reference_mm[1]):
            raise ValueError("Invalid texture-guidance physical or temporal bounds")
        box_mm = engine.topology["box_mm"]
        editable, cell_mm = cap_interior(layout, box_mm, self.protected_mm, editable_mask)
        self.editable = torch.as_tensor(editable, device=engine.device)[None, None]
        self.editable_cells = int(editable.sum())
        self.target, self.reference_proof = ring_reference_std(
            frozen_ring.to(device=engine.device), layout, box_mm, self.reference_mm, direction
        )
        self.highpass = _radii(HIGHPASS_RADIUS_MM, cell_mm, direction)
        self.local = _radii(LOCAL_STD_RADIUS_MM, cell_mm, direction)
        self.alphas = engine.pipe.scheduler.alphas_cumprod.to(device=engine.device, dtype=torch.float32)
        self.trace = []

    def __call__(self, index, timestep, latent, epsilon):
        if not (self.first <= index <= self.last and (index - self.first) % self.stride == 0):
            return epsilon
        alpha = self.alphas.index_select(0, timestep.to(device=latent.device).reshape(1))[0]
        a, sigma = alpha.sqrt(), (1 - alpha).sqrt()
        clean = (latent.float() - sigma * epsilon.float()) / a
        change, active = uplift_clean_latent(
            clean, self.target, self.editable, gamma=self.gamma, highpass_radius=self.highpass,
            local_radius=self.local, direction=self.direction,
        )
        reduction = (-1,) if self.direction == "tangent" else (-2, -1)
        drift = change.mean(dim=reduction).abs().max()
        self.trace.append((index, active.sum(), drift))
        guided = (epsilon.float() - (a / sigma.clamp_min(1e-6)) * change).to(dtype=epsilon.dtype)
        return torch.where(active, guided, epsilon)

    def evidence(self):
        return {
            "mode": self.direction, "gamma": self.gamma, "protected_mm": self.protected_mm,
            "reference_mm": list(self.reference_mm), "highpass_radius_mm": HIGHPASS_RADIUS_MM,
            "local_std_radius_mm": LOCAL_STD_RADIUS_MM, "highpass_radius_latents": list(self.highpass),
            "local_std_radius_latents": list(self.local), "eligible_latent_cells": self.editable_cells,
            "reference_std_per_channel": self.target.flatten().tolist(),
            "no_update_above_fraction_of_reference": 0.9 if self.direction == "tangent" else 1.0,
            "executed_steps": [index for index, _, _ in self.trace],
            "updated_channels_or_cells": [int(count) for _, count, _ in self.trace],
            "max_abs_guarded_mean_change_float32": max((float(drift) for _, _, drift in self.trace), default=0.0),
            "mean_preservation": "per_row_and_channel" if self.direction == "tangent" else "per_channel",
            **self.reference_proof,
        }


def texture_guidance(engine, layout, frozen_ring, gamma, settings, *, editable_mask=None, direction="isotropic"):
    return TextureGuidance(engine, layout, frozen_ring, gamma, settings,
                           editable_mask=editable_mask, direction=direction)
