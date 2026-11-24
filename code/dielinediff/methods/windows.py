"""Pack six upright face windows against a common latent atlas."""

from dataclasses import dataclass

import numpy as np
import torch

EDGES = ("top", "right", "bottom", "left")


def latent_rect(rect, scale=8):
    """Rasterize half-open bounds at the same centers used to sample labels."""
    return tuple((value + scale - 1 - scale // 2) // scale for value in rect)


def strip_coordinates(rect, edge, along, depth):
    """Indices counted inward from the indicated face edge."""
    x0, y0, x1, y1 = rect
    if edge == "top":
        return x0 + along, y0 + depth
    if edge == "right":
        return x1 - 1 - depth, y0 + along
    if edge == "bottom":
        return x0 + along, y1 - 1 - depth
    return x0 + depth, y0 + along


@dataclass
class FaceWindow:
    name: str
    rotation: int
    shape: tuple
    read_ids: torch.Tensor
    write_ids: torch.Tensor
    write_weights: torch.Tensor
    bleed_destinations: torch.Tensor
    bleed_sources: torch.Tensor

    def gather(self, flat):
        tile = flat.index_select(-1, self.read_ids).unflatten(-1, self.shape)
        return torch.rot90(tile, self.rotation, dims=(-2, -1))

    def merge(self, tile, total, count):
        tile = torch.rot90(tile, -self.rotation, dims=(-2, -1)).flatten(-2)
        total.index_add_(-1, self.write_ids, tile.float() * self.write_weights)
        count.index_add_(-1, self.write_ids, self.write_weights)
        if self.bleed_sources.numel():
            color = tile.index_select(-1, self.bleed_sources).float()
            total.index_add_(-1, self.bleed_destinations, color)
            count.index_add_(-1, self.bleed_destinations, torch.ones_like(color[0, 0]))


def build_windows(topology, bleed_owner, shape, margin, synchronized, device):
    if margin < 1:
        raise ValueError("The edge band must be at least one latent pixel")
    rects = {name: latent_rect(face["rect"]) for name, face in topology["faces"].items()}
    adjacent = {}
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        adjacent[a["face"], a["edge"]] = b["face"], b["edge"], seam["reverse"]
        adjacent[b["face"], b["edge"]] = a["face"], a["edge"], seam["reverse"]

    atlas_h, atlas_w = shape
    windows = []
    for identifier, (name, face) in enumerate(topology["faces"].items()):
        x0, y0, x1, y1 = rects[name]
        width, height = x1 - x0, y1 - y0
        if min(width, height) < 2 * margin or max(width, height) + 2 * margin > 128:
            raise ValueError(f"Face {name} is incompatible with the configured SDXL window")
        yy, xx = np.mgrid[-margin:height + margin, -margin:width + margin]
        own_x = np.clip(xx, 0, width - 1) + x0
        own_y = np.clip(yy, 0, height - 1) + y0
        read = own_y * atlas_w + own_x
        core = (0 <= xx) & (xx < width) & (0 <= yy) & (yy < height)
        weights = core.astype(np.float32)
        for edge in EDGES:
            along, depth, length = {
                "top": (xx, -yy - 1, width),
                "right": (yy, xx - width, height),
                "bottom": (xx, yy - height, width),
                "left": (yy, -xx - 1, height),
            }[edge]
            band = (0 <= along) & (along < length) & (0 <= depth) & (depth < margin)
            if not synchronized:
                continue
            other, other_edge, reverse = adjacent[name, edge]
            other_rect = rects[other]
            ox0, oy0, ox1, oy1 = other_rect
            other_length = ox1 - ox0 if other_edge in ("top", "bottom") else oy1 - oy0
            normalized = (along[band] + 0.5) / length
            if reverse:
                normalized = 1 - normalized
            corresponding = np.minimum((normalized * other_length).astype(int), other_length - 1)
            strip_x, strip_y = strip_coordinates(other_rect, other_edge, corresponding, depth[band])
            read[band] = strip_y * atlas_w + strip_x
            weights[band] = (margin - depth[band]) / (margin + 1)

        atlas_x = np.clip(xx + x0, 0, atlas_w - 1)
        atlas_y = np.clip(yy + y0, 0, atlas_h - 1)
        bleed = (~core) & (bleed_owner[atlas_y, atlas_x] == identifier + 1)
        bleed_targets = atlas_y[bleed] * atlas_w + atlas_x[bleed]
        read[bleed & (weights == 0)] = (atlas_y * atlas_w + atlas_x)[bleed & (weights == 0)]

        windows.append(FaceWindow(
            name, face["upright_rotation"] // 90, tuple(read.shape),
            torch.as_tensor(read.reshape(-1), dtype=torch.long, device=device),
            torch.as_tensor(read.reshape(-1), dtype=torch.long, device=device),
            torch.as_tensor(weights.reshape(-1), dtype=torch.float32, device=device),
            torch.as_tensor(bleed_targets, dtype=torch.long, device=device),
            torch.as_tensor(np.flatnonzero(bleed), dtype=torch.long, device=device),
        ))
    return windows
