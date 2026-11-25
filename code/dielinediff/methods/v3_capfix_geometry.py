"""Physical cap-only resampling on the immutable E02 ring registration."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .v3_geometry import EDGES, LATENT_SCALE
from .windows import strip_coordinates


def _span(rect, edge):
    x0, y0, x1, y1 = rect
    return x1 - x0 if edge in ("top", "bottom") else y1 - y0


def _strip(rect, edge, length, band, halo):
    x0, y0, x1, y1 = rect
    along = np.arange(length)[None, :]
    depth = np.arange(band)[:, None] + halo
    if edge == "top":
        return y0 - 1 - depth, x0 + along
    if edge == "right":
        return y0 + along, x1 + depth
    if edge == "bottom":
        return y1 + depth, x0 + along
    return y0 + along, x0 - 1 - depth


def _pixel_rect(rect):
    return tuple(value * LATENT_SCALE for value in rect)


@dataclass(frozen=True)
class CapFixLayout:
    """A scaled cap canvas retaining the original ring's read-only edge coordinates."""

    original: object
    ring: object
    scale: int
    band: int
    latent_shape: tuple
    center_rect: tuple
    known_mask: np.ndarray
    bleed_cells: dict

    @property
    def face(self):
        return self.original.face

    @property
    def width_px(self):
        return self.latent_shape[1] * LATENT_SCALE

    @property
    def height_px(self):
        return self.latent_shape[0] * LATENT_SCALE

    def read_frozen_ring(self, ring_latents):
        original = self.original.read_frozen_ring(ring_latents)
        if self.scale == 1:
            return original
        frozen = ring_latents.new_zeros((*ring_latents.shape[:-2], *self.latent_shape))
        for edge in EDGES:
            original_span = _span(self.original.center_rect, edge)
            source_y, source_x = _strip(self.original.center_rect, edge, original_span, self.band,
                                        self.original.bleed_cells[edge])
            source_y = torch.as_tensor(source_y, device=ring_latents.device)
            source_x = torch.as_tensor(source_x, device=ring_latents.device)
            values = original[..., source_y, source_x]
            span = _span(self.center_rect, edge)
            fixed = F.interpolate(values, size=(self.band * self.scale, span),
                                  mode="bilinear", align_corners=False)
            dy, dx = _strip(self.center_rect, edge, span, self.band * self.scale, self.bleed_cells[edge])
            dy = torch.as_tensor(dy, device=ring_latents.device)
            dx = torch.as_tensor(dx, device=ring_latents.device)
            frozen[..., dy, dx] = fixed
        return frozen

    def read_frozen_ring_rgb(self, ring_image):
        """Read oriented full-resolution known pixels for the native inpainting input."""
        ring_rgb = np.asarray(ring_image)
        if ring_rgb.dtype != np.uint8 or ring_rgb.shape != (self.ring.height_px, self.ring.width_px, 3):
            raise ValueError("Expected the RGB image of the original frozen ring")
        known = np.zeros((self.height_px, self.width_px, 3), dtype=np.uint8)
        rect = _pixel_rect(self.center_rect)
        for edge in EDGES:
            side, side_edge, reverse, _ = self.original.edge_sources[edge]
            side_rect = _pixel_rect(self.ring.face_rects[side])
            length = _span(_pixel_rect(self.original.center_rect), edge)
            along = np.arange(length)[None, :]
            if reverse:
                along = length - 1 - along
            depth = np.arange(self.band * LATENT_SCALE)[:, None]
            sx, sy = strip_coordinates(side_rect, side_edge, along, depth)
            pixels = ring_rgb[sy, sx]
            target_length = _span(rect, edge)
            target_band = self.band * LATENT_SCALE * self.scale
            if pixels.shape[:2] != (target_band, target_length):
                pixels = np.asarray(Image.fromarray(pixels).resize(
                    (target_length, target_band), Image.Resampling.BILINEAR,
                ))
            dy, dx = _strip(rect, edge, target_length, target_band, self.bleed_cells[edge] * LATENT_SCALE)
            known[dy, dx] = pixels
        return Image.fromarray(known)


def build_capfix_layout(original, ring, band, scale):
    """Scale a cap without recomputing or moving any angular boundaries of its ring."""
    if scale not in (1, 2):
        raise ValueError("Cap scale must be 1 or 2")
    x0, y0, x1, y1 = original.center_rect
    if scale == 1:
        return CapFixLayout(original, ring, 1, band, original.latent_shape, original.center_rect,
                            original.known_mask, dict(original.bleed_cells))
    core_w = max(64, 2 * (x1 - x0))
    core_h = max(64, 2 * (y1 - y0))
    x0_new = 2 * x0 - (core_w - 2 * (x1 - x0)) // 2
    y0_new = 2 * y0 - (core_h - 2 * (y1 - y0)) // 2
    rect = (x0_new, y0_new, x0_new + core_w, y0_new + core_h)
    halos = {edge: amount * 2 for edge, amount in original.bleed_cells.items()}
    required_h = rect[3] + 2 * band + halos["bottom"]
    required_w = rect[2] + 2 * band + halos["right"]
    height = 8 * math.ceil(max(2 * original.latent_shape[0], required_h) / 8)
    width = 8 * math.ceil(max(2 * original.latent_shape[1], required_w) / 8)
    if rect[0] < 2 * band + halos["left"] or rect[1] < 2 * band + halos["top"]:
        raise ValueError("Scaled cap has insufficient room before its core")
    known = np.zeros((height, width), dtype=bool)
    for edge in EDGES:
        cy, cx = _strip(rect, edge, _span(rect, edge), band * 2, halos[edge])
        known[cy, cx] = True
    return CapFixLayout(original, ring, 2, band, known.shape, rect, known, halos)


def _print_rectangles(center_rect, bleed_cells):
    x0, y0, x1, y1 = _pixel_rect(center_rect)
    yield x0, y0, x1, y1
    for edge, rect in {
        "top": (x0, y0 - bleed_cells["top"] * LATENT_SCALE, x1, y0),
        "right": (x1, y0, x1 + bleed_cells["right"] * LATENT_SCALE, y1),
        "bottom": (x0, y1, x1, y1 + bleed_cells["bottom"] * LATENT_SCALE),
        "left": (x0 - bleed_cells["left"] * LATENT_SCALE, y0, x0, y1),
    }.items():
        if bleed_cells[edge]:
            yield rect


def cap_at_original_scale(layout, image):
    """Downsample only the owned print core and its free bleed, never the known strips."""
    if image.size != (layout.width_px, layout.height_px):
        raise ValueError("Decoded cap size does not match its sampling layout")
    if layout.scale == 1:
        return layout.original, image
    original = layout.original
    restored = Image.new("RGB", (original.width_px, original.height_px))
    pairs = zip(_print_rectangles(layout.center_rect, layout.bleed_cells),
                _print_rectangles(original.center_rect, original.bleed_cells), strict=True)
    for incoming, outgoing in pairs:
        rect_size = (outgoing[2] - outgoing[0], outgoing[3] - outgoing[1])
        crop = image.crop(incoming).resize(rect_size, Image.Resampling.LANCZOS)
        restored.paste(crop, outgoing[:2])
    return original, restored
