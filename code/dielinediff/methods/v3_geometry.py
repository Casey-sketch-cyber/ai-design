"""Register v3 ring and frozen-neighbor cap canvases to the print dieline."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch

from dielinediff.dieline.geometry import REGION_LABELS
from .windows import strip_coordinates


LATENT_SCALE = 8
SDXL_ALIGNMENT = 64
SIDES = ("L", "F", "R", "B")
CAPS = ("T", "Btm")
EDGES = ("top", "right", "bottom", "left")
BLEED = REGION_LABELS["Bleed"]


def _aligned(value, alignment=SDXL_ALIGNMENT):
    return math.ceil(value / alignment) * alignment


def _atlas_labels(region_map):
    labels = np.asarray(region_map)
    if labels.ndim != 2 or min(labels.shape) < LATENT_SCALE:
        raise ValueError("The atlas needs a two-dimensional region map")
    return labels


def atlas_canvas_size(region_map):
    """Return (width, height) for a flat UNet canvas padded on the right and bottom."""
    height, width = _atlas_labels(region_map).shape
    return _aligned(width), _aligned(height)


def atlas_region_ids(region_map):
    """Return the aligned B2-regional latent face labels; zero is the global context."""
    labels = _atlas_labels(region_map)
    width, height = atlas_canvas_size(labels)
    padded = np.zeros((height, width), dtype=np.uint8)
    data = np.where((labels >= 1) & (labels <= 6), labels, 0)
    padded[:labels.shape[0], :labels.shape[1]] = data
    mid = LATENT_SCALE // 2
    return padded[mid::LATENT_SCALE, mid::LATENT_SCALE].copy()


@dataclass(frozen=True)
class RingLayout:
    width_px: int
    height_px: int
    latent_shape: tuple
    body_rows: tuple
    face_rects: dict
    region_ids: np.ndarray
    buffer_mm: tuple


def _neighbors(topology):
    result = {}
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        result[a["face"], a["edge"]] = b["face"], b["edge"], seam["reverse"]
        result[b["face"], b["edge"]] = a["face"], a["edge"], seam["reverse"]
    return result


def build_ring_layout(topology, width_px, height_px):
    """Arrange the four side faces in one cyclic latent-aligned body band."""
    if any(not isinstance(size, int) or size < SDXL_ALIGNMENT or size % SDXL_ALIGNMENT
           for size in (width_px, height_px)):
        raise ValueError("Ring width and height must be positive multiples of 64 pixels")
    neighbors = _neighbors(topology)
    for name, next_name in zip(SIDES, SIDES[1:] + SIDES[:1], strict=True):
        if neighbors[name, "right"] != (next_name, "left", False):
            raise ValueError(f"The expected upright ring seam {name}/{next_name} is missing")
    dims = topology["box_mm"]
    widths = np.asarray([dims["width"], dims["length"], dims["width"], dims["length"]])
    circumference = float(widths.sum())
    latent_w, latent_h = width_px // LATENT_SCALE, height_px // LATENT_SCALE
    boundaries = np.rint(np.r_[0, np.cumsum(widths)] * latent_w / circumference).astype(int)
    body_nominal = dims["height"] * latent_w / circumference
    buffer = round((latent_h - body_nominal) / 2)
    body = (buffer, latent_h - buffer)
    if buffer <= 0 or body[0] >= body[1] or np.any(np.diff(boundaries) < 1):
        raise ValueError("The ring resolution leaves no room for an aligned body and buffers")
    body_ppm = (body[1] - body[0]) * LATENT_SCALE / dims["height"]
    buffer_mm = buffer * LATENT_SCALE / body_ppm
    if not 10 - 1e-7 <= buffer_mm <= 12 + 1e-7:
        raise ValueError(f"The aligned ring buffer must be 10-12 mm per side, got {buffer_mm:.3f} mm")
    labels = np.empty((latent_h, latent_w), dtype=np.uint8)
    rects = {}
    for index, name in enumerate(SIDES):
        left, right = int(boundaries[index]), int(boundaries[index + 1])
        rects[name] = (left, body[0], right, body[1])
        labels[:, left:right] = REGION_LABELS[name]
    return RingLayout(width_px, height_px, labels.shape, body, rects, labels, (buffer_mm, buffer_mm))


@dataclass(frozen=True)
class CapLayout:
    face: str
    width_px: int
    height_px: int
    latent_shape: tuple
    center_rect: tuple
    known_mask: np.ndarray
    ring_ids: np.ndarray
    ring_shape: tuple
    bleed_cells: dict
    edge_sources: dict

    def read_frozen_ring(self, ring_latents):
        """Gather the four immutable edge strips into cap order; leave free slots zero."""
        if tuple(ring_latents.shape[-2:]) != self.ring_shape:
            raise ValueError(f"{self.face} needs ring latent dimensions {self.ring_shape}")
        device = ring_latents.device
        indices = torch.as_tensor(self.ring_ids.clip(min=0).ravel(), dtype=torch.long, device=device)
        frozen = ring_latents.flatten(-2).index_select(-1, indices).unflatten(-1, self.latent_shape)
        mask = torch.as_tensor(self.known_mask, device=device)
        return torch.where(mask, frozen, 0)


def _native_fold(a, edge, b, b_edge, reverse):
    if reverse:
        return False
    x0, y0, x1, y1 = a
    ox0, oy0, ox1, oy1 = b
    return {
        "top": b_edge == "bottom" and y0 == oy1 and (x0, x1) == (ox0, ox1),
        "right": b_edge == "left" and x1 == ox0 and (y0, y1) == (oy0, oy1),
        "bottom": b_edge == "top" and y1 == oy0 and (x0, x1) == (ox0, ox1),
        "left": b_edge == "right" and x0 == ox1 and (y0, y1) == (oy0, oy1),
    }[edge]


def build_cap_layout(topology, ring, face, band_latent=16):
    """Place the cap and its printable bleed between four read-only side strips."""
    if face not in CAPS:
        raise ValueError(f"Expected one of {CAPS}")
    if not isinstance(band_latent, int) or band_latent < 16:
        raise ValueError("Each frozen neighbor strip needs at least sixteen latent cells")
    rects = ring.face_rects
    core_w, core_h = rects["F"][2] - rects["F"][0], rects["L"][2] - rects["L"][0]
    hidden = _neighbors(topology)
    atlas_faces = topology["faces"]
    box = topology["box_mm"]
    bleed_x = math.ceil(box["bleed"] * core_w / box["length"])
    bleed_y = math.ceil(box["bleed"] * core_h / box["width"])
    halos = {}
    for edge in EDGES:
        side, side_edge, reverse = hidden[face, edge]
        attached = _native_fold(atlas_faces[face]["rect"], edge, atlas_faces[side]["rect"],
                                side_edge, reverse)
        # The cap's real bleed is free to denoise. Virtual side strips cannot be exported as bleed.
        halos[edge] = 0 if attached else bleed_y if edge in ("top", "bottom") else bleed_x
    wanted_w = core_w + 2 * band_latent + halos["left"] + halos["right"]
    wanted_h = core_h + 2 * band_latent + halos["top"] + halos["bottom"]
    width_px = _aligned(max(wanted_w * LATENT_SCALE, 512))
    height_px = _aligned(max(wanted_h * LATENT_SCALE, 512))
    width, height = width_px // LATENT_SCALE, height_px // LATENT_SCALE
    x0 = band_latent + halos["left"] + (width - wanted_w) // 2
    y0 = band_latent + halos["top"] + (height - wanted_h) // 2
    x1, y1 = x0 + core_w, y0 + core_h
    source = np.full((height, width), -1, dtype=np.int64)
    edge_sources = {}
    for edge in EDGES:
        side, side_edge, reverse = hidden[face, edge]
        ox0, oy0, ox1, oy1 = rects[side]
        length = core_w if edge in ("top", "bottom") else core_h
        side_length = ox1 - ox0 if side_edge in ("top", "bottom") else oy1 - oy0
        side_depth = oy1 - oy0 if side_edge in ("top", "bottom") else ox1 - ox0
        if side_length != length or band_latent > side_depth:
            raise ValueError(f"Incompatible ring registration for {face} {edge} / {side} {side_edge}")
        along = np.arange(length)[None, :]
        depth = np.arange(band_latent)[:, None]
        ring_along = length - 1 - along if reverse else along
        rx, ry = strip_coordinates(rects[side], side_edge, ring_along, depth)
        phase = depth + halos[edge]
        if edge == "top":
            cx, cy = x0 + along, y0 - 1 - phase
        elif edge == "right":
            cx, cy = x1 + phase, y0 + along
        elif edge == "bottom":
            cx, cy = x0 + along, y1 + phase
        else:
            cx, cy = x0 - 1 - phase, y0 + along
        source[cy, cx] = ry * ring.latent_shape[1] + rx
        edge_sources[edge] = (side, side_edge, reverse, halos[edge])
    return CapLayout(face, width_px, height_px, source.shape, (x0, y0, x1, y1), source >= 0,
                     source, ring.latent_shape, halos, edge_sources)


def _owners(topology, labels, bleed_owner):
    owner = np.where((labels >= 1) & (labels <= 6), labels, 0)
    owner = np.where(labels == BLEED, bleed_owner, owner).astype(np.uint8)
    remaining = (labels != 0) & (labels != BLEED) & (owner == 0)
    yy, xx = np.nonzero(remaining)
    if len(xx):
        distances = []
        tags = []
        for face in SIDES + CAPS:
            x0, y0, x1, y1 = topology["faces"][face]["rect"]
            dx = np.maximum(np.maximum(x0 - xx - 0.5, xx + 0.5 - x1), 0)
            dy = np.maximum(np.maximum(y0 - yy - 0.5, yy + 0.5 - y1), 0)
            distances.append(dx * dx + dy * dy)
            tags.append(REGION_LABELS[face])
        owner[yy, xx] = np.asarray(tags, dtype=np.uint8)[np.argmin(distances, axis=0)]
    return owner


def _image_rgb(image, expected):
    array = np.asarray(image)
    if array.shape != (*expected, 3) or array.dtype != np.uint8:
        raise ValueError(f"Expected an RGB uint8 image with shape {(*expected, 3)}")
    return array


def _source_points(atlas_rect, latent_rect, x, y):
    x0, y0, x1, y1 = atlas_rect
    sx0, sy0, sx1, sy1 = (value * LATENT_SCALE for value in latent_rect)
    sx = sx0 + (x + 0.5 - x0) * (sx1 - sx0) / (x1 - x0)
    sy = sy0 + (y + 0.5 - y0) * (sy1 - sy0) / (y1 - y0)
    return sx, sy


def _bilinear(image, x, y, periodic_x=False):
    """Sample at pixel-center coordinates with horizontal wrapping only for the ring."""
    height, width = image.shape[:2]
    x = x - 0.5
    y = np.clip(y - 0.5, 0, height - 1)
    if not periodic_x:
        x = np.clip(x, 0, width - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    xfrac, yfrac = (x - x0)[:, None], (y - y0)[:, None]
    x1 = np.minimum(x0 + 1, width - 1)
    if periodic_x:
        x1 = (x0 + 1) % width
        x0 %= width
    y1 = np.minimum(y0 + 1, height - 1)
    above = (1 - xfrac) * image[y0, x0] + xfrac * image[y0, x1]
    below = (1 - xfrac) * image[y1, x0] + xfrac * image[y1, x1]
    return np.rint((1 - yfrac) * above + yfrac * below).clip(0, 255).astype(np.uint8)


def _cap_printable(layout, x, y):
    l, t, r, b = (v * LATENT_SCALE for v in layout.center_rect)
    included_x = (x >= l) & (x < r)
    included_y = (y >= t) & (y < b)
    top = t - LATENT_SCALE * layout.bleed_cells["top"]
    right = r + LATENT_SCALE * layout.bleed_cells["right"]
    bottom = b + LATENT_SCALE * layout.bleed_cells["bottom"]
    left = l - LATENT_SCALE * layout.bleed_cells["left"]
    north = (y >= top) & (y < t)
    east = (x >= r) & (x < right)
    south = (y >= b) & (y < bottom)
    west = (x >= left) & (x < l)
    valid = (included_x & (included_y | north | south)) | (included_y & (east | west))
    # Keep the resampling footprint inside the actual core or its one-sided free bleed.
    # It must not include virtual context or the discarded corners at a subpixel boundary.
    x_low, x_high = np.where(west, left, l), np.where(east, right, r)
    y_low, y_high = np.where(north, top, t), np.where(south, bottom, b)
    return valid, np.clip(x[valid], x_low[valid] + 0.5, x_high[valid] - 0.5), np.clip(
        y[valid], y_low[valid] + 0.5, y_high[valid] - 0.5,
    )


def compose_atlas(topology, region_map, bleed_owner, ring_layout, ring_image, caps, *, return_support=False):
    """Composite registered source pixels without editing any frozen ring or running preflight."""
    labels = _atlas_labels(region_map)
    bleed_owner = np.asarray(bleed_owner)
    image_spec = topology["image"]
    if (labels.shape != (image_spec["height"], image_spec["width"]) or bleed_owner.shape != labels.shape):
        raise ValueError("The region and bleed maps must match the atlas registration")
    owners = _owners(topology, labels, bleed_owner)
    atlas = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    support = np.zeros(labels.shape, dtype=np.uint8)
    ring = _image_rgb(ring_image, (ring_layout.height_px, ring_layout.width_px))
    for face in SIDES + CAPS:
        tag = REGION_LABELS[face]
        y, x = np.nonzero(owners == tag)
        rect = topology["faces"][face]["rect"]
        if face in SIDES:
            sx, sy = _source_points(rect, ring_layout.face_rects[face], x, y)
            valid = (sy >= 0.5) & (sy <= ring_layout.height_px - 0.5)
            values = _bilinear(ring, sx[valid], sy[valid], periodic_x=True)
        else:
            if face not in caps:
                continue
            cap, image = caps[face]
            cap_rgb = _image_rgb(image, (cap.height_px, cap.width_px))
            if cap.face != face:
                raise ValueError(f"Misregistered cap {cap.face} for face {face}")
            sx, sy = _source_points(rect, cap.center_rect, x, y)
            valid, sx, sy = _cap_printable(cap, sx, sy)
            values = _bilinear(cap_rgb, sx, sy)
        atlas[y[valid], x[valid]] = values
        support[y[valid], x[valid]] = tag
    output = Image.fromarray(atlas)
    return (output, support) if return_support else output


def compose_support_counts(region_map, bleed_owner, support):
    """Count actual generated support by print region and nominal bleed owner."""
    labels = _atlas_labels(region_map)
    bleed_owner = np.asarray(bleed_owner)
    support = np.asarray(support)
    if labels.shape != support.shape or labels.shape != bleed_owner.shape:
        raise ValueError("Support and bleed ownership must match the region map")

    def count(mask):
        return {"supported": int(np.count_nonzero((support != 0) & mask)), "total": int(mask.sum())}

    return {
        "regions": {name: count(labels == tag) for name, tag in REGION_LABELS.items()},
        "bleed_owners": {name: count((labels == BLEED) & (bleed_owner == REGION_LABELS[name]))
                         for name in SIDES + CAPS},
    }
