"""Construct E01b window slots on a shared latent atlas."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from dielinediff.dieline.geometry import REGION_LABELS
from .windows import strip_coordinates


SCALE = 8
RING_WINDOW = 128
MIN_WINDOW = 64
SIDES = ("F", "R", "B", "L")
FACE_NAMES = (*SIDES, "T", "Btm")
BLEED = REGION_LABELS["Bleed"]


@dataclass
class Window:
    name: str
    primary_face: str
    faces: tuple
    shape: tuple
    read_ids: torch.Tensor
    write_ids: torch.Tensor
    write_sources: torch.Tensor
    write_weights: torch.Tensor

    def gather(self, flat):
        """Read the tile already oriented in the window's viewing coordinates."""
        return flat.index_select(-1, self.read_ids).unflatten(-1, self.shape)

    def merge(self, tile, total, count):
        """Accumulate each prediction into exactly the atlas slot it read."""
        values = tile.flatten(-2).index_select(-1, self.write_sources).float()
        total.index_add_(-1, self.write_ids, values * self.write_weights)
        count.index_add_(-1, self.write_ids, self.write_weights)


@dataclass
class WindowPlan:
    windows: tuple
    shape: tuple
    labels: np.ndarray
    required: torch.Tensor
    contributions: np.ndarray

    def assert_covered(self, count):
        """Reject any uncovered or non-finite non-fixed latent cell."""
        if count.shape != self.required.shape:
            raise ValueError(f"Expected flattened coverage of shape {self.required.shape}")
        missing = self.required & ((count <= 0) | ~torch.isfinite(count))
        if torch.any(missing):
            offsets = torch.nonzero(missing, as_tuple=True)[0]
            width = self.shape[1]
            sample = [(value % width, value // width) for value in offsets[:5].tolist()]
            raise RuntimeError(f"{offsets.numel()} uncovered latent cells, including {sample}")


def _latent_rect(rect):
    # Integer UV bounds are exclusive on the right and bottom; positions sample at 4 + 8n.
    return tuple((value + SCALE // 2 - 1) // SCALE for value in rect)


def _get_neighbors(topology):
    neighbors = {}
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        neighbors[a["face"], a["edge"]] = b["face"], b["edge"], seam["reverse"]
        neighbors[b["face"], b["edge"]] = a["face"], a["edge"], seam["reverse"]
    return neighbors


def _window(name, primary, faces, read, writes, device, rotation=0):
    if rotation:
        read, writes = np.rot90(read, rotation), np.rot90(writes, rotation)
    height, width = read.shape
    if min(height, width) < MIN_WINDOW:
        raise ValueError(f"{name} is smaller than {MIN_WINDOW * SCALE} px on its short side")
    rows = 0.15 + 0.85 * np.sin(np.pi * (np.arange(height) + 0.5) / height)
    columns = 0.15 + 0.85 * np.sin(np.pi * (np.arange(width) + 0.5) / width)
    sources = np.flatnonzero(writes)
    if not len(sources):
        raise ValueError(f"{name} does not write to any latent cells")
    return Window(
        name, primary, tuple(faces), read.shape,
        torch.as_tensor(read.flatten(), dtype=torch.long, device=device),
        torch.as_tensor(read.flatten()[sources], dtype=torch.long, device=device),
        torch.as_tensor(sources, dtype=torch.long, device=device),
        torch.as_tensor((rows[:, None] * columns).ravel()[sources], dtype=torch.float32, device=device),
    )


def _atlas_grid(x, y, shape, fallback):
    height, width = shape
    valid = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    return np.where(valid, y * width + x, fallback), valid


def _ring_windows(rects, neighbors, labels, stride, pad, fallback, device):
    face = "F"
    columns, column_faces = [], []
    for _ in SIDES:
        x0, y0, x1, y1 = rects[face]
        columns.extend(range(x0, x1))
        column_faces.extend([face] * (x1 - x0))
        other, edge, reverse = neighbors[face, "right"]
        if other not in SIDES or edge != "left" or reverse:
            raise ValueError("Side ring requires consistently upright, aligned left/right seams")
        face = other
    if face != "F" or set(column_faces) != set(SIDES):
        raise ValueError("The four body faces do not form one closed ring")
    y0, y1 = rects["F"][1::2]
    if any(rects[name][1::2] != (y0, y1) for name in SIDES):
        raise ValueError("E01b needs aligned side faces")
    if y1 - y0 != RING_WINDOW or len(columns) < RING_WINDOW:
        raise ValueError("E01b expects a body height of 1024 px and a complete side ring")
    columns = np.asarray(columns)
    column_faces = np.asarray(column_faces)
    circumference = len(columns)
    # Top- and bottom-shifted 1024-pixel windows include actual bleed while retaining the SDXL window size.
    starts = (y0 - pad, y0 + pad)
    windows = []
    flat_labels = labels.ravel()
    for offset in range(0, circumference, stride):
        segment = (offset + np.arange(RING_WINDOW)) % circumference
        present = tuple(dict.fromkeys(column_faces[segment].tolist()))
        primary = str(column_faces[segment[RING_WINDOW // 2]])
        for vertical, start in zip(("upper", "lower"), starts, strict=True):
            y = start + np.arange(RING_WINDOW)[:, None]
            x = columns[segment][None, :]
            read, valid = _atlas_grid(x, y, labels.shape, fallback)
            regions = flat_labels[read]
            writes = valid & np.isin(regions, [BLEED, *(REGION_LABELS[name] for name in SIDES)])
            windows.append(_window(f"ring-{offset * SCALE}-{vertical}", primary, present,
                                   read, writes, device))
    return windows


def _lid_window(name, faces, rects, neighbors, labels, band, pad, fallback, device):
    x0, y0, x1, y1 = rects[name]
    width, height = x1 - x0, y1 - y0
    margin = band + pad
    yy, xx = np.mgrid[-margin:height + margin, -margin:width + margin]
    read, valid = _atlas_grid(x0 + xx, y0 + yy, labels.shape, fallback)
    core = (0 <= xx) & (xx < width) & (0 <= yy) & (yy < height)
    writes = core | (valid & (labels.ravel()[read] == BLEED))
    atlas_width = labels.shape[1]
    adjacent = [name]
    for edge, (along, depth, length) in {
        "top": (xx, -yy - 1, width),
        "right": (yy, xx - width, height),
        "bottom": (xx, yy - height, width),
        "left": (yy, -xx - 1, height),
    }.items():
        other, other_edge, reverse = neighbors[name, edge]
        adjacent.append(other)
        # The first pad cells retain true exterior coordinates; the complete neighbor band starts beyond them.
        slot = (0 <= along) & (along < length) & (pad <= depth) & (depth < margin)
        ox0, oy0, ox1, oy1 = rects[other]
        other_length = ox1 - ox0 if other_edge in ("top", "bottom") else oy1 - oy0
        position = (along[slot] + 0.5) / length
        if reverse:
            position = 1 - position
        corresponding = np.minimum((position * other_length).astype(int), other_length - 1)
        x, y = strip_coordinates(rects[other], other_edge, corresponding, depth[slot] - pad)
        read[slot] = y * atlas_width + x
        writes[slot] = True
    rotation = faces[name]["upright_rotation"] // 90
    return _window(f"lid-{name}", name, adjacent, read, writes, device, rotation)


def _nearest_faces(rects, shape):
    yy, xx = np.indices(shape)
    distances = []
    for x0, y0, x1, y1 in rects.values():
        dx = np.maximum(np.maximum(x0 - xx, xx - (x1 - 1)), 0)
        dy = np.maximum(np.maximum(y0 - yy, yy - (y1 - 1)), 0)
        distances.append(dx**2 + dy**2)
    indices = np.asarray([REGION_LABELS[name] for name in rects], dtype=np.uint8)
    return indices[np.argmin(distances, axis=0)]


def _coverage(windows, shape):
    counts = np.zeros(int(np.prod(shape)), dtype=np.int32)
    for window in windows:
        np.add.at(counts, window.write_ids.cpu().numpy(), 1)
    return counts.reshape(shape)


def _fill_hem_windows(windows, labels, owners, device):
    # A large flap can carry the cut contour beyond the body/lid windows. Such slots still
    # read and write their own atlas coordinates through standard spatial denoising windows.
    missing = (labels == BLEED) & (_coverage(windows, labels.shape) == 0)
    height, width = labels.shape
    tile_h, tile_w = min(height, RING_WINDOW), min(width, RING_WINDOW)
    rows = np.rint(np.linspace(0, height - tile_h, int(np.ceil(height / tile_h)))).astype(int)
    columns = np.rint(np.linspace(0, width - tile_w, int(np.ceil(width / tile_w)))).astype(int)
    area = np.arange(height * width).reshape(labels.shape)
    for y0 in rows:
        for x0 in columns:
            ys, xs = slice(y0, y0 + tile_h), slice(x0, x0 + tile_w)
            needs_coverage = missing[ys, xs]
            if np.any(needs_coverage):
                modal_region = np.bincount(owners[ys, xs][needs_coverage]).argmax()
                primary = next(name for name in FACE_NAMES if REGION_LABELS[name] == modal_region)
                # Blend across overlaps; every slot is the same physical pixel on read and write.
                writes = labels[ys, xs] == BLEED
                windows.append(_window(f"rim-{x0}-{y0}", primary, (primary,),
                                       area[ys, xs], writes, device))
                missing[ys, xs] &= ~writes
    return windows


def _independent_windows(faces, rects, labels, owners, fallback, device):
    windows = []
    fixed = ~np.isin(labels, [BLEED, *(REGION_LABELS[name] for name in FACE_NAMES)])
    for name, (x0, y0, x1, y1) in rects.items():
        own = (labels == REGION_LABELS[name]) | ((labels == BLEED) & (owners == REGION_LABELS[name]))
        ys, xs = np.nonzero(own)
        x0, y0 = min(x0, xs.min()), min(y0, ys.min())
        x1, y1 = max(x1, xs.max() + 1), max(y1, ys.max() + 1)
        extra_x, extra_y = max(0, MIN_WINDOW - (x1 - x0)), max(0, MIN_WINDOW - (y1 - y0))
        x0, x1 = x0 - extra_x // 2, x1 + (extra_x + 1) // 2
        y0, y1 = y0 - extra_y // 2, y1 + (extra_y + 1) // 2
        yy, xx = np.mgrid[y0:y1, x0:x1]
        native, valid = _atlas_grid(xx, yy, labels.shape, fallback)
        writes = valid & own.ravel()[native]
        # Padding never sees an independently generated exterior face or another owner's bleed.
        read = np.where(writes | (valid & fixed.ravel()[native]), native, fallback)
        rotation = faces[name]["upright_rotation"] // 90
        windows.append(_window(f"face-{name}", name, (name,), read, writes, device, rotation))
    return windows


def build_windows(topology, region_map, stride_px=512, lid_band_latent=16, synchronized=True, device="cuda"):
    """Build the fixed E01b geometry's gather/scatter mappings."""
    if not isinstance(stride_px, int) or not SCALE <= stride_px <= 512 or stride_px % SCALE:
        raise ValueError("The ring stride must be a positive multiple of eight and at most 512 px")
    if not isinstance(lid_band_latent, int) or lid_band_latent < 16:
        raise ValueError("The lid neighbor bands must be at least sixteen latent cells")
    image = topology["image"]
    width, height = image["width"], image["height"]
    if (width % SCALE or height % SCALE or np.shape(region_map) != (height, width)):
        raise ValueError("Expected the registered E01b region map padded to multiples of eight")
    labels = np.asarray(region_map)[SCALE // 2::SCALE, SCALE // 2::SCALE].copy()
    fallback = np.flatnonzero(labels == REGION_LABELS["Outside"])[0]
    faces = topology["faces"]
    rects = {name: _latent_rect(faces[name]["rect"]) for name in FACE_NAMES}
    owners = _nearest_faces(rects, labels.shape)
    pad = int(np.ceil(topology["box_mm"]["bleed"] * image["pixels_per_mm"] / SCALE))
    if synchronized:
        neighbors = _get_neighbors(topology)
        windows = _ring_windows(rects, neighbors, labels, stride_px // SCALE, pad, fallback, device)
        windows.extend(_lid_window(name, faces, rects, neighbors, labels, lid_band_latent,
                                   pad, fallback, device) for name in ("T", "Btm"))
        _fill_hem_windows(windows, labels, owners, device)
    else:
        windows = _independent_windows(faces, rects, labels, owners, fallback, device)
    coverage = _coverage(windows, labels.shape)
    required = np.isin(labels, [BLEED, *(REGION_LABELS[name] for name in FACE_NAMES)])
    plan = WindowPlan(tuple(windows), labels.shape, labels,
                      torch.as_tensor(required.ravel(), dtype=torch.bool, device=device), coverage)
    plan.assert_covered(torch.as_tensor(coverage.ravel(), device=device))
    return plan


def save_coverage_heatmap(plan, path):
    """Draw the number of positive-weight predictions for each latent atlas cell."""
    palette = np.asarray(((249, 250, 252), (205, 235, 228), (143, 204, 198),
                          (79, 158, 174), (37, 106, 150), (36, 57, 110)), dtype=np.uint8)
    maximum = int(plan.contributions.max())
    steps = np.ceil(5 * plan.contributions / max(maximum, 1)).astype(np.uint8)
    pixels = palette[steps]
    required = plan.required.cpu().numpy().reshape(plan.shape)
    pixels[~required] = (229, 231, 235)
    pixels[required & (plan.contributions == 0)] = (202, 30, 52)
    map_image = Image.fromarray(pixels)
    map_image = map_image.resize((map_image.width * 3, map_image.height * 3), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (map_image.width + 36, map_image.height + 96), "white")
    canvas.paste(map_image, (18, 46))
    draw = ImageDraw.Draw(canvas)
    text = ImageFont.load_default(size=17)
    dark = (35, 45, 58)
    draw.text((18, 14), "E01b latent window coverage", font=text, fill=dark)
    draw.text((18, map_image.height + 60),
              f"Required cells: {int(required.sum())}   |   Max predictions per cell: {maximum}   |   Gray: fixed",
              font=ImageFont.load_default(size=13), fill=dark)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
