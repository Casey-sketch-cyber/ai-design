"""Attach cap cores to their frozen physical neighbors and export registered bleed."""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from dielinediff.dieline.geometry import REGION_LABELS

from .v3_capfix_geometry import _span, _strip, build_capfix_layout
from .v3_geometry import BLEED, CAPS, EDGES, SIDES, _native_fold, _neighbors, compose_atlas
from .windows import strip_coordinates


def _attached_ring_ids(original, band):
    source = np.full(original.latent_shape, -1, dtype=original.ring_ids.dtype)
    for edge in EDGES:
        length = _span(original.center_rect, edge)
        former = _strip(original.center_rect, edge, length, band, original.bleed_cells[edge])
        current = _strip(original.center_rect, edge, length, band, 0)
        original_ids = original.ring_ids[former]
        if np.any(original_ids < 0) or np.any(source[current] >= 0):
            raise ValueError(f"Invalid frozen strip registration for {original.face}/{edge}")
        source[current] = original_ids
    known_ids = source[source >= 0]
    if np.unique(known_ids).size != known_ids.size or known_ids.size != np.count_nonzero(original.known_mask):
        raise ValueError(f"Reattaching {original.face} lost or duplicated registered ring cells")
    return source


def build_capgeom_layout(original, ring, band, scale=2):
    """Keep C3's canvas and core fixed while attaching all four frozen strips."""
    legacy = build_capfix_layout(original, ring, band, scale)
    ring_ids = _attached_ring_ids(original, band)
    no_bleed = dict.fromkeys(EDGES, 0)
    attached = replace(original, known_mask=ring_ids >= 0, ring_ids=ring_ids,
                       bleed_cells=no_bleed,
                       edge_sources={edge: (*source[:3], 0) for edge, source in original.edge_sources.items()})
    mask = np.zeros(legacy.latent_shape, dtype=bool)
    for edge in EDGES:
        length = _span(legacy.center_rect, edge)
        mask[_strip(legacy.center_rect, edge, length, scale * band, 0)] = True
    expected = 2 * scale * band * (_span(legacy.center_rect, "top") + _span(legacy.center_rect, "left"))
    if np.count_nonzero(mask) != expected:
        raise ValueError(f"Frozen strips overlap for {original.face}")
    return replace(legacy, original=attached, known_mask=mask, bleed_cells=no_bleed)


@dataclass(frozen=True)
class FrozenBleedPatch:
    cap: str
    edge: str
    side: str
    destination: tuple
    source: tuple


def _edge_offsets(rect, edge, x, y):
    x0, y0, x1, y1 = rect
    if edge == "top":
        return x - x0, y0 - y - 1
    if edge == "right":
        return y - y0, x - x1
    if edge == "bottom":
        return x - x0, y - y1
    return y - y0, x0 - x - 1


def iter_frozen_cap_bleed(topology, region_map, bleed_owner, faces=CAPS):
    """Pair exterior bleed pixels with physical atlas pixels on their adjoining sides."""
    bounds = {face: spec["rect"] for face, spec in topology["faces"].items()}
    neighbors = _neighbors(topology)
    limit_px = float(topology["box_mm"]["bleed"]) * float(topology["image"]["pixels_per_mm"])
    for cap in faces:
        if cap not in CAPS:
            raise ValueError(f"Expected a cap face, got {cap}")
        tag = REGION_LABELS[cap]
        y, x = np.nonzero((region_map == BLEED) & (bleed_owner == tag))
        rect = bounds[cap]
        for edge in EDGES:
            side, side_edge, reverse = neighbors[cap, edge]
            side_rect = bounds[side]
            if _native_fold(rect, edge, side_rect, side_edge, reverse):
                continue
            length = _span(rect, edge)
            if side not in SIDES or _span(side_rect, side_edge) != length:
                raise ValueError(f"Unequal physical edge lengths for {cap}/{edge} and {side}/{side_edge}")
            along, depth = _edge_offsets(rect, edge, x, y)
            # A pixel center at depth d is d + 0.5 px beyond the cut.
            # Tangential bounds exclude both corner quadrants.
            valid = (0 <= along) & (along < length) & (0 <= depth) & (depth + 0.5 <= limit_px)
            registration = length - 1 - along[valid] if reverse else along[valid]
            sx, sy = strip_coordinates(side_rect, side_edge, registration, depth[valid])
            yield FrozenBleedPatch(cap, edge, side, (y[valid], x[valid]), (sy, sx))


def compose_capgeom_atlas(topology, region_map, bleed_owner, ring_layout, ring_image, caps,
                          *, return_support=False):
    """Compose the tight cap cores and crop their owned bleed from the frozen ring."""
    if any(any(layout.bleed_cells.values()) for layout, _ in caps.values()):
        raise ValueError("Frozen cap bleed requires the tight cap layout")
    image, support = compose_atlas(topology, region_map, bleed_owner, ring_layout, ring_image, caps,
                                   return_support=True)
    atlas = np.array(image)
    for patch in iter_frozen_cap_bleed(topology, np.asarray(region_map), np.asarray(bleed_owner), caps):
        if np.any(support[patch.destination]) or not np.all(support[patch.source] == REGION_LABELS[patch.side]):
            raise ValueError(f"Invalid frozen cap bleed ownership for {patch.cap}/{patch.edge}")
        # The adjacent side was sampled from the frozen RGB ring by compose_atlas at this exact
        # physical resolution. Read those bytes instead of resizing a separate context strip.
        atlas[patch.destination] = atlas[patch.source]
        support[patch.destination] = REGION_LABELS[patch.cap]
    image = Image.fromarray(atlas)
    return (image, support) if return_support else image
