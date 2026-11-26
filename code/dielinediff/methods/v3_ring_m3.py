"""Phase control and expanded physical registration for the Task 17 ring sampler."""

from dataclasses import replace

import numpy as np

from .v3_geometry import LATENT_SCALE, build_ring_layout


EXPANDED_RING_SIZE = (1280, 1152)
_ORIGINAL_RING_SIZE = (1280, 1024)


def extended_ring_layout(topology, size):
    """Translate the original body down by eight latent rows on the expanded ring."""
    width, height = size
    if (width, height) != EXPANDED_RING_SIZE:
        return build_ring_layout(topology, width, height)

    original = build_ring_layout(topology, *_ORIGINAL_RING_SIZE)
    extra = (height - original.height_px) // (2 * LATENT_SCALE)
    top, bottom = (value + extra for value in original.body_rows)
    body_density = (bottom - top) * LATENT_SCALE / float(topology["box_mm"]["height"])
    buffer_px = top * LATENT_SCALE, height - bottom * LATENT_SCALE
    labels = np.pad(original.region_ids, ((extra, extra), (0, 0)), mode="edge")
    faces = {
        face: (x0, y0 + extra, x1, y1 + extra)
        for face, (x0, y0, x1, y1) in original.face_rects.items()
    }
    return replace(
        original, height_px=height, latent_shape=labels.shape, body_rows=(top, bottom),
        face_rects=faces, region_ids=labels, buffer_mm=tuple(value / body_density for value in buffer_px),
    )
