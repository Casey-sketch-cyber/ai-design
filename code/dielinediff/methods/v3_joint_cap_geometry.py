"""Geometry and physical export for the registered Task 20 ring/cap samplers."""

from dataclasses import dataclass, replace
import math

import numpy as np
from PIL import Image

from dielinediff.dieline.geometry import REGION_LABELS

from .v3_capgeom_geometry import compose_capgeom_atlas
from .v3_geometry import CAPS, LATENT_SCALE, SIDES, RingLayout, _aligned, _native_fold, _neighbors


# Atlas orientation: T extends upward from F; Btm extends downward from B.
NATIVE = {"T": ("F", "top", "bottom"), "Btm": ("B", "bottom", "top")}


@dataclass(frozen=True)
class JointCanvas:
    ring: RingLayout
    cap_rects_px: dict


@dataclass(frozen=True)
class OutpaintCanvas:
    size_px: tuple
    known_rows_latent: tuple
    source_rows_latent: tuple
    cap_rect_px: tuple


@dataclass(frozen=True)
class JointStageOne:
    ring: RingLayout
    ring_image: Image.Image
    caps: dict
    proof: dict
    ring_latent: object = None


def _side_ring(topology, size, body_rows):
    width_px, height_px = size
    if any(size < 64 or size % 64 for size in (width_px, height_px)):
        raise ValueError("Joint SDXL canvas dimensions must align to 64 pixels")
    width, height = width_px // LATENT_SCALE, height_px // LATENT_SCALE
    low, high = body_rows
    if not 0 < low < high < height:
        raise ValueError("Joint ring body must be strictly inside the canvas")
    dims = topology["box_mm"]
    steps = np.asarray([dims["width"], dims["length"], dims["width"], dims["length"]])
    edges = np.rint(np.r_[0, np.cumsum(steps)] * width / steps.sum()).astype(int)
    if np.any(np.diff(edges) < 1):
        raise ValueError("Joint ring requires four distinct side faces")
    neighbors = _neighbors(topology)
    for face, next_face in zip(SIDES, SIDES[1:] + SIDES[:1], strict=True):
        if neighbors[face, "right"] != (next_face, "left", False):
            raise ValueError(f"Missing upright cyclic adjacency {face}/{next_face}")
    for face, (side, side_edge, cap_edge) in NATIVE.items():
        if (neighbors[side, side_edge] != (face, cap_edge, False)
                or not _native_fold(topology["faces"][side]["rect"], side_edge,
                                    topology["faces"][face]["rect"], cap_edge, False)):
            raise ValueError(f"Task 20 needs {face} natively attached to {side} in atlas orientation")
    labels = np.empty((height, width), dtype=np.uint8)
    rects = {}
    for i, face in enumerate(SIDES):
        l, r = int(edges[i]), int(edges[i + 1])
        labels[:, l:r] = REGION_LABELS[face]
        rects[face] = l, low, r, high
    mm_per_row = float(dims["height"]) / (high - low)
    return RingLayout(width_px, height_px, (height, width), (low, high), rects, labels,
                      (low * mm_per_row, (height - high) * mm_per_row))


def ring_layout_from_metadata(topology, proof):
    ring = _side_ring(topology, tuple(proof["ring_size_px"]), tuple(proof["body_rows_latent"]))
    if "face_rects_latent" in proof and {
        face: list(rect) for face, rect in ring.face_rects.items()
    } != proof["face_rects_latent"]:
        raise ValueError("Stored native joint ring registration disagrees with the final topology")
    return ring


def _cap_rectangles(ring):
    rects = ring.face_rects
    depth = rects["L"][2] - rects["L"][0]
    left, top, right, _ = rects["F"]
    t = left, top - depth, right, top
    left, _, right, bottom = rects["B"]
    btm = left, bottom, right, bottom + depth
    if t[1] < 0 or btm[3] > ring.latent_shape[0]:
        raise ValueError("The physical cap crop must fit in the joint extensions")
    return {face: tuple(v * LATENT_SCALE for v in rect) for face, rect in zip(CAPS, (t, btm), strict=True)}


def build_joint_ring_layout(topology, width_px=1280, cap_buffer_px=64):
    if width_px % 64 or width_px < 512 or cap_buffer_px < 0:
        raise ValueError("Joint width must be 64-pixel aligned and buffer nonnegative")
    dims = topology["box_mm"]
    ppm = width_px / (2 * (dims["length"] + dims["width"]))
    # Keep body, extensions and both SDXL sizes aligned. The physical cap core
    # uses the registered L/R span; the available extension uses a ceiling.
    body = max(64, round(dims["height"] * ppm / 64) * 64)
    extension = _aligned(math.ceil(dims["width"] * ppm) + cap_buffer_px)
    ring = _side_ring(topology, (width_px, body + 2 * extension),
                      (extension // LATENT_SCALE, (extension + body) // LATENT_SCALE))
    return JointCanvas(ring, _cap_rectangles(ring))


def coarse_joint_layout(topology, full, scale):
    if not 0 < scale < 1:
        raise ValueError("Coarse layout scale must be strictly between zero and one")
    original = full.ring
    width_px = _aligned(math.ceil(original.width_px * scale))
    height_px = _aligned(math.ceil(original.height_px * scale))
    top = round(original.body_rows[0] * height_px / original.height_px)
    ring = _side_ring(topology, (width_px, height_px), (top, height_px // LATENT_SCALE - top))
    return JointCanvas(ring, _cap_rectangles(ring))


def build_outpaint_layout(topology, ring, face, strip_height_px=320, cap_buffer_px=64):
    if (face not in NATIVE or not 64 <= strip_height_px or strip_height_px % 64
            or strip_height_px > LATENT_SCALE * (ring.body_rows[1] - ring.body_rows[0])
            or cap_buffer_px < 0):
        raise ValueError("Invalid oriented outpainting strip geometry")
    dims = topology["box_mm"]
    ppm = ring.width_px / (2 * (dims["length"] + dims["width"]))
    extension = _aligned(math.ceil(dims["width"] * ppm) + cap_buffer_px)
    strip = strip_height_px // LATENT_SCALE
    extension_rows = extension // LATENT_SCALE
    low, high = ring.body_rows
    side = NATIVE[face][0]
    left, _, right, _ = ring.face_rects[side]
    depth = ring.face_rects["L"][2] - ring.face_rects["L"][0]
    if not 0 < depth <= extension_rows:
        raise ValueError("Outpainting extension does not contain the physical cap")
    if face == "T":
        known = extension_rows, extension_rows + strip
        source = low, low + strip
        rect = left, extension_rows - depth, right, extension_rows
    else:
        known = 0, strip
        source = high - strip, high
        rect = left, strip, right, strip + depth
    return OutpaintCanvas((ring.width_px, strip_height_px + extension), known, source,
                          tuple(v * LATENT_SCALE for v in rect))


def ring_registration(ring):
    return {
        "ring_size_px": [ring.width_px, ring.height_px],
        "body_rows_latent": list(ring.body_rows),
        "face_rects_latent": {face: list(rect) for face, rect in ring.face_rects.items()},
        "native_folds": {face: list(edge) for face, edge in NATIVE.items()},
    }


@dataclass(frozen=True)
class _PrintCap:
    face: str
    width_px: int
    height_px: int

    @property
    def center_rect(self):
        return 0, 0, self.width_px // LATENT_SCALE, self.height_px // LATENT_SCALE

    @property
    def bleed_cells(self):
        return dict.fromkeys(("top", "right", "bottom", "left"), 0)


def _projection_ring(result):
    ring = result.ring
    if result.ring_image.size != (ring.width_px, ring.height_px):
        raise ValueError("Joint ring pixels do not match the registered canvas")
    low, high = ring.body_rows
    context = result.proof.get("retained_side_buffer_px")
    if context is None:
        return ring, result.ring_image
    context_rows = math.ceil(context / LATENT_SCALE)
    start = max(0, low - context_rows)
    stop = min(ring.latent_shape[0], high + context_rows)
    if start == 0 and stop == ring.latent_shape[0]:
        return ring, result.ring_image
    mm_per_row = ring.buffer_mm[0] / low
    rects = {face: (l, t - start, r, b - start) for face, (l, t, r, b) in ring.face_rects.items()}
    trimmed = replace(ring, height_px=(stop - start) * LATENT_SCALE,
                      latent_shape=(stop - start, ring.latent_shape[1]),
                      body_rows=(low - start, high - start), face_rects=rects,
                      region_ids=ring.region_ids[start:stop],
                      buffer_mm=((low - start) * mm_per_row, (stop - high) * mm_per_row))
    image = result.ring_image.crop((0, start * LATENT_SCALE, ring.width_px, stop * LATENT_SCALE))
    return trimmed, image


def compose_joint_stage1(topology, labels, owners, result, *, return_support=False):
    ring, ring_image = _projection_ring(result)
    caps = {}
    for face in CAPS:
        image = result.caps[face]
        width, height = image.size
        if image.mode != "RGB" or min(width, height) < LATENT_SCALE or width % 8 or height % 8:
            raise ValueError("Joint cap core must be RGB with latent-aligned dimensions")
        caps[face] = (_PrintCap(face, width, height), image)
    return compose_capgeom_atlas(topology, labels, owners, ring, ring_image, caps,
                                 return_support=return_support)
