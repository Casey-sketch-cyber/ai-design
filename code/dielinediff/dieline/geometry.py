"""Build a reverse-tuck-end dieline and its folded six-face UV mesh."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import contains_xy
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


REGION_LABELS = {
    "Outside": 0,
    "F": 1,
    "B": 2,
    "L": 3,
    "R": 4,
    "T": 5,
    "Btm": 6,
    "Glue": 7,
    "Tuck": 8,
    "Dust": 9,
    "Bleed": 10,
}
EDGES = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


@dataclass(frozen=True)
class BoxSpec:
    """Nominal exterior dimensions and editable flap sizes, in millimetres."""

    length: float = 70.0
    width: float = 45.0
    height: float = 160.0
    paper_thickness: float = 0.45
    bleed: float = 3.0
    safety: float = 3.0
    glue_width: float = 12.0
    tuck_depth: float = 12.0
    dust_depth: float = 20.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if 2 * self.safety >= min(self.length, self.width, self.height):
            raise ValueError("The safety distance leaves at least one face empty")
        if self.paper_thickness >= min(self.length, self.width, self.height):
            raise ValueError("Paper thickness must be smaller than all box dimensions")
        if self.dust_depth >= self.length or self.tuck_depth >= self.width:
            raise ValueError("Flap depths must fit inside the corresponding closure")


@dataclass(frozen=True)
class Geometry:
    """Vector geometry in millimetres and a registered atlas specification."""

    topology: dict
    faces_mm: dict
    flaps: dict
    outline: Polygon
    bleed_outline: Polygon
    folds: tuple


def _folded_positions(spec):
    x, y, z = spec.length / 2, spec.height / 2, spec.width / 2
    return {
        "F": ((-x, y, z), (x, y, z), (x, -y, z), (-x, -y, z)),
        "B": ((x, y, -z), (-x, y, -z), (-x, -y, -z), (x, -y, -z)),
        "L": ((-x, y, -z), (-x, y, z), (-x, -y, z), (-x, -y, -z)),
        "R": ((x, y, z), (x, y, -z), (x, -y, -z), (x, -y, z)),
        "T": ((-x, y, -z), (x, y, -z), (x, y, z), (-x, y, z)),
        "Btm": ((x, -y, -z), (-x, -y, -z), (-x, -y, z), (x, -y, z)),
    }


def _adjacencies(positions):
    pending = {}
    adjacent = []
    for face, corners in positions.items():
        for edge, (start, end) in EDGES.items():
            segment = (corners[start], corners[end])
            key = frozenset(segment)
            descriptor = {"face": face, "edge": edge}
            if key in pending:
                previous, previous_segment = pending.pop(key)
                adjacent.append({"a": previous, "b": descriptor, "reverse": previous_segment != segment})
            else:
                pending[key] = (descriptor, segment)
    if pending or len(adjacent) != 12:
        raise ValueError("The folded mesh does not have twelve two-sided edges")
    return adjacent


def build_geometry(spec=BoxSpec(), short_side=1024, *, pixels_per_mm=None):
    """Create a dieline whose atlas has square pixels and dimensions divisible by eight."""
    if pixels_per_mm is None:
        if not isinstance(short_side, int) or short_side < 64 or short_side % 8:
            raise ValueError("short_side must be an integer multiple of eight, at least 64")
    elif not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be finite and positive")
    length, width, height = spec.length, spec.width, spec.height
    top = width + max(spec.tuck_depth, spec.dust_depth - width)
    bottom = top + height
    x_f, x_r, x_b, x_end = width, width + length, 2 * width + length, 2 * (width + length)
    faces = {
        "F": (x_f, top, x_r, bottom),
        "B": (x_b, top, x_end, bottom),
        "L": (0, top, x_f, bottom),
        "R": (x_r, top, x_b, bottom),
        "T": (x_f, top - width, x_r, top),
        "Btm": (x_b, bottom, x_end, bottom + width),
    }
    tuck = spec.tuck_depth
    dust = spec.dust_depth
    glue = spec.glue_width
    tuck_bevel = min(tuck / 3 + spec.paper_thickness, length / 8)
    dust_bevel = min(dust / 3, width / 8)
    glue_bevel = min(glue / 2, height / 8)
    flaps = {
        "Glue": [Polygon(((x_end, top), (x_end + glue, top + glue_bevel),
                          (x_end + glue, bottom - glue_bevel), (x_end, bottom)))],
        "Tuck": [
            Polygon(((x_f, top - width), (x_f + tuck_bevel, top - width - tuck),
                     (x_r - tuck_bevel, top - width - tuck), (x_r, top - width))),
            Polygon(((x_b, bottom + width), (x_end, bottom + width),
                     (x_end - tuck_bevel, bottom + width + tuck),
                     (x_b + tuck_bevel, bottom + width + tuck))),
        ],
        "Dust": [],
    }
    for x0, x1 in ((0, x_f), (x_r, x_b)):
        flaps["Dust"].extend((
            Polygon(((x0, top), (x0 + dust_bevel, top - dust),
                     (x1 - dust_bevel, top - dust), (x1, top))),
            Polygon(((x0, bottom), (x1, bottom), (x1 - dust_bevel, bottom + dust),
                     (x0 + dust_bevel, bottom + dust))),
        ))
    outline = unary_union([*(box(*rect) for rect in faces.values()),
                           *(shape for shapes in flaps.values() for shape in shapes)])
    bleed_outline = outline.buffer(spec.bleed, quad_segs=8)
    min_x, min_y, max_x, max_y = bleed_outline.bounds
    span_x, span_y = max_x - min_x, max_y - min_y
    scale = short_side / min(span_x, span_y) if pixels_per_mm is None else pixels_per_mm
    image_width = 8 * math.ceil((span_x * scale - 1e-8) / 8)
    image_height = 8 * math.ceil((span_y * scale - 1e-8) / 8)
    origin_x = min_x - (image_width / scale - span_x) / 2
    origin_y = min_y - (image_height / scale - span_y) / 2

    def pixel_rect(rect):
        x0, y0, x1, y1 = rect
        return [round((x0 - origin_x) * scale), round((y0 - origin_y) * scale),
                round((x1 - origin_x) * scale), round((y1 - origin_y) * scale)]

    positions = _folded_positions(spec)
    topology = {
        "format_version": 1,
        "box_mm": asdict(spec),
        "image": {"width": image_width, "height": image_height, "pixels_per_mm": scale,
                  "origin_mm": [origin_x, origin_y]},
        "faces": {},
        "adjacencies": _adjacencies(positions),
    }
    for name, rect in faces.items():
        x0, y0, x1, y1 = pixel_rect(rect)
        left, upper, right, lower = rect
        safe_rect = pixel_rect((left + spec.safety, upper + spec.safety,
                                right - spec.safety, lower - spec.safety))
        if min(x1 - x0, y1 - y0) < 2 or min(safe_rect[2] - safe_rect[0], safe_rect[3] - safe_rect[1]) < 1:
            raise ValueError(f"Face {name} or its safety inset is too small for this atlas")
        topology["faces"][name] = {
            "rect": [x0, y0, x1, y1],
            "safe_rect": safe_rect,
            "uv": [x0 / image_width, 1 - y1 / image_height,
                   x1 / image_width, 1 - y0 / image_height],
            "upright_rotation": 180 if name == "Btm" else 0,
            "corners_mm": positions[name],
        }
    folds = [((x, top), (x, bottom)) for x in (x_f, x_r, x_b, x_end)]
    folds.extend((((x_f, top), (x_r, top)), ((x_b, bottom), (x_end, bottom)),
                  ((x_f, top - width), (x_r, top - width)),
                  ((x_b, bottom + width), (x_end, bottom + width))))
    for x0, x1 in ((0, x_f), (x_r, x_b)):
        folds.extend((((x0, top), (x1, top)), ((x0, bottom), (x1, bottom))))
    return Geometry(topology, faces, flaps, outline, bleed_outline, tuple(folds))


def _format_points(points):
    return " ".join(f"{x:.6f},{y:.6f}" for x, y in points)


def _write_svg(geometry, path):
    image = geometry.topology["image"]
    scale = image["pixels_per_mm"]
    x, y = image["origin_mm"]
    width, height = image["width"] / scale, image["height"] / scale
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.6f}mm" height="{height:.6f}mm"'
        f' viewBox="{x:.6f} {y:.6f} {width:.6f} {height:.6f}" fill="none">',
        '<g id="cut" stroke="#cf3030" stroke-width="0.30">',
        f'<polygon points="{_format_points(geometry.outline.exterior.coords)}"/>',
        '</g>',
        '<g id="crease" stroke="#006fa0" stroke-width="0.25" stroke-dasharray="2 1">',
    ]
    for fold in geometry.folds:
        svg.append(f'<polyline points="{_format_points(fold)}"/>')
    svg.extend((
        '</g>',
        '<g id="bleed" stroke="#b24bd1" stroke-width="0.22" stroke-dasharray="1 1">',
        f'<polygon points="{_format_points(geometry.bleed_outline.exterior.coords)}"/>',
        '</g>',
        '<g id="safe" stroke="#007747" stroke-width="0.20" stroke-dasharray="1 1">',
    ))
    safety = geometry.topology["box_mm"]["safety"]
    for name, (x0, y0, x1, y1) in geometry.faces_mm.items():
        svg.append(f'<rect id="safe-{name}" x="{x0 + safety:.6f}" y="{y0 + safety:.6f}"'
                   f' width="{x1 - x0 - 2 * safety:.6f}" height="{y1 - y0 - 2 * safety:.6f}"/>')
    svg.extend(("</g>", "</svg>"))
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_labels(geometry, destination):
    image = geometry.topology["image"]
    height, width = image["height"], image["width"]
    origin_x, origin_y = image["origin_mm"]
    columns = origin_x + (np.arange(width) + 0.5) / image["pixels_per_mm"]
    rows = origin_y + (np.arange(height) + 0.5) / image["pixels_per_mm"]
    xs, ys = columns[None, :], rows[:, None]
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[contains_xy(geometry.bleed_outline, xs, ys)] = REGION_LABELS["Bleed"]
    for name, shapes in geometry.flaps.items():
        for shape in shapes:
            labels[contains_xy(shape, xs, ys)] = REGION_LABELS[name]
    for name, face in geometry.topology["faces"].items():
        x0, y0, x1, y1 = face["rect"]
        labels[y0:y1, x0:x1] = REGION_LABELS[name]
    mask = labels == REGION_LABELS["Bleed"]
    # A cut interior pixel can lie on the rounding discrepancy between the SVG and quantized rectangles.
    unassigned_inside = mask & contains_xy(geometry.outline, xs, ys)
    selected = np.nonzero(mask)
    bleed_x, bleed_y = columns[selected[1]], rows[selected[0]]
    distances = []
    owners = []
    for name, (x0, y0, x1, y1) in geometry.faces_mm.items():
        dx = np.maximum(np.maximum(x0 - bleed_x, bleed_x - x1), 0)
        dy = np.maximum(np.maximum(y0 - bleed_y, bleed_y - y1), 0)
        distances.append(dx * dx + dy * dy)
        owners.append(REGION_LABELS[name])
    nearest = np.asarray(owners, dtype=np.uint8)[np.argmin(distances, axis=0)]
    bleed_owner = np.zeros_like(labels)
    bleed_owner[selected] = nearest
    bleed_owner[unassigned_inside] = 0
    # Rectangles are quantized once for the UV contract; a fractional inboard sliver has no atlas face.
    labels[unassigned_inside] = REGION_LABELS["Outside"]
    Image.fromarray(labels).save(destination / "region_map.png")
    Image.fromarray(bleed_owner).save(destination / "bleed_owner.png")
    (destination / "region_labels.json").write_text(
        json.dumps(REGION_LABELS, indent=2) + "\n", encoding="utf-8")


def _write_obj(topology, destination):
    lines = ["# Six exterior surfaces, UV atlas uses dieline coordinates",
             "mtllib box.mtl", "o RTE", "usemtl dieline_atlas"]
    for index, (name, face) in enumerate(topology["faces"].items()):
        u0, v0, u1, v1 = face["uv"]
        lines.append(f"# {name}")
        for x, y, z in face["corners_mm"]:
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
        for u, v in ((u0, v1), (u1, v1), (u1, v0), (u0, v0)):
            lines.append(f"vt {u:.12f} {v:.12f}")
        first, second, third, fourth = (4 * index + offset for offset in (1, 2, 3, 4))
        lines.append(f"f {first}/{first} {fourth}/{fourth} {third}/{third}")
        lines.append(f"f {first}/{first} {third}/{third} {second}/{second}")
    (destination / "box.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "box.mtl").write_text(
        "newmtl dieline_atlas\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd box_atlas.png\n",
        encoding="utf-8",
    )
    image = topology["image"]
    Image.new("RGB", (image["width"], image["height"]), "white").save(destination / "box_atlas.png")


def write_geometry(geometry, destination):
    """Write the registered SVG, label maps, topology JSON, and OBJ/MTL assets."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_svg(geometry, destination / "dieline.svg")
    _write_labels(geometry, destination)
    _write_obj(geometry.topology, destination)
    (destination / "topology.json").write_text(
        json.dumps(geometry.topology, indent=2) + "\n", encoding="utf-8")
    return destination
