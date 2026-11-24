"""Generate a checker atlas with upright arrows and verify its real OBJ renders."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dielinediff.dieline.geometry import REGION_LABELS
from .orthographic import render_views


ARROW = (190, 29, 61)
CHECKER_LIGHT = (246, 234, 194)
CHECKER_DARK = (67, 133, 146)
TITLES = {
    "F": "F (front)", "B": "B (back)", "L": "L (left)", "R": "R (right)",
    "T": "T (top)", "Btm": "Btm (bottom)", "iso": "Isometric",
}


def _checker(points):
    # Coincident 3D points sample one common pattern regardless of their UV island.
    phase = np.array((2.7, 6.1, 3.4))
    index = np.floor((points + phase) / 14).astype(int).sum(axis=-1) % 2
    return np.where(index[..., None] == 0, CHECKER_LIGHT, CHECKER_DARK).astype(np.uint8)


def _face_patch(face):
    x0, y0, x1, y1 = face["rect"]
    width, height = x1 - x0, y1 - y0
    tl, tr, _, bl = np.asarray(face["corners_mm"])
    x = (np.arange(width) + 0.5) / width
    y = (np.arange(height) + 0.5) / height
    points = tl + x[None, :, None] * (tr - tl) + y[:, None, None] * (bl - tl)
    return Image.fromarray(_checker(points))


def _add_arrow(patch, face, name):
    # The annotations are placed in the physically intended viewing orientation.
    angle = face["upright_rotation"]
    patch = patch.rotate(angle, expand=True)
    width, height = patch.size
    points = [(0.50, 0.24), (0.70, 0.43), (0.57, 0.43), (0.57, 0.58),
              (0.43, 0.58), (0.43, 0.43), (0.30, 0.43)]
    draw = ImageDraw.Draw(patch)
    draw.polygon([(round(x * width), round(y * height)) for x, y in points], fill=ARROW)
    font = ImageFont.load_default(size=max(11, round(min(width, height) * 0.17)))
    draw.text((width / 2, height * 0.69), name, anchor="mm", fill=(34, 38, 48),
              font=font, stroke_width=1, stroke_fill=(252, 248, 233))
    return patch.rotate(-angle, expand=True)


def checker_atlas(topology, region_map):
    """Color six face islands using one 3D pattern, with separately testable arrows."""
    colors = {
        "Outside": (231, 234, 238), "Bleed": (196, 192, 215), "Glue": (255, 255, 255),
        "Tuck": (255, 255, 255), "Dust": (255, 255, 255),
    }
    atlas = np.zeros(region_map.shape + (3,), dtype=np.uint8)
    for name, color in colors.items():
        atlas[region_map == REGION_LABELS[name]] = color
    atlas = Image.fromarray(atlas)
    for name, face in topology["faces"].items():
        atlas.paste(_add_arrow(_face_patch(face), face, name), face["rect"][:2])
    return atlas


def _edge_pixels(atlas, face, edge):
    x0, y0, x1, y1 = face["rect"]
    if edge in ("left", "right"):
        return atlas[y0:y1, x0 if edge == "left" else x1 - 1]
    return atlas[y0 if edge == "top" else y1 - 1, x0:x1]


def verify_selfcheck(topology, atlas, views):
    """Check all twelve folded seams and the arrow in each exported view."""
    atlas = np.asarray(atlas)
    seam_report = []
    for edge in topology["adjacencies"]:
        first, second = edge["a"], edge["b"]
        a = _edge_pixels(atlas, topology["faces"][first["face"]], first["edge"])
        b = _edge_pixels(atlas, topology["faces"][second["face"]], second["edge"])
        if edge["reverse"]:
            b = b[::-1]
        # Rectangles with the same physical edge may quantize a pixel differently.
        samples = (np.arange(max(len(a), len(b))) + 0.5) / max(len(a), len(b))
        equal = (a[(samples * len(a)).astype(int)] == b[(samples * len(b)).astype(int)]).all(axis=1)
        agreement = float(equal.mean())
        if agreement < 0.97:
            raise AssertionError(f"Checker seam {first} / {second}: only {agreement:.1%} matches")
        seam_report.append({"a": first, "b": second, "checker_pixel_agreement": agreement})
    arrow_report = {}
    for name in topology["faces"]:
        pixels = np.asarray(views[name])
        arrow = (pixels == ARROW).all(axis=-1)
        rows = np.flatnonzero(arrow.any(axis=1))
        if not len(rows):
            raise AssertionError(f"No arrow rendered in view {name}")
        top, bottom = rows[0], rows[-1]
        point_width = arrow[top:top + 3].sum(axis=1).max()
        stem_width = arrow[bottom - 2:bottom + 1].sum(axis=1).min()
        if not (top < pixels.shape[0] / 2 and point_width < stem_width / 2):
            raise AssertionError(f"Arrow in view {name} is not upright")
        arrow_report[name] = True
    return {"seams": seam_report, "upright": arrow_report}


def _gallery(atlas, views, path):
    panel = 210
    gap = 10
    cover_width = 420
    width = cover_width + 4 * (panel + gap) + gap
    height = 56 + 2 * (panel + 25) + 84
    page = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(page)
    heading = ImageFont.load_default(size=23)
    label = ImageFont.load_default(size=15)
    ink = (30, 40, 48)
    draw.text((14, 12), "RTE dieline and folded OBJ self-check", fill=ink, font=heading)
    draw.text((14, 49), "Shared 3D checker pattern in the unfolded atlas", fill=ink, font=label)
    atlas_preview = atlas.copy()
    atlas_preview.thumbnail((cover_width - 22, height - 95), Image.Resampling.LANCZOS)
    page.paste(atlas_preview, ((cover_width - atlas_preview.width) // 2, 71))
    for index, name in enumerate(("F", "R", "T", "B", "L", "Btm")):
        x = cover_width + (index % 3) * (panel + gap)
        y = 54 + (index // 3) * (panel + 25)
        preview = views[name].resize((panel, panel), Image.Resampling.LANCZOS)
        page.paste(preview, (x, y))
        draw.text((x + panel // 2, y + panel + 11), TITLES[name], anchor="mm", fill=ink, font=label)
    iso_x = cover_width + 3 * (panel + gap)
    iso_y = 54 + (panel + 25) // 2
    page.paste(views["iso"].resize((panel, panel), Image.Resampling.LANCZOS), (iso_x, iso_y))
    draw.text((iso_x + panel // 2, iso_y + panel + 11), TITLES["iso"],
              anchor="mm", fill=ink, font=label)
    draw.text((14, height - 27), "Face checker colors meet along folded seams. Each red arrow points up.",
              fill=ink, font=label)
    page.save(path, optimize=True)


def write_selfcheck(geometry, output, report=None):
    geometry, output = Path(geometry), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    topology = json.loads((geometry / "topology.json").read_text(encoding="utf-8"))
    with Image.open(geometry / "region_map.png") as region_map:
        atlas = checker_atlas(topology, np.asarray(region_map))
    atlas_path = output / "selfcheck-atlas.png"
    atlas.save(atlas_path, optimize=True)
    views = render_views(geometry, atlas_path, output)
    results = verify_selfcheck(topology, atlas, views)
    (output / "selfcheck.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _gallery(atlas, views, report or output / "selfcheck.png")
    return results
