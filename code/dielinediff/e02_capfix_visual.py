"""Original-pixel cap-edge evidence and folded comparisons for Task 08."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .e01 import read_json
from .e02_visual import _preview, render_native_creases
from .metrics import _oriented_strip
from .metrics.physical import _edges_on_sheet
from .render import render_views


CAPS = ("T", "Btm")
EDGES = ("top", "right", "bottom", "left")
INK = (30, 45, 56)
MARK = (24, 147, 155)


def _example_comparisons(experiment, chosen, original_method, original_title, current_title):
    methods = ("b2_regional", original_method, chosen)
    titles = ("B2-regional", original_title, current_title)
    records = {(job.atlas.method, job.atlas.brief["id"]): job
               for job in experiment.samples(methods=methods)}
    output = experiment.output / "examples"
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    width, height = 310, 315
    for brief in experiment.briefs[:3]:
        name = brief["id"]
        page = Image.new("RGB", (3 * width + 32, 2 * height + 90), "white")
        draw = ImageDraw.Draw(page)
        draw.text((16, 12), f"{name} / seed 42 / Preflight atlas and folded view", fill=INK, font=font)
        for i, (method, title) in enumerate(zip(methods, titles, strict=True)):
            sample = records[method, name]
            dest = output / name / method
            render_views(sample.atlas.geometry, sample.preflight, dest, size=350, views=["iso"])
            x = 16 + i * width
            draw.text((x, 44), title, fill=INK, font=small)
            for row, file in enumerate((sample.preflight, dest / "folded-iso.png")):
                with Image.open(file) as source:
                    image = _preview(source.convert("RGB"), (width - 15, height - 15))
                y = 65 + row * height + (height - image.height) // 2
                page.paste(image, (x + (width - image.width) // 2, y))
        page.save(output / f"{name}-capfix.jpg", quality=90, optimize=True)


def _cap_neighbors(topology, face):
    result = {}
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        if b["face"] == face:
            a, b = b, a
        if a["face"] == face:
            result[a["edge"]] = b["face"], b["edge"], seam["reverse"]
    return result


def _edge_tiles(faces, face, edge, neighbor, other_edge, reverse, width):
    cap = _oriented_strip(faces[face], edge, width).transpose(1, 0, 2)
    adjacent = _oriented_strip(faces[neighbor], other_edge, width)
    if reverse:
        adjacent = adjacent[::-1]
    side = adjacent.transpose(1, 0, 2)[::-1]
    return Image.fromarray(np.ascontiguousarray(side)), Image.fromarray(np.ascontiguousarray(cap))


def _cap_edges(experiment, chosen, original_method, original_title, current_title):
    topology = read_json(experiment.geometry / "topology.json")
    bounds = {name: face["rect"] for name, face in topology["faces"].items()}
    directions = {face: _cap_neighbors(topology, face) for face in CAPS}
    both = list(experiment.samples(methods=(original_method, chosen)))
    cases = {(j.atlas.brief["id"], j.atlas.method): j for j in both}
    output = experiment.output / "cap-edges"
    output.mkdir(parents=True, exist_ok=True)
    bar = 25
    span = max(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in (bounds[name] for name in CAPS))
    font = ImageFont.load_default(size=14)
    header = ImageFont.load_default(size=17)
    for brief in experiment.briefs:
        name = brief["id"]
        page = Image.new("RGB", (2 * span + 62, 114 + 8 * 75), "white")
        draw = ImageDraw.Draw(page)
        draw.text((16, 10), f"{name} / Cap-edge pairs at native scale / Side above, cap below", fill=INK, font=header)
        draw.text((16, 39), original_title, fill=INK, font=font)
        draw.text((span + 48, 39), current_title, fill=INK, font=font)
        for i, method in enumerate((original_method, chosen)):
            with Image.open(cases[name, method].atlas.path) as atlas:
                faces = {face: np.asarray(atlas.crop(rect).convert("RGB")) for face, rect in bounds.items()}
            origin = 16 + i * (span + 32)
            for f, face in enumerate(CAPS):
                for e, edge in enumerate(EDGES):
                    neighbor, other_edge, reverse = directions[face][edge]
                    first, second = _edge_tiles(faces, face, edge, neighbor, other_edge, reverse, bar)
                    y = 77 + (4 * f + e) * 75
                    original = _edges_on_sheet(bounds[face], edge)
                    counterpart = _edges_on_sheet(bounds[neighbor], other_edge)
                    label = f"{face} {edge} / {neighbor} {other_edge}"
                    if frozenset(original) == frozenset(counterpart):
                        label += " (native fold)"
                    draw.text((origin, y - 5), label, anchor="lb", fill=INK, font=font)
                    page.paste(first, (origin, y))
                    page.paste(second, (origin, y + bar))
                    draw.line((origin - 10, y + bar, origin - 4, y + bar), width=2, fill=MARK)
        page.save(output / f"{name}-cap-edges.png", optimize=True)


def render_capfix(experiment, chosen, *, original_method="v3_1280",
                  original_title="Task 07 v3", current_title="Task 08 cap fix"):
    _example_comparisons(experiment, chosen, original_method, original_title, current_title)
    render_native_creases(experiment, chosen)
    _cap_edges(experiment, chosen, original_method, original_title, current_title)
