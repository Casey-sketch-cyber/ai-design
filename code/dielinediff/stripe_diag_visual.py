"""Render lossless native-pixel evidence for frozen stripe v1 peak folds."""

from collections.abc import Mapping
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import _oriented_strip
from .metrics.physical import PIXELS_PER_MM, STRIPE_MM, STRIPE_REFERENCE_OFFSET_MM, _edges_on_sheet


_INK = (24, 39, 52)
_PAPER = (247, 249, 250)
_MARK = (21, 137, 153)


def _part(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip(".-") or "fold"


def _is_peak(value):
    return str(value).lower() in ("true", "1")


def _fold_strips(topology, atlas, a, b, depth):
    """Read both strips with depth increasing inward and matched tangential order."""
    rects = topology["faces"]
    a_face, a_edge = a
    b_face, b_edge = b
    first = _edges_on_sheet(rects[a_face]["rect"], a_edge)
    second = _edges_on_sheet(rects[b_face]["rect"], b_edge)
    if frozenset(first) != frozenset(second):
        raise ValueError(f"Expected a native dieline fold: {a_face}/{b_face}")
    for seam in topology["adjacencies"]:
        keys = {(seam[name]["face"], seam[name]["edge"]) for name in ("a", "b")}
        if keys == {a, b}:
            reverse = seam["reverse"]
            break
    else:
        raise ValueError(f"Native fold missing from topology: {a_face}/{b_face}")

    def read(face, edge):
        rect = tuple(rects[face]["rect"])
        pixels = np.asarray(atlas.crop(rect).convert("RGB"))
        return _oriented_strip(pixels, edge, depth).transpose(1, 0, 2)

    strip_a, strip_b = read(*a), read(*b)
    if strip_a.shape != strip_b.shape or strip_a.shape[0] != depth:
        raise ValueError(f"Insufficient length or depth at native fold {a_face}/{b_face}")
    return strip_a, strip_b[:, ::-1] if reverse else strip_b


def _paste(page, crop, x, y):
    page.paste(Image.fromarray(np.ascontiguousarray(crop)), (x, y))


def render_peak(row, source, destination):
    """Publish one selected fold, including both sides' v1 energy windows."""
    topology = json.loads((source.geometry / "topology.json").read_text(encoding="utf-8"))
    image_spec = topology["image"]
    ppm = float(image_spec["pixels_per_mm"])
    crease_stop = round(STRIPE_MM * ppm)
    reference_start = round(STRIPE_REFERENCE_OFFSET_MM * ppm)
    reference_stop = round((STRIPE_REFERENCE_OFFSET_MM + STRIPE_MM) * ppm)
    if not 0 < crease_stop <= reference_start < reference_stop:
        raise ValueError("The native grid cannot represent the physical stripe windows")
    sides = [(row[f"{axis}_face"], row[f"{axis}_edge"]) for axis in ("a", "b")]
    with Image.open(source.path) as atlas:
        if atlas.size != (image_spec["width"], image_spec["height"]):
            raise ValueError(f"Native atlas size does not match topology: {source.path}")
        first, second = _fold_strips(topology, atlas, *sides, reference_stop)

    span = first.shape[1]
    window_height = max(crease_stop, reference_stop - reference_start)
    gap = 22
    left, right = 23, 23 + span + gap
    overview_y = 92
    bands_y = overview_y + 2 * reference_stop + 59
    band_line = 45 + window_height
    page = Image.new("RGB", (max(700, right + span + 23), bands_y + 2 * band_line + 22), _PAPER)
    draw = ImageDraw.Draw(page)
    caption = ImageFont.load_default(size=14)
    large = ImageFont.load_default(size=19)
    fold = " / ".join(face for face, _ in sides)
    draw.text((left, 13), f"{source.brief['id']} | seed {source.seed} | {source.method} | {fold} | peak v1 {float(row['v1']):.3f}",
              font=large, fill=_INK)
    draw.text((left, 42), f"Native pixels at {ppm:.4g} px/mm. Energies from frozen v1 at {PIXELS_PER_MM:g} px/mm.",
              font=caption, fill=_INK)
    overview_label = "Full native fold, " + " / ".join(f"{face} {edge}" for face, edge in sides)
    draw.text((left, overview_y - 21), overview_label + " (above / below)", font=caption, fill=_INK)
    _paste(page, first[::-1], left, overview_y)
    _paste(page, second, left, overview_y + reference_stop)
    for x in (left - 10, left + span + 4):
        draw.line((x, overview_y + reference_stop, x + 6, overview_y + reference_stop),
                  fill=_MARK, width=2)
    draw.text((left, bands_y - 41), f"Near fold | 0-{STRIPE_MM:g} mm | native depth 0:{crease_stop}",
              font=caption, fill=_INK)
    draw.text((right, bands_y - 41),
              f"Reference | {STRIPE_REFERENCE_OFFSET_MM:g}-{STRIPE_REFERENCE_OFFSET_MM + STRIPE_MM:g} mm"
              f" | native depth {reference_start}:{reference_stop}", font=caption, fill=_INK)
    for index, (axis, band) in enumerate(zip(("a", "b"), (first, second), strict=True)):
        face, edge = sides[index]
        y = bands_y + index * band_line
        for x, values, metric in (
            (left, band[:crease_stop], "crease_energy"),
            (right, band[reference_start:reference_stop], "reference_energy_v1"),
        ):
            energy = float(row[f"{axis}_{metric}"])
            draw.text((x, y - 20), f"{face} {edge} | E = {energy:.4e}", font=caption, fill=_INK)
            _paste(page, values, x, y)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    page.save(destination, format="PNG", optimize=True)
    return destination


def render_peaks(rows, sources, output_dir):
    """Render selected per-box v1 maxima from the scorer's fold rows and archived atlases.

    Sources can be an iterable of unique SourceAtlas records, or a map keyed
    by (cohort, method, brief_id, seed) when a method spans several cohorts.
    """
    if not isinstance(sources, Mapping):
        lookup = {}
        for source in sources:
            key = (source.method, source.brief["id"], int(source.seed))
            if key in lookup:
                raise ValueError(f"Duplicate visual source {key}; supply cohort-qualified keys")
            lookup[key] = source
    else:
        lookup = sources
    output_dir = Path(output_dir)
    published = []
    for row in rows:
        if not _is_peak(row["controls_v1"]):
            continue
        key = (row["method"], row["brief_id"], int(row["seed"]))
        cohort_key = (row["cohort"], *key)
        source = lookup.get(cohort_key, lookup.get(key))
        if source is None:
            raise ValueError(f"No original atlas for peak fold {cohort_key}")
        prefix = output_dir / _part(row["cohort"]) / _part(row["method"])
        filename = f"{_part(row['brief_id'])}-{int(row['seed'])}-{_part(row['fold'])}.png"
        published.append(render_peak(row, source, prefix / filename))
    return published


_HISTORY_METHODS = (
    ("v2_task03", "Task 03 E01b v2"),
    ("v2_task04", "Task 04 crease fix v2"),
    ("v2", "Task 05 bottom frame v2"),
    ("b2_post", "B2-post baseline"),
)


def render_history_controls(sources, output_dir):
    """Render matched B/Btm crops without using scores to pick historical controls."""
    selected_methods = {method for method, _ in _HISTORY_METHODS}
    available = {(job.brief["id"], job.method): job for job in sources
                 if int(job.seed) == 42 and job.method in selected_methods}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    published = []
    for number in range(1, 6):
        brief = f"E01-{number:03d}"
        views = []
        for method, title in _HISTORY_METHODS:
            source = available[brief, method]
            topology = json.loads((source.geometry / "topology.json").read_text(encoding="utf-8"))
            ppm = float(topology["image"]["pixels_per_mm"])
            depth = round(7 * ppm)
            with Image.open(source.path) as atlas:
                first, second = _fold_strips(topology, atlas, ("B", "bottom"), ("Btm", "top"), depth)
            x0, y0, x1, _ = topology["faces"]["Btm"]["rect"]
            label = f"{title} | {ppm:.4g} px/mm | y={y0}, x={x0}..{x1 - 1}"
            views.append((label, first, second))
        span = max(view[1].shape[1] for view in views)
        size = max(view[1].shape[0] for view in views)
        col_width = span + 25
        row_height = 2 * size + 59
        page = Image.new("RGB", (2 * col_width + 10, 47 + 2 * row_height), _PAPER)
        draw = ImageDraw.Draw(page)
        draw.text((20, 12), f"{brief} | seed 42 | B / Btm raw original pixels", font=font, fill=_INK)
        for index, (label, first, second) in enumerate(views):
            x, y = 20 + col_width * (index % 2), 60 + row_height * (index // 2)
            depth, width = first.shape[:2]
            draw.text((x, y - 22), label, font=font, fill=_INK)
            _paste(page, first[::-1], x, y)
            _paste(page, second, x, y + depth)
            for tick in (x - 10, x + width + 4):
                draw.line((tick, y + depth, tick + 6, y + depth), fill=_MARK, width=2)
        path = destination / f"{brief}-42-native-controls.png"
        page.save(path, format="PNG", optimize=True)
        published.append(path)
    return published
