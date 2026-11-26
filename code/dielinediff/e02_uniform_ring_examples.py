"""Registered two-seed atlas comparisons and full native cap-edge strips."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import Sample
from .e02_capfix_visual import CAPS, EDGES, INK, MARK, _cap_neighbors, _edge_tiles
from .e02_capgeom_eval import _cohort
from .e02_sources import SourceAtlas
from .e02_visual import _preview
from .metrics.physical import _edges_on_sheet
from .render import render_views


SEEDS = (42, 1234)
FOCUS = ("E01-001", "E01-003", "E01-004", "E01-007", "E01-008")
COMPARISONS = ("E01-001", "E01-003", "E01-007")
STRIP_WIDTH = 25
COLUMNS = ("B2-regional", "Task 13 G5", "Task 15")


def _historical(exp, brief, seed):
    directory = "g5" if seed == 42 else "seed_1234"
    key = f"{brief['id']}-{seed}"
    atlas = SourceAtlas("g5", brief, seed, exp.previous / directory / f"{key}.png",
                        exp.geometry)
    evaluation = exp.previous / "evaluation" / directory
    return Sample(atlas, evaluation / f"{key}.metrics.json",
                  evaluation / f"{key}.preflight.png")


def _evidence(job, geometry_sha):
    atlas = job.atlas
    raw = file_hash(atlas.path)
    expected = {"atlas_sha256": raw, "run_sha256": file_hash(atlas.metadata),
                "geometry_sha256": geometry_sha}
    if (file_hash(atlas.geometry / "topology.json") != geometry_sha
            or read_json(job.score)["input"] != expected):
        raise ValueError(f"Comparison atlas does not match its registered score: {atlas.path}")
    return {"raw_source": str(atlas.path), "raw_sha256": raw,
            "preflight_source": str(job.preflight), "preflight_sha256": file_hash(job.preflight),
            "score_sha256": file_hash(job.score)}


def _save_image(image, destination, **options):
    temporary = destination.with_suffix(f".{os.getpid()}.tmp{destination.suffix}")
    image.save(temporary, **options)
    temporary.replace(destination)


def _comparison(jobs, brief_id, seed, chosen, output):
    width, height = 310, 295
    image = Image.new("RGB", (3 * width + 74, 2 * height + 78), "white")
    draw = ImageDraw.Draw(image)
    header = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text((18, 13), f"{brief_id} / seed {seed}", fill=INK, font=header)
    for row, title in enumerate(("Raw\natlas", "Folded\npreflight")):
        draw.multiline_text((12, 62 + row * height + height // 2 - 14), title,
                            fill=INK, font=small, spacing=4)
    titles = (*COLUMNS[:2], f"Task 15 final {chosen.upper()}")
    for column, (job, title) in enumerate(zip(jobs, titles, strict=True)):
        x = 69 + column * width
        draw.text((x + width // 2, 50), title, anchor="mm", fill=INK, font=small)
        render_dir = output / "folded" / brief_id / f"column-{column + 1}"
        folded = render_views(job.atlas.geometry, job.preflight, render_dir,
                              size=360, views=("iso",))["iso"]
        with Image.open(job.atlas.path) as raw:
            tiles = [_preview(raw.convert("RGB"), (width - 22, height - 16)),
                     _preview(folded, (width - 22, height - 16))]
        for row, tile in enumerate(tiles):
            image.paste(tile, (x + (width - tile.width) // 2,
                               66 + row * height + (height - tile.height) // 2))
    path = output / f"{brief_id}-comparison.jpg"
    _save_image(image, path, quality=94, optimize=True)
    return path


def _cap_edges(jobs, brief_id, seed, chosen, topology, output):
    bounds = {face: tuple(entry["rect"]) for face, entry in topology["faces"].items()}
    neighbors = {face: _cap_neighbors(topology, face) for face in CAPS}
    span = max(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in (bounds[face] for face in CAPS))
    stride = 2 * STRIP_WIDTH + 27
    image = Image.new("RGB", (3 * (span + 36) + 12, 92 + 8 * stride), "white")
    draw = ImageDraw.Draw(image)
    header = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text((18, 12), f"{brief_id} / seed {seed} / Native RGB pixels / Side above, cap below",
              fill=INK, font=header)
    titles = (*COLUMNS[:2], f"Task 15 final {chosen.upper()}")
    for column, (job, title) in enumerate(zip(jobs, titles, strict=True)):
        x = 18 + column * (span + 36)
        draw.text((x, 44), title, fill=INK, font=small)
        with Image.open(job.atlas.path) as atlas:
            faces = {face: np.asarray(atlas.crop(box).convert("RGB")) for face, box in bounds.items()}
        for index, (face, edge) in enumerate((face, edge) for face in CAPS for edge in EDGES):
            neighbor, other_edge, reverse = neighbors[face][edge]
            side, cap = _edge_tiles(faces, face, edge, neighbor, other_edge, reverse, STRIP_WIDTH)
            if side.size != cap.size or side.height != STRIP_WIDTH:
                raise ValueError(f"Native edge dimensions diverge: {face} {edge}")
            y = 87 + index * stride
            first = frozenset(_edges_on_sheet(bounds[face], edge))
            second = frozenset(_edges_on_sheet(bounds[neighbor], other_edge))
            label = f"{face} {edge} / {neighbor} {other_edge}"
            if first == second:
                label += " (native fold)"
            draw.text((x, y - 6), label, anchor="lb", fill=INK, font=small)
            image.paste(side, (x, y))
            image.paste(cap, (x, y + STRIP_WIDTH))
            draw.line((x - 10, y + STRIP_WIDTH, x - 4, y + STRIP_WIDTH), width=2, fill=MARK)
    path = output / f"{brief_id}-cap-edges.png"
    _save_image(image, path, optimize=True)
    return path


def export_examples(exp, groups_by_seed, chosen, output):
    """Export fixed three-way visual comparisons and both sets of native cap edges."""
    output = Path(output).resolve()
    if not output.is_relative_to(exp.output.resolve()):
        raise ValueError("Comparison destination must be in the current Task 15 run")
    geometry_file = exp.geometry / "topology.json"
    topology, geometry_sha = read_json(geometry_file), file_hash(geometry_file)
    manifest = {"selection": chosen, "comparison_briefs": list(COMPARISONS),
                "focus_briefs": list(FOCUS), "native_edge_width_per_side_px": STRIP_WIDTH,
                "native_edge_rendering": "Raw atlas strips with rigid reorientation, no resampling",
                "columns": [*COLUMNS[:2], f"Task 15 final {chosen.upper()}"],
                "geometry_sha256": geometry_sha, "seeds": {}}
    for seed in SEEDS:
        groups = groups_by_seed[seed]
        incumbent = "b2_regional" if "b2_regional" in groups else "incumbent"
        baseline = _cohort(groups[incumbent], seed=seed)
        final = _cohort(groups[chosen], seed=seed)
        directory = output / f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest["seeds"][str(seed)] = entries = {}
        for brief_id in FOCUS:
            baseline_job, _ = baseline[brief_id]
            final_job, _ = final[brief_id]
            jobs = (baseline_job, _historical(exp, final_job.atlas.brief, seed), final_job)
            entry = {"sources": [_evidence(job, geometry_sha) for job in jobs]}
            edges = _cap_edges(jobs, brief_id, seed, chosen, topology, directory)
            entry["native_cap_edges"] = edges.relative_to(output).as_posix()
            if brief_id in COMPARISONS:
                preview = _comparison(jobs, brief_id, seed, chosen, directory)
                entry["comparison"] = preview.relative_to(output).as_posix()
            entries[brief_id] = entry
    write_json(output / "manifest.json", manifest)
    return manifest
