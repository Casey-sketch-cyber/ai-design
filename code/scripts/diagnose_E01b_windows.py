"""Record actual E01b gather coordinates and positive write masks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from dielinediff.dieline.geometry import REGION_LABELS


def window_module(revision):
    if revision is None:
        from dielinediff.methods import windows_v2
        return windows_v2

    relative = "code/dielinediff/methods/windows_v2.py"
    repository = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"], cwd=repository, text=True,
    )
    name = "dielinediff.methods._diagnostic_revision"
    module = ModuleType(name)
    module.__package__ = "dielinediff.methods"
    sys.modules[name] = module
    exec(compile(source, f"{revision}:{relative}", "exec"), module.__dict__)
    return module


def coordinates(plan):
    y, x = np.indices(plan.shape)
    height, width = plan.shape
    return torch.as_tensor(np.stack((x / (width - 1), y / (height - 1), plan.labels / 10)),
                           dtype=torch.float32).flatten(1)[None]


def window_mapping(window, flat):
    pixels = window.gather(flat)[0].permute(1, 2, 0).numpy()
    write = np.zeros(np.prod(window.shape), dtype=bool)
    write[window.write_sources.numpy()] = True
    return np.rint(pixels * 255).astype(np.uint8), write.reshape(window.shape)


def write_montage(plan, destination, caption):
    """Place every window coordinate image beside the same image with writes marked."""
    flat = coordinates(plan)
    width, height = 304, 160
    columns = 3
    rows = (len(plan.windows) + columns - 1) // columns
    page = Image.new("RGB", (columns * width + 20, rows * height + 65), (250, 251, 252))
    draw = ImageDraw.Draw(page)
    heading = ImageFont.load_default(size=20)
    label = ImageFont.load_default(size=13)
    ink = (31, 44, 55)
    draw.text((10, 8), caption, font=heading, fill=ink)
    draw.text((10, 35), "Left: gathered atlas coordinates (red x, green y, blue region)."
                        " Right: amber tint marks positive writes.", font=label, fill=ink)
    for index, window in enumerate(plan.windows):
        x = 10 + index % columns * width
        y = 64 + index // columns * height
        pixels, write = window_mapping(window, flat)
        colored = pixels.copy()
        colored[write] = np.rint(pixels[write] * 0.61 + np.array([255, 185, 30]) * 0.39).astype(np.uint8)
        for offset, image in ((0, pixels), (146, colored)):
            tile = Image.fromarray(image)
            step = min(137 / tile.width, 136 / tile.height)
            size = tuple(max(1, round(side * step)) for side in tile.size)
            page.paste(tile.resize(size, Image.Resampling.NEAREST), (x + offset, y + 21))
        draw.text((x, y), f"{window.name}  {window.shape[1]} x {window.shape[0]} latent", font=label, fill=ink)
    page.save(destination, optimize=True)


def native_edges(plan, topology):
    """Trace the crop-connected F/T and B/Btm normals using the actual gathered IDs."""
    width = plan.shape[1]
    details = {}
    for name, edge, adjacent in (("T", "bottom", "F"), ("Btm", "top", "B")):
        window = next(window for window in plan.windows if window.name in (f"lid-{name}", f"frame-{name}"))
        read = window.read_ids.numpy().reshape(window.shape)
        write = np.zeros(read.size, dtype=bool)
        write[window.write_sources.numpy()] = True
        write = write.reshape(window.shape)
        rotation = topology["faces"][name]["upright_rotation"] // 90 if window.name == "lid-Btm" else 0
        read, write = np.rot90(read, -rotation), np.rot90(write, -rotation)
        bounds = [(value + 3) // 8 for value in topology["faces"][name]["rect"]]
        panel_h, panel_w = bounds[3] - bounds[1], bounds[2] - bounds[0]
        margin_y, margin_x = (read.shape[0] - panel_h) // 2, (read.shape[1] - panel_w) // 2
        column = margin_x + panel_w // 2
        rows = (range(margin_y - 1, -1, -1) if edge == "top"
                else range(margin_y + panel_h, read.shape[0]))
        samples = [{"x": int(read[row, column] % width), "y": int(read[row, column] // width),
                    "write": bool(write[row, column])} for row in rows]
        ring_writes = np.unique(np.concatenate([ring.write_ids.numpy() for ring in plan.windows
                                               if ring.name.startswith("ring-")]))
        overlap = np.intersect1d(window.write_ids.numpy(), ring_writes)
        regions = plan.labels.ravel()[overlap]
        details[f"{name}-{adjacent}"] = {
            "native_outward_trace": samples,
            "window_rotation_degrees": rotation * 90,
            "shared_lid_cells": int(np.count_nonzero(regions == REGION_LABELS[name])),
            "shared_side_cells": int(np.count_nonzero(regions == REGION_LABELS[adjacent])),
        }
    label = REGION_LABELS["Btm"]
    details["ring_bottom_context"] = {
        window.name: {
            "reads_btm": int(np.count_nonzero(plan.labels.ravel()[window.read_ids.numpy()] == label)),
            "writes_btm": int(np.count_nonzero(plan.labels.ravel()[window.write_ids.numpy()] == label)),
        }
        for window in plan.windows if window.name.startswith("ring-")
    }
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--transition-mm", type=float)
    parser.add_argument("--caption", default="E01b window mapping")
    args = parser.parse_args()
    module = window_module(args.revision)
    topology = json.loads((args.geometry / "topology.json").read_text(encoding="utf-8"))
    with Image.open(args.geometry / "region_map.png") as image:
        options = {} if args.transition_mm is None else {"transition_mm": args.transition_mm}
        plan = module.build_windows(topology, np.asarray(image), device="cpu", **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_montage(plan, args.output, args.caption)
    args.output.with_suffix(".json").write_text(
        json.dumps(native_edges(plan, topology), indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
