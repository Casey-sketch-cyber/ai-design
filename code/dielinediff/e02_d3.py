"""Isolate the v2 Outside anchor in the five matched Task 05 samples."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from .dieline.geometry import REGION_LABELS
from .e01 import write_json
from .e01b import coverage_summary
from .methods.ours_tf_v2 import DieLineDiffV2
from .metrics.e01b import preflight_atlas
from .metrics.physical import PhysicalEvaluator


def run(root, device):
    base = root / "outputs/e02/step0/d3"
    base.mkdir(parents=True, exist_ok=True)
    old = root / "outputs/e01b/bottom-frame/runs/b598a9650a8a"
    geometry = root / "outputs/e01b/geometry"
    lut = root / "cache/icc/iso-coated-v2-lut.npy"
    topology = json.loads((geometry / "topology.json").read_text())
    config_path = Path("code/configs/E01b.yaml")
    config = yaml.safe_load(config_path.read_text())
    briefs = [json.loads(line) for line in Path("data/briefs/E01_briefs.jsonl").read_text().splitlines()][:5]
    assert [item["id"] for item in briefs] == [f"E01-{n:03d}" for n in range(1, 6)]
    archived = json.loads((old / "inputs.json").read_text())
    assert archived["config"] == {key: value for key, value in config.items() if key != "frozen"}
    assert archived["lut_sha256"] == sha256(lut.read_bytes()).hexdigest()
    with Image.open(geometry / "region_map.png") as src:
        labels = np.array(src)
    evaluator = PhysicalEvaluator(geometry, learned=object())
    engine = None
    comparisons = []
    results = []
    for brief in briefs:
        name = f"{brief['id']}-42"
        destination = base / f"{name}.png"
        source = old / "v2" / f"{name}.png"
        reference = json.loads(source.with_suffix(".run.json").read_text())
        if not destination.is_file() or not destination.with_suffix(".run.json").is_file():
            if engine is None:
                engine = DieLineDiffV2(root / "cache/models",
                                      {"directory": geometry, "topology": topology}, config, device, lut)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            atlas = engine.generate(brief, 42, anchor_outside=False)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            atlas.save(destination.with_suffix(".tmp.png"))
            destination.with_suffix(".tmp.png").replace(destination)
            write_json(destination.with_suffix(".run.json"), {
                "brief": brief["id"], "seed": 42, "protocol": "e02-d3",
                "variant": "no-outside-anchor", "base_image_sha256": sha256(source.read_bytes()).hexdigest(),
                "elapsed_seconds": elapsed, "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "coverage": coverage_summary(engine.last_coverage, config["steps"]),
            })
        pictures = []
        for variant, path, metadata in (
            ("task05", source, reference),
            ("outside_unanchored", destination, json.loads(destination.with_suffix(".run.json").read_text())),
        ):
            with Image.open(path) as src:
                rgb = np.array(src.convert("RGB"))
            preflight = preflight_atlas(rgb, labels, brief["main_color"])
            if variant == "outside_unanchored":
                Image.fromarray(preflight).save(destination.with_suffix(".preflight.png"))
            picture = Image.fromarray(preflight)
            pictures.append(picture)
            bleed = labels == REGION_LABELS["Bleed"]
            white = np.all(rgb[bleed] >= 250, axis=1)
            score = evaluator.measure(rgb, lpips=False)
            results.append({
                "brief_id": brief["id"], "seed": 42, "method": variant,
                "bleed_fill_pct": score["bleed_fill_pct"], "native_white_bleed_pct": 100 * white.mean(),
                "stripe_index": score["stripe_index"], "back_bottom_stripe_index": score["back_bottom_stripe_index"],
                "elapsed_seconds": metadata["elapsed_seconds"], "peak_memory_mib": metadata["peak_memory_mib"],
            })
        comparisons.append((brief["id"], pictures))
        print(f"{name} bleed {results[-2]['bleed_fill_pct']:.2f} -> {results[-1]['bleed_fill_pct']:.2f}",
              flush=True)
    with (base / "d3.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    # Same registered 1:1 crop, reaching from the free left edge into the body.
    x0, y0, _, y1 = topology["faces"]["L"]["rect"]
    y = (y0 + y1) // 2
    crop = (max(0, x0 - 42), y - 144, x0 + 90, y + 144)
    tiles = [image.crop(crop).transpose(Image.Transpose.ROTATE_90)
             for _, variants in comparisons for image in variants]
    width, height = tiles[0].size
    panel = Image.new("RGB", (2 * width + 66, 5 * (height + 21) + 68), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=14)
    draw.text((17, 10), "D3 / Native left cut-edge crops, same preflight", font=font, fill="black")
    for i, title in enumerate(("Task 05 control", "Outside anchor removed")):
        draw.text((17 + i * (width + 24), 31), title, font=font, fill="black")
    for i, (brief_id, _) in enumerate(comparisons):
        yy = 60 + i * (height + 21)
        draw.text((17, yy - 13), brief_id, font=font, fill="black")
        for j in range(2):
            panel.paste(tiles[2 * i + j], (17 + j * (width + 24), yy))
    panel.save(base / "d3-native-crops.png", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.root is None:
        parser.error("source code/scripts/e01_env.sh or provide --root")
    run(args.root, args.device)
