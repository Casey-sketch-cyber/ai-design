#!/usr/bin/env python3
"""Reproduce the Task 05 interventions against the pinned Task 04 implementation."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import detrend
import torch
import yaml

from dielinediff.e01 import read_json, write_json
from dielinediff.e01b import coverage_summary
from dielinediff.metrics.physical import PhysicalEvaluator, stripe_energy


REVISION = "f1ef63a5a11561ea7267c032532f3bec279fd7d6"
NAMES = ("task04", "task04-no-bottom-write", "task04-real-context")
LABELS = ("Task 04 control", "No bottom-window writes within 2 mm of fold",
          "Ring reads real bottom pixels; bottom still rotated")


def task04_source(repo, path):
    """Read the historical files without checking out or modifying production code."""
    return subprocess.check_output(
        ["git", "show", f"{REVISION}:{path}"], cwd=repo, encoding="utf-8"
    )


def load_probe_module(name, source):
    module_name = f"dielinediff.methods.{name}"
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, module_name, "exec"), vars(module))
    return module


def replace_once(source, line, replacement):
    if source.count(line) != 1:
        raise RuntimeError(f"Could not identify the Task 04 probe site: {line!r}")
    return source.replace(line, replacement)


def remove_bottom_writes(plan, topology, windows_v2, pixels_per_mm):
    width = plan.shape[1]
    x0, y0, x1, _ = windows_v2._latent_rect(topology["faces"]["Btm"]["rect"])
    transition = 2 * pixels_per_mm / windows_v2.SCALE
    keep_windows = []
    removed = []
    for window in plan.windows:
        if window.primary_face != "Btm" or window.name != "lid-Btm":
            keep_windows.append(window)
            continue
        ids = window.write_ids
        x, y = ids % width, ids // width
        mask = ((x0 <= x) & (x < x1) & ((y - y0 + 0.5).abs() <= transition))
        removed.append(ids[mask])
        keep_windows.append(replace(window, write_ids=ids[~mask], write_sources=window.write_sources[~mask],
                                    write_weights=window.write_weights[~mask]))
    removed = torch.cat(removed)
    coverage = windows_v2._coverage(keep_windows, plan.shape)
    required = plan.required.clone()
    uncovered = removed[torch.as_tensor(coverage.ravel(), device=required.device)[removed] == 0]
    required[uncovered] = False
    result = replace(plan, windows=tuple(keep_windows), required=required, contributions=coverage)
    return result, removed.numel(), uncovered.numel()

def diagnose(repo, root, geometry, directory):
    topology = read_json(geometry / "topology.json")
    with Image.open(geometry / "region_map.png") as image:
        region_map = np.array(image)
    with (repo / "data/briefs/E01_briefs.jsonl").open(encoding="utf-8") as stream:
        brief = json.loads(next(stream))
    if brief["id"] != "E01-001":
        raise ValueError("The diagnostics require E01-001 first in the brief archive")
    config = yaml.safe_load(task04_source(repo, "code/configs/E01b.yaml"))
    window_source = task04_source(repo, "code/dielinediff/methods/windows_v2.py")
    windows_v2 = load_probe_module("_task04_windows", window_source)
    fixed_read = '            read = np.where(regions == REGION_LABELS["Btm"], fallback, read)\n'
    real_reads = load_probe_module("_task04_real_reads", replace_once(window_source, fixed_read, ""))
    engine_source = task04_source(repo, "code/dielinediff/methods/ours_tf_v2.py")
    engine_source = replace_once(
        engine_source, "from .windows_v2 import build_windows, save_coverage_heatmap",
        "from ._task04_windows import build_windows, save_coverage_heatmap",
    )
    decode = "        decoded = self.decode(latent)[0].permute(1, 2, 0)\n"
    capture = "        self.diagnostic_latent = latent.detach().float().cpu().numpy()\n"
    module = load_probe_module("_task04_capture", replace_once(engine_source, decode, capture + decode))

    torch.cuda.set_device(0)
    engine = module.DieLineDiffV2(root / "cache/models", {"directory": geometry, "topology": topology},
                                  config, "cuda", root / "cache/icc/iso-coated-v2-lut.npy")
    control = engine.plans[True]
    restored = real_reads.build_windows(topology, region_map, device="cuda")
    for before, after in zip(control.windows, restored.windows, strict=True):
        if before.name != after.name or not torch.equal(before.write_ids, after.write_ids):
            raise RuntimeError("Real-context probe unexpected change to write coordinates")
    plans = ((control, 0, 0),
             remove_bottom_writes(control, topology, windows_v2, config["pixels_per_mm"]),
             (restored, 0, 0))
    for name, (plan, num_removed, missing) in zip(NAMES, plans, strict=True):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        engine.plans[True] = plan
        engine.active = plan.required
        engine.required = int(plan.required.sum())
        start = perf_counter()
        image = engine.generate(brief, 42)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        image.save(directory / f"{name}.png")
        np.save(directory / f"{name}.npy", engine.diagnostic_latent)
        write_json(directory / f"{name}.json", {
            "brief": brief["id"], "seed": 42, "name": name,
            "baseline_commit": REVISION,
            "coverage": coverage_summary(engine.last_coverage, config["steps"]),
            "actual_coverage": int(control.required.sum()) - missing,
            "removed_bottom_writes": num_removed,
            "minutes": seconds / 60, "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        })
        print(f"{name} {seconds:.1f}s removed={num_removed} uncovered={missing}", flush=True)


def plot(root, geometry, directory):
    physical = PhysicalEvaluator(geometry, learned=object())
    x0, yfold, x1, y1 = physical.native_topology["faces"]["Btm"]["rect"]
    crop = (x0 + 8, yfold - 40, x1 - 8, yfold + 40)
    old_png = root / "outputs/e01b/crease-fix/runs/bc079eabca00/v2/E01-001-42.png"
    images = [Image.open(directory / f"{name}.png").convert("RGB") for name in NAMES]
    latents = [np.load(directory / f"{name}.npy")[0] for name in NAMES]
    ly, lx0, lx1 = yfold // 8, (x0 + 8) // 8, (x1 - 8) // 8
    swaths = [latent[:, ly - 8:ly + 8, lx0:lx1] for latent in latents]
    bound = np.quantile(np.abs(swaths[0]), 0.985, axis=(1, 2)).reshape(4, 1, 1)
    small = ImageFont.load_default(size=15)
    large = ImageFont.load_default(size=19)
    white, dark = (255, 255, 255), (29, 43, 56)
    navy, red = (23, 49, 82), (201, 67, 61)
    fold_color = (25, 149, 162)

    edge_panel = Image.new("RGB", (crop[2] - crop[0] + 88, 398), white)
    edge_draw = ImageDraw.Draw(edge_panel)
    edge_draw.text((18, 9), "E01-001, seed 42  /  Back-to-bottom fold", font=large, fill=dark)
    edge_draw.text((18, 37), "Native RGB, no resizing. Teal ticks mark the fold.", font=small, fill=dark)
    for i, (image, label) in enumerate(zip(images, LABELS, strict=True)):
        gy = 80 + i * 103
        edge_draw.text((34, gy - 19), label, font=small, fill=dark)
        patch = image.crop(crop)
        edge_panel.paste(patch, (34, gy))
        edge_draw.line((13, gy + 40, 29, gy + 40), width=2, fill=fold_color)
        edge_draw.line((34 + patch.width + 4, gy + 40, 34 + patch.width + 20, gy + 40),
                       width=2, fill=fold_color)
    edge_panel.save(directory / "diagnostics-native.png", optimize=True)

    latent_panel = Image.new("RGB", (4 * (lx1 - lx0) * 4 + 94, 389), white)
    latent_draw = ImageDraw.Draw(latent_panel)
    latent_draw.text((18, 9), "Final denoised latent before VAE  /  Four channels", font=large, fill=dark)
    latent_draw.text((18, 37), "4x nearest-neighbor. Blue is negative, red positive. Each channel uses the same scale in all rows.",
                     font=small, fill=dark)
    for k, (latent, label) in enumerate(zip(swaths, LABELS, strict=True)):
        vertical = 108 + k * 94
        latent_draw.text((26, vertical - 20), label, font=small, fill=dark)
        normalized = (latent / bound).clip(-1, 1)
        for c, channel in enumerate(normalized):
            origin = 26 + c * ((lx1 - lx0) * 4 + 12)
            negative = np.minimum(channel[:, :, None], 0) * (np.array(white) - navy)
            positive = np.maximum(channel[:, :, None], 0) * (np.array(red) - white)
            patch = Image.fromarray(np.rint(np.array(white) + negative + positive).astype(np.uint8))
            patch = patch.resize((patch.width * 4, patch.height * 4), Image.Resampling.NEAREST)
            latent_panel.paste(patch, (origin, vertical))
            latent_draw.line((origin - 4, vertical + 32, origin - 1, vertical + 32),
                             width=2, fill=fold_color)
            latent_draw.line((origin + patch.width, vertical + 32, origin + patch.width + 3, vertical + 32),
                             width=2, fill=fold_color)
            if k == 0:
                latent_draw.text((origin, 61), f"Channel {c}, +/-{float(bound[c, 0, 0]):.2f}",
                                 font=small, fill=dark)
    latent_panel.save(directory / "diagnostics-latent.png", optimize=True)

    def period_strength(rgb, y, offset, width):
        # Per-row detrending removes gradual motif and illumination gradients without resizing.
        # Coherent Fourier coefficient at 1/8 px, normalized as RGB levels at native resolution.
        band = rgb[y + offset:y + offset + width, crop[0]:crop[2]].mean(axis=2)
        pattern = np.exp(-2j * np.pi * np.arange(band.shape[1]) / 8)
        return float(2 * abs((detrend(band, axis=1) @ pattern).mean()) / len(pattern))

    scores = {}
    for name, image, latent in zip(NAMES, images, latents, strict=True):
        rgb = np.array(image)
        row_steps = np.abs(np.diff(latent[:, ly - 5:ly + 6, lx0:lx1], axis=1)).mean(axis=(0, 2))
        stripe, back_bottom = stripe_energy(physical._physical_faces(rgb), physical.creases)
        scores[name] = {
            **json.loads((directory / f"{name}.json").read_text()),
            "stripe_index": stripe, "back_bottom_stripe_index": back_bottom,
            "latent_fold_crossing_mean_absolute_step": float(row_steps[4]),
            "latent_neighbor_steps_mean_absolute": float(np.delete(row_steps, 4).mean()),
            "native_8px_coherent_amplitude_rgb": {
                "back_fold_8px": period_strength(rgb, yfold, -8, 8),
                "bottom_fold_8px": period_strength(rgb, yfold, 0, 8),
                "back_reference_8px": period_strength(rgb, yfold, -64, 8),
                "bottom_reference_8px": period_strength(rgb, yfold, 56, 8),
            },
            "sha256": {
                suffix: sha256((directory / f"{name}.{suffix}").read_bytes()).hexdigest()
                for suffix in ("png", "npy")
            },
        }
    if old_png.is_file():
        with Image.open(old_png) as original:
            control = np.array(images[0])
            scores["task04"]["agrees_with_archived_pixels"] = bool(np.array_equal(np.array(original), control))
            scores["task04"]["max_difference_from_archived_pixels"] = int(
                np.max(np.abs(control.astype(np.int16) - np.array(original).astype(np.int16)))
            )
    write_json(directory / "diagnostics-summary.json", {
        "brief": "E01-001", "seed": 42, "native_crop_xyxy": list(crop), "fold_y_px": yfold,
        "latent_crop_xyxy": [lx0, ly - 8, lx1, ly + 8], "scores": scores,
    })
    print(f"Diagnostics and figures saved in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.root is None:
        parser.error("source code/scripts/e01_env.sh or provide --root")
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    geometry = root / "outputs/e01b/geometry"
    directory = root / "outputs/e01b/bottom-frame/diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        diagnose(repo, root, geometry, directory)
    plot(root, geometry, directory)


if __name__ == "__main__":
    main()
