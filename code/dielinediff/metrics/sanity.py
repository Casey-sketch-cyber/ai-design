"""Run the E01 metric sanity checks on paired synthetic textures."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import Evaluator, FACES, _flap_crops


def _bold_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _title_panel(image, text):
    picture = Image.fromarray(image.copy())
    margin = min(picture.size) // 7
    font = _bold_font(max(12, margin // 2))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((margin, margin, picture.width - margin, 3 * margin),
                   fill=(251, 245, 225))
    draw.text((picture.width // 2, 2 * margin), text, anchor="mm", font=font, fill=(34, 23, 23))
    return np.array(picture)


def _synthetic_variants(good, evaluator):
    labels = evaluator.labels
    bad_seam = good.copy()
    bad_orientation = good.copy()
    good_orientation = good.copy()
    clean_light = good.copy()
    bad_light = good.copy()
    ambiguous_light = good.copy()
    low_gamut = good.copy()
    high_gamut = good.copy()
    for index, rect in enumerate(evaluator.topology["faces"].values()):
        left, top, right, bottom = rect["rect"]
        source = good[top:bottom, left:right]
        bad_seam[top:bottom, left:right] = np.roll(
            source, (source.shape[0] // 5, source.shape[1] // (index + 2)), axis=(0, 1)
        )
        turns = rect["upright_rotation"] // 90
        caption = _title_panel(np.rot90(source, turns), "THIS SIDE UP")
        good_orientation[top:bottom, left:right] = np.rot90(caption, -turns)
        bad_orientation[top:bottom, left:right] = np.rot90(caption, 2 - turns)
        shade = np.linspace(0.30, 1.0, len(source))
        ambiguous_light[top:bottom, left:right] = np.rint(source.astype(float) * shade[:, None, None])
        yy, xx = np.indices(source.shape[:2])
        fine_print = ((xx // 5 + yy // 5) % 2) * 28 + 142
        print_rgb = np.stack([fine_print + 35, fine_print, fine_print - 30], axis=-1)
        clean_light[top:bottom, left:right] = print_rgb
        bad_light[top:bottom, left:right] = np.rint(print_rgb * shade[:, None, None])
        low_gamut[top:bottom, left:right] = (126, 129, 127)
        high_gamut[top:bottom, left:right] = (0, 255, 0)
    bad_glue = good.copy()
    bad_glue[labels == 7] = (32, 70, 150)
    bad_flap = good.copy()
    y, x = np.indices(labels.shape)
    pattern = ((y // 2 + x // 2) % 2) == 1
    bad_flap[np.isin(labels, [8, 9]) & pattern] = (13, 11, 10)
    clean_flap = good.copy()
    clean_flap[np.isin(labels, [8, 9])] = 255
    text_flap = clean_flap.copy()
    tuck_components, count = ndimage.label(labels == 8)
    canvas = Image.fromarray(text_flap)
    draw = ImageDraw.Draw(canvas)
    if count:
        largest = int(np.argmax(np.bincount(tuck_components.ravel())[1:])) + 1
        yy, xx = np.where(tuck_components == largest)
        font = _bold_font(min(29, max(12, (int(yy.max()) - int(yy.min())) // 2)))
        draw.text((int(np.median(xx)), int(np.median(yy))), "PRINT", font=font, anchor="mm",
                  fill=(20, 20, 20))
    text_flap = np.where((labels == 8)[..., None], np.asarray(canvas), text_flap)
    empty = good.copy()
    copied = good.copy()
    y, x = evaluator.bleed_y, evaluator.bleed_x
    boundary = good[evaluator.near_y, evaluator.near_x]
    empty[y, x] = 255
    copied[y, x] = boundary
    good[y, x] = np.clip(boundary.astype(np.int16) - np.array([9, 7, 5]), 0, 255)
    return {
        "seam": bad_seam, "orientation_good": good_orientation, "orientation_bad": bad_orientation,
        "light_clean": clean_light, "light": bad_light, "light_ambiguous": ambiguous_light,
        "glue": bad_glue, "flap": bad_flap, "bleed_empty": empty,
        "bleed_copy": copied, "low_gamut": low_gamut, "high_gamut": high_gamut,
        "clean_flap": clean_flap, "text_flap": text_flap,
    }


def _face_images(evaluator, rgb):
    return [
        Image.fromarray(np.rot90(face, evaluator.topology["faces"][name]["upright_rotation"] // 90))
        for name, face in evaluator._crop_faces(rgb).items()
    ]


def run(args):
    import torch
    from dielinediff.color import project_rgb

    geometry = Path(args.geometry)
    enabled = tuple(args.learned.split(",")) if args.learned else ()
    evaluator = Evaluator(geometry, args.lut, args.device, learned=())
    learned = evaluator.learned
    good = np.array(Image.open(args.atlas).convert("RGB"))
    pairs = _synthetic_variants(good, evaluator)
    time_start = time.perf_counter()
    reference = evaluator.measure(good, elapsed_seconds=0.0, peak_memory_mib=0.0)
    elapsed = time.perf_counter() - time_start
    results = {}

    def record(field, good_value, bad_value, direction):
        if good_value is None or bad_value is None:
            passed = None
        else:
            difference = float(good_value) - float(bad_value)
            passed = difference < 0 if direction == "lower" else difference > 0
        results[field] = {"good": good_value, "bad": bad_value,
                          "good_expected": direction, "passed": passed}

    for field, bad, direction in (
        ("seam_sdr", "seam", "lower"),
        ("glue_ink_pct", "glue", "lower"),
        ("flap_edge_pct", "flap", "lower"),
        ("bleed_fill_pct", "bleed_empty", "higher"),
     ):
        record(field, reference[field], evaluator.measure(pairs[bad])[field], direction)
    record("lighting_residual", evaluator.measure(pairs["light_clean"])["lighting_residual"],
           evaluator.measure(pairs["light"])["lighting_residual"], "lower")
    diagnostics = {"checkerboard_light": {
        "base": reference["lighting_residual"],
        "with_shading": evaluator.measure(pairs["light_ambiguous"])["lighting_residual"],
        "interpretation": "Darkening can mute large pre-existing design gradients.",
    }}
    record("bleed_reject_copy", reference["bleed_fill_pct"],
           evaluator.measure(pairs["bleed_copy"])["bleed_fill_pct"], "higher")
    low = evaluator.measure(pairs["low_gamut"])
    high = evaluator.measure(pairs["high_gamut"])
    for field in ("oog_pct", "roundtrip_delta_e00"):
        record(field, low[field], high[field], "lower")

    grid = torch.linspace(0, 1, 33)
    rgb_lattice = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
    input_pixels = torch.tensor([[0.19, 0.42, 0.95], [0.01, 0.97, 0.12]], requires_grad=True)
    identity = project_rgb(input_pixels, rgb_lattice)
    identity.sum().backward()
    lut_ok = torch.allclose(identity, input_pixels) and torch.allclose(
        input_pixels.grad, torch.ones_like(input_pixels)
    )
    results["lut_identity_and_grad"] = {"passed": bool(lut_ok)}
    results["telemetry_passthrough"] = {
        "elapsed_seconds": elapsed, "passed": reference["elapsed_seconds"] == 0.0
        and reference["peak_memory_mib"] == 0.0,
    }

    good_images = _face_images(evaluator, pairs["orientation_good"])
    upside_down = _face_images(evaluator, pairs["orientation_bad"])
    matched = ["a checkerboard and a big upward arrow on a flat packaging design"] * len(FACES)
    unrelated = ["a photograph of a red sports car driving on a race track"] * len(FACES)
    if "lpips" in enabled:
        evaluator.enabled = {"lpips"}
        record("seam_lpips", evaluator.measure(good)["seam_lpips"],
               evaluator.measure(pairs["seam"])["seam_lpips"], "lower")
    if "clip" in enabled:
        upright, clip_match = learned.clip_scores(good_images, matched)
        inverted, _ = learned.clip_scores(upside_down, matched)
        _, clip_mismatch = learned.clip_scores(good_images, unrelated)
        record("upright_rate", upright, inverted, "higher")
        record("clip_score", clip_match, clip_mismatch, "higher")
    if "hps" in enabled:
        record("hps_v2", learned.hps_score(good_images, matched),
               learned.hps_score(good_images, unrelated), "higher")
    if "ocr" in enabled:
        counts = []
        for variant in ("clean_flap", "text_flap"):
            crops = _flap_crops(pairs[variant], evaluator.labels)
            counts.append(sum(len(learned.ocr_lines(crop)) for crop in crops))
        record("flap_ocr_count", *counts, "lower")

        def rate(images):
            angles = [angle for crop in images for _, angle in learned.ocr_lines(crop)
                      if angle in (0, 180)]
            return sum(angle == 0 for angle in angles) / len(angles) if angles else None

        record("ocr_upright_rate", rate(good_images), rate(upside_down), "higher")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"enabled": enabled, "checks": results, "diagnostics": diagnostics}, indent=2) + "\n")
    print(destination)
    return any(check["passed"] is not True for check in results.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learned", default="lpips,clip,hps,ocr")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
