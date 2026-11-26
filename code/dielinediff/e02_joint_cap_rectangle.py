"""Recheck the E01-009 seed-42 bottom-cap rectangle on archived stage inputs."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_cap_energy_curves import CapEnergyCurves, TASK17
from .e02_capfix_ring import bytes_digest
from .methods.v3_capfix_geometry import cap_at_original_scale
from .methods.v3_capgeom_geometry import build_capgeom_layout, compose_capgeom_atlas
from .methods.v3_capgeom_init import harmonic_fill
from .methods.v3_ring_midscale_cap import inject_texture_at_scale


CASE = "E01-009-42"
FACE = "Btm"
# Approximate native-pixel ROIs registered from the historical E0 review image.
EDGE_LOCATORS = {"upper": {"axis": 0, "index": 34, "along": (180, 405)},
                 "right": {"axis": 1, "index": 412, "along": (47, 165)}}


def _profile(image, roi):
    lightness = rgb2lab(np.asarray(image).astype(np.float32) / 255)[..., 0]
    values = lightness.T if roi["axis"] else lightness
    a, b = roi["along"]
    center = roi["index"]
    row = values[center - 6:center + 7, a:b].mean(axis=1)
    return {"positions": list(range(center - 6, center + 7)), "mean_Lstar": row.tolist(),
            "nearest_low_high_3px_difference_Lstar": float(row[6:9].mean() - row[3:6].mean())}


def diagnose_rectangle(root, output):
    root, output = Path(root), Path(output)
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Save Task 20 source evidence on the project data disk")
    output.mkdir(parents=True, exist_ok=True)
    e02 = root / "outputs/e02"
    archive = e02 / "ring-m2m3/runs" / TASK17
    raw = archive / "u1_sigma28" / f"{CASE}.png"
    preflight = archive / "evaluation/u1_sigma28" / f"{CASE}.preflight.png"
    metadata = raw.with_suffix(".run.json")
    record = read_json(metadata)
    case_cache = e02 / "capfix/rings/0979bc4c77ef" / f"{CASE}.npz"
    proof = read_json(case_cache.with_suffix(".json"))
    if (file_hash(raw) != record["raw_sha256"] or file_hash(case_cache) != proof["cache_sha256"]
            or proof["cache_sha256"] != record["ring"]["cache_sha256"]):
        raise ValueError("Task 17 rectangle atlas and R0 cache must match its archived proof")
    with np.load(case_cache, allow_pickle=False) as package:
        ring = package["rgb"]
    if bytes_digest(ring) != proof["ring_rgb_sha256"]:
        raise ValueError("The frozen ring source has changed")

    common = read_json(e02 / "step1/c6bb770094d3/inputs.json")["generation_config"]["common"]
    geometry = root / "outputs/e01b/geometry"
    cap = CapEnergyCurves(geometry, cap_band=int(common.get("cap_band", 16)))
    rect = tuple(cap.topology["faces"][FACE]["rect"])
    candidate = record["candidate"]
    layout = build_capgeom_layout(cap.caps[FACE], cap.ring, cap.cap_band, candidate["cap_scale"])
    ring_img = Image.fromarray(ring)
    context = np.asarray(layout.read_frozen_ring_rgb(ring_img))
    known = layout.known_mask.repeat(8, 0).repeat(8, 1)
    x0, y0, x1, y1 = (8 * x for x in layout.center_rect)
    harmonic, harmonic_proof = harmonic_fill(context, known)
    initial, injection_proof = inject_texture_at_scale(
        harmonic, context, layout, sigma_px_at_scale_1=candidate["texture_sigma_px_at_scale_1"])
    archived_proof = record["details"]["caps"][FACE]["sdedit"]
    fields = ("applied_core_residual_rms", "applied_center_residual_rms", "applied_edge_residual_rms")
    for key in fields:
        np.testing.assert_allclose(injection_proof[key], archived_proof["texture_injection"][key],
                                   rtol=5e-6, atol=1e-8)

    def project(value):
        source = Image.fromarray(np.rint(value * 255).astype(np.uint8))
        payload = {FACE: cap_at_original_scale(layout, source)}
        return compose_capgeom_atlas(cap.topology, cap.labels, cap.owners, cap.ring, ring_img, payload,
                                     return_support=True)

    stages = {}
    with Image.open(raw) as image:
        stages["raw"] = image.convert("RGB").crop(rect)
    with Image.open(preflight) as image:
        stages["preflight"] = image.convert("RGB").crop(rect)
    source_mask, support = project(initial)
    stages["initial"] = source_mask.crop(rect)
    stages["harmonic"] = project(harmonic)[0].crop(rect)
    mask_core_known_px = int(known[y0:y1, x0:x1].sum())
    l, t, r, b = rect
    all_cap_support = bool((support[t:b, l:r] == 6).all())
    if mask_core_known_px or not all_cap_support or not np.array_equal(stages["raw"], stages["preflight"]):
        raise ValueError("Expected one contiguous generated bottom cap unaffected by preflight.")
    context_image = Image.fromarray(np.where(known[..., None], context, 196).astype(np.uint8))
    smaller = context_image.copy()
    smaller.thumbnail((448, 320), resample=Image.Resampling.LANCZOS)
    stages["known_context_overview"] = smaller
    stages["r0_back"] = source_mask.crop(tuple(cap.topology["faces"]["B"]["rect"]))
    stages["known_context_model"] = context_image

    evidence = {"case": CASE, "face": FACE, "cap_rect_native": list(rect),
                "initial_body_float32_sha256": bytes_digest(initial),
                "mask_core_model_known_pixels": mask_core_known_px, "cap_has_single_composition_support": all_cap_support,
                "raw_equals_preflight_entire_cap": True,
                "initial_diagnostic_residuals": {name: injection_proof[name] for name in fields},
                "harmonic_solver": harmonic_proof,
                "candidate": candidate, "source_sha256": {str(path): file_hash(path) for path in (
                    raw, preflight, metadata, case_cache, case_cache.with_suffix(".json"), geometry / "topology.json")},
                "profiles": {}, "lossless_crop_sha256": {}}
    for name, image in stages.items():
        dest = output / f"{name}.png"
        image.save(dest, optimize=True)
        evidence["lossless_crop_sha256"][name] = file_hash(dest)
        if image.size == stages["raw"].size:
            evidence["profiles"][name] = {edge: _profile(image, loc) for edge, loc in EDGE_LOCATORS.items()}
    sheet = Image.new("RGB", (940, 676), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for i, name in enumerate(("harmonic", "initial", "raw", "preflight")):
        col, row = i % 2, i // 2
        x, y = 12 + 464 * col, 30 + 332 * row
        draw.text((x, y - 22), f"{CASE} / {name}", font=font, fill=(30, 42, 50))
        sheet.paste(stages[name], (x, y))
    sheet.save(output / "rectangle-stage-comparison.png", optimize=True)
    sheet.save(output / "rectangle-stage-preview.jpg", quality=91, optimize=True)
    write_json(output / "rectangle-stage-proof.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose_rectangle(args.root, args.output)


if __name__ == "__main__":
    main()
