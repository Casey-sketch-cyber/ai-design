"""Evaluate paired Task 15 rings before committing to any cap candidate."""

from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_capfix_ring import SIDE_LABELS, TASK07_RING, archived_rgb, bytes_digest
from .e02_capgeom_eval import _crease_rows
from .e02_captexture_eval import _write_csv
from .e02_uniform_ring_baseline import regional_negative
from .methods.sdxl import prompt_for
from .methods.v3_geometry import build_ring_layout, compose_atlas
from .metrics import DEFAULT_THRESHOLDS, _aligned_strips, _lpips_pairs
from .metrics.e02_physical import E02PhysicalEvaluator
from .metrics.e02_repetition import DinoRepetition
from .metrics.image_reward import ImageRewardScorer
from .metrics.learned import LearnedMetrics
from .metrics.physical import LPIPS_MM, PIXELS_PER_MM, SDR_MM


FACES = ("F", "B", "L", "R")
SEEDS = (42, 1234)
BRIEFS = {f"E01-{n:03d}" for n in range(1, 11)}
RULES = {
    "wrap_lb_delta_e00": ("wrap_lb_delta_e00_increase", 1),
    "wrap_lb_lpips": ("wrap_lb_lpips_increase", 1),
    "clip_score": ("clip_score_decrease", -1),
    "image_reward": ("image_reward_decrease", -1),
}
FIELDS = ("ring_only_box_floor", *RULES, "side_bleed_fill_pct", "side_bleed_fill_v2_pct",
          "m4_repeat_score", "ring_elapsed_seconds", "ring_peak_memory_mib")


def _score_plan(experiment, device):
    if set(experiment.r0) != set(SEEDS) or {row["id"] for row in experiment.briefs} != BRIEFS:
        raise ValueError("Task 15 requires both seeds on the ten registered briefs.")
    cases = []
    for seed in SEEDS:
        for brief in experiment.briefs:
            identifier = f"{brief['id']}-{seed}"
            original_cache = experiment.r0[seed] / f"{identifier}.npz"
            original_proof = read_json(original_cache.with_suffix(".json"))
            for method, directory in (("r0", experiment.r0[seed]), ("r1", experiment.r1)):
                cache = directory / f"{identifier}.npz"
                metadata = cache.with_suffix(".json")
                proof = read_json(metadata)
                signature = file_hash(cache)
                if signature != proof["cache_sha256"] or (proof["brief_id"], proof["seed"]) != (brief["id"], seed):
                    raise ValueError(f"Ring cache identity failed for {method}/{identifier}.")
                if method == "r1" and (
                    proof["r0_cache_sha256"] != original_proof["cache_sha256"]
                    or proof["rng_state_sha256"] != original_proof["rng_state_sha256"]
                    or proof["prompt"]["negative_prompt"] != regional_negative(experiment.negative_suffix)
                ):
                    raise ValueError(f"R1 ring does not pair with the registered R0 for {identifier}.")
                cases.append((method, seed, brief, cache, proof, file_hash(metadata)))
    identity = {
        "device": str(device),
        "measurer_sha256": file_hash(Path(__file__)),
        "metric_config": experiment.metric_config,
        "geometry_sha256": {name: file_hash(experiment.geometry / name)
                            for name in ("topology.json", "region_map.png", "bleed_owner.png")},
        "ring_sources": {f"{method}/{brief['id']}-{seed}": {
            "cache_sha256": proof["cache_sha256"], "record_sha256": metadata
        } for method, seed, brief, _, proof, metadata in cases},
    }
    return cases, identity


def _projection(cache, proof, topology, labels, bleed_owner, layout, body):
    with np.load(cache, allow_pickle=False) as data:
        latent, pixels, state = data["latent"], data["rgb"], data["rng_state"]
    values = {"latent_sha256": bytes_digest(latent), "ring_rgb_sha256": bytes_digest(pixels),
              "rng_state_sha256": bytes_digest(state)}
    atlas, support = compose_atlas(topology, labels, bleed_owner, layout, Image.fromarray(pixels), {},
                                   return_support=True)
    rgb = np.asarray(atlas)
    supported = np.isin(support, SIDE_LABELS)
    values.update(body_rgb_sha256=bytes_digest(rgb[body]), ring_supported_rgb_sha256=bytes_digest(rgb[supported]),
                  side_pixels=int(body.sum()), ring_supported_pixels=int(supported.sum()))
    if not np.all(supported[body]) or any(proof[key] != value for key, value in values.items()):
        raise ValueError(f"Ring cache contents do not match the registered projection: {cache}.")
    return atlas, rgb, supported


def _side_bleed(evaluator, rgb, faces, labs):
    eligible = np.logical_or.reduce([evaluator.bleed_refs[name][0] for name in FACES])
    if not eligible.any():
        raise ValueError("The registered side-only bleed denominator is empty.")
    sampled, valid, delta = evaluator._sample_bleed_lab(rgb, labs)
    limits = DEFAULT_THRESHOLDS
    nonwhite = (sampled < limits["bleed_white_rgb"]).any(axis=1)
    close = delta <= limits["bleed_match_delta_e"]
    changed = delta >= limits["bleed_change_delta_e"]
    base = valid & nonwhite & close
    old = base & changed
    new = base & (evaluator._face_is_flat(faces)
                  | (changed & evaluator._bleed_normal_has_structure(sampled, valid)))
    return {
        "side_bleed_fill_pct": float(100 * old[eligible].mean()),
        "side_bleed_fill_v2_pct": float(100 * new[eligible].mean()),
        "side_bleed_samples": int(eligible.sum()),
    }


def _native_sides(atlas, topology):
    images = []
    for name in FACES:
        spec = topology["faces"][name]
        native = np.asarray(atlas.crop(tuple(spec["rect"])))
        images.append(Image.fromarray(np.ascontiguousarray(np.rot90(native, spec["upright_rotation"] // 90))))
    return images


def _original_rgb(experiment, seed, brief):
    identifier = f"{brief['id']}-{seed}.png"
    if seed == 42:
        return archived_rgb(experiment.original / "v3_1280" / identifier)
    archive = experiment.previous / "seed_1234" / identifier
    metadata = read_json(archive.with_suffix(".run.json"))
    if ((metadata["method"], metadata["seed"], metadata["brief_id"], metadata["validation_of"])
            != ("seed_1234", 1234, brief["id"], "g5") or metadata["raw_sha256"] != file_hash(archive)):
        raise ValueError(f"Invalid Task 13 G5 validation archive for {identifier}.")
    with Image.open(archive) as image:
        return np.asarray(image.convert("RGB"))


def _measure(case, experiment, evaluator, reward, dino, topology, labels, bleed_owner, layout, body):
    method, seed, brief, path, proof, _ = case
    atlas, rgb, support = _projection(path, proof, topology, labels, bleed_owner, layout, body)
    if method == "r0":
        original = _original_rgb(experiment, seed, brief)
        if not np.array_equal(rgb[support], original[support]):
            raise ValueError(f"R0 side artwork differs from the reviewed archive: {brief['id']}-{seed}.")
    faces = evaluator._physical_faces(rgb)
    labs = {name: rgb2lab(face.astype(np.float32) / 255) for name, face in faces.items()}
    seams = [seam for seam, selected in zip(evaluator.seams, evaluator.groups["wrap_lb"], strict=True) if selected]
    if len(seams) != 1:
        raise ValueError("The frozen topology must contain the L–B seam exactly once.")
    a, b = _aligned_strips(labs, seams[0], round(SDR_MM * PIXELS_PER_MM))
    lpips = evaluator.learned.seam_lpips(_lpips_pairs(faces, seams, round(LPIPS_MM * PIXELS_PER_MM)))
    side_views = _native_sides(atlas, topology)
    prompts = [prompt_for(brief)] * len(side_views)
    _, clip = evaluator.learned.clip_scores(side_views, prompts)
    repeats, = dino.measure_batch([(atlas, topology)])
    folds = _crease_rows(evaluator, faces)
    base = {"method": method, "seed": seed, "brief_id": brief["id"]}
    fields = {
        **base,
        "ring_only_box_floor": max(row["ring_only_fold_floor"] for row in folds),
        "wrap_lb_delta_e00": float(deltaE_ciede2000(a, b).mean()),
        "wrap_lb_lpips": lpips, "clip_score": clip, "image_reward": reward.score(side_views, prompts),
        **_side_bleed(evaluator, rgb, faces, labs), **repeats,
        **{key: proof[key] for key in ("ring_elapsed_seconds", "ring_peak_memory_mib",
                                      "cache_sha256", "body_rgb_sha256", "ring_supported_rgb_sha256")},
    }
    return fields, [{**base, **fold} for fold in folds]


def decide_ring(rows, margins):
    """R1 must lower both stripe floors without passing any adverse-quality margin."""
    seeds = {}
    for seed in SEEDS:
        grouped = {}
        for method in ("r0", "r1"):
            values = [row for row in rows if row["seed"] == seed and row["method"] == method]
            grouped[method] = {row["brief_id"]: row for row in values}
            if len(values) != len(BRIEFS) or set(grouped[method]) != BRIEFS:
                raise ValueError(f"Task 15 ring decision requires all ten {method} cases at seed {seed}.")
        means = {method: {key: mean(row[key] for row in values.values()) for key in FIELDS}
                 for method, values in grouped.items()}
        changes = {key: mean(grouped["r1"][brief][key] - row[key]
                             for brief, row in grouped["r0"].items()) for key in FIELDS}
        floor = {"r0": means["r0"]["ring_only_box_floor"], "r1": means["r1"]["ring_only_box_floor"],
                 "r1_minus_r0": changes["ring_only_box_floor"], "passed": changes["ring_only_box_floor"] < 0}
        guards = {key: {"r1_minus_r0": changes[key], "adverse_margin": margins[bound],
                        "passed": sign * changes[key] <= margins[bound]}
                  for key, (bound, sign) in RULES.items()}
        seeds[str(seed)] = {"n": len(BRIEFS), "means": means, "paired_delta_r1_minus_r0": changes,
                            "stripe_floor": floor, "quality_guards": guards,
                            "passed": floor["passed"] and all(row["passed"] for row in guards.values())}
    keep = all(row["passed"] for row in seeds.values())
    return {"selected": "r1" if keep else "r0", "seeds": seeds,
            "reason": "both_seeds_pass" if keep else "one_or_more_ring_checks_failed"}


def measure_rings(experiment, device):
    cases, input_signature = _score_plan(experiment, device)
    output = experiment.output
    report_path = output / "ring-scores.json"
    saved = read_json(report_path) if report_path.is_file() else {}
    if saved.get("inputs") == input_signature:
        rows, folds = saved["scores"], saved["folds"]
    else:
        topology = read_json(experiment.geometry / "topology.json")
        with Image.open(experiment.geometry / "region_map.png") as image:
            labels = np.asarray(image)
        with Image.open(experiment.geometry / "bleed_owner.png") as image:
            bleed_owner = np.asarray(image)
        body = np.isin(labels, SIDE_LABELS)
        layout = build_ring_layout(topology, *TASK07_RING)
        learned = LearnedMetrics(device, DEFAULT_THRESHOLDS["ocr_confidence"])
        evaluator = E02PhysicalEvaluator(experiment.geometry, device=device, learned=learned,
                                          config=experiment.metric_config["physical"])
        reward = ImageRewardScorer(device=device)
        dino = DinoRepetition(experiment.dino_model, experiment.metric_config["m4"], device=device)
        rows, folds = [], []
        for case in cases:
            row, fields = _measure(case, experiment, evaluator, reward, dino, topology, labels, bleed_owner,
                                   layout, body)
            rows.append(row)
            folds.extend(fields)
            print(f"Ring scores {row['method']} {row['brief_id']}-{row['seed']}", flush=True)
    result = decide_ring(rows, experiment.config["ring_quality_guard"])
    _write_csv(output / "ring-scores.csv", rows)
    _write_csv(output / "ring-folds.csv", folds)
    write_json(report_path, {"inputs": input_signature, "scores": rows, "folds": folds})
    write_json(output / "ring-selection.json", result)
    return result
