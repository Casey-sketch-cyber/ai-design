"""Select a Task 17 ring with frozen E02 scores and descriptive normal profiles."""

import math
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from . import e02_uniform_ring_ring_eval as task15
from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_captexture_eval import _write_csv
from .methods.v3_ring_m3 import extended_ring_layout
from .metrics import _oriented_strip
from .metrics.physical import PIXELS_PER_MM


SEEDS = task15.SEEDS
BRIEFS = task15.BRIEFS
RULES = {
    **task15.RULES,
    "m4_repeat_score": ("m4_repeat_score_increase", 1),
    "side_bleed_fill_pct": ("side_bleed_fill_pct_decrease", -1),
    "side_bleed_fill_v2_pct": ("side_bleed_fill_v2_pct_decrease", -1),
}
PROFILE_NEAR_MM = 3
PROFILE_DEPTH_MM = 9
PROFILE_DESCRIPTION = {
    "scale": "CIELAB L* on the frozen 4 px/mm UV grid",
    "depth": "0–3 mm minus 3–9 mm; pixel centers sampled on each real side inward from the crease",
    "gradient": ("Maximum absolute derivative (L*/mm) of the tangential mean profile. The across-fold "
                 "peak and the boundary jump are given only when both adjacent sides are in the ring."),
    "decision": "Descriptive only; cap pixels are absent from this ring-stage measurement.",
}


def _normal_fields(profile, ppm):
    near = round(PROFILE_NEAR_MM * ppm)
    if near < 1 or len(profile) <= near:
        raise ValueError("The 9 mm physical normal profile does not fit on the face.")
    edge = float(profile[:near].mean())
    interior = float(profile[near:].mean())
    gradient = np.abs(np.diff(profile)) * ppm
    peak = int(gradient.argmax())
    return {
        "mean_Lstar_0_3mm": edge,
        "mean_Lstar_3_9mm": interior,
        "near_minus_reference_Lstar": edge - interior,
        "normal_peak_Lstar_per_mm": float(gradient[peak]),
        "normal_peak_inward_mm": (peak + 1) / ppm,
    }


def fold_normal_profile(evaluator, faces):
    """Measure five folds from the caller's physical RGB faces, including available caps."""
    ring = set(task15.FACES)
    if set(faces) not in (ring, ring | {"T", "Btm"}):
        raise ValueError("Normal profiles need all four ring faces, with or without both caps.")
    labs = {name: rgb2lab(face.astype(np.float32) / 255) for name, face in faces.items()}
    ppm = PIXELS_PER_MM
    depth = round(PROFILE_DEPTH_MM * ppm)
    rows = []
    for seam in evaluator.creases:
        first, second = seam["a"], seam["b"]
        label = f"{first['face']}-{second['face']}"
        profiles = []
        for flank, side, other in (("a", first, second), ("b", second, first)):
            face = side["face"]
            if face not in labs:
                continue
            strip = _oriented_strip(labs[face], side["edge"], depth)
            if strip.shape[1] != depth:
                raise ValueError(f"The physical crease has insufficient inward depth: {face}/{side['edge']}.")
            profile = strip[:, :, 0].mean(axis=0, dtype=np.float64)
            profiles.append(profile)
            rows.append({
                "fold": label, "flank": flank, "face": face, "edge": side["edge"],
                "other_face": other["face"], "pixels_per_mm": ppm,
                **_normal_fields(profile, ppm),
                "across_profile_peak_Lstar_per_mm": None,
                "across_profile_peak_signed_mm": None,
                "across_boundary_jump_Lstar": None,
                "across_boundary_gradient_Lstar_per_mm": None,
            })
        if len(profiles) == 2:
            left, right = profiles
            across = np.abs(np.diff(np.concatenate((left[::-1], right)))) * ppm
            maximum = int(across.argmax())
            cross_fields = {
                "across_profile_peak_Lstar_per_mm": float(across[maximum]),
                "across_profile_peak_signed_mm": (maximum + 1 - depth) / ppm,
                "across_boundary_jump_Lstar": float(right[0] - left[0]),
                "across_boundary_gradient_Lstar_per_mm": float(across[depth - 1]),
            }
            for row in rows[-2:]:
                row.update(cross_fields)
    expected_sides = 8 if set(faces) == ring else 10
    if len(rows) != expected_sides or len({row["fold"] for row in rows}) != 5:
        raise ValueError("The physical side profiles do not match the five original native folds.")
    return rows


def normal_profiles(evaluator, rgb, *, method, seed, brief_id):
    """Describe only the ring sides of the projected atlas."""
    faces = evaluator._physical_faces(rgb)
    native = fold_normal_profile(evaluator, {face: faces[face] for face in task15.FACES})
    base = {"method": method, "seed": seed, "brief_id": brief_id}
    return [{**base, **row} for row in native]


def decide_ring(rows, margins, order):
    """Compare ten paired cases per seed, then rank the eligible two-seed means."""
    if len(order) != 5 or len(set(order)) != 5 or order[0] != "r0":
        raise ValueError("Register R0 followed by the four Task 17 challengers.")
    if set(margins) != {bound for bound, _ in RULES.values()}:
        raise ValueError("The ring quality guards must match all seven registered margins.")
    if any(not math.isfinite(value) or value < 0 for value in margins.values()):
        raise ValueError("Ring guard margins must be finite and nonnegative.")
    table = {}
    for row in rows:
        key = row["method"], row["seed"], row["brief_id"]
        if key in table or any(not math.isfinite(row[field]) for field in task15.FIELDS):
            raise ValueError(f"Invalid or repeated Task 17 ring scores for {key}.")
        table[key] = row
    wanted = {(method, seed, brief) for method in order for seed in SEEDS for brief in BRIEFS}
    if table.keys() != wanted:
        raise ValueError("The ring decision requires five variants by two seeds by ten registered briefs.")
    methods = {}
    for method in order:
        by_seed = {}
        for seed in SEEDS:
            originals = [table["r0", seed, brief] for brief in sorted(BRIEFS)]
            across = [table[method, seed, brief] for brief in sorted(BRIEFS)]
            means = {field: mean(row[field] for row in across) for field in task15.FIELDS}
            deltas = {field: mean(new[field] - old[field] for new, old in zip(across, originals, strict=True))
                      for field in task15.FIELDS}
            floor = {"r0": mean(row["ring_only_box_floor"] for row in originals),
                     "candidate": means["ring_only_box_floor"],
                     "delta_vs_r0": deltas["ring_only_box_floor"],
                     "passed": deltas["ring_only_box_floor"] < 0}
            guards = {}
            for field, (bound, sign) in RULES.items():
                adverse = sign * deltas[field]
                limit = margins[bound]
                guards[field] = {"delta_vs_r0": deltas[field], "adverse_change": adverse,
                                 "adverse_margin": limit,
                                 "passed": adverse <= limit or math.isclose(adverse, limit, rel_tol=0, abs_tol=1e-12)}
            by_seed[str(seed)] = {"n": len(BRIEFS), "means": means, "paired_delta_vs_r0": deltas,
                                  "stripe_floor": floor, "quality_guards": guards,
                                  "passed": floor["passed"] and all(v["passed"] for v in guards.values())}
        methods[method] = {"seeds": by_seed, "eligible": all(row["passed"] for row in by_seed.values()),
                           "equal_seed_mean_ring_only_box_floor": mean(
                               row["means"]["ring_only_box_floor"] for row in by_seed.values()
                           )}
    eligible = sorted((method for method in order[1:] if methods[method]["eligible"]), key=lambda method: (
        methods[method]["equal_seed_mean_ring_only_box_floor"], order.index(method),
    ))
    return {"selected": eligible[0] if eligible else "r0", "eligible": eligible, "method_order": list(order),
            "reason": "lowest_two_seed_floor_among_eligible" if eligible else "no_candidate_passes_both_seeds",
            "methods": methods}


def _registered_inputs(experiment, device):
    order = tuple(experiment.config["ring_candidate_order"])
    if set(order) != set(experiment.ring_paths) or any(set(paths) != set(SEEDS) for paths in experiment.ring_paths.values()):
        raise ValueError("Ring paths must match exactly the registered candidates and two seeds.")
    if len(experiment.briefs) != len(BRIEFS) or {brief["id"] for brief in experiment.briefs} != BRIEFS:
        raise ValueError("Only E01-001 through E01-010 are registered here.")
    geometry = experiment.geometry
    shared = {
        "device": str(device), "algorithm_sha256": file_hash(Path(__file__)),
        "task15_measurer_sha256": file_hash(Path(task15.__file__)),
        "geometry_sha256": {name: file_hash(geometry / name)
                            for name in ("topology.json", "region_map.png", "bleed_owner.png")},
        "metric_config": experiment.metric_config,
    }
    cases = []
    for seed in SEEDS:
        for brief in experiment.briefs:
            identifier = f"{brief['id']}-{seed}"
            r0_path = experiment.ring_paths["r0"][seed] / f"{identifier}.npz"
            r0_digest = file_hash(r0_path)
            r0_atlas = (experiment.original / "v3_1280" if seed == 42 else experiment.previous / "seed_1234"
                        ) / f"{identifier}.png"
            for method in order:
                path = experiment.ring_paths[method][seed] / f"{identifier}.npz"
                sidecar = path.with_suffix(".json")
                proof = read_json(sidecar)
                signature = r0_digest if method == "r0" else file_hash(path)
                if ((proof["brief_id"], proof["seed"], proof["cache_sha256"]) != (brief["id"], seed, signature)
                        or (method != "r0" and (proof["method"], proof["r0_cache_sha256"]) != (method, r0_digest))):
                    raise ValueError(f"Task 17 ring proof does not match its registered case: {method}/{identifier}.")
                size = tuple(proof.get("ring_size_px", task15.TASK07_RING)) if method == "r0" else tuple(proof["ring_size_px"])
                if size not in ((1280, 1024), (1280, 1152)):
                    raise ValueError(f"Unexpected ring dimensions for {method}/{identifier}: {size}.")
                identity = {**shared, "method": method, "seed": seed, "brief_id": brief["id"],
                            "ring_size_px": list(size), "cache_sha256": signature, "record_sha256": file_hash(sidecar)}
                if method == "r0":
                    identity["archive_sha256"] = file_hash(r0_atlas)
                    identity["archive_record_sha256"] = file_hash(r0_atlas.with_suffix(".run.json"))
                cases.append(((method, seed, brief, path, proof, identity["record_sha256"]), size, identity))
    return cases, order


def _models(experiment, device):
    learned = task15.LearnedMetrics(device, task15.DEFAULT_THRESHOLDS["ocr_confidence"])
    evaluator = task15.E02PhysicalEvaluator(experiment.geometry, device=device, learned=learned,
                                           config=experiment.metric_config["physical"])
    reward = task15.ImageRewardScorer(device=device)
    dino = task15.DinoRepetition(experiment.dino_model, experiment.metric_config["m4"], device=device)
    return evaluator, reward, dino


def _cached_score(experiment, case, size, identity, images, layouts, models):
    method, seed, brief, path, proof, _ = case
    checkpoint = experiment.output / "ring-evaluation" / method / f"{brief['id']}-{seed}.json"
    if checkpoint.is_file():
        cached = read_json(checkpoint)
        if cached["input"] == identity:
            return cached
    if not models:
        models.extend(_models(experiment, identity["device"]))
    evaluator, reward, dino = models
    topology, labels, bleed_owner, body = images
    if size not in layouts:
        layouts[size] = extended_ring_layout(topology, size)
    layout = layouts[size]
    score, folds = task15._measure(case, experiment, evaluator, reward, dino, topology,
                                  labels, bleed_owner, layout, body)
    _, rgb, _ = task15._projection(path, proof, topology, labels, bleed_owner, layout, body)
    profiles = normal_profiles(evaluator, rgb, method=method, seed=seed, brief_id=brief["id"])
    score.update(ring_width_px=size[0], ring_height_px=size[1])
    result = {"input": identity, "score": score, "folds": folds, "normal_profiles": profiles}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint, result)
    print(f"Task 17 ring scored {method} {brief['id']}-{seed}", flush=True)
    return result


def measure_rings(experiment, device):
    """Verify caches, resume per case, export descriptions, then select a ring."""
    cases, order = _registered_inputs(experiment, device)
    topology = read_json(experiment.geometry / "topology.json")
    with Image.open(experiment.geometry / "region_map.png") as image:
        labels = np.asarray(image)
    with Image.open(experiment.geometry / "bleed_owner.png") as image:
        owner = np.asarray(image)
    body = np.isin(labels, task15.SIDE_LABELS)
    layouts, models = {}, []
    scores, folds, profiles, evidence = [], [], [], {}
    for case, size, identity in cases:
        record = _cached_score(experiment, case, size, identity, (topology, labels, owner, body), layouts, models)
        scores.append(record["score"])
        folds.extend(record["folds"])
        profiles.extend(record["normal_profiles"])
        method, seed, brief, _, _, _ = case
        evidence[f"{method}/{brief['id']}-{seed}"] = identity
    decision = decide_ring(scores, experiment.config["ring_quality_guard"], order)
    root = experiment.output
    _write_csv(root / "ring-scores.csv", scores)
    _write_csv(root / "ring-folds.csv", folds)
    _write_csv(root / "ring-normal-profile.csv", profiles)
    write_json(root / "ring-scores.json", {"inputs": evidence, "scores": scores, "folds": folds,
                                           "normal_profiles": profiles, "normal_profile_definition": PROFILE_DESCRIPTION})
    write_json(root / "ring-selection.json", decision)
    return decision
