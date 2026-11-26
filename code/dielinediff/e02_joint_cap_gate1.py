"""Task 20 Stage 1 matched two-seed gate on native J4 outputs."""

import csv
from math import isfinite
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

from .e01 import prompt_for, read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_capfix_ring import TASK07_RING, bytes_digest
from .e02_capgeom_eval import _crease_rows
from .e02_uniform_ring_eval import _energy_at_half_size, _ratio
from .metrics import DEFAULT_THRESHOLDS, _aligned_strips, _lpips_pairs
from .metrics.e02_physical import E02PhysicalEvaluator
from .metrics.e02_repetition import DinoRepetition, SIDE_ORDER
from .metrics.learned import LearnedMetrics
from .metrics.physical import LPIPS_MM, PIXELS_PER_MM, SDR_MM
from .methods.v3_geometry import build_ring_layout, compose_atlas
from .stripe_diag import face_energy


SEEDS = (42, 1234)
BRIEFS = tuple(f"E01-{n:03d}" for n in range(1, 11))
RING_NAMES = ("F-L", "F-R", "B-R", "B-L")
RING_NAMES_BY_PAIR = {frozenset(name.split("-")): name for name in RING_NAMES}
CAP_FIELDS = {"F-T": "cap_ft_stripe", "B-Btm": "cap_bbtm_stripe"}
BANNER_CRITERION = "Two or more independent vertically stacked panels separated by a pale or white straight separator."
MARGINS = {"ring_delta_e00_factor": 1.05, "ring_lpips_factor": 1.05,
           "floor_increase": 0.0, "banner_increase": 0, "m4_increase": .02,
           "cap_side_ratio_max": 1.15, "seconds_max": 60, "peak_mib_max": 20 * 1024}


def _cohort(rows, seed):
    subset = [row for row in rows if row["seed"] == seed]
    result = {row["brief_id"]: row for row in subset}
    if len(subset) != len(BRIEFS) or set(result) != set(BRIEFS):
        raise ValueError(f"Gate 1 needs exactly ten matched registered cases at seed {seed}")
    return result


def _bounded(value, limit):
    if isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"Nonfinite Gate 1 value {value}")
    return {"value": value, "limit": limit, "passed": value <= limit + 1e-12}


def decide_gate1(r0_rows, candidate_rows, r0_banner, candidate_banner=None):
    """Apply the preregistered limits independently within each seed."""
    result = {}
    for seed in SEEDS:
        original, candidate = (_cohort(pool, seed) for pool in (r0_rows, candidate_rows))
        first, second = list(original.values()), list(candidate.values())
        fixed = {"ring_delta_e00": MARGINS["ring_delta_e00_factor"],
                 "ring_lpips": MARGINS["ring_lpips_factor"]}
        extra = {"ring_only_box_floor": MARGINS["floor_increase"],
                 "m4_repeat_score": MARGINS["m4_increase"]}
        checks = {}
        for metric, margin in (fixed | extra).items():
            base = mean(row[metric] for row in first)
            limit = base * margin if metric in fixed else base + margin
            checks[metric] = {"r0": base, **_bounded(mean(row[metric] for row in second), limit)}
        per_case = {}
        for brief, row in candidate.items():
            per_case[brief] = {**{field: _bounded(row[field], MARGINS["cap_side_ratio_max"])
                                  for field in CAP_FIELDS.values()},
                               "elapsed_seconds": _bounded(row["elapsed_seconds"], MARGINS["seconds_max"]),
                               "peak_allocated_mib": _bounded(row["peak_allocated_mib"], MARGINS["peak_mib_max"]),
                               "stage2_remaining_seconds": MARGINS["seconds_max"] - row["elapsed_seconds"]}
        for field in (*CAP_FIELDS.values(), "elapsed_seconds", "peak_allocated_mib"):
            values = {brief: data[field]["value"] for brief, data in per_case.items()}
            check = _bounded(max(values.values()), per_case[BRIEFS[0]][field]["limit"])
            checks[field] = {**check, "mean": mean(values.values()),
                             "failed_briefs": [brief for brief, data in per_case.items()
                                               if not data[field]["passed"]]}
        objective = all(check["passed"] for check in checks.values())
        count = None
        if candidate_banner is not None:
            count = sum(candidate_banner[str(seed)][case]["stacked_banner_white_line"] for case in BRIEFS)
        baseline = sum(r0_banner[str(seed)][case]["stacked_banner_white_line"] for case in BRIEFS)
        checks["banner_count"] = {"r0": baseline, "value": count, "limit": baseline,
                                  "passed": count <= baseline if count is not None else None}
        result[str(seed)] = {"n": len(BRIEFS), "checks": checks, "per_case": per_case,
                             "objective_passed": objective,
                             "passed": objective and checks["banner_count"]["passed"] is True}
    return {"seeds": result, "complete": candidate_banner is not None,
            "objective_passed": all(seed["objective_passed"] for seed in result.values()),
            "passed": all(seed["passed"] for seed in result.values())}


def _ring_seams(topology):
    selected = {RING_NAMES_BY_PAIR[pair]: seam for seam in topology["adjacencies"]
                if (pair := frozenset((seam["a"]["face"], seam["b"]["face"]))) in RING_NAMES_BY_PAIR}
    if set(selected) != set(RING_NAMES):
        raise ValueError("The physical box must have all four side-to-side adjacencies")
    return selected


def _measure(evaluator, dino, atlas, brief, *, caps):
    rgb = evaluator._native_atlas(atlas)
    faces = evaluator._physical_faces(rgb)
    labs = {face: rgb2lab(faces[face].astype(np.float32) / 255) for face in SIDE_ORDER}
    seam_map = _ring_seams(evaluator.topology)
    seams = list(seam_map.values())
    widths = round(SDR_MM * PIXELS_PER_MM), round(LPIPS_MM * PIXELS_PER_MM)
    delta = [float(deltaE_ciede2000(*_aligned_strips(labs, seam, widths[0])).mean()) for seam in seams]
    distances = evaluator.learned.seam_lpips(_lpips_pairs(faces, seams, widths[1]), per_seam=True)
    fold_rows = _crease_rows(evaluator, faces)
    folds = {row["fold"]: row for row in fold_rows}
    sides = [atlas.crop(tuple(evaluator.native_topology["faces"][face]["rect"])) for face in SIDE_ORDER]
    prompt = prompt_for(brief)
    _, average_clip = evaluator.learned.clip_scores(sides, [prompt] * len(sides))
    _, front_clip = evaluator.learned.clip_scores([sides[SIDE_ORDER.index('F')]], [prompt])
    m4, = dino.measure_batch([(atlas, evaluator.native_topology)])
    values = {**m4, "ring_delta_e00": mean(delta), "ring_lpips": mean(float(x) for x in distances),
              "ring_only_box_floor": max(fold["ring_only_fold_floor"] for fold in fold_rows),
              "side_clip_score": average_clip, "front_clip_score": front_clip}
    side_fields = (*SIDE_ORDER, "T", "Btm") if caps else SIDE_ORDER
    for face in side_fields:
        source = face_energy(faces[face])
        coarse = _energy_at_half_size(faces[face])
        values[f"{face}_sharpness_energy"] = source
        values[f"{face}_sharpness_energy_2x"] = coarse
        values[f"{face}_sharpness_retention_2x"] = _ratio(coarse, source)
    whole_side = mean(values[f"{face}_sharpness_energy"] for face in SIDE_ORDER)
    whole_side_2x = mean(values[f"{face}_sharpness_energy_2x"] for face in SIDE_ORDER)
    values.update(ring_sharpness_energy=whole_side, ring_sharpness_energy_2x=whole_side_2x,
                  ring_sharpness_retention_2x=_ratio(whole_side_2x, whole_side))
    for fold, name in CAP_FIELDS.items():
        row = folds[fold]
        values[name] = next(row[f"{flank}_ratio"] for flank in ("a", "b") if row[f"{flank}_role"] == "cap") if caps else None
    for name, color, lpips in zip(seam_map, delta, distances, strict=True):
        values[f"{name}_delta_e00"], values[f"{name}_lpips"] = color, float(lpips)
    return values, fold_rows


def _r0_atlas(seed, brief, geometry, labels, bleed_owner, directory):
    cache = Path(directory) / f"{brief}-{seed}.npz"
    proof = read_json(cache.with_suffix(".json"))
    digest = file_hash(cache)
    if (proof["cache_sha256"], proof["brief_id"], proof["seed"]) != (digest, brief, seed):
        raise ValueError(f"R0 cache proof differs for {brief}-{seed}")
    with np.load(cache, allow_pickle=False) as stored:
        canvas = stored["rgb"]
    if bytes_digest(canvas) != proof["ring_rgb_sha256"]:
        raise ValueError(f"R0 RGB proof differs for {brief}-{seed}")
    atlas, support = compose_atlas(geometry, labels, bleed_owner, build_ring_layout(geometry, *TASK07_RING),
                                    Image.fromarray(canvas), {}, return_support=True)
    if bytes_digest(np.asarray(atlas)[np.isin(labels, [1, 2, 3, 4])]) != proof["body_rgb_sha256"]:
        raise ValueError(f"R0 atlas body differs for {brief}-{seed}")
    if bytes_digest(np.asarray(atlas)[np.isin(support, [1, 2, 3, 4])]) != proof["ring_supported_rgb_sha256"]:
        raise ValueError(f"R0 support differs for {brief}-{seed}")
    return atlas, digest


def _stage1_inputs(output, recipe, seed, brief):
    stem = output / "stage1" / recipe / f"seed-{seed}" / f"{brief}-{seed}"
    atlas = stem.with_suffix(".png")
    meta = stem.with_suffix(".run.json")
    canvas = stem.with_suffix(".ring.png")
    record = read_json(meta)
    values = {"raw_sha256": file_hash(atlas), "ring_sha256": file_hash(canvas),
              "run_sha256": file_hash(meta)}
    sources = {"atlas": atlas, "ring": canvas,
               "T": stem.with_suffix(".cap-T.png"), "Btm": stem.with_suffix(".cap-Btm.png")}
    if "latent" in record["source_sha256"]:
        sources["latent"] = stem.with_suffix(".ring.npy")
    hashes = {name: file_hash(path) for name, path in sources.items()}
    if ((record["recipe"], record["brief_id"], record["seed"]) != (recipe, brief, seed)
            or record["source_sha256"] != hashes or record["raw_sha256"] != hashes["atlas"]):
        raise ValueError(f"J4 Stage 1 asset proof differs for {atlas}")
    values["source_sha256"] = stable_hash(hashes)
    return atlas, canvas, record, values


def _banner_sources(output, recipe, folder):
    request = {"recipe": recipe, "criterion": BANNER_CRITERION, "seeds": {}}
    for seed in SEEDS:
        target = folder / f"seed-{seed}"
        target.mkdir(parents=True, exist_ok=True)
        members = {}
        for brief in BRIEFS:
            _, canvas, record, identity = _stage1_inputs(output, recipe, seed, brief)
            proof = record["metadata"]
            with Image.open(canvas) as source:
                if list(source.size) != list(proof["ring_size_px"]):
                    raise ValueError(f"J4 canvas has unexpected size at {canvas}")
                first, last = (int(row) * 8 for row in proof["body_rows_latent"])
                if not 0 <= first < last <= source.height:
                    raise ValueError(f"Corrupt ring body registration at {canvas}")
                destination = target / f"{brief}-{seed}-body.png"
                source.crop((0, first, source.width, last)).save(destination, optimize=True)
            members[brief] = {**identity, "review_view": str(destination.relative_to(folder)),
                              "review_view_sha256": file_hash(destination)}
        request["seeds"][str(seed)] = members
    request["fingerprint"] = stable_hash(request)
    write_json(folder / "banner-request.json", request)
    return request


def _verify_review(review_path, request):
    if review_path is None:
        return None
    review = read_json(review_path)
    if review.get("fingerprint") != request["fingerprint"] or set(review.get("seeds", {})) != set(request["seeds"]):
        raise ValueError("J4 banner review must bind the exact Stage 1 request")
    for seed, entries in request["seeds"].items():
        labels = review["seeds"][seed]
        if set(labels) != set(entries):
            raise ValueError(f"Incomplete Gate 1 banner review at seed {seed}")
        for brief, source in entries.items():
            item = labels[brief]
            if (item["ring_sha256"] != source["ring_sha256"] or not item["notes"].strip()
                    or type(item["stacked_banner_white_line"]) is not bool):
                raise ValueError(f"Unbound Gate 1 banner identification for {brief}-{seed}")
    return review["seeds"]


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score_gate1(experiment, recipe, *, device, banner_review=None):
    """Measure two Stage 1 populations and publish a source-bound stop/continue decision."""
    output = Path(experiment.output)
    folder = output / "gate1" / recipe
    folder.mkdir(parents=True, exist_ok=True)
    request = _banner_sources(output, recipe, folder)
    current_banner = _verify_review(banner_review, request)
    source_banner = Path("docs/experiments/E02/ring-m2m3-banner-review.json")
    r0_banner = read_json(source_banner)["methods"]["r0"]
    learned = LearnedMetrics(device, DEFAULT_THRESHOLDS["ocr_confidence"])
    evaluator = E02PhysicalEvaluator(experiment.geometry, device=device, learned=learned,
                                    config=experiment.metric_config["physical"])
    topology = evaluator.native_topology
    with Image.open(Path(experiment.geometry) / "region_map.png") as image:
        labels = np.array(image)
    with Image.open(Path(experiment.geometry) / "bleed_owner.png") as image:
        bleed_owner = np.array(image)
    dino = None
    case_map = {brief["id"]: brief for brief in experiment.briefs}
    all_rows, fold_rows = [], []
    for method in ("r0", recipe):
        for seed in SEEDS:
            for brief in BRIEFS:
                if method == "r0":
                    atlas, digest = _r0_atlas(seed, brief, topology, labels,
                                             bleed_owner, experiment.r0[seed])
                    note = r0_banner[str(seed)][brief]
                    if note["cache_sha256"] != digest or type(note["stacked_banner_white_line"]) is not bool:
                        raise ValueError(f"R0 human banner evidence has changed for {brief}-{seed}")
                    identity = {"cache_sha256": digest}
                    marker = Path(experiment.r0[seed]) / f"{brief}-{seed}.json"
                    identity["proof_sha256"] = file_hash(marker)
                    runtime = {}
                else:
                    atlas_path, _, record, identity = _stage1_inputs(output, recipe, seed, brief)
                    with Image.open(atlas_path) as image:
                        atlas = image.convert("RGB")
                    runtime = {key: float(record[key]) for key in ("elapsed_seconds", "peak_allocated_mib")}
                base = {"method": method, "seed": seed, "brief_id": brief}
                fingerprint = {**identity, **base, "source_sha256": file_hash(Path(__file__)),
                               "geometry_sha256": file_hash(Path(experiment.geometry) / "topology.json"),
                               "frozen_metrics": stable_hash(experiment.metric_config), "device": str(device)}
                score_dir = output / "gate1" / method / "rows"
                score_dir.mkdir(parents=True, exist_ok=True)
                file = score_dir / f"{brief}-{seed}.json"
                cached = read_json(file) if file.is_file() else {}
                if cached.get("inputs") == fingerprint:
                    row, folds = cached["row"], cached["folds"]
                else:
                    if dino is None:
                        dino = DinoRepetition(experiment.dino_model, experiment.metric_config["m4"], device=device)
                    row, folds = _measure(evaluator, dino, atlas, case_map[brief], caps=method != "r0")
                    write_json(file, {"inputs": fingerprint, "row": row, "folds": folds})
                all_rows.append({**base, **identity, **row, **runtime})
                fold_rows.extend({**base, **fold} for fold in folds)
                print(f"Gate 1 {method} {brief}-{seed}", flush=True)
    original = [row for row in all_rows if row["method"] == "r0"]
    candidate = [row for row in all_rows if row["method"] == recipe]
    result = decide_gate1(original, candidate, r0_banner, current_banner)
    for seed in SEEDS:
        scored = result["seeds"][str(seed)]
        per_method = {"r0": _cohort(original, seed), recipe: _cohort(candidate, seed)}
        diagnostics = {}
        for method, cases in per_method.items():
            faces = (*SIDE_ORDER, "T", "Btm") if method != "r0" else SIDE_ORDER
            keys = ("side_clip_score", "front_clip_score", "ring_sharpness_energy",
                    "ring_sharpness_energy_2x", "ring_sharpness_retention_2x",
                    *(f"{face}_sharpness_{field}" for face in faces
                      for field in ("energy", "energy_2x", "retention_2x")))
            diagnostics[method] = {key: mean(row[key] for row in cases.values() if row[key] is not None)
                                   if any(row[key] is not None for row in cases.values()) else None for key in keys}
        scored["diagnostics"] = diagnostics
        base = diagnostics["r0"]["ring_sharpness_energy"]
        measured = diagnostics[recipe]["ring_sharpness_energy"]
        failed_cap_folds = [fold for fold, field in CAP_FIELDS.items() if not scored["checks"][field]["passed"]]
        scored["softness_diagnostic"] = {
            "r0_ring_absolute_energy": base, "candidate_ring_absolute_energy": measured,
            "relative_energy_vs_r0": _ratio(measured, base),
            "failed_cap_folds": failed_cap_folds,
            "retry_for_softness": measured < base and bool(failed_cap_folds),
        }
    retry = any(seed["softness_diagnostic"]["retry_for_softness"] for seed in result["seeds"].values())
    result.update(recipe=recipe, retry_for_softness=retry, banner_request_sha256=file_hash(folder / "banner-request.json"),
                  r0_banner_sha256=file_hash(source_banner), banner_review_pending=current_banner is None,
                  frozen_metric_source_sha256=file_hash(Path(__file__).with_name("metrics") / "e02_physical.py"))
    _csv(folder / "gate1-scores.csv", all_rows)
    _csv(folder / "gate1-folds.csv", fold_rows)
    write_json(folder / "gate1.json", result)
    return result
