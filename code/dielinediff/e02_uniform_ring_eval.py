"""Task 15 texture guards and same-seed evaluation on frozen E02 scores."""

from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np

from .e01 import write_json
from .e02 import BENCHMARK_BASELINES, VARIANTS, gate as e02_gate, measure as evaluate
from .e02_capfix_eval import BRIEF_IDS, runtime_budget
from .e02_capgeom_eval import _cohort, long_tail, stripe_sources
from .e02_captexture_eval import (
    CLIP_MINIMUM,
    ENERGIES,
    INTERIOR_FRACTION,
    PRIMARY_DIRECTIONS,
    SELECTION_FIELDS,
    TEXTURE_FIELDS,
    _means,
    _number,
    _score_rows,
    _texture_fields,
    _values,
    _write_csv,
)
from .metrics.e02_physical import E02PhysicalEvaluator
from .stripe_diag import OFFSET, face_energy


SEEDS = (42, 1234)
G5 = "g5"
REFERENCE = "b2_regional"
RING_FACES = ("F", "L", "B", "R")
CAP_RATIO_LIMIT = 2.0
IMAGE_REWARD_MARGIN = 0.10
RATIOS = ("top_cap_to_ring_texture_ratio", "bottom_cap_to_ring_texture_ratio")
MULTISCALE = ("top_texture_energy_2x", "bottom_texture_energy_2x",
              "top_texture_retention_2x", "bottom_texture_retention_2x")
EXTRA_TEXTURE_FIELDS = (
    *(f"{face}_interior_texture_energy" for face in RING_FACES),
    "ring_interior_texture_energy", *RATIOS, *MULTISCALE,
)
SELECTION_POLICY = {
    "seed": 42, "variant": "preflight", "reference": G5, "guard_reference": REFERENCE,
    "clip_minimum": CLIP_MINIMUM,
    "interior_fraction_vs_original_b2_regional": INTERIOR_FRACTION,
    "maximum_cap_ratio_vs_own_ring_each_box": CAP_RATIO_LIMIT,
    "image_reward_margin_vs_original_b2_regional": IMAGE_REWARD_MARGIN,
    "runtime": "Every case stays within 60 s / 20 GiB.",
    "primary_directions": PRIMARY_DIRECTIONS,
    "ranking": ["more_primary_wins_vs_g5", "lower_frozen_stripe_v1",
                "no_visible_native_artifacts_if_exact_v1_tied", "fewer_native_artifacts",
                "registered_method_order"],
    "gate_seeds": list(SEEDS),
}


def _energy_at_half_size(face):
    """Repeat Task 13's 2×2 box average after the same 9 mm interior crop."""
    interior = face[OFFSET:-OFFSET, OFFSET:-OFFSET].astype(np.float32) / 255
    h, w = (n // 2 for n in interior.shape[:2])
    if min(h, w) < 2:
        raise ValueError("A face is too small for the two-scale interior diagnostic.")
    reduced = interior[:2 * h, :2 * w].reshape(h, 2, w, 2, 3).mean(axis=(1, 3))
    return float((np.square(np.diff(reduced, axis=0)).mean() + np.square(np.diff(reduced, axis=1)).mean()) / 2)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _measured_texture(evaluator, faces):
    values = _texture_fields(evaluator, faces)
    sides = {face: face_energy(faces[face]) for face in RING_FACES}
    ring = mean(sides.values())
    top, bottom = (values[key] for key in ENERGIES)
    coarse_top, coarse_bottom = (_energy_at_half_size(faces[cap]) for cap in ("T", "Btm"))
    return {
        **values, **{f"{face}_interior_texture_energy": v for face, v in sides.items()},
        "ring_interior_texture_energy": ring,
        "top_cap_to_ring_texture_ratio": _ratio(top, ring),
        "bottom_cap_to_ring_texture_ratio": _ratio(bottom, ring),
        "top_texture_energy_2x": coarse_top, "bottom_texture_energy_2x": coarse_bottom,
        "top_texture_retention_2x": _ratio(coarse_top, top),
        "bottom_texture_retention_2x": _ratio(coarse_bottom, bottom),
    }


def textures(groups, *, seed=42):
    """Measure paired face regions, which are untouched by E02 preflight."""
    evaluators, data = {}, {}
    for method, rows in groups.items():
        values = {}
        for brief, (job, _) in _cohort(rows, seed=seed).items():
            path = job.atlas.geometry
            if path not in evaluators:
                evaluators[path] = E02PhysicalEvaluator(path, device="cpu")
            evaluator = evaluators[path]
            faces = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            values[brief] = _measured_texture(evaluator, faces)
        data[method] = values
    return data


def _method_mean(rows, texture, *, seed=42, variant="preflight"):
    values = _values(rows, texture, seed=seed, variant=variant)
    aggregate = _means(values, (*SELECTION_FIELDS, *TEXTURE_FIELDS))
    for field in EXTRA_TEXTURE_FIELDS:
        present = [_number(case[field], field) for case in values.values() if case[field] is not None]
        aggregate[field] = mean(present) if present else None
    return aggregate


def _hard_constraints(rows, means, texture, reference, config, *, seed):
    runtime = runtime_budget(rows, config)
    clip = {"value": means["clip_score"], "minimum": CLIP_MINIMUM,
            "passed": means["clip_score"] >= CLIP_MINIMUM}
    reward_floor = reference["image_reward"] - IMAGE_REWARD_MARGIN
    reward = {"value": means["image_reward"], "minimum": reward_floor,
              "passed": means["image_reward"] >= reward_floor}
    lid_minimum = {
        key: {"value": means[key], "minimum": INTERIOR_FRACTION * reference[key],
              "passed": means[key] >= INTERIOR_FRACTION * reference[key]} for key in ENERGIES
    }
    over_limit = []
    for brief in _cohort(rows, seed=seed):
        entry = texture[brief]
        ring = _number(entry["ring_interior_texture_energy"], "ring_interior_texture_energy")
        maximum = CAP_RATIO_LIMIT * ring
        for name, key in zip(("top", "bottom"), ENERGIES, strict=True):
            cap = _number(entry[key], key)
            if cap > maximum:
                over_limit.append({"brief_id": brief, "cap": name, "value": cap, "maximum": maximum,
                                   "cap_to_ring_ratio": _ratio(cap, ring)})
    lid_maximum = {"ratio_limit": CAP_RATIO_LIMIT, "over_limit": over_limit, "passed": not over_limit}
    return {"hardware": runtime, "clip": clip, "image_reward": reward,
            "minimum_texture": lid_minimum, "maximum_texture": lid_maximum,
            "passed": all((runtime["passed"], clip["passed"], reward["passed"], lid_maximum["passed"],
                           *(value["passed"] for value in lid_minimum.values())))}


def gate_for_seed(experiment, groups, chosen, *, seed, visual=None):
    """Apply the original gate to the original five baselines for this seed."""
    if seed not in SEEDS:
        raise ValueError("Task 15 is registered only at seeds 42 and 1234.")
    for method in (chosen, *BENCHMARK_BASELINES):
        _cohort(groups[method], seed=seed)
    result = e02_gate(experiment, groups, chosen, visual)
    result["checks"]["runtime"].update(runtime_budget(groups[chosen], experiment.config["gate"]))
    result["seed"] = seed
    result["passed"] = all(check["passed"] for check in result["checks"].values())
    return result


def _review_counts(groups, tied, config, reviews):
    if not isinstance(reviews, dict) or not set(tied).issubset(reviews):
        return None
    experiment = SimpleNamespace(config={"gate": config})
    counts = {}
    for method in tied:
        review = reviews[method]
        samples = review.get("samples") if isinstance(review, dict) else None
        if not isinstance(samples, dict) or set(samples) != BRIEF_IDS or any(
            not isinstance(row, dict) or not isinstance(row.get("visible_artifact"), bool)
            or not isinstance(row.get("notes"), str) or not row["notes"].strip()
            for row in samples.values()
        ):
            return None
        verified = gate_for_seed(experiment, groups, method, seed=42, visual=review)
        if verified["visual_review"] is None:
            return None
        counts[method] = sum(row["visible_artifact"] for row in samples.values())
    return counts


def select_candidates(groups42, candidates, gate_config, *, texture=None, visual_reviews=None, incumbent_method=G5, content_guards=None):
    """Select a same-ring candidate using the registered layered cap policy."""
    methods = tuple(dict.fromkeys((incumbent_method, *candidates)))
    if REFERENCE in methods:
        raise ValueError("B2-regional is a guard reference, outside the same-ring candidate set.")
    names = (REFERENCE, *methods)
    if texture is None:
        texture = textures({method: groups42[method] for method in names})
    data = {method: _method_mean(groups42[method], texture[method]) for method in names}
    reference = data[REFERENCE]
    incumbent = data[incumbent_method]
    results = {}
    if content_guards is not None and set(content_guards) != set(methods):
        raise ValueError("A cap content result is required for every registered candidate")
    wins_field, delta_field = f"wins_vs_{incumbent_method}", f"delta_vs_{incumbent_method}"
    for method in methods:
        averages = data[method]
        delta = {key: averages[key] - incumbent[key] for key in PRIMARY_DIRECTIONS}
        wins = [key for key, direction in PRIMARY_DIRECTIONS.items() if direction * delta[key] > 0]
        guards = _hard_constraints(groups42[method], averages, texture[method], reference, gate_config, seed=42)
        if content_guards is not None:
            content = content_guards[method]
            guards["cap_content"] = content
            guards["passed"] = guards["passed"] and content["passed"]
        results[method] = {"means": averages, wins_field: wins, delta_field: delta,
                           "win_count": len(wins), "hard_constraints": guards, "eligible": guards["passed"]}
    eligible = [method for method in methods if results[method]["eligible"]]
    highest = max((results[method]["win_count"] for method in eligible), default=None)
    primary_ties = [method for method in eligible if results[method]["win_count"] == highest]
    lowest = min((data[method]["stripe_index"] for method in primary_ties), default=None)
    stripe_ties = [method for method in primary_ties if data[method]["stripe_index"] == lowest]
    counts = _review_counts(groups42, stripe_ties, gate_config, visual_reviews) if len(stripe_ties) > 1 else None
    pending = len(stripe_ties) > 1 and counts is None
    if not stripe_ties:
        selected, reason = None, "no_eligible_candidates"
    elif pending:
        selected, reason = stripe_ties[0], "awaiting_tie_visual_review"
    elif counts is not None:
        selected = min(stripe_ties, key=lambda method: (counts[method], methods.index(method)))
        reason = "reviewed_exact_stripe_tie"
    elif len(primary_ties) > 1:
        selected, reason = stripe_ties[0], "lowest_frozen_stripe_v1"
    else:
        selected, reason = stripe_ties[0], f"most_primary_wins_vs_{incumbent_method}"
    return {"selected": selected, "reason": reason, "method_order": list(methods),
            "eligible": eligible, "primary_ties": primary_ties, "stripe_ties": stripe_ties,
            "requires_tie_visual_review": pending, "native_artifact_counts": counts,
            "b2_regional_seed_42_means": reference, "methods": results}


def _paired_final(groups, texture, chosen):
    earlier = _values(groups[42][chosen], texture[42][chosen])
    later = _values(groups[1234][chosen], texture[1234][chosen], seed=1234)
    changes = {}
    for key in (*SELECTION_FIELDS, *TEXTURE_FIELDS, *EXTRA_TEXTURE_FIELDS):
        valid = [brief for brief in earlier if earlier[brief][key] is not None and later[brief][key] is not None]
        changes[key] = {
            "measured_pairs": len(valid),
            "mean_seed_42": mean(earlier[b][key] for b in valid) if valid else None,
            "mean_seed_1234": mean(later[b][key] for b in valid) if valid else None,
            "mean_paired_delta_1234_minus_42": mean(later[b][key] - earlier[b][key] for b in valid) if valid else None,
        }
    return {"selected": chosen, "variant": "preflight", "metrics": changes}


def write_results(experiment, groups_by_seed, chosen, visual=None, *, selection=None, texture_by_seed=None, results_file="results-uniform-ring.csv"):
    """Publish seed-local original gates and the supplemental texture diagnostics."""
    if 42 not in groups_by_seed or set(groups_by_seed) - set(SEEDS):
        raise ValueError("Publish seed 42, optionally followed by the seed 1234 validation.")
    if selection is not None and selection["selected"] != chosen:
        raise ValueError("The evaluated method differs from the registered selection.")
    if 1234 in groups_by_seed and selection is not None and selection["requires_tie_visual_review"]:
        raise ValueError("Resolve the exact stripe tie before starting seed 1234.")
    texture_by_seed = dict(texture_by_seed or {})
    for seed, groups in groups_by_seed.items():
        if seed not in texture_by_seed:
            texture_by_seed[seed] = textures(groups, seed=seed)
    reviews = visual or {}
    means, tails, stripes, gates = {}, {}, {}, {}
    metric_rows, fold_rows = [], []
    for seed, groups in sorted(groups_by_seed.items()):
        texture = texture_by_seed[seed]
        seed_key = str(seed)
        means[seed_key] = {
            method: {variant: _method_mean(rows, texture[method], seed=seed, variant=variant) for variant in VARIANTS}
            for method, rows in groups.items()
        }
        tails[seed_key] = long_tail(groups, seed=seed)
        new_folds, sources = stripe_sources(groups, seed=seed)
        stripes[seed_key] = sources
        fold_rows.extend(new_folds)
        metric_rows.extend(_score_rows(groups, texture, seed))
        if chosen is not None:
            gates[seed_key] = gate_for_seed(experiment, groups, chosen, seed=seed,
                                            visual=reviews.get(seed, reviews.get(seed_key)))
    complete = set(gates) == {str(seed) for seed in SEEDS}
    verdict = {"selected": chosen, "seeds": gates, "complete": complete,
               "passed": complete and all(gate["passed"] for gate in gates.values())}
    summary = {"selection": selection, "methods_by_seed": means, "tail_by_seed": tails,
               "stripe_sources_by_seed": stripes}
    if complete:
        summary["cross_seed_verification"] = _paired_final(groups_by_seed, texture_by_seed, chosen)
    output = Path(experiment.output)
    _write_csv(output / results_file, metric_rows)
    _write_csv(output / "stripe-sources.csv", fold_rows)
    write_json(output / "evaluation-summary.json", summary)
    write_json(output / "gate.json", verdict)
    return verdict
