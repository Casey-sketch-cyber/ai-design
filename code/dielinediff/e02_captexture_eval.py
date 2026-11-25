"""Task 13 selection and texture diagnostics on frozen E02 measurements."""

import csv
import math
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from .e01 import write_json
from .e02 import BENCHMARK_BASELINES, VARIANTS, measure as evaluate
from .e02_capfix_eval import BRIEF_IDS, cap_six, gate, runtime_budget
from .e02_capgeom_eval import _cohort, cross_seed, long_tail, stripe_sources
from .metrics import _aligned_strips
from .metrics.e02_physical import E02PhysicalEvaluator
from .metrics.physical import LPIPS_MM, PIXELS_PER_MM
from .stripe_diag import face_energy
from .stripe_diag_lpips import _edge_energy, _splits


CLIP_MINIMUM = 0.6362
INTERIOR_FRACTION = 0.5
PRIMARY_DIRECTIONS = {
    "cap_6_delta_e00": -1, "cap_6_lpips": -1,
    "non_native_7_sdr": -1, "non_native_7_lpips": -1,
    "bleed_fill_pct": 1, "bleed_fill_v2_pct": 1,
}
ENERGIES = ("top_interior_texture_energy", "bottom_interior_texture_energy")
TEXTURE_FIELDS = (*ENERGIES, "top_near_seam_energy_gap", "bottom_near_seam_energy_gap",
                  "cap_6_near_seam_energy_gap", "wrap_lb_near_seam_energy_gap")
SELECTION_FIELDS = (*PRIMARY_DIRECTIONS, "stripe_index", "clip_score", "image_reward",
                    "elapsed_seconds", "peak_memory_mib")
SELECTION_POLICY = {
    "reference": "g1", "variant": "preflight", "seed": 42,
    "clip_minimum": CLIP_MINIMUM, "interior_fraction_vs_b2_regional": INTERIOR_FRACTION,
    "runtime": "Every case stays within unchanged 60 s / 20 GiB limits.",
    "primary_directions": PRIMARY_DIRECTIONS,
    "ranking": ["more_primary_wins_vs_g1", "no_visible_native_artifacts", "lower_frozen_stripe_v1",
                "fewer_native_artifacts", "registered_method_order"],
    "tie_review": "Every method tied on the primary win count needs a bound ten-case native review.",
    "provisional": "Before all tied methods are reviewed, show the lowest frozen stripe v1 without finalizing.",
    "fallback": "Keep archived G1 if every new candidate is ineligible.",
}


def _texture_fields(evaluator, faces):
    width = round(LPIPS_MM * PIXELS_PER_MM)
    gaps = {}
    for group, mask in _splits(evaluator).items():
        distances = []
        for seam, included in zip(evaluator.seams, mask, strict=True):
            if included:
                first, second = _aligned_strips(faces, seam, width)
                distances.append(abs(_edge_energy(first) - _edge_energy(second)))
        gaps[group] = distances
    if tuple(map(len, gaps.values())) != (1, 3, 3):
        raise ValueError("Task 13 requires L–B and six non-native cap seams.")
    return {
        "top_interior_texture_energy": face_energy(faces["T"]),
        "bottom_interior_texture_energy": face_energy(faces["Btm"]),
        "top_near_seam_energy_gap": mean(gaps["top_three"]),
        "bottom_near_seam_energy_gap": mean(gaps["bottom_three"]),
        "cap_6_near_seam_energy_gap": mean(gaps["top_three"] + gaps["bottom_three"]),
        "wrap_lb_near_seam_energy_gap": gaps["wrap_lb"][0],
    }


def textures(groups, *, seed=42):
    """Measure appearance-face support only, shared by raw and preflight."""
    evaluators, methods = {}, {}
    for method, rows in groups.items():
        values = {}
        for brief, (job, _) in _cohort(rows, seed=seed).items():
            geometry = job.atlas.geometry
            if geometry not in evaluators:
                evaluators[geometry] = E02PhysicalEvaluator(geometry, device="cpu")
            evaluator = evaluators[geometry]
            faces = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            values[brief] = _texture_fields(evaluator, faces)
        methods[method] = values
    return methods


def _number(value, metric):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Task 13 needs a finite value for {metric}.")
    return float(value)


def _values(rows, texture, *, seed=42, variant="preflight"):
    current = _cohort(rows, seed=seed)
    if set(current) != set(texture):
        raise ValueError("Task 13 texture and score cohorts disagree.")
    values = {}
    for brief, (_, pair) in current.items():
        scores = pair[variant]
        values[brief] = {**scores, **cap_six(scores), **texture[brief]}
    return values


def _means(values, fields):
    return {metric: mean(_number(scores.get(metric), metric) for scores in values.values()) for metric in fields}


def _hard_constraints(rows, means, energy_minima, gate_config):
    hardware = runtime_budget(rows, gate_config)
    entries = {
        label: {"value": means[metric], "minimum": energy_minima[metric],
                "passed": means[metric] >= energy_minima[metric]}
        for label, metric in zip(("top", "bottom"), ENERGIES, strict=True)
    }
    return {
        "hardware": hardware,
        "clip": {"value": means["clip_score"], "minimum": CLIP_MINIMUM,
                 "passed": means["clip_score"] >= CLIP_MINIMUM},
        "texture": entries,
        "passed": (hardware["passed"] and means["clip_score"] >= CLIP_MINIMUM
                   and all(entry["passed"] for entry in entries.values())),
    }


def _artifact_counts(groups, names, gate_config, reviews):
    if not isinstance(reviews, dict) or not set(names).issubset(reviews):
        return None
    experiment = SimpleNamespace(config={"gate": gate_config})
    counts = {}
    for name in names:
        review = reviews[name]
        samples = review.get("samples") if isinstance(review, dict) else None
        if not isinstance(samples, dict) or set(samples) != BRIEF_IDS or any(
            not isinstance(row, dict) or not isinstance(row.get("visible_artifact"), bool)
            or not isinstance(row.get("notes"), str) or not row["notes"].strip()
            for row in samples.values()
        ):
            return None
        result = gate(experiment, groups, name, review)
        if result["visual_review"] is None:
            return None
        counts[name] = sum(row["visible_artifact"] for row in samples.values())
    return counts


def select_candidates(groups42, candidates, gate_config, *, texture=None, visual_reviews=None):
    """Apply the registered hard filters, then rank how many objectives beat G1."""
    candidates = tuple(candidates)
    if len(set(candidates)) != len(candidates) or "g1" in candidates or "b2_regional" in candidates:
        raise ValueError("The Task 13 candidate list must contain distinct challengers.")
    methods = tuple(dict.fromkeys(("b2_regional", "g1", *candidates)))
    if texture is None:
        texture = textures({name: groups42[name] for name in methods})
    means = {name: _means(_values(groups42[name], texture[name]), (*SELECTION_FIELDS, *TEXTURE_FIELDS))
             for name in methods}
    minima = {name: INTERIOR_FRACTION * means["b2_regional"][name] for name in ENERGIES}
    reference = means["g1"]
    reports = {}
    for name in ("g1", *candidates):
        data = means[name]
        wins = [key for key, direction in PRIMARY_DIRECTIONS.items() if direction * (data[key] - reference[key]) > 0]
        checks = _hard_constraints(groups42[name], data, minima, gate_config)
        reports[name] = {"means": data, "hard_constraints": checks, "eligible": checks["passed"],
                         "wins_vs_g1": wins, "win_count": len(wins),
                         "delta_vs_g1": {key: data[key] - reference[key] for key in PRIMARY_DIRECTIONS}}
    eligible = [name for name in candidates if reports[name]["eligible"]]
    highest = max((reports[name]["win_count"] for name in eligible), default=None)
    primary_ties = [name for name in eligible if reports[name]["win_count"] == highest]
    lowest = min((means[name]["stripe_index"] for name in primary_ties), default=None)
    stripe_ties = [name for name in primary_ties if means[name]["stripe_index"] == lowest]
    counts = _artifact_counts(groups42, primary_ties, gate_config, visual_reviews) if len(primary_ties) > 1 else None
    pending = len(primary_ties) > 1 and counts is None
    if not primary_ties:
        selected, reason = "g1", "fallback_no_new_candidate_passes_hard_constraints"
    elif len(primary_ties) == 1:
        selected, reason = primary_ties[0], "most_primary_wins_vs_g1"
    elif pending:
        selected, reason = stripe_ties[0], "awaiting_tie_visual_review"
    else:
        selected = min(primary_ties, key=lambda name: (
            counts[name] > 0, means[name]["stripe_index"], counts[name], primary_ties.index(name),
        ))
        reason = "reviewed_primary_tie"
    return {"selected": selected, "reason": reason, "selection_seed": 42,
            "method_order": list(candidates), "eligible": eligible, "primary_ties": primary_ties,
            "requires_tie_visual_review": pending, "stripe_ties": stripe_ties,
            "native_artifact_counts": counts,
            "texture_minimum": minima, "b2_regional_means": means["b2_regional"], "methods": reports}


def _validation_groups(groups1234, chosen):
    expected = {"g1", chosen, "seed_1234"}
    if "g1" not in groups1234 or set(groups1234) - expected:
        raise ValueError("Seed 1234 requires the chosen configuration and archived G1.")
    if chosen != "g1" and {"seed_1234", chosen}.issubset(groups1234):
        raise ValueError("Seed 1234 has more than one selected-configuration cohort.")
    selected = "seed_1234" if "seed_1234" in groups1234 else chosen
    if selected not in groups1234:
        raise ValueError("Missing seed 1234 scores for the chosen configuration.")
    return selected, groups1234[selected], groups1234["g1"]


def _paired_differences(reference, candidate, fields, reference_label, candidate_label):
    return {
        metric: {
            reference_label: mean(_number(v[metric], metric) for v in reference.values()),
            candidate_label: mean(_number(v[metric], metric) for v in candidate.values()),
            "mean_paired_difference": mean(_number(candidate[b][metric], metric) - _number(v[metric], metric)
                                           for b, v in reference.items()),
        }
        for metric in fields
    }


def _validation_summary(groups42, groups1234, texture42, texture1234, chosen):
    name, selected, reference = _validation_groups(groups1234, chosen)
    result = cross_seed(groups42, {chosen: selected}, chosen)
    against_g1 = {}
    for variant in VARIANTS:
        first = _values(groups42[chosen], texture42[chosen], variant=variant)
        second = _values(selected, texture1234[name], seed=1234, variant=variant)
        g1 = _values(reference, texture1234["g1"], seed=1234, variant=variant)
        extra = _paired_differences(first, second, TEXTURE_FIELDS, "mean_seed_42", "mean_seed_1234")
        for metric, values in extra.items():
            result["variants"][variant][metric] = {
                "mean_seed_42": values["mean_seed_42"], "mean_seed_1234": values["mean_seed_1234"],
                "mean_paired_delta_1234_minus_42": values["mean_paired_difference"],
                "paired_deltas": {b: second[b][metric] - first[b][metric] for b in first},
            }
        against_g1[variant] = _paired_differences(
            g1, second, (*SELECTION_FIELDS, *TEXTURE_FIELDS),
            "mean_g1_seed_1234", "mean_selected_seed_1234",
        )
    result["versus_g1_seed_1234"] = against_g1
    result["selected_source_group"] = name
    result["reuses_archived_g1"] = name == "g1"
    return result


def _score_rows(groups, all_texture, seed, *, roles=None):
    for method, entries in groups.items():
        for brief, (job, record) in _cohort(entries, seed=seed).items():
            role, label = roles[method] if roles else ("selection", method)
            base = {"method": label, "seed": seed, "brief_id": brief, "evaluation_role": role,
                    "continuous": job.atlas.brief["continuous"]}
            for variant in VARIANTS:
                scores = record[variant]
                yield {**base, "variant": variant, **scores, **cap_six(scores), **all_texture[method][brief]}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_results(experiment, groups42, groups1234, chosen, visual=None, *, texture42=None, selection=None):
    """Keep the unchanged five checks and publish Task 13 measurements and hard filters."""
    if selection is not None and selection["selected"] != chosen:
        raise ValueError("Final Task 13 artifacts must bind to the selected configuration.")
    if groups1234 is not None and selection is not None and selection["requires_tie_visual_review"]:
        raise ValueError("Resolve all primary ties on native folds before validating seed 1234.")
    if texture42 is None:
        texture42 = textures(groups42)
    result = gate(experiment, groups42, chosen, visual)
    summaries = {
        method: {variant: _means(_values(rows, texture42[method], variant=variant),
                                 (*SELECTION_FIELDS, *TEXTURE_FIELDS)) for variant in VARIANTS}
        for method, rows in groups42.items()
    }
    minima = {key: INTERIOR_FRACTION * summaries["b2_regional"]["preflight"][key] for key in ENERGIES}
    result["selection_hard_constraints"] = _hard_constraints(
        groups42[chosen], summaries[chosen]["preflight"], minima, experiment.config["gate"],
    )
    focus42 = {name: rows for name, rows in groups42.items() if name not in BENCHMARK_BASELINES}
    fold_rows, source42 = stripe_sources(focus42)
    summary = {
        "methods_seed_42": summaries,
        "hard_constraints_seed_42": {
            method: _hard_constraints(rows, summaries[method]["preflight"], minima, experiment.config["gate"])
            for method, rows in groups42.items()
        },
        "selection_seed_42_tail": long_tail(groups42),
        "stripe_sources_seed_42": source42, "selection": selection,
    }
    rows = list(_score_rows(groups42, texture42, 42))
    if groups1234 is not None:
        name, selected, _ = _validation_groups(groups1234, chosen)
        texture1234 = textures(groups1234, seed=1234)
        summary["cross_seed_verification"] = _validation_summary(
            groups42, groups1234, texture42, texture1234, chosen,
        )
        caseload = {chosen: selected} if chosen == "g1" else {chosen: selected, "g1": groups1234["g1"]}
        validation_folds, validation_source = stripe_sources(caseload, seed=1234)
        summary["stripe_sources_seed_1234"] = validation_source
        fold_rows.extend(validation_folds)
        roles = {"g1": ("task10_reference", "g1"), name: ("validation", chosen)}
        rows.extend(_score_rows(groups1234, texture1234, 1234, roles=roles))
    output = Path(experiment.output)
    _write_csv(output / "results-captexture.csv", rows)
    _write_csv(output / "stripe-sources.csv", fold_rows)
    write_json(output / "evaluation-summary.json", summary)
    write_json(output / "gate.json", result)
    return result
