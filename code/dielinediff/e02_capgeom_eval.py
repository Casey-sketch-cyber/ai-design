"""Describe Task 10 with frozen E02 measurements and Task 08 voting."""

from collections import Counter
import csv
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np

from .e01 import write_json
from .e02 import BENCHMARK_BASELINES, VARIANTS, measure as evaluate
from .e02_capfix_eval import BRIEF_IDS, DIRECTIONS, SELECTION_POLICY as TASK08_POLICY
from .e02_capfix_eval import cap_six, gate, select_addition, select_prompts
from .metrics.physical import PhysicalEvaluator, _stripe_side

TAIL_METRICS = ("cap_6_delta_e00", "stripe_index")
CROSS_SEED_METRICS = (*DIRECTIONS, "bleed_fill_pct", "bleed_fill_v2_pct", "hps_v2", "peak_memory_mib")
QUANTILE = "Hyndman–Fan type 7 (linear; at n=10, x[8] + 0.1 × (x[9]−x[8]) in zero-based ascending order)"
SELECTION_POLICY = {
    "task08": TASK08_POLICY,
    "g2": "Test both strengths against the same incumbent. Compare eligible improvements head to head.",
    "g2_tiebreak": ["cap_6_delta_e00", "cap_6_lpips", "elapsed_seconds", "strength_0.75"],
    "validation": "Seed 1234 is descriptive and excluded from selection and the original five gates.",
}


def compare_step(groups42, incumbent, challenger, gate_config, *, stage):
    """Apply Task 08's unchanged retain-if-net-improvement decision."""
    stage = stage.upper()
    previous = {"G1": "C2", "G2": "C3", "G3": "C4"}
    if stage not in previous:
        raise ValueError("Expected G1, G2 or G3.")
    record = select_addition(groups42, incumbent, challenger, gate_config, stage=previous[stage])
    return {**record, "stage": stage}


def select_strength(groups42, incumbent, low, high, gate_config):
    """Compare two C5 strengths independently on the same retained configuration."""
    comparisons = {name: compare_step(groups42, incumbent, name, gate_config, stage="G2")
                   for name in (low, high)}
    eligible = [name for name in (low, high) if comparisons[name]["selected"] == name]
    head_to_head = None
    if len(eligible) == 2:
        head_to_head = {**select_prompts(groups42, low, high, gate_config), "stage": "G2"}
        selected, reason = head_to_head["selected"], head_to_head["reason"]
    elif eligible:
        selected, reason = eligible[0], "one_strength_improved"
    else:
        selected, reason = incumbent, "no_strength_improved"
    return {"stage": "G2", "reference": incumbent, "selected": selected, "reason": reason,
            "comparisons": comparisons, "head_to_head": head_to_head}


def _cohort(entries, *, seed):
    indexed = {job.atlas.brief["id"]: (job, scores) for job, scores in entries if job.atlas.seed == seed}
    if len(entries) != len(BRIEF_IDS) or indexed.keys() != BRIEF_IDS:
        raise ValueError(f"Expected exactly E01-001 through E01-010 at seed {seed}.")
    return dict(sorted(indexed.items()))


def _score_values(measured, variant):
    scores = measured[variant]
    return {**scores, **cap_six(scores)}


def _value(scores, metric):
    value = scores.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Expected a finite Task 10 score for {metric}.")
    return float(value)


def long_tail(groups, *, seed=42):
    """Report worst two boxes and a declared linear p90 for the two tail endpoints."""
    methods = {}
    for method, entries in groups.items():
        scores = {brief: _score_values(record, "preflight")
                  for brief, (_, record) in _cohort(entries, seed=seed).items()}
        metrics = {}
        for metric in TAIL_METRICS:
            ranked = sorted(((brief, _value(s, metric)) for brief, s in scores.items()),
                            key=lambda pair: (-pair[1], pair[0]))
            values = [pair[1] for pair in ranked]
            metrics[metric] = {
                "mean": mean(values),
                "p90": float(np.quantile(values, 0.9, method="linear")),
                "worst_two": [{"brief_id": brief, "value": value} for brief, value in ranked[:2]],
            }
        methods[method] = metrics
    return {"seed": seed, "n": len(BRIEF_IDS), "variant": "preflight",
            "p90_definition": QUANTILE, "methods": methods}


def _validation_group(groups1234, chosen):
    if set(groups1234) == {chosen}:
        return groups1234[chosen]
    if set(groups1234) == {"seed_1234"}:
        return groups1234["seed_1234"]
    raise ValueError("Seed 1234 must contain only the frozen selected configuration.")


def cross_seed(groups42, groups1234, chosen):
    """Describe matched brief differences for the selected configuration only."""
    first = _cohort(groups42[chosen], seed=42)
    validation = _cohort(_validation_group(groups1234, chosen), seed=1234)
    variants = {}
    for variant in VARIANTS:
        first_values = {brief: _score_values(record, variant) for brief, (_, record) in first.items()}
        second_values = {brief: _score_values(record, variant) for brief, (_, record) in validation.items()}
        result = {}
        for metric in CROSS_SEED_METRICS:
            a = {brief: _value(scores, metric) for brief, scores in first_values.items()}
            b = {brief: _value(scores, metric) for brief, scores in second_values.items()}
            paired = {brief: b[brief] - a[brief] for brief in a}
            result[metric] = {"mean_seed_42": mean(a.values()), "mean_seed_1234": mean(b.values()),
                              "mean_paired_delta_1234_minus_42": mean(paired.values()),
                              "paired_deltas": paired}
        variants[variant] = result
    return {"selected": chosen, "n": len(BRIEF_IDS), "variants": variants,
            "seed_1234_tail": long_tail({chosen: _validation_group(groups1234, chosen)}, seed=1234)}


def _crease_rows(evaluator, faces):
    rows = []
    for seam in evaluator.creases:
        a, b = seam["a"], seam["b"]
        a_ratio, b_ratio = (_stripe_side(faces[side["face"]], side["edge"]) for side in (a, b))
        a_role, b_role = ("cap" if side["face"] in ("T", "Btm") else "ring" for side in (a, b))
        excess = {"ring": 0.0, "cap": 0.0}
        excess[a_role] += (a_ratio - 1) / 2
        excess[b_role] += (b_ratio - 1) / 2
        rows.append({
            "fold": f"{a['face']}-{b['face']}",
            "a_face": a["face"], "a_edge": a["edge"], "a_role": a_role, "a_ratio": a_ratio,
            "b_face": b["face"], "b_edge": b["edge"], "b_role": b_role, "b_ratio": b_ratio,
            "fold_stripe_index": (a_ratio + b_ratio) / 2,
            "ring_excess": excess["ring"], "cap_excess": excess["cap"],
            "ring_only_fold_floor": 1 + excess["ring"], "cap_only_fold_floor": 1 + excess["cap"],
        })
    if len(rows) != 5:
        raise ValueError("The E02 physical protocol requires five native folds.")
    return rows


def _source_summary(cases):
    mean_fields = ("whole_stripe_index", "ring_only_box_floor", "cap_only_box_floor",
                   "ring_excess_at_max", "cap_excess_at_max")
    return {
        **{f"mean_{field}": mean(case[field] for case in cases.values()) for field in mean_fields},
        "mean_decrease_if_cap_ratios_are_one": mean(case["whole_stripe_index"] - case["ring_only_box_floor"]
                                                   for case in cases.values()),
        "mean_decrease_if_ring_ratios_are_one": mean(case["whole_stripe_index"] - case["cap_only_box_floor"]
                                                    for case in cases.values()),
        "controlling_folds": dict(Counter(case["controlling_fold"] for case in cases.values())),
        "controlling_fold_larger_excess": dict(Counter(case["larger_excess"] for case in cases.values())),
        "cases": cases,
    }


def stripe_sources(groups, *, seed=42):
    """Decompose the frozen 3 mm crease index and bound cap-only changes."""
    evaluators = {}
    fold_rows = []
    summary = {}
    for method, entries in groups.items():
        cases = {}
        for brief, (job, measured) in _cohort(entries, seed=seed).items():
            geometry = job.atlas.geometry
            if geometry not in evaluators:
                evaluators[geometry] = PhysicalEvaluator(geometry, device="cpu")
            evaluator = evaluators[geometry]
            faces = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            rows = _crease_rows(evaluator, faces)
            strongest = max(rows, key=lambda row: row["fold_stripe_index"])
            actual = strongest["fold_stripe_index"]
            expected = _value(measured["preflight"], "stripe_index")
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Native stripe output disagrees with the saved E02 score for {method}/{brief}.")
            ring, cap = strongest["ring_excess"], strongest["cap_excess"]
            source = "tie" if math.isclose(ring, cap, abs_tol=1e-10) else ("ring" if ring > cap else "cap")
            cases[brief] = {
                "whole_stripe_index": actual,
                "controlling_fold": strongest["fold"],
                "ring_excess_at_max": ring, "cap_excess_at_max": cap, "larger_excess": source,
                "ring_only_box_floor": max(row["ring_only_fold_floor"] for row in rows),
                "cap_only_box_floor": max(row["cap_only_fold_floor"] for row in rows),
            }
            for row in rows:
                fold_rows.append({"method": method, "seed": seed, "brief_id": brief, **row,
                                  "controls_whole_stripe": row is strongest, **cases[brief]})
        summary[method] = _source_summary(cases)
    return fold_rows, {
        "seed": seed, "n": len(BRIEF_IDS), "methods": summary,
        "definition": ("Per-fold stripe index is 1 + ring excess + cap excess. Each box takes the maximum "
                       "across five folds. The ring-only and cap-only floors independently set the other "
                       "side's ratio to one and then recompute the maximum."),
    }


def banner_summary(entries, sources, review, fingerprint):
    """Summarize native ring annotations without implying causation."""
    if review is None:
        return None
    if not isinstance(review, dict) or review.get("fingerprint") != fingerprint:
        raise ValueError("The banner review must match the current selected atlas fingerprint.")
    samples = review.get("samples")
    if not isinstance(samples, dict) or samples.keys() != BRIEF_IDS or any(
        not isinstance(item, dict) or not isinstance(item.get("stacked_banner_white_line"), bool)
        or not isinstance(item.get("notes"), str) or not item["notes"].strip()
        for item in samples.values()
    ):
        raise ValueError("The banner review needs ten explicit observations for seed 42.")
    scores = _cohort(entries, seed=42)
    groups = {}
    for label, flag in (("present", True), ("absent", False)):
        ids = sorted(brief for brief, item in samples.items() if item["stacked_banner_white_line"] is flag)
        groups[label] = {
            "n": len(ids), "brief_ids": ids,
            "mean_stripe_index": mean(_value(scores[b][1]["preflight"], "stripe_index") for b in ids) if ids else None,
            "mean_ring_only_box_floor": mean(sources["cases"][b]["ring_only_box_floor"] for b in ids) if ids else None,
            "mean_image_reward": mean(_value(scores[b][1]["preflight"], "image_reward") for b in ids) if ids else None,
        }
    difference = {
        key: groups["present"][key] - groups["absent"][key]
        for key in ("mean_stripe_index", "mean_ring_only_box_floor", "mean_image_reward")
        if groups["present"][key] is not None and groups["absent"][key] is not None
    }
    return {"seed": 42, "observational": True, "groups": groups,
            "present_minus_absent": difference, "review": review}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_sources(path, methods, shown, threshold):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(shown), 1, figsize=(10, 3.4 * len(shown)),
                             squeeze=False, sharey=True, layout="constrained")
    for axis, name in zip(axes[:, 0], shown, strict=True):
        cases = list(methods[name]["cases"].values())
        x = np.arange(len(cases))
        ring = np.array([row["ring_excess_at_max"] for row in cases])
        cap = np.array([row["cap_excess_at_max"] for row in cases])
        axis.bar(x, ring, bottom=1, color="#356887", label="Ring excess at maximum")
        axis.bar(x, cap, bottom=1 + ring, color="#d58353", label="Cap excess at maximum")
        axis.scatter(x, [row["ring_only_box_floor"] for row in cases], zorder=3, s=21,
                     facecolors="white", edgecolors="#293b45", label="Ring-only box floor")
        axis.axhline(threshold, color="#454545", linestyle="--", linewidth=1, label="Ten-box mean limit")
        axis.set_ylim(bottom=0.95)
        axis.set_xticks(x, [f"{i:03d}" for i in range(1, len(cases) + 1)])
        axis.set_ylabel("Stripe index")
        axis.set_title(name)
    axes[0, 0].legend(ncol=2, loc="upper left")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_results(experiment, groups42, groups1234, chosen, visual=None, banner_review=None):
    """Write the Task 10 outputs while keeping the five acceptance checks on seed 42."""
    output = Path(experiment.output)
    decision = gate(experiment, groups42, chosen, visual)
    threshold = decision["checks"]["fold_stripe"]["threshold"]
    descriptions = long_tail(groups42)
    groups_v3 = {name: entries for name, entries in groups42.items() if name not in BENCHMARK_BASELINES}
    fold_rows, source42 = stripe_sources(groups_v3)
    summary = {"selection_seed_42_tail": descriptions, "stripe_sources_seed_42": source42}
    for row in source42["methods"].values():
        row["ring_only_floor_exceeds_mean_limit"] = row["mean_ring_only_box_floor"] > threshold
    summary["banner_review"] = banner_summary(groups42[chosen], source42["methods"][chosen],
                                              banner_review, decision["fingerprint"])
    all_trials = [(42, groups42)]
    if groups1234 is not None:
        verification = {chosen: _validation_group(groups1234, chosen)}
        summary["cross_seed_verification"] = cross_seed(groups42, groups1234, chosen)
        validation_folds, validation_source = stripe_sources(verification, seed=1234)
        summary["stripe_sources_seed_1234"] = validation_source
        fold_rows.extend(validation_folds)
        all_trials.append((1234, verification))
    rows = []
    for seed, groups in all_trials:
        for method, cases in groups.items():
            for brief, (job, measured) in _cohort(cases, seed=seed).items():
                for variant in VARIANTS:
                    rows.append({"method": method, "variant": variant, "brief_id": brief, "seed": seed,
                                 "continuous": job.atlas.brief["continuous"], **_score_values(measured, variant)})
    _write_csv(output / "results-capgeom.csv", rows)
    _write_csv(output / "stripe-sources.csv", fold_rows)
    shown = tuple(dict.fromkeys(name for name in ("c3", chosen) if name in groups_v3))
    _plot_sources(output / "stripe-sources.png", source42["methods"], shown, threshold)
    write_json(output / "evaluation-summary.json", summary)
    write_json(output / "gate.json", decision)
    return decision
