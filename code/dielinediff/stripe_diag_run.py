"""Run Task 12 against frozen archives. All measurements execute on CPU."""

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from .e01 import write_json
from .e01b import file_hash
from .e02_capgeom_eval import _write_csv
from .e02_freeze import verify
from .metrics.physical import PhysicalEvaluator
from .stripe_diag import CALIBRATION_PERCENTILE, METRICS, _ratio, collect_terms, compare_references, measure_atlas
from .stripe_diag_sources import TRIAL_BRIEFS, registered_atlases, registered_synthetic_source


REVIEW_METHODS = ("g1", "g2_075", "g2_085")
MATERIALS = 32
STRIPE_REFERENCE = Path("docs/experiments/E02/gate-capgeom.json")


def run_corpus(jobs, output):
    evaluators = {}
    archives = []
    texture_sources = []
    calibration = []
    for job in jobs:
        if job.geometry not in evaluators:
            evaluators[job.geometry] = PhysicalEvaluator(job.geometry, device="cpu")
        evaluator = evaluators[job.geometry]
        if file_hash(job.path) != job.raw_sha256:
            raise ValueError(f"Archived source image has changed: {job.path}")
        with Image.open(job.path) as encoded:
            faces = evaluator._physical_faces(evaluator._native_atlas(encoded))
        interiors, rows = collect_terms(faces, evaluator.creases)
        archives.append((job, interiors, rows))
        if registered_synthetic_source(job):
            texture_sources.append((f"{job.method}/{job.identifier}", faces))
        if job.cohort == "history":
            calibration.extend(row[f"{side}_reference_energy_v1"] for row in rows for side in ("a", "b"))
        if len(archives) % 100 == 0:
            print(f"Read {len(archives)}/{len(jobs)} archived atlases", flush=True)
    if len(texture_sources) != MATERIALS or not calibration:
        raise ValueError("The diagnostic does not have the registered physical calibration and texture sources.")
    floor = float(np.quantile(calibration, CALIBRATION_PERCENTILE, method="linear"))
    box_rows, fold_rows = evaluate_archives(archives, floor)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "stripe-folds.csv", fold_rows)
    _write_csv(output / "real-atlases.csv", box_rows)
    details = {
        "historical_atlases": len(calibration) // 10, "historical_side_references": len(calibration),
        "percentile": CALIBRATION_PERCENTILE, "reference_floor": floor,
        "references_below_floor": sum(value < floor for value in calibration),
        "constant_v1_regularizer_kept": True,
    }
    return box_rows, fold_rows, floor, details, texture_sources, evaluator.creases


def evaluate_archives(archives, floor):
    box_rows, fold_rows = [], []
    for job, interiors, terms in archives:
        registered = {"cohort": job.cohort, "method": job.method, "brief_id": job.brief_id, "seed": job.seed,
                      "atlas_sha256": job.raw_sha256}
        rows = compare_references(interiors, terms, floor)
        peaks = {metric: next(row for row in rows if row[f"controls_{metric}"]) for metric in METRICS}
        if job.expected_v1 is not None and not math.isclose(
            peaks["v1"]["v1"], job.expected_v1, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"Recomputed frozen v1 differs from the archive for {job.method}/{job.identifier}.")
        box_rows.append({
            "record_type": "real", **registered,
            **{metric: row[metric] for metric, row in peaks.items()},
            **{f"{metric}_peak_fold": row["fold"] for metric, row in peaks.items()},
            **{k: v for k, v in peaks["v1"].items() if k.startswith(("a_", "b_"))},
            **{f"interior_energy_{name}": value for name, value in interiors.items()},
        })
        fold_rows.extend({**registered, **row} for row in rows)
    return box_rows, fold_rows


def cohort_results(rows):
    groups = defaultdict(list)
    for row in rows:
        seed = "all" if row["cohort"] == "history" else str(row["seed"])
        key = row["cohort"], row["method"], seed
        groups[key].append(row)
    summary = []
    for (cohort, method, seed), data in sorted(groups.items()):
        summary.append({"cohort": cohort, "method": method, "seed": seed, "n": len(data),
                        **{metric: mean(row[metric] for row in data) for metric in METRICS}})
    for metric in METRICS:
        positions = {key: i + 1 for i, (key, _) in enumerate(sorted(
            enumerate(summary), key=lambda element: element[1][metric]
        ))}
        for index, row in enumerate(summary):
            row[f"{metric}_rank"] = positions[index]
    correlations = {metric: float(spearmanr([row["v1"] for row in summary],
                                            [row[metric] for row in summary]).statistic)
                    for metric in METRICS[1:]}
    matched = defaultdict(list)
    for row in rows:
        if (row["seed"] == 42 and row["brief_id"] in TRIAL_BRIEFS
                and row["cohort"] != "failure_control" and row["method"] != "v2"):
            matched[row["method"]].append(row)
    control = matched["b2_regional"]
    paired = []
    for method, data in sorted(matched.items()):
        if len(data) != 10:
            raise ValueError(f"Expected a complete paired 10-brief trial for {method}.")
        by_brief = {row["brief_id"]: row for row in data}
        paired.append({"method": method, "n": len(data), **{
            metric: mean(by_brief[row["brief_id"]][metric] - row[metric] for row in control)
            for metric in METRICS
        }})
    return summary, correlations, paired


def reference_effects(rows):
    by_key = {(r["method"], r["brief_id"], r["fold"]): r for r in rows
              if r["cohort"] == "task10" and r["seed"] == 42 and r["method"] in REVIEW_METHODS}
    summary = {}
    for fold, side in (("F-T", "b"), ("B-Btm", "b")):
        brief_ids = sorted({brief for method, brief, fold_name in by_key if fold_name == fold})
        by_method = {method: [by_key[method, brief, fold] for brief in brief_ids] for method in REVIEW_METHODS}
        for rows_for_method in by_method.values():
            if len(rows_for_method) != 10 or any(r[f"{side}_face"] not in ("T", "Btm") for r in rows_for_method):
                raise ValueError(f"The diagnostic expected the cap side of {fold}.")
        names = ("crease_energy", "reference_energy_v1", "v1")
        levels = {method: {name: mean(r[f"{side}_{name}"] for r in values) for name in names}
                  for method, values in by_method.items()}
        changes = []
        area, ref = f"{side}_crease_energy", f"{side}_reference_energy_v1"
        for low, high in zip(by_method["g2_075"], by_method["g2_085"], strict=True):
            e0, e1, r0, r1 = low[area], high[area], low[ref], high[ref]
            original, updated = _ratio(e0, r0), _ratio(e1, r1)
            edge_only, ref_only = _ratio(e1, r0), _ratio(e0, r1)
            changes.append(((edge_only - original + updated - ref_only) / 2,
                            (ref_only - original + updated - edge_only) / 2))
        summary[fold] = {"levels": levels, "edge_energy_effect": mean(a for a, _ in changes),
                         "reference_energy_effect": mean(b for _, b in changes)}
    return summary


def main():
    from .e01 import read_json
    from .stripe_diag_lpips import run_lpips
    from .stripe_diag_synthetic import plot_synthetic, run_synthetic
    from .stripe_diag_visual import render_history_controls, render_peaks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    args = parser.parse_args()
    if args.root is None:
        parser.error("The AutoDL data root is required.")
    root = args.root.resolve()
    output = root / "outputs/e02/stripe-diagnostic"
    _, _, freeze = verify(root)
    jobs = list(registered_atlases(root, freeze))
    boxes, folds, floor, calibration, textures, creases = run_corpus(jobs, output)
    original_bound = read_json(STRIPE_REFERENCE)["checks"]["fold_stripe"]["threshold"]
    synthetic = run_synthetic(textures, creases, measure_atlas, floor, output, threshold=original_bound)
    plot_synthetic(output / "synthetic-samples.csv", output / "synthetic-boxplot.png", original_bound)
    run_lpips(jobs, output)
    review_jobs = [job for job in jobs if job.method in REVIEW_METHODS and job.seed == 42]
    review_folds = [row for row in folds if row["method"] in REVIEW_METHODS and row["seed"] == 42]
    images = render_peaks(review_folds, review_jobs, output / "peak-crops")
    controls = render_history_controls(jobs, output / "historical-crops")
    print(f"Rendered {len(images)} peak comparisons and {len(controls)} historical control plates", flush=True)
    methods, ranks, paired = cohort_results(boxes)
    with (output / "synthetic-samples.csv").open(newline="", encoding="utf-8") as stream:
        synthetic_rows = [
            {"record_type": "synthetic", **{metric: row[f"{metric}_box"] for metric in METRICS}, **row}
            for row in csv.DictReader(stream)
        ]
    _write_csv(output / "stripe-diag.csv", [*boxes, *synthetic_rows])
    write_json(output / "stripe-diagnostic.json", {
        "metrics_commit": freeze["metrics_commit"], "parameter_sha256": freeze["parameter_sha256"],
        "n_real": len(boxes), "n_synthetic": len(synthetic_rows), "reference_v1_numeric_only": original_bound,
        "calibration": calibration, "method_means": methods, "method_rank_spearman": ranks,
        "paired_seed_42_deltas_to_b2_regional": paired, "reference_effects": reference_effects(folds),
        "synthetic": synthetic, "lpips_output": "lpips-diagnostic.json",
        "archive_failures_bb": [row for row in folds if row["fold"] == "B-Btm" and (
            row["cohort"] == "failure_control" or row["method"] == "v2"
        )],
    })
    print(f"{len(boxes)} archived atlases, {len(synthetic_rows)} synthetic controls. Saved {output}", flush=True)


if __name__ == "__main__":
    main()
