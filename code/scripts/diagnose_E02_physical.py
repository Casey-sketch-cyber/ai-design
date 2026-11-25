#!/usr/bin/env python3
"""Summarize E02 D1 and D2 from paired historical physical score CSVs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from dielinediff.e01 import read_json, write_json
from dielinediff.e02_history import read_briefs
from dielinediff.e02_sources import REFERENCE_METHODS

D1_FIELDS = (
    "inside_dE00", "sdr_denominator", "seam_sdr", "seam_lpips",
    *(f"{group}_{metric}" for group in ("native_5", "non_native_7", "wrap_lb")
      for metric in ("delta_e00", "sdr", "lpips")),
    "cross_seam_edge_cont", "m2_salient_seam_count",
)
D2_FIELDS = ("d2_white_halo_pct", "d2_flat_bg_trap_pct", "d2_color_jump_pct", "d2_invalid_pct")
BLEED_FIELDS = ("bleed_fill_pct", "bleed_fill_v2_pct")
FIELDS = D1_FIELDS + D2_FIELDS + BLEED_FIELDS
LABELS = {"b2": "B2", "b2_post": "B2-post", "b3": "SyncMVD", "b4": "Paint3D",
          "v1": "v1", "v2": "v2"}


def read_results(paths):
    scores = {}
    reference = None
    for path in paths:
        evidence = read_json(path.with_name("inputs.json"))
        signature = {name: evidence[name] for name in ("thresholds", "implementation")}
        if reference is None:
            reference = signature
        if signature != reference:
            raise ValueError(f"Mixed physical formulas or cutoffs: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = row["method"], row["brief_id"], int(row["seed"])
                record = {field: float(row[field]) for field in FIELDS}
                record["atlas_sha256"] = row["atlas_sha256"]
                if not row["atlas_sha256"] or (key in scores and record != scores[key]):
                    raise ValueError(f"Inconsistent archived result: {key}")
                scores[key] = record
    return scores, reference


def cohort(scores, briefs, seeds, methods, paired_methods):
    ids = [brief["id"] for brief in briefs]
    if not ids:
        raise ValueError("A diagnostic cohort contains no briefs.")
    missing = [(method, brief, seed) for method in methods for brief in ids for seed in seeds
               if (method, brief, seed) not in scores]
    if missing:
        raise ValueError(f"Missing {len(missing)} paired scores; first missing result: {missing[0]}")
    # Each brief has the same weight. Seeds are replicates within brief.
    per_brief = {
        method: {
            brief: {field: mean(scores[method, brief, seed][field] for seed in seeds) for field in FIELDS}
            for brief in ids
        }
        for method in methods
    }
    rows = []
    for method, results in per_brief.items():
        values = {field: mean(row[field] for row in results.values()) for field in FIELDS}
        failure = sum(values[field] for field in D2_FIELDS)
        if abs(failure + values["bleed_fill_pct"] - 100) > 1e-5:
            raise ValueError(f"D2 buckets do not match the legacy denominator: {method}")
        rows.append({
            "method": method, "briefs": len(ids), "per_seed_images": len(ids) * len(seeds),
            **values,
            "d2_fraction_among_failures_pct": {
                field: 100 * values[field] / failure if failure else None for field in D2_FIELDS
            },
        })
    contrasts = []
    for method in paired_methods:
        contrasts.append({
            "reference": "b2", "method": method, "paired_briefs": len(ids),
            "mean_candidate_minus_b2": {
                field: mean(per_brief[method][brief][field] - per_brief["b2"][brief][field] for brief in ids)
                for field in FIELDS
            },
        })
    return {
        "brief_ids": ids, "seeds": seeds, "methods": rows, "paired_deltas": contrasts,
    }


def plot_d1(results, output):
    import matplotlib.pyplot as plt

    groups = ("native_5", "non_native_7", "wrap_lb")
    row_titles = ("E01b final trial · five briefs, seed 42", "E01 holdout · forty briefs, three seeds")
    files = [results["e01b_final_trial"]["methods"]]
    if "e01_holdout" in results:
        files.append(results["e01_holdout"]["methods"])
    fig, axes = plt.subplots(len(files), 4, figsize=(15, 3.6 * len(files)), layout="constrained", squeeze=False)
    for ri, all_methods in enumerate(files):
        keep = ("b2", "b2_post", "v2") if ri == 0 else ("b2", "b2_post", "v1")
        rows = [entry for entry in all_methods if entry["method"] in keep]
        for axis, metric, title in zip(axes[ri, :3], ("delta_e00", "sdr", "lpips"),
                                       ("Absolute ΔE00", "SDR", "LPIPS")):
            x = range(len(groups))
            for offset, row in enumerate(rows):
                axis.bar([n + 0.24 * (offset - 1) for n in x], [row[f"{g}_{metric}"] for g in groups],
                         width=0.23, label=LABELS[row["method"]])
            axis.set_xticks(list(x), ("Native 5", "Folded 7", "L–B"))
            axis.set_title(title)
            axis.set_ylim(bottom=0)
        axes[ri, 0].set_ylabel(row_titles[ri])
        axes[ri, 0].legend(loc="upper left")
        axis = axes[ri, 3]
        axis.bar([LABELS[row["method"]] for row in rows], [row["inside_dE00"] for row in rows])
        axis.set_title("Within-face ΔE00")
        axis.set_ylim(bottom=0)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_d2(results, output):
    import matplotlib.pyplot as plt

    files = [results["e01b_final_trial"]["methods"]]
    if "e01_holdout" in results:
        files.append(results["e01_holdout"]["methods"])
    fig, axes = plt.subplots(1, len(files), figsize=(7.6 * len(files), 4.1), layout="constrained", squeeze=False)
    colors = ("#367560", "#d9d9d9", "#dfae62", "#b35e60", "#65556f")
    bands = ("Legacy pass", "White edge", "ΔE00 < 0.8", "ΔE00 > 9", "Invalid")
    for i, (axis, rows) in enumerate(zip(axes[0], files)):
        labels = [LABELS[row["method"]] for row in rows]
        bottom = [0] * len(rows)
        for field, label, color in zip(("bleed_fill_pct", *D2_FIELDS), bands, colors):
            values = [row[field] for row in rows]
            axis.bar(labels, values, bottom=bottom, label=label, color=color)
            bottom = [a + b for a, b in zip(bottom, values)]
        axis.scatter(labels, [row["bleed_fill_v2_pct"] for row in rows], marker="D",
                     s=21, label="M3 pass", color="#17253c", zorder=5)
        axis.set_title(("E01b final trial", "E01 holdout")[i])
        axis.set_ylabel("Eligible 3 mm bleed samples (%)")
        axis.set_ylim(0, 100)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    scores, signature = read_results(args.scores)
    briefs = read_briefs()
    results = {
        "protocol": signature,
        "e01b_final_trial": cohort(scores, briefs[:5], [42], REFERENCE_METHODS, ("b2_post", "v2")),
    }
    if not args.pilot_only:
        holdout = briefs[10:]
        results.update(
            e01_holdout=cohort(scores, holdout, [42, 1234, 2026], REFERENCE_METHODS[:-1], ("b2_post", "v1")),
            e01_holdout_continuous=cohort(
                scores, [brief for brief in holdout if brief["continuous"]], [42, 1234, 2026],
                REFERENCE_METHODS[:-1], ("b2_post", "v1"),
            ),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "d1-d2.json", results)
    plot_d1(results, args.output / "d1-seams.png")
    plot_d2(results, args.output / "d2-bleed.png")


if __name__ == "__main__":
    main()
