"""Paired CPU stripe diagnostics on registered crops from archived packaging art."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import _oriented_strip
from .metrics.physical import PIXELS_PER_MM


METRICS = ("v1", "v2_a", "v2_b", "v2_c")
CAP_FOLDS = (frozenset(("F", "T")), frozenset(("B", "Btm")))
STRIPE_KINDS = ("uniform_bright", "uniform_dark", "modulated")
CONTRASTS = (24, 36, 48)
FADE_MM = 8.0
MILD_SHIFT_RGB = 5
PAIR_TOLERANCE = 1e-6


def _oriented(face, edge):
    depth = face.shape[0] if edge in ("top", "bottom") else face.shape[1]
    return _oriented_strip(face, edge, depth)


def _face_from_strip(template, edge, strip):
    face = np.empty_like(template)
    _oriented(face, edge)[:] = strip
    return face


def _half_cosine(depth, length):
    fraction = np.minimum(np.arange(depth, dtype=np.float64) / length, 1)
    return (0.5 + 0.5 * np.cos(np.pi * fraction))[None, :, None]


def _to_uint8(values):
    return np.rint(values).astype(np.uint8)


def _patch_range(faces):
    if len(faces) != 6 or any(face.dtype != np.uint8 or face.ndim != 3 or face.shape[2] != 3
                              for face in faces.values()):
        raise ValueError("Synthetic diagnostics need six registered uint8 RGB faces")
    # Reserve exactly 48 RGB levels in both directions. A constant tangential
    # offset remains constant after insertion; saturation cannot fake texture.
    return {key: _to_uint8(face.astype(np.float64) * (159 / 255) + 48)
            for key, face in faces.items()}


def _cap_controls(faces, seam):
    """Construct the continuation, flat control, and matched textured cap."""
    a, b = seam["a"], seam["b"]
    cap, side = (a, b) if a["face"] in ("T", "Btm") else (b, a)
    cap_shape = _oriented(faces[cap["face"]], cap["edge"]).shape
    side_strip = _oriented(faces[side["face"]], side["edge"])
    length, depth, _ = cap_shape
    if side_strip.shape[0] != length or side_strip.shape[1] < depth:
        raise ValueError(f"Unregistered physical cap crease: {side['face']} to {cap['face']}")
    texture = side_strip[:, :depth].astype(np.float64)
    flat = np.rint(np.median(texture[:, :round(3 * PIXELS_PER_MM)], axis=(0, 1)))

    fade = _half_cosine(depth, round(FADE_MM * PIXELS_PER_MM))
    a_cap = _to_uint8(flat + (texture - flat) * fade)

    # C has a controlled mild colour step. Blend the side texture into a
    # constant edge over two mm, using the same side for C, D and D's control.
    softened_side = _to_uint8(side_strip + (flat - side_strip) * _half_cosine(
        side_strip.shape[1], round(2 * PIXELS_PER_MM)
    ))
    mild = flat + np.where(flat <= 207 - MILD_SHIFT_RGB, MILD_SHIFT_RGB, -MILD_SHIFT_RGB)
    c_cap = np.broadcast_to(_to_uint8(mild), cap_shape)
    d_control = _to_uint8(texture + (mild - texture) * _half_cosine(
        depth, round(2 * PIXELS_PER_MM)
    ))
    cap_face = faces[cap["face"]]

    def atlas(strip, soften):
        oriented = strip[::-1] if seam["reverse"] else strip
        result = {**faces, cap["face"]: _face_from_strip(cap_face, cap["edge"], oriented)}
        if soften:
            result[side["face"]] = _face_from_strip(faces[side["face"]], side["edge"], softened_side)
        return result

    return atlas, a_cap, c_cap, d_control


def _apply_stripe(strip, pair, kind, width, contrast):
    if kind == "modulated":
        period = (4, 6, 8)[(pair // 3) % 3]
        phase = np.random.default_rng(20261125 + pair).uniform(0, 2 * np.pi)
        delta = contrast * (0.65 + 0.35 * np.sin(np.arange(len(strip)) * 2 * np.pi / period + phase))
        sign = 1 if pair % 2 == 0 else -1
    else:
        period = None
        sign = 1 if kind == "uniform_bright" else -1
        delta = np.full(len(strip), contrast)
    delta = np.rint(delta * sign).astype(np.int16)
    painted = strip.astype(np.int16)
    painted[:, :width] += delta[:, None, None]
    if np.any((painted < 0) | (painted > 255)):
        raise AssertionError("Synthetic stripe clipped its reserved RGB headroom")
    return painted.astype(np.uint8), {
        "stripe_kind": kind, "stripe_width_px": width, "nominal_contrast_rgb": contrast,
        "actual_peak_rgb": int(np.max(np.abs(delta))), "actual_mean_abs_rgb": float(np.abs(delta).mean()),
        "stripe_sign": "bright" if sign > 0 else "dark", "tangential_period_px": period,
    }


def cases(faces, seam, pair):
    """Yield A–D derived crops. D0 is D's auxiliary texture-matched negative."""
    build_atlas, a_cap, c_cap, d_control = _cap_controls(_patch_range(faces), seam)
    empty = {"stripe_kind": "none", "stripe_width_px": 0, "nominal_contrast_rgb": 0,
             "actual_peak_rgb": 0, "actual_mean_abs_rgb": 0.0, "stripe_sign": None,
             "tangential_period_px": None}
    for name, strip, soften in (("A", a_cap, False), ("C", c_cap, True), ("D0", d_control, True)):
        yield name, build_atlas(strip, soften), empty
    width = 2 + pair % 7
    contrast = CONTRASTS[(pair + pair // 7) % len(CONTRASTS)]
    for kind in STRIPE_KINDS:
        for name, strip, soften in (("B", a_cap, False), ("D", d_control, True)):
            painted, metadata = _apply_stripe(strip, pair, kind, width, contrast)
            yield name, build_atlas(painted, soften), metadata


def _scores(rows, seam):
    rows = tuple(rows)
    target_faces = frozenset((seam["a"]["face"], seam["b"]["face"]))
    target, = (row for row in rows if frozenset((row["a_face"], row["b_face"])) == target_faces)
    cap_side = "a" if target["a_face"] in ("T", "Btm") else "b"
    record = {"fold": target["fold"],
              "cap_crease_energy": float(target[f"{cap_side}_crease_energy"]),
              "cap_reference_energy_v1": float(target[f"{cap_side}_reference_energy_v1"])}
    for metric in METRICS:
        peak = max(rows, key=lambda row: float(row[metric]))
        record[f"{metric}_fold"] = float(target[metric])
        record[f"{metric}_box"] = float(peak[metric])
        record[f"{metric}_peak_fold"] = peak["fold"]
    return record


def _distribution(values, threshold):
    values = np.asarray(values, dtype=np.float64)
    q25, median, q75, p90 = np.percentile(values, [25, 50, 75, 90])
    hits = int(np.count_nonzero(values > threshold))
    return {"n": len(values), "mean": float(values.mean()), "q25": float(q25),
            "median": float(median), "q75": float(q75), "p90": float(p90),
            "min": float(values.min()), "max": float(values.max()),
            "above_n": hits, "above_rate": hits / len(values)}


def _ordered(positive, negative):
    equal = np.isclose(positive, negative, atol=PAIR_TOLERANCE, rtol=PAIR_TOLERANCE)
    return (positive > negative) & ~equal, equal


def _discrimination(positives, negatives, threshold):
    positives = np.asarray(positives, dtype=np.float64)
    negatives = np.asarray(negatives, dtype=np.float64)
    higher, tied = _ordered(positives, negatives)
    all_higher, all_tied = _ordered(positives[:, None], negatives[None, :])
    true_n = int(np.count_nonzero(positives > threshold))
    false_n = int(np.count_nonzero(negatives > threshold))
    return {
        "n_pairs": len(positives),
        "true_positive_n": true_n, "true_positive_rate": true_n / len(positives),
        "false_positive_n": false_n, "false_positive_rate": false_n / len(negatives),
        "pair_higher_n": int(higher.sum()), "pair_higher_rate": float(higher.mean()),
        "pair_tie_n": int(tied.sum()), "pair_tie_rate": float(tied.mean()),
        "auc": float(all_higher.mean() + 0.5 * all_tied.mean()),
    }


def summarize(rows, threshold):
    controls = {(row["pair"], row["case"]): row for row in rows if row["stripe_kind"] == "none"}
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["case"], row["stripe_kind"]].append(row)
    report = {"threshold": threshold, "pair_tolerance": PAIR_TOLERANCE,
              "distributions": [], "discrimination": []}
    comparisons = (("B", "A"), ("D", "C"), ("D", "D0"))
    for metric in METRICS:
        for scope in ("fold", "box"):
            field = f"{metric}_{scope}"
            for (case, stripe_kind), group in sorted(by_group.items()):
                report["distributions"].append({
                    "metric": metric, "scope": scope, "case": case, "stripe_kind": stripe_kind,
                    **_distribution([sample[field] for sample in group], threshold),
                })
            for positive, negative in comparisons:
                for kind in STRIPE_KINDS:
                    group = by_group[positive, kind]
                    strata = [(None, group)]
                    strata.extend((contrast, [row for row in group if row["nominal_contrast_rgb"] == contrast])
                                  for contrast in CONTRASTS)
                    for contrast, samples in strata:
                        if not samples:
                            continue
                        positives = [row[field] for row in samples]
                        negatives = [controls[row["pair"], negative][field] for row in samples]
                        report["discrimination"].append({
                            "metric": metric, "scope": scope, "positive": positive, "negative": negative,
                            "stripe_kind": kind, "nominal_contrast_rgb": contrast,
                            **_discrimination(positives, negatives, threshold),
                        })
    return report


def run_synthetic(faces_by_source, creases, measure, reference_floor, output_dir, *, threshold):
    """Run only numeric diagnostics from (archive ID, six physical faces) pairs."""
    creases = tuple(creases)
    threshold = float(threshold)
    reference_floor = float(reference_floor)
    targets = [next(seam for seam in creases if frozenset((seam["a"]["face"], seam["b"]["face"])) == pair)
               for pair in CAP_FOLDS]
    rows, sources = [], set()
    for pair, (source, faces) in enumerate(faces_by_source):
        if source in sources:
            raise ValueError(f"Synthetic pairing requires unique archive IDs: {source}")
        sources.add(source)
        seam = targets[pair % len(targets)]
        for name, constructed, metadata in cases(faces, seam, pair):
            scores = _scores(measure(constructed, creases, reference_floor), seam)
            rows.append({"pair": pair, "source": source, "case": name, **metadata, **scores})
    if len(sources) < 20:
        raise ValueError("Synthetic diagnostics require at least 20 distinct archived textures")

    report = summarize(rows, threshold)
    report["design"] = {
        "paired_sources": len(sources), "total_numeric_rows": len(rows), "random_seed_offset": 20261125,
        "pixels_per_mm": PIXELS_PER_MM, "baseline_headroom_rgb": [48, 207],
        "fade_mm": FADE_MM, "flat_fold_step_rgb": MILD_SHIFT_RGB,
        "synthetic_stripe_widths_px": list(range(2, 9)),
        "nominal_contrasts_rgb": list(CONTRASTS), "reference_floor": reference_floor,
        "primary_cases": ["A", "B", "C", "D"], "D0": "matched textured cap without a stripe",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "synthetic-samples.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (destination / "synthetic-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def plot_synthetic(samples_csv, destination, threshold):
    """Plot score distributions from numeric results without rendering the fixtures."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

    with Path(samples_csv).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {
        "A": ("A", "none"), "B+": ("B", "uniform_bright"), "B−": ("B", "uniform_dark"),
        "B~": ("B", "modulated"), "C": ("C", "none"), "D+": ("D", "uniform_bright"),
        "D−": ("D", "uniform_dark"), "D~": ("D", "modulated"),
    }
    colors = ("#9DA9B2", "#68A89F", "#68A89F", "#258777", "#9DA9B2", "#E4AA72", "#E4AA72", "#BE672E")
    maximum = max(float(row[f"{metric}_fold"]) for row in rows for metric in METRICS)
    top = max(maximum, threshold) * 1.15
    ticks = [2 ** exponent for exponent in range(int(np.log2(top)) + 1)]
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 6.6), sharex=True, sharey=True, layout="constrained")
    figure.suptitle("Paired synthetic controls at the edited fold", fontsize=14, fontweight="bold")
    for index, (metric, axis) in enumerate(zip(METRICS, axes.flat, strict=True)):
        key = f"{metric}_fold"
        values = [[float(row[key]) for row in rows if (row["case"], row["stripe_kind"]) == group]
                  for group in groups.values()]
        artists = axis.boxplot(values, patch_artist=True, widths=0.56, tick_labels=list(groups),
                               medianprops={"color": "#1B232A", "linewidth": 1.5},
                               flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for artist, color in zip(artists["boxes"], colors, strict=True):
            artist.set_facecolor(color)
            artist.set_edgecolor("#46535C")
        axis.axhline(threshold, color="#96554C", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_ylim(0.94, top)
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.grid(axis="y", alpha=0.22)
        axis.set_title(metric.replace("_", "-"))
        if index % 2 == 0:
            axis.set_ylabel("Stripe score · log scale")
        axis.spines[["top", "right"]].set_visible(False)
    figure.supxlabel("A continuation · C flat · + bright uniform · − dark uniform · ~ modulated\n"
                     f"Dashed reference {threshold:.3f} reuses the historical v1 number. "
                     "Candidate cutoffs remain uncalibrated.",
                     fontsize=9, color="#46535C")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
