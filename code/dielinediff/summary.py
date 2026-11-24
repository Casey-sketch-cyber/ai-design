"""Summarize measured E01 results and render deterministic matched examples."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
from scipy import stats


METHODS = ("b2", "b3", "b4", "ours", "no_sync", "no_projection")
SEEDS = (42, 1234, 2026)
METRICS = {
    "seam_sdr": "lower",
    "seam_lpips": "lower",
    "upright_rate": "higher",
    "glue_ink_pct": "lower",
    "flap_edge_pct": "lower",
    "flap_ocr_count": "lower",
    "bleed_fill_pct": "higher",
    "lighting_residual": "lower",
    "oog_pct": "lower",
    "roundtrip_delta_e00": "lower",
    "clip_score": "higher",
    "hps_v2": "higher",
    "elapsed_seconds": "lower",
    "peak_memory_mib": "lower",
}
PHASE_LENGTH = {"pilot": 5, "tune": 10, "full": 40}
EXAMPLE_NAME = re.compile(r"(.+?)-(-?\d+)\.png")


def _brief_source(path, phase):
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as stream:
        briefs = [json.loads(line) for line in stream if line.strip()]
    if len(briefs) != 50 or len({brief["id"] for brief in briefs}) != 50:
        raise ValueError("Expected the E01 manifest with 50 unique briefs")
    if any(type(brief["continuous"]) is not bool for brief in briefs):
        raise ValueError("The continuous field must be a boolean")
    if phase == "full":
        selected = briefs[10:]
    else:
        selected = briefs[:PHASE_LENGTH[phase]]
    return {brief["id"]: brief["continuous"] for brief in selected}


def _boolean(text):
    values = {"true": True, "1": True, "false": False, "0": False}
    try:
        return values[text.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Invalid continuous flag {text!r}") from error


def _number(text):
    if text is None or text.strip().lower() in ("", "none", "null"):
        return None
    val = float(text)
    return val if math.isfinite(val) else None


def _numbered_brief_in_phase(brief, phase):
    suffix = brief.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return True
    number = int(suffix)
    return number > 10 if phase == "full" else 1 <= number <= PHASE_LENGTH[phase]


def _read_rows(path, phase, seeds, planned):
    split = "holdout" if phase == "full" else "tune"
    metadata = {}
    selected = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "brief_id", "seed", "split", "continuous"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected CSV fields: {', '.join(sorted(required))}")
        for row in reader:
            if row["split"] not in {"holdout", "tune"}:
                raise ValueError(f"Invalid split {row['split']!r}")
            if row["split"] != split or int(row["seed"]) not in seeds:
                continue
            method, brief, seed = row["method"], row["brief_id"], int(row["seed"])
            if planned is None and not _numbered_brief_in_phase(brief, phase):
                continue
            continuous = _boolean(row["continuous"])
            if planned is not None:
                if brief not in planned:
                    raise ValueError(f"Brief {brief} is outside the {phase} manifest")
                if continuous != planned[brief]:
                    raise ValueError(f"Continuous flag differs from the manifest for {brief}")
            if brief in metadata and metadata[brief] != continuous:
                raise ValueError(f"Inconsistent continuous flags for {brief}")
            metadata[brief] = continuous
            key = (method, brief, seed)
            if key in selected:
                raise ValueError(f"Duplicate result {key}")
            selected[key] = {metric: _number(row.get(metric)) for metric in METRICS}
    return selected, metadata


def _interval(values):
    if len(values) < 2:
        return None
    average = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    width = float(stats.t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return [average - width, average + width]


def _aggregate(rows, method, brief_ids, seeds, metric, planned):
    complete = {}
    measured = 0
    for brief in brief_ids:
        values = [rows.get((method, brief, seed), {}).get(metric) for seed in seeds]
        measured += sum(value is not None for value in values)
        if all(value is not None for value in values):
            complete[brief] = float(np.mean(values))
    vals = list(complete.values())
    summary = {
        "n": len(vals), "planned_n": planned, "measured_seed_values": measured,
        "mean": float(np.mean(vals)) if vals else None, "ci95": _interval(vals),
    }
    return summary, complete


def _paired(reference, comparator):
    matched = sorted(reference.keys() & comparator.keys())
    differences = np.array([reference[key] - comparator[key] for key in matched])
    n = len(matched)
    result = {"n": n, "mean_difference": None, "median_difference": None,
              "direction": None, "p_wilcoxon": None}
    if not n:
        return result
    average = float(differences.mean())
    direction = "higher" if average > 0 else "lower" if average < 0 else "equal"
    probability = None
    if n >= 2:
        if np.all(differences == 0):
            probability = 1.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Sample size too small.*", category=UserWarning)
                probability = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    result.update(mean_difference=average, median_difference=float(np.median(differences)),
                  direction=direction, p_wilcoxon=probability)
    return result


def build_summary(results, phase, seeds=SEEDS, briefs=None, methods=None):
    """Analyze one phase with equal weighting per complete brief."""
    seeds = tuple(seeds) if phase == "full" else tuple(seeds[:1])
    if not seeds or len(set(seeds)) != len(seeds) or (phase == "full" and len(seeds) != 3):
        raise ValueError("Full E01 needs three distinct seeds; pilot or tune needs a first seed")
    methods = tuple(methods or (METHODS if phase == "full" else METHODS[:4]))
    planned = _brief_source(briefs, phase)
    rows, observed = _read_rows(results, phase, seeds, planned)
    brief_flags = planned if planned is not None else observed
    subsets = {
        "overall": sorted(brief_flags),
        "cross_panel": sorted(brief for brief, crosses in brief_flags.items() if crosses),
    }
    report = {"phase": phase, "unit": "brief", "seeds": list(seeds),
              "reference": "ours", "subsets": {}}
    for group, ids in subsets.items():
        expected = len(ids) if planned is not None else PHASE_LENGTH[phase] if group == "overall" else None
        target = {"planned_briefs": expected, "observed_briefs": len(set(ids) & observed.keys()),
                  "descriptive": [], "comparisons": []}
        for metric, favorable in METRICS.items():
            means = {}
            for method in methods:
                summary, means[method] = _aggregate(rows, method, ids, seeds, metric, expected)
                target["descriptive"].append({"method": method, "metric": metric,
                                              "favorable": favorable, **summary})
            if phase != "full" or "ours" not in means:
                continue
            for other, values in means.items():
                if other == "ours":
                    continue
                pair = _paired(means["ours"], values)
                target["comparisons"].append({"reference": "ours", "comparator": other,
                                              "metric": metric, "favorable": favorable, **pair})
        report["subsets"][group] = target
    return report


def _display(value):
    if value is None:
        return "NA"
    return f"{value:.5g}"


def format_markdown(report):
    """Present units, per-metric missingness, and directional paired results."""
    # Keep this display text aligned with the Chinese E01 report.
    phases = {"full": "正式测试集", "pilot": "冒烟实验", "tune": "调参阶段"}
    groups = {"overall": "全部简报", "cross_panel": "跨面连续子集"}
    directions = {"higher": "较高", "lower": "较低", "equal": "持平", None: "NA"}
    favorable = {"higher": "越高越好", "lower": "越低越好"}
    phase, seeds = report["phase"], report["seeds"]
    lines = [f"# E01 {phases[phase]}", "",
             f"统计单位为简报。种子为 {', '.join(map(str, seeds))}。"]
    if phase == "full":
        lines.append("每份简报的三个种子均有测量时才纳入对应指标。95% 置信区间用简报均值的 t 区间。"
                     "配对 Wilcoxon 检验基于两方法共同具备的简报，采用双侧未校正 p 值。")
    else:
        lines.append("本表用于阶段检查。95% 区间基于有结果的简报，显著性检验留待正式测试集。")
    lines.append("NA 表示缺少测量值、配对样本或区间估计所需样本。")
    for key, subset in report["subsets"].items():
        planned = subset["planned_briefs"]
        lines.extend(("", f"## {groups[key]}", "",
                      f"计划简报 {_display(planned)} 份，CSV 可见 {subset['observed_briefs']} 份。", "",
                      "| 指标 | 方法 | 指标方向 | 完整简报数 | 有效种子测量数 | 均值 | 95% CI |",
                      "| --- | --- | --- | ---: | ---: | ---: | --- |"))
        for row in subset["descriptive"]:
            ci = row["ci95"]
            ci_text = f"[{_display(ci[0])}, {_display(ci[1])}]" if ci else "NA"
            lines.append(f"| {row['metric']} | {row['method']} | {favorable[row['favorable']]} |"
                         f" {row['n']} | {row['measured_seed_values']} | {_display(row['mean'])} | {ci_text} |")
        if subset["comparisons"]:
            lines.extend(("", "差值用 ours 减对照方法，方向按平均差值判定。", "",
                          "| 指标 | 对照 | 共同简报数 | 平均差值 | 差值中位数 | ours 方向 | Wilcoxon p |",
                          "| --- | --- | ---: | ---: | ---: | --- | ---: |"))
        for row in subset["comparisons"]:
            lines.append(f"| {row['metric']} | {row['comparator']} | {row['n']} |"
                         f" {_display(row['mean_difference'])} | {_display(row['median_difference'])} |"
                         f" {directions[row['direction']]} | {_display(row['p_wilcoxon'])} |")
    return "\n".join(lines) + "\n"


def _gallery(geometry, images, path):
    from PIL import Image, ImageDraw, ImageFont

    from .render.orthographic import VIEWS, load_mesh, render_texture

    mesh = load_mesh(geometry / "box.obj")
    height = 45 + 325 * len(images)
    page = Image.new("RGB", (1090, height), (250, 251, 252))
    draw = ImageDraw.Draw(page)
    header = ImageFont.load_default(size=22)
    label = ImageFont.load_default(size=15)
    ink = (28, 38, 46)
    for x, title in ((160, "Dieline texture"), (490, "Folded front"), (790, "Folded isometric")):
        draw.text((x + 140, 19), title, anchor="mm", font=header, fill=ink)
    for index, (method, path_to_image) in enumerate(images.items()):
        y = 45 + index * 325
        draw.text((14, y + 142), method, font=header, fill=ink)
        with Image.open(path_to_image) as atlas:
            preview = atlas.convert("RGB")
            preview.thumbnail((310, 290), Image.Resampling.LANCZOS)
            page.paste(preview, (160 + (310 - preview.width) // 2, y + (290 - preview.height) // 2))
            for x, name in ((490, "F"), (790, "iso")):
                folded = render_texture(mesh, atlas, *VIEWS[name], size=290)
                page.paste(folded, (x, y))
        draw.text((160, y + 297), path_to_image.name, font=label, fill=ink)
    page.save(path, format="JPEG", quality=84, optimize=True)


def write_examples(run, geometry, destination, phase, seed=42, methods=None, briefs=None):
    """Choose the first three unique common briefs for the requested seed."""
    run, geometry, destination = Path(run), Path(geometry), Path(destination)
    methods = methods or (METHODS if phase == "full" else METHODS[:4])
    allowed = _brief_source(briefs, phase)
    available = {}
    for method in methods:
        files = {}
        for file in (run / method).glob("*.png"):
            parsed = EXAMPLE_NAME.fullmatch(file.name)
            if parsed is None or int(parsed[2]) != seed or not file.with_suffix(".run.json").is_file():
                continue
            brief = parsed[1]
            accepted = _numbered_brief_in_phase(brief, phase) if allowed is None else brief in allowed
            if accepted:
                files[brief] = file
        available[method] = files
    included = {method: files for method, files in available.items() if files}
    common = set.intersection(*(set(files) for files in included.values())) if included else set()
    selected = sorted(common)[:3]
    destination.mkdir(parents=True, exist_ok=True)
    figures = []
    for brief in selected:
        name = f"example-{phase}-{brief}-seed{seed}.jpg"
        _gallery(geometry, {method: files[brief] for method, files in included.items()}, destination / name)
        figures.append(name)
    manifest = {
        "phase": phase, "seed": seed, "chosen_briefs": selected, "figures": figures,
        "expected_figure_count": 3, "common_brief_count": len(common),
        "included_methods": list(included), "missing_methods": sorted(set(methods) - set(included)),
        "available_brief_counts": {method: len(files) for method, files in available.items()},
    }
    (destination / f"examples-{phase}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("stats", help="Summarize one phase's results CSV")
    summary.add_argument("--results", type=Path, required=True)
    summary.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    summary.add_argument("--format", choices=("markdown", "json"), default="markdown")
    examples = commands.add_parser("examples", help="Create matched atlas/folded-example images")
    examples.add_argument("--run", type=Path, required=True)
    examples.add_argument("--geometry", type=Path, required=True)
    examples.add_argument("--seed", type=int, default=SEEDS[0])
    for command in (summary, examples):
        command.add_argument("--phase", choices=tuple(PHASE_LENGTH), required=True)
        command.add_argument("--briefs", type=Path)
        command.add_argument("--methods", nargs="+", choices=METHODS)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "examples":
        write_examples(args.run, args.geometry, args.output, args.phase, args.seed, args.methods, args.briefs)
        return
    report = build_summary(args.results, args.phase, args.seeds, args.briefs, args.methods)
    content = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
               if args.format == "json" else format_markdown(report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
