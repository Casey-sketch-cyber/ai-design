"""Disaggregate frozen E02 LPIPS on the four paired Task 12 methods."""

import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .e01 import read_json, write_json
from .e01b import file_hash
from .metrics import DEFAULT_THRESHOLDS, _aligned_strips, _lpips_pairs
from .metrics.e02_physical import E02PhysicalEvaluator
from .metrics.learned import LearnedMetrics
from .metrics.physical import LPIPS_MM, PIXELS_PER_MM


METHODS = ("b2_regional", "g1", "g2_075", "g2_085")
NAMES = {"b2_regional": "B2-regional", "g1": "G1", "g2_075": "G2 · 0.75", "g2_085": "G2 · 0.85"}
COHORTS = {"b2_regional": "task07", "g1": "task10", "g2_075": "task10", "g2_085": "task10"}
BRIEFS = {f"E01-{i:03d}" for i in range(1, 11)}
REGIONS = ("wrap_lb", "top_three", "bottom_three")
REGION_NAMES = {"wrap_lb": "L–B", "top_three": "顶盖三缝", "bottom_three": "底盖三缝"}
EDGE_ORDER = ("L–B", "B–T", "L–T", "R–T", "F–Btm", "L–Btm", "R–Btm")
CPU_RECHECK_ATOL = 2e-4
JOBS_SEED = 42


def _registered(sources):
    jobs = {}
    for source in sources:
        if (source.method not in METHODS or source.seed != JOBS_SEED or source.brief_id not in BRIEFS
                or source.cohort != COHORTS[source.method]):
            continue
        key = source.method, source.brief_id
        if key in jobs:
            raise ValueError(f"Duplicate Task 12 LPIPS source: {key}.")
        jobs[key] = source
    wanted = {(method, brief) for method in METHODS for brief in BRIEFS}
    if jobs.keys() != wanted:
        missing = sorted(wanted - jobs.keys())
        raise ValueError(f"Task 12 needs forty original paired atlases. Missing: {missing}.")
    return jobs


def _weight_hashes():
    import lpips
    import torch
    from torchvision.models import AlexNet_Weights

    source = AlexNet_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    alexnet = Path(torch.hub.get_dir()) / "checkpoints" / source
    lpips_alex = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
    for item in (alexnet, lpips_alex):
        if not item.is_file():
            raise FileNotFoundError(f"Task 12 uses the existing CPU LPIPS weights. Missing: {item}")
    hashes = {item.name: file_hash(item) for item in (alexnet, lpips_alex)}
    if not hashes[alexnet.name].startswith("7be5be79"):
        raise ValueError("The cached AlexNet weights failed their published SHA-256 prefix.")
    return hashes


def _archive(source):
    raw_sha = file_hash(source.path)
    geometry_sha = file_hash(source.geometry / "topology.json")
    score = read_json(source.score)
    inputs = score["input"]
    if (raw_sha != source.raw_sha256 or raw_sha != inputs["atlas_sha256"]
            or geometry_sha != inputs["geometry_sha256"]):
        raise ValueError(f"Stale Task 12 archive input: {source.method}/{source.brief_id}.")
    return score["variants"], {"raw_sha256": raw_sha, "score_sha256": file_hash(source.score),
                               "geometry_sha256": geometry_sha}


def _splits(evaluator):
    cap = evaluator.groups["non_native_7"]
    return {
        "wrap_lb": evaluator.groups["wrap_lb"],
        "top_three": cap & evaluator.caps["T"],
        "bottom_three": cap & evaluator.caps["Btm"],
    }


def _frozen_groups(evaluator, per_edge):
    return {
        "seam_lpips": per_edge.mean(),
        "non_native_7_lpips": per_edge[evaluator.groups["non_native_7"]].mean(),
        "wrap_lb_lpips": per_edge[evaluator.groups["wrap_lb"]].mean(),
        "top_seam_lpips": per_edge[evaluator.caps["T"]].mean(),
        "bottom_seam_lpips": per_edge[evaluator.caps["Btm"]].mean(),
    }


def _confirm_frozen(source, archive, recomputed):
    largest = 0.0
    for variant in ("raw", "preflight"):
        for key, actual in recomputed.items():
            saved = archive[variant][key]
            if not math.isclose(saved, actual, abs_tol=CPU_RECHECK_ATOL, rel_tol=0):
                raise ValueError(f"CPU LPIPS differs from frozen {variant} {key} for "
                                 f"{source.method}/{source.brief_id}: {actual:g} vs {saved:g}.")
            largest = max(largest, abs(saved - actual))
    return largest


def _edge_energy(strip):
    scaled = strip.astype(np.float32) / 255
    return float(mean(float(np.square(np.diff(scaled, axis=axis)).mean()) for axis in (0, 1)))


def _named_seam(seam, group):
    if group == "wrap_lb":
        return "L–B"
    faces = (seam["a"]["face"], seam["b"]["face"])
    lid = "T" if group == "top_three" else "Btm"
    side = faces[0] if faces[0] != lid else faces[1]
    return f"{side}–{lid}"


def _case(source, evaluator, face_energy):
    archive, provenance = _archive(source)
    faces = evaluator._physical_faces(evaluator._native_atlas(source.path))
    width = round(LPIPS_MM * PIXELS_PER_MM)
    per_edge = evaluator.learned.seam_lpips(_lpips_pairs(faces, evaluator.seams, width), per_seam=True)
    recomputed = _frozen_groups(evaluator, per_edge)
    error = _confirm_frozen(source, archive, recomputed)
    base = {"method": source.method, "brief_id": source.brief_id, "seed": source.seed}
    texture = {
        **base,
        **{f"interior_{face}": float(face_energy(pixels)) for face, pixels in faces.items()},
        "ring_repeat_score": archive["preflight"]["m4_repeat_score"],
    }
    rows = []
    for group, indices in _splits(evaluator).items():
        for index in np.flatnonzero(indices):
            seam = evaluator.seams[index]
            a, b = seam["a"], seam["b"]
            a_strip, b_strip = _aligned_strips(faces, seam, width)
            energy_a, energy_b = _edge_energy(a_strip), _edge_energy(b_strip)
            rows.append({
                **base, "group": group, "seam": _named_seam(seam, group),
                "a_face": a["face"], "a_edge": a["edge"], "a_energy": energy_a,
                "b_face": b["face"], "b_edge": b["edge"], "b_energy": energy_b,
                "reverse": seam["reverse"], "lpips": float(per_edge[index]),
                "abs_energy_difference": abs(energy_a - energy_b),
            })
    if len(rows) != 7 or {row["seam"] for row in rows} != set(EDGE_ORDER):
        raise ValueError("Task 12 seven-seam grouping does not match the frozen box topology.")
    if not math.isclose(mean(row["lpips"] for row in rows), recomputed["non_native_7_lpips"], abs_tol=1e-12):
        raise ValueError("The seven returned seams disagree with the frozen LPIPS aggregate.")
    return rows, texture, error, provenance


def _group_data(rows, textures):
    grouped = {}
    for method in METHODS:
        data = [row for row in rows if row["method"] == method]
        texture = [row for row in textures if row["method"] == method]
        edges = {edge: [r for r in data if r["seam"] == edge] for edge in EDGE_ORDER}
        regions = {group: [r for r in data if r["group"] == group] for group in REGIONS}
        lids = regions["top_three"] + regions["bottom_three"]
        energies = {f"mean_interior_{face}": mean(r[f"interior_{face}"] for r in texture)
                    for face in ("F", "B", "L", "R", "T", "Btm")}
        grouped[method] = {
            "seam_means": {k: mean(r["lpips"] for r in v) for k, v in edges.items()},
            "region_means": {k: mean(r["lpips"] for r in v) for k, v in regions.items()},
            "seven_seams": mean(r["lpips"] for r in data),
            "six_cap_seams": mean(r["lpips"] for r in lids),
            **energies,
            "mean_ring_interior": mean(energies[f"mean_interior_{face}"] for face in ("F", "B", "L", "R")),
            "mean_ring_repeat_score": mean(r["ring_repeat_score"] for r in texture),
            "mean_near_cap_energy_difference": mean(r["abs_energy_difference"] for r in lids),
            "mean_top_energy_difference": mean(r["abs_energy_difference"] for r in regions["top_three"]),
            "mean_bottom_energy_difference": mean(r["abs_energy_difference"] for r in regions["bottom_three"]),
        }
    baseline = grouped["b2_regional"]
    base_data = [row for row in rows if row["method"] == "b2_regional"]
    for method, item in grouped.items():
        data = [row for row in rows if row["method"] == method]
        edge_gaps = {edge: item["seam_means"][edge] - baseline["seam_means"][edge] for edge in EDGE_ORDER}
        item["pair_gaps_vs_b2_regional"] = edge_gaps
        item["seven_gap_contributions_vs_b2_regional"] = {
            group: (item["region_means"][group] - baseline["region_means"][group]) * count / 7
            for group, count in zip(REGIONS, (1, 3, 3), strict=True)
        }
        item["better_cases_vs_b2_regional"] = {}
        for group in REGIONS:
            item["better_cases_vs_b2_regional"][group] = sum(
                mean(r["lpips"] for r in data if r["brief_id"] == brief and r["group"] == group)
                < mean(r["lpips"] for r in base_data if r["brief_id"] == brief and r["group"] == group)
                for brief in BRIEFS
            )
    return grouped


def _table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(value) for value in values) + " |"
    return "\n".join((line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)))


def _number(value, digits=4, signed=False):
    return f"{value:{'+' if signed else ''}.{digits}f}".replace("-", "−")


def _section(summary):
    groups = summary["methods"]
    baseline, case = groups["b2_regional"], groups["g2_085"]
    gap = case["seven_seams"] - baseline["seven_seams"]
    explanation = case["seven_gap_contributions_vs_b2_regional"]
    max_two = sorted(EDGE_ORDER, key=lambda edge: (-case["pair_gaps_vs_b2_regional"][edge], edge))[:2]
    top_count = case["better_cases_vs_b2_regional"]["top_three"]
    bottom_count = case["better_cases_vs_b2_regional"]["bottom_three"]
    parts = [
        "## 七条非原生缝的 LPIPS 与盖面纹理",
        "同一批 E01-001 至 E01-010（种子 42）。使用冻结的物理网格和 LPIPS AlexNet v0.1，"
        "按每侧 5 mm 条带逐缝重算。顶盖与底盖分别只计三个非原生邻边。"
        "报告中的旧顶盖、底盖汇总各含第四条原生折线。"
        f"逐边重组成原有指标后，与存档的最大绝对差为 {summary['largest_frozen_score_difference']:.2e}。",
        _table(("方法", "L–B", "顶盖三缝", "底盖三缝", "盖面六缝", "七缝"), (
            (NAMES[method], *(_number(groups[method]["region_means"][k]) for k in REGIONS),
             _number(groups[method]["six_cap_seams"]), _number(groups[method]["seven_seams"]))
            for method in METHODS
        )),
        _table(("原邻接缝", *(NAMES[m] for m in METHODS), "G2 · 0.85 − B2-regional"), (
            (edge, *(_number(groups[m]["seam_means"][edge]) for m in METHODS),
             _number(case["pair_gaps_vs_b2_regional"][edge], signed=True))
            for edge in EDGE_ORDER
        )),
        f"G2 · 0.85 的七缝 LPIPS 相对 B2-regional 差 {_number(gap, signed=True)}。"
        "按七缝权重拆开后，"
        + "、".join(f"{REGION_NAMES[k]} {_number(explanation[k], signed=True)}" for k in REGIONS)
        + "。"
        f"正向差值最大的两条缝为 {'、'.join(max_two)}。按简报配对，"
        f"顶盖三缝有 {top_count}/10 盒低于 B2-regional，底盖三缝有 {bottom_count}/10 盒低于 B2-regional。",
        "### 盖面面内纹理",
        "在 4 px/mm 网格上，面内排除四周 9 mm，归一化 sRGB 后分别计算横、纵一阶差分平方均值，"
        "两轴等权。配对能量差用跨缝两侧各 5 mm 的条带做同样计算后取绝对差。"
        "表中能量均以 10⁻³ 为单位。M4 重复度来自同盒侧面和正面两半的相似度。",
        _table(("方法", "环带面内平均", "顶盖面内", "底盖面内", "顶三缝两侧能量差",
                "底三缝两侧能量差", "M4 重复度"), (
            (NAMES[method], *(_number(groups[method][key] * 1e3, 3) for key in
                              ("mean_ring_interior", "mean_interior_T", "mean_interior_Btm",
                               "mean_top_energy_difference", "mean_bottom_energy_difference")),
             _number(groups[method]["mean_ring_repeat_score"]))
            for method in METHODS
        )),
        f"B2-regional 与 G2 · 0.85 的底盖面内能量分别为 "
        f"{_number(baseline['mean_interior_Btm'] * 1e3, 3)} 与 "
        f"{_number(case['mean_interior_Btm'] * 1e3, 3)}，底三缝的两侧能量差分别为 "
        f"{_number(baseline['mean_bottom_energy_difference'] * 1e3, 3)} 与 "
        f"{_number(case['mean_bottom_energy_difference'] * 1e3, 3)}。"
        "在这十盒中，更平滑的底盖、更大的跨缝能量差与更高的底盖 LPIPS 同时出现。"
        f"G2 · 0.85 的顶盖面内能量也很低，顶盖三缝 LPIPS 为 {_number(case['region_means']['top_three'])}，"
        f"略低于 B2-regional 的 {_number(baseline['region_means']['top_three'])}。"
        "梯度能量同时受图案、对比和噪声影响，M4 记录侧面的重复度。"
        "观察结果支持位置相关的纹理关联。要估计同花纹的因果效应，仍需在其余条件一致时单独改变重复程度。",
    ]
    return "\n\n".join(parts) + "\n"


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_lpips(sources, target_dir, *, face_energy=None):
    """Write paired LPIPS, texture, and archival verification data on CPU."""
    if face_energy is None:
        from .stripe_diag import face_energy

    jobs = _registered(sources)
    weights = _weight_hashes()
    learned = LearnedMetrics("cpu", DEFAULT_THRESHOLDS["ocr_confidence"])
    evaluators, evidence = {}, {}
    rows, texture, errors = [], [], []
    for method in METHODS:
        for brief in sorted(BRIEFS):
            source = jobs[method, brief]
            geometry = source.geometry
            if geometry not in evaluators:
                evaluators[geometry] = E02PhysicalEvaluator(geometry, device="cpu", learned=learned)
            seams, fields, error, provenance = _case(source, evaluators[geometry], face_energy)
            rows.extend(seams)
            texture.append(fields)
            errors.append(error)
            evidence[f"{method}/{brief}"] = provenance
        print(f"Task 12 LPIPS verified 10 {method} atlases on CPU.", flush=True)
    summary = {
        "seed": JOBS_SEED, "n_briefs_per_method": len(BRIEFS), "device": "cpu",
        "width_mm_each_side": LPIPS_MM, "pixels_per_mm": PIXELS_PER_MM,
        "model_sha256": weights, "original_images": evidence,
        "frozen_score_cpu_absolute_tolerance": CPU_RECHECK_ATOL,
        "largest_frozen_score_difference": max(errors),
        "methods": _group_data(rows, texture),
    }
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_table(target / "lpips-seams.csv", rows)
    _write_table(target / "lpips-texture.csv", texture)
    write_json(target / "lpips-diagnostic.json", summary)
    (target / "lpips-section.md").write_text(_section(summary), encoding="utf-8")
    return summary
