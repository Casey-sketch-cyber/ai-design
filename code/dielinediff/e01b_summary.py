"""Summarize the paired E01b variants and render native-geometry examples."""

from .e01 import read_json
from .summary import _aggregate, _paired


MAIN = (
    ("seam_sdr", "接缝 SDR"),
    ("seam_lpips", "接缝 LPIPS"),
    ("bleed_fill_pct", "出血填充率 %"),
    ("oog_pct", "色域外像素 %"),
    ("roundtrip_delta_e00", "平均往返 ΔE00"),
    ("clip_score", "CLIPScore"),
    ("hps_v2", "HPS v2"),
    ("image_reward", "ImageReward"),
    ("elapsed_seconds", "单盒耗时 s"),
    ("peak_memory_mib", "峰值分配显存 MiB"),
)
STRIPE = ("stripe_index", "折边条带指数")
LID = (
    ("top_seam_sdr", "顶面接缝 SDR"),
    ("top_seam_lpips", "顶面接缝 LPIPS"),
    ("top_upright_rate", "顶面 CLIP 正立率"),
    ("bottom_seam_sdr", "底面接缝 SDR"),
    ("bottom_seam_lpips", "底面接缝 LPIPS"),
    ("bottom_upright_rate", "底面 CLIP 正立率"),
    ("back_bottom_stripe_index", "背面—底面压痕条带指数"),
)
SUPPLEMENT = (("lighting_residual", "低频亮度梯度"), ("upright_rate", "CLIP 正立率"))
RAW_ONLY = ("glue_ink_pct", "粘口油墨覆盖率 %")
NATIVE_METRIC_KEYS = tuple(key for key, _ in (RAW_ONLY, *MAIN, *SUPPLEMENT))
METRIC_KEYS = NATIVE_METRIC_KEYS + tuple(key for key, _ in (STRIPE, *LID))


def _number(value):
    return "NA" if value is None else f"{value:.5g}"


def _ci(bounds):
    return "NA" if bounds is None else f"[{_number(bounds[0])}, {_number(bounds[1])}]"


def _variant(rows, briefs, methods, seeds, phase, metrics):
    indexed = {}
    for row in rows:
        key = row["method"], row["brief_id"], row["seed"]
        if key in indexed:
            raise ValueError(f"Duplicate measurement {key}")
        indexed[key] = row
    result = {}
    for group, panel in (("overall", False), ("cross_panel", True)):
        ids = [row["id"] for row in briefs if not panel or row["continuous"]]
        if not ids:
            continue
        stats = {"planned_n": len(ids), "metrics": {}}
        for metric, label in metrics:
            samples = {}
            descriptions = {}
            for method in methods:
                description, samples[method] = _aggregate(indexed, method, ids, seeds, metric, len(ids))
                descriptions[method] = description
            pairs = {}
            if phase == "full" and "v2" in samples:
                for method in methods:
                    if method != "v2":
                        pairs[method] = _paired(samples["v2"], samples[method])
            stats["metrics"][metric] = {"label": label, "descriptive": descriptions, "paired": pairs}
        result[group] = stats
    return result


def build_summary(rows, briefs, methods, seeds, phase, physical=False):
    selected = list(briefs[10:] if phase == "full" else briefs[:5 if phase == "pilot" else 10])
    population = {brief["id"] for brief in selected}
    variants = {}
    for variant in ("raw", "preflight"):
        table_metrics = (RAW_ONLY, *MAIN) if variant == "raw" else MAIN
        if physical:
            table_metrics = (*table_metrics, STRIPE, *LID)
        variants[variant] = _variant(
            [row for row in rows if row["variant"] == variant and row["brief_id"] in population],
            selected, methods, seeds, phase, (*table_metrics, *SUPPLEMENT),
        )
    return {"phase": phase, "seeds": list(seeds), "physical": physical, "variants": variants}


def _tables(metrics):
    lines = ["| 指标 | 方法 | 完整简报数 | 均值 | 95% CI |",
             "| --- | --- | ---: | ---: | --- |"]
    for metric in metrics:
        for method, entry in metric["descriptive"].items():
            lines.append(f"| {metric['label']} | {method} | {entry['n']} |"
                         f" {_number(entry['mean'])} | {_ci(entry['ci95'])} |")
    if any(metric["paired"] for metric in metrics):
        lines += ["", "| 指标 | 对照方法 | 配对简报数 | v2 减对照均值 | 双侧 Wilcoxon p |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for metric in metrics:
            for method, entry in metric["paired"].items():
                lines.append(f"| {metric['label']} | {method} | {entry['n']} |"
                             f" {_number(entry['mean_difference'])} | {_number(entry['p_wilcoxon'])} |")
    return lines


def format_tables(value):
    phases = {"pilot": "前 5 条试跑", "tune": "前 10 条调参", "full": "留出集"}
    groups = {"overall": "全部简报", "cross_panel": "跨面连续子集"}
    lines = [f"# E01b {phases[value['phase']]}统计", "",
             "正式统计采用 E01-011 至 E01-050，每条简报先平均三个种子。95% CI 使用简报均值的 t 区间。"
             "Wilcoxon 检验以共同具备数据的简报配对，报告双侧未校正 p 值。"]
    if value["phase"] != "full":
        lines += ["", "本阶段用于验收或调参，显著性检验保留给正式留出集。"]
    for variant, subsets in value["variants"].items():
        lines.extend(("", f"## {variant}", ""))
        for subset, result in subsets.items():
            lines.extend((f"### {groups[subset]}", "", f"计划简报 {result['planned_n']} 份。", ""))
            metrics = result["metrics"]
            names = [key for key, _ in MAIN]
            if value.get("physical"):
                names.insert(2, STRIPE[0])
            if variant == "raw":
                names.insert(0, RAW_ONLY[0])
            lines.extend(_tables([metrics[name] for name in names]))
            lines.extend(("", "#### 补充指标", "",
                          "大面积色彩变化可能触发亮度梯度。CLIP 方向判断也会漏检倒置。", ""))
            lines.extend(_tables([metrics[name] for name, _ in SUPPLEMENT]))
            if value.get("physical"):
                lines.extend(("", "#### 顶面与底面", "",
                              "各盖面的接缝覆盖其四条三维邻边。接缝按物理尺度计算，"
                              "正立率仍基于各原图裁面。背面—底面指数针对实际压痕。", ""))
                lines.extend(_tables([metrics[name] for name, _ in LID]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_examples(experiment, phase):
    from PIL import Image, ImageDraw, ImageFont
    from .render import render_views

    first = 10 if phase == "full" else 0
    destination = experiment.run / phase / "examples"
    for brief in experiment.briefs[first:first + 3]:
        name = f"{brief['id']}-42"
        old = experiment.legacy_outputs / "ours" / f"{name}.png"
        new = experiment.run / "v2" / f"{name}.png"
        if not old.is_file() or not new.is_file():
            continue
        directory = destination / name
        directory.mkdir(parents=True, exist_ok=True)
        for method, geometry, source in (("v1", experiment.legacy_geometry, old),
                                          ("v2", experiment.geometry, new)):
            render_views(geometry, source, directory / method, 440, ["iso"])
        panel = Image.new("RGB", (980, 1000), "white")
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
        for row, method, atlas in ((0, "v1", old), (1, "v2", new)):
            draw.text((24, 500 * row + 10), f"{brief['id']} · seed 42 · {method}", fill="black", font=font)
            for column, path in enumerate((atlas, directory / method / "folded-iso.png")):
                with Image.open(path) as source:
                    texture = source.convert("RGB")
                    texture.thumbnail((450, 440), Image.Resampling.LANCZOS)
                    x = column * 490 + (490 - texture.width) // 2
                    y = row * 500 + 49 + (440 - texture.height) // 2
                    panel.paste(texture, (x, y))
        output = destination / f"v1-v2-{name}.jpg"
        panel.save(output, quality=87, optimize=True)


def render_native_fold_reviews(experiment, destination):
    """Show every real atlas crease at one source pixel per output pixel."""
    from PIL import Image, ImageDraw, ImageFont

    from .metrics.physical import _edges_on_sheet

    topology = read_json(experiment.geometry / "topology.json")
    faces = topology["faces"]
    creases = []
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        endpoints = _edges_on_sheet(faces[a["face"]]["rect"], a["edge"])
        other = _edges_on_sheet(faces[b["face"]]["rect"], b["edge"])
        if frozenset(endpoints) == frozenset(other):
            creases.append((a["face"], b["face"], endpoints))

    font = ImageFont.load_default(size=16)
    ink, tick = (29, 43, 56), (25, 149, 162)
    radius = 38
    longest = max(max(abs(x1 - x0), abs(y1 - y0)) for _, _, ((x0, y0), (x1, y1)) in creases)
    for brief in experiment.briefs[:5]:
        path = experiment.pilot_run / "v2" / f"{brief['id']}-42.png"
        if not path.is_file():
            continue
        with Image.open(path) as source:
            atlas = source.convert("RGB")
        panel = Image.new("RGB", (longest + 40, 60 + (radius * 2 + 26) * len(creases)), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((20, 11), f"{brief['id']} / seed 42 / Native crease crops", fill=ink, font=font)
        for index, (a, b, ((x0, y0), (x1, y1))) in enumerate(creases):
            y = 54 + index * (radius * 2 + 26)
            caption = f"{a} / {b}"
            if y0 == y1:
                crop = atlas.crop((x0, y0 - radius, x1, y0 + radius))
            else:
                crop = atlas.crop((x0 - radius, y0, x0 + radius, y1)).transpose(Image.Transpose.ROTATE_90)
                caption += " (90-degree rotation for viewing)"
            draw.text((20, y - 20), caption, font=font, fill=ink)
            panel.paste(crop, (20, y))
            for x in (7, 25 + crop.width):
                draw.line((x, y + radius, x + 8, y + radius), fill=tick, width=2)
        panel.save(destination / f"{brief['id']}-edges.png", optimize=True)


def render_edge_reviews(experiment):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    destination = experiment.pilot_run / "edge-review"
    destination.mkdir(parents=True, exist_ok=True)
    if experiment.bottom_frame:
        render_native_fold_reviews(experiment, destination)
        return
    with Image.open(experiment.geometry / "region_map.png") as labels:
        active = np.asarray(labels) > 0
    ys, xs = np.where(active)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    centers = []
    for y in (y0, y1):
        band = np.where(active[max(0, y - 2):y + 3])[1]
        centers.append((int(np.median(band)), int(y)))
    for x in (x0, x1):
        band = np.where(active[:, max(0, x - 2):x + 3])[0]
        centers.append((int(x), int(np.median(band))))
    # Also expose the body seams and lid folds at native resolution.
    faces = read_json(experiment.geometry / "topology.json")["faces"]
    left, top, right, bottom = faces["F"]["rect"]
    centers.extend(((left, (top + bottom) // 2), (right, (top + bottom) // 2)))
    for name in ("T", "Btm"):
        left, top, right, bottom = faces[name]["rect"]
        centers.append(((left + right) // 2, bottom if name == "T" else top))
    size = 208
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    for brief in experiment.briefs[:5]:
        path = experiment.pilot_run / "v2" / f"{brief['id']}-42.png"
        if not path.is_file():
            continue
        with Image.open(path) as source:
            atlas = source.convert("RGB")
            panel = Image.new("RGB", (860, 472), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((12, 8), f"{brief['id']} · seed 42 · v2 · edge crops at 1:1 scale", fill="black", font=font)
            for index, (x, y) in enumerate(centers):
                left = min(max(0, x - size // 2), atlas.width - size)
                top = min(max(0, y - size // 2), atlas.height - size)
                panel.paste(atlas.crop((left, top, left + size, top + size)),
                            (6 + 214 * (index % 4), 42 + 214 * (index // 4)))
            panel.save(destination / f"{brief['id']}-edges.png")
