"""Re-evaluate archived E01 and task-03 atlases under the crease-fix protocol."""

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from .e01 import read_json, validate_briefs, write_json
from .e01b import (GEOMETRY_FILES, LEARNED, file_hash, physical_implementation_hash, stable_hash)
from .methods.sdxl import prompt_for


E01_METHODS = {
    "b2": "b2", "b3": "b3", "b4": "b4", "v1": "ours",
    "v1_no_sync": "no_sync", "v1_no_projection": "no_projection",
}
SEEDS = (42, 1234, 2026)
TASK03_RUN = "4b57f6e33a53"
TASK03_LEGACY_CACHE = "553a10692ea7"


@dataclass(frozen=True)
class ArchivedAtlas:
    experiment: str
    method: str
    brief: dict
    seed: int
    image: Path
    previous: Path | None

    @property
    def key(self):
        return self.experiment, self.method, self.brief["id"], self.seed

    @property
    def geometry(self):
        return "legacy" if self.experiment == "E01" else "new"

    @property
    def metadata(self):
        return self.image.with_suffix(".run.json")

    @property
    def immutable_input(self):
        return {"atlas_sha256": file_hash(self.image), "run_sha256": file_hash(self.metadata),
                "learned": list(LEARNED)}


def atlas_catalog(root, briefs, legacy_outputs):
    archived = root / "outputs" / "e01b"
    prior = archived / "legacy-metrics" / TASK03_LEGACY_CACHE
    items = []
    for method, directory in E01_METHODS.items():
        for brief in briefs:
            for seed in SEEDS:
                name = f"{brief['id']}-{seed}"
                image = legacy_outputs / directory / (name + ".png")
                cached = prior / method / (name + ".metrics.json")
                items.append(ArchivedAtlas("E01", method, brief, seed, image,
                                           cached if cached.is_file() else None))
    for brief in briefs[:5]:
        image = archived / "runs" / TASK03_RUN / "v2" / f"{brief['id']}-42.png"
        items.append(ArchivedAtlas("E01b-task03", "v2", brief, 42, image,
                                   image.with_suffix(".metrics.json")))
    # The archived pilot overlap is needed first for the new v2 comparison.
    return sorted(items, key=lambda item: (item.previous is None, item.key))


def output_record(output, stage, item):
    return output / stage / item.experiment / item.method / (item.image.stem + ".json")


def reuse_prior(item, provenance):
    if item.previous is None or not item.previous.is_file():
        return None
    prior = read_json(item.previous)
    if prior.get("input") != provenance:
        raise ValueError(f"Task 03 archive does not match its saved fingerprints: {item.image}")
    return prior["variants"]


def reward_backfill(items, output, legacy_geometry, new_geometry, lut):
    from .metrics import Evaluator, face_prompts, rgb_atlas
    from .metrics.image_reward import ImageRewardScorer

    shape = {name: Evaluator(path, lut, device="cpu", learned=())
             for name, path in (("legacy", legacy_geometry), ("new", new_geometry))}
    scorer = None
    written = reused = 0
    for index, item in enumerate(items, 1):
        destination = output_record(output, "image-reward", item)
        provenance = item.immutable_input
        if destination.is_file() and read_json(destination).get("input") == provenance:
            reused += 1
            continue
        prior = reuse_prior(item, provenance)
        score = prior["raw"]["image_reward"] if prior else None
        if score is None:
            if scorer is None:
                scorer = ImageRewardScorer(device="cpu")
            evaluator = shape[item.geometry]
            images = evaluator._upright_images(evaluator._crop_faces(rgb_atlas(item.image)))
            score = scorer.score(images, face_prompts(prompt_for(item.brief)))
            source = "measured"
            written += 1
        else:
            source = "task03-matching-atlas-sha256"
            reused += 1
        if not math.isfinite(score):
            raise ValueError(f"Invalid ImageReward for {item.image}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, {"input": provenance, "image_reward": score, "source": source})
        if index % 10 == 0 or index == len(items):
            print(f"ImageReward {index}/{len(items)}; newly measured {written}; reused {reused}", flush=True)
    print(f"ImageReward finished {len(items)}/{len(items)}; newly measured {written}; reused {reused}", flush=True)


def physical_backfill(items, output, legacy_geometry, new_geometry):
    from .metrics import DEFAULT_THRESHOLDS, rgb_atlas
    from .metrics.learned import LearnedMetrics
    from .metrics.physical import PhysicalEvaluator

    implementation = physical_implementation_hash()
    learned = LearnedMetrics("cpu", DEFAULT_THRESHOLDS["ocr_confidence"])
    evaluators = {method: PhysicalEvaluator(path, device="cpu", learned=learned)
                  for method, path in (("legacy", legacy_geometry), ("new", new_geometry))}
    measured = reused = 0
    for index, item in enumerate(items, 1):
        destination = output_record(output, "physical", item)
        provenance = {**item.immutable_input, "physical_implementation_sha256": implementation}
        if destination.is_file() and read_json(destination).get("input") == provenance:
            reused += 1
        else:
            atlas = rgb_atlas(item.image)
            evaluator = evaluators[item.geometry]
            metrics = {**evaluator.measure(atlas), **evaluator.top_bottom_orientation(atlas)}
            if any(value is None or not math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Incomplete millimetre-scale metrics for {item.image}: {metrics!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_json(destination, {"input": provenance, "metrics": metrics})
            measured += 1
        if index % 10 == 0 or index == len(items):
            print(f"Physical {index}/{len(items)}; newly measured {measured}; resumed {reused}", flush=True)


NATIVE_FIELDS = (
    "upright_rate", "glue_ink_pct", "flap_edge_pct", "lighting_residual", "oog_pct",
    "roundtrip_delta_e00", "clip_score", "hps_v2", "elapsed_seconds", "peak_memory_mib",
)
FACE_QUALITY = ("upright_rate", "clip_score", "hps_v2")


def canonical_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (row["method"], row["brief_id"], int(row["seed"])): row
            for row in csv.DictReader(stream)
        }


def check_archived_native(item, raw, tracked):
    if item.image.stat().st_mtime_ns > item.image.with_suffix(".metrics.json").stat().st_mtime_ns:
        return False
    if not {"clip", "hps"}.issubset(raw.get("learned_backends", ())):
        return False
    if any(float(raw[name]) != float(tracked[name]) for name in NATIVE_FIELDS):
        return False
    run = read_json(item.metadata)
    return all(float(raw[name]) == float(run[name]) for name in ("elapsed_seconds", "peak_memory_mib"))


def archived_native_variants(item, evaluator, tracked):
    from .metrics import rgb_atlas
    from .metrics.e01b import preflight_atlas

    source = rgb_atlas(item.image)
    prepared = preflight_atlas(source, evaluator.labels, item.brief["main_color"])
    prompt = prompt_for(item.brief)
    costs = read_json(item.metadata)
    costs = {name: costs[name] for name in ("elapsed_seconds", "peak_memory_mib")}
    previous = read_json(item.image.with_suffix(".metrics.json"))
    if check_archived_native(item, previous, tracked):
        raw = {name: previous[name] for name in NATIVE_FIELDS}
        origin = "E01-tracked-native-scores"
    else:
        fallback = evaluator.measure(source, prompt, learned=("clip", "hps"), **costs)
        raw = {name: fallback[name] for name in NATIVE_FIELDS}
        origin = "CPU-native-rescore"
    measured = evaluator.measure(prepared, prompt, learned=(), **costs)
    preflight = {name: measured[name] for name in NATIVE_FIELDS if name != "glue_ink_pct"}
    preflight.update({name: raw[name] for name in FACE_QUALITY})
    return {"raw": raw, "preflight": preflight}, origin


def legacy_cache(output, briefs_path, old_geometry, legacy_outputs, lut):
    inputs = {"briefs_sha256": file_hash(briefs_path),
              "legacy_geometry": {name: file_hash(old_geometry / name) for name in GEOMETRY_FILES},
              "legacy_outputs": str(legacy_outputs), "lut_sha256": file_hash(lut)}
    return output.parent / "legacy-metrics" / stable_hash(inputs)[:12]


def assemble(items, output, legacy_geometry, lut, tracked_csv, shared):
    import numpy as np
    from .metrics import Evaluator
    from .metrics.e01b import RESET_LABELS

    tracked = canonical_rows(tracked_csv)
    geometry = Evaluator(legacy_geometry, lut, device="cpu", learned=())
    if any(mask.any() for mask in geometry._crop_faces(np.isin(geometry.labels, RESET_LABELS)).values()):
        raise ValueError("Preflight overlaps a crop whose native scores would be reused")
    implementation = physical_implementation_hash()
    complete = 0
    for item in items:
        ph_path = output_record(output, "physical", item)
        ir_path = output_record(output, "image-reward", item)
        if not ph_path.is_file() or not ir_path.is_file():
            continue
        provenance = item.immutable_input
        versioned = {**provenance, "physical_implementation_sha256": implementation}
        physical = read_json(ph_path)
        reward = read_json(ir_path)
        if physical["input"] != versioned or reward["input"] != provenance:
            raise ValueError(f"Stale backfill intermediate: {item.image}")
        assembled = output_record(output, "evaluations", item)
        existing = read_json(assembled) if assembled.is_file() else None
        if existing is not None and existing["input"] == versioned:
            result = existing
        else:
            prior = reuse_prior(item, provenance)
            if prior is not None:
                scores = prior
                origin = "task03-matching-atlas-sha256"
            else:
                key = E01_METHODS[item.method], item.brief["id"], item.seed
                scores, origin = archived_native_variants(item, geometry, tracked[key])
            for values in scores.values():
                values.update(physical["metrics"])
                values["image_reward"] = reward["image_reward"]
            result = {"input": versioned, "variants": scores,
                      "native_source": origin, "image_reward_source": reward["source"]}
            assembled.parent.mkdir(parents=True, exist_ok=True)
            write_json(assembled, result)
        if item.experiment == "E01" and item.method in ("b2", "b3", "b4", "v1"):
            destination = shared / item.method / (item.image.stem + ".metrics.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or read_json(destination).get("input") != versioned:
                write_json(destination, result)
        complete += 1
        if complete % 20 == 0:
            print(f"Assembled {complete}/{len(items)} old atlases", flush=True)
    print(f"Assembled {complete}/{len(items)} old atlases", flush=True)


SCALE_FIELDS = ("seam_sdr", "seam_lpips", "bleed_fill_pct")
QUALITY_FIELDS = ("clip_score", "hps_v2", "image_reward")


def save_csv(destination, rows, fields):
    temp = destination.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(destination)


def backfill_rows(items, output):
    measurement = {}
    for item in items:
        path = output_record(output, "evaluations", item)
        if not path.is_file():
            raise ValueError(f"Missing assembled evaluation {path}")
        record = read_json(path)
        versioned = {**item.immutable_input, "physical_implementation_sha256": physical_implementation_hash()}
        if record["input"] != versioned:
            raise ValueError(f"Backfill record no longer matches its input and metric version: {path}")
        measurement[item.key] = record
    return measurement


def metric_rows(items, records):
    for item in items:
        record = records[item.key]
        for variant, values in record["variants"].items():
            brief = item.brief
            yield {"experiment": item.experiment, "method": item.method, "variant": variant,
                   "brief_id": brief["id"], "seed": item.seed, "continuous": brief["continuous"],
                   "split": "holdout" if int(brief["id"].split("-")[-1]) > 10 else "tune",
                   "native_score_source": record["native_source"],
                   "image_reward_source": record["image_reward_source"], **values}


def scale_comparison(items, records):
    from collections import defaultdict
    from statistics import mean

    pair_groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        current = records[item.key]["variants"]
        brief_id = item.brief["id"]
        historical = None
        if item.experiment == "E01" and int(brief_id.split("-")[-1]) > 10:
            old = {"raw": read_json(item.image.with_suffix(".metrics.json"))}
            historical = "E01 holdout"
        if historical is not None:
            for metric in (*SCALE_FIELDS, *QUALITY_FIELDS):
                previous, proposed = old["raw"].get(metric), current["raw"].get(metric)
                if previous is not None and proposed is not None:
                    pair_groups[(historical, item.method, "raw", metric)][brief_id].append((previous, proposed))
        if item.previous is not None:
            old_variants = reuse_prior(item, item.immutable_input)
            for variant, previous in old_variants.items():
                for metric in (*SCALE_FIELDS, *QUALITY_FIELDS):
                    if previous.get(metric) is None or current[variant].get(metric) is None:
                        continue
                    pair_groups[("E01b task03 pilot", item.method, variant, metric)][brief_id].append(
                        (previous[metric], current[variant][metric]))
    rows = []
    for (cohort, method, variant, metric), samples in sorted(pair_groups.items()):
        expected = 3 if cohort == "E01 holdout" else 1
        if any(len(pairs) != expected for pairs in samples.values()):
            raise ValueError(f"Incomplete matched before/after seeds for {cohort}/{method}/{metric}")
        old = mean(mean(x for x, _ in pairs) for pairs in samples.values())
        new = mean(mean(y for _, y in pairs) for pairs in samples.values())
        rows.append({"cohort": cohort, "method": method, "variant": variant, "metric": metric,
                     "briefs": len(samples), "old_mean": old, "new_mean": new, "new_minus_old": new - old})
    return rows


def collect(items, briefs, output):
    from collections import Counter
    from .e01b_summary import METRIC_KEYS, build_summary, format_tables

    records = backfill_rows(items, output)
    rows = list(metric_rows(items, records))
    metadata = ("experiment", "method", "variant", "brief_id", "seed", "split", "continuous",
                "native_score_source", "image_reward_source")
    extra = sorted({key for row in rows for key in row} - set(metadata + METRIC_KEYS))
    save_csv(output / "results.csv", rows, metadata + METRIC_KEYS + tuple(extra))
    compare = scale_comparison(items, records)
    save_csv(output / "scale-comparison.csv", compare,
             ("cohort", "method", "variant", "metric", "briefs", "old_mean", "new_mean", "new_minus_old"))
    lines = ["# 旧口径与统一物理尺度对照", "",
             "接缝色差与 LPIPS 采用 0.5 mm 和 5 mm 窄带，出血 3 mm。"
             "物理评测使用 4 px/mm，图文质量沿用原图裁面。E01 留出集按简报平均三个种子。", "",
             "| 批次 | 方法 | 变体 | 指标 | 简报 | 旧均值 | 新均值 | 新−旧 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in compare:
        if row["metric"] not in SCALE_FIELDS:
            continue
        lines.append(f"| {row['cohort']} | {row['method']} | {row['variant']} | {row['metric']} |"
                     f" {row['briefs']} | {row['old_mean']:.5g} | {row['new_mean']:.5g} |"
                     f" {row['new_minus_old']:+.5g} |")
    (output / "SCALE-COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    e01 = build_summary([row for row in rows if row["experiment"] == "E01"],
                        briefs, tuple(E01_METHODS), SEEDS, "full", physical=True)
    task03_keys = {item.key for item in items if item.previous is not None}
    task03 = build_summary([
        row for row in rows if (row["experiment"], row["method"], row["brief_id"], row["seed"]) in task03_keys
    ], briefs, ("b2", "b3", "b4", "v1", "v2"), SEEDS[:1], "pilot", physical=True)
    for name, result in (("E01-PHYSICAL", e01), ("E01B-TASK03-PHYSICAL", task03)):
        write_json(output / (name + ".json"), result)
        (output / (name + ".md")).write_text(format_tables(result), encoding="utf-8")
    provenance = {
        "atlases": len(items), "variant_rows": len(rows),
        "native_score_sources": dict(Counter(record["native_source"] for record in records.values())),
        "image_reward_sources": dict(Counter(record["image_reward_source"] for record in records.values())),
        "physical_implementation_sha256": physical_implementation_hash(),
    }
    write_json(output / "provenance-summary.json", provenance)
    print(provenance, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reward", "physical", "assemble", "collect", "all"))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "DIELINEDIFF_ROOT", "/root/autodl-tmp/ai-design")))
    parser.add_argument("--briefs", type=Path, default=Path("data/briefs/E01_briefs.jsonl"))
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.stage in ("collect", "all") and (args.pilot_only or args.shards != 1):
        parser.error("collect and all require the complete catalog")
    root = args.root.resolve()
    briefs = [json.loads(line) for line in args.briefs.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_briefs(briefs)
    previous = root / "outputs" / "e01" / "full-52dd11c30fc1"
    output = root / "outputs" / "e01b" / "crease-fix" / "reanalysis"
    items = atlas_catalog(root, briefs, previous)
    if args.pilot_only:
        items = [item for item in items if item.previous is not None]
    items = items[args.shard::args.shards]
    old_geometry = root / "outputs" / "geometry"
    new_geometry = root / "outputs" / "e01b" / "geometry"
    lut = root / "cache" / "icc" / "iso-coated-v2-lut.npy"
    if args.stage in ("reward", "all"):
        reward_backfill(items, output, old_geometry, new_geometry, lut)
    if args.stage in ("physical", "all"):
        physical_backfill(items, output, old_geometry, new_geometry)
    if args.stage in ("assemble", "all"):
        shared = legacy_cache(output, args.briefs, old_geometry, previous, lut)
        assemble(items, output, old_geometry, lut, Path("docs/experiments/E01/results.csv"), shared)
    if args.stage in ("collect", "all"):
        collect(items, briefs, output)


if __name__ == "__main__":
    main()
