"""Run the frozen E02 Step 1 on the registered ten briefs, with no holdout generation."""

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from PIL import Image
import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_freeze import METRICS_CONFIG, verify
from .e02_history import read_briefs
from .e02_sources import SourceAtlas, sources


DEFAULT_CONFIG = Path("code/configs/E02_step1.yaml")
RING_METHODS = ("v3_1280", "v3_1472")
CORE_NEW = ("b2_regional", *RING_METHODS)
SHARED = "v3_shared"
BENCHMARK_BASELINES = ("b2", "b2_post", "b2_regional", "b3", "b4")
SIZES = {"v3_1280": [1280, 1024], "v3_1472": [1472, 1152]}
CALIBRATED_METRICS = ("clip_score", "image_reward", "hps_v2", "non_native_7_sdr",
                      "non_native_7_lpips", "m4_repeat_score")
VARIANTS = ("raw", "preflight")
GPU_METHODS = (*CORE_NEW, SHARED)


@dataclass(frozen=True)
class Sample:
    atlas: SourceAtlas
    score: Path
    preflight: Path

    @property
    def generation(self):
        return self.atlas.method in GPU_METHODS

    def validate_generation(self):
        atlas = self.atlas
        if not atlas.path.is_file() or not atlas.metadata.is_file():
            return False
        record = read_json(atlas.metadata)
        return (record["brief_id"] == atlas.brief["id"] and record["seed"] == atlas.seed
                and record["method"] == atlas.method and record["raw_sha256"] == file_hash(atlas.path))


class Experiment:
    def __init__(self, root, path):
        self.root = root.resolve()
        self.config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (self.config["protocol"] != "e02-v3-step1" or self.config["seed"] != 42
                or self.config["ring_sizes"] != SIZES or self.config["common"]["steps"] != 30):
            raise ValueError("E02 Step 1 is registered to seed 42, 30 steps and the two consensus ring sizes.")
        self.metric_config, self.dino_model, freeze = verify(self.root)
        self.briefs = read_briefs()
        self.legacy = self.root / "outputs/geometry"
        self.geometry = self.root / "outputs/e01b/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        payload = {
            "generation_config": self.config,
            "briefs": stable_hash(self.briefs[:10]),
            "metrics": stable_hash(freeze),
            "geometry": {str(g.name): file_hash(g) for g in sorted(self.geometry.glob("*"))
                         if g.is_file() and g.name in ("topology.json", "region_map.png", "bleed_owner.png")},
            "lut_sha256": file_hash(self.lut),
            "generation_implementation": {
                name: file_hash(Path("code/dielinediff/methods") / name)
                for name in ("v3.py", "v3_geometry.py", "v3_circular.py", "v3_attention.py",
                             "ours_tf.py", "sdxl.py", "b2_controlnet.py")
            },
        }
        self.output = self.root / "outputs/e02/step1" / stable_hash(payload)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "inputs.json", payload)
        self.selection_path = self.output / "resolution-selection.json"
        self.chosen_path = self.output / "candidate-selection.json"

    def selection(self):
        if not self.selection_path.is_file():
            raise ValueError("Measure and select both registered ring resolutions first.")
        record = read_json(self.selection_path)
        if record["selected"] not in RING_METHODS:
            raise ValueError("Invalid E02 resolution selection")
        scores = list(self.samples(methods=RING_METHODS))
        actual = {j.atlas.method + "/" + j.atlas.identifier: file_hash(j.score) for j in scores if j.score.is_file()}
        if record["scores_sha256"] != actual:
            raise ValueError("The resolution selection no longer matches both score sets")
        return record["selected"]

    def ring_size(self, name):
        if name == SHARED:
            name = self.selection()
        return self.config["ring_sizes"].get(name, SIZES[RING_METHODS[0]])

    def samples(self, *, methods=None, include_shared=False):
        selected = methods if methods is not None else (*CORE_NEW, *((SHARED,) if include_shared else ()))
        known = set(selected)
        historical = methods is None or bool(known - set(GPU_METHODS))
        if historical:
            for atlas in sources(self.root, self.briefs, trial_only=True):
                if methods is None or atlas.method in known:
                    yield self._sample(atlas)
        for method in selected:
            if method not in GPU_METHODS:
                continue
            for brief in self.briefs[:10]:
                path = self.output / method / f"{brief['id']}-42.png"
                yield self._sample(SourceAtlas(method, brief, 42, path, self.geometry))

    def _sample(self, atlas):
        d = self.output / "evaluation" / atlas.method
        return Sample(atlas, d / f"{atlas.identifier}.metrics.json", d / f"{atlas.identifier}.preflight.png")


def generate(experiment, methods, device):
    import torch
    from .methods.v3 import DieLineDiffV3

    hardware = torch.device(device)
    if hardware.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Step 1 generation requires the NVIDIA GPU.")
    torch.cuda.set_device(hardware.index or 0)
    for method in methods:
        if method not in GPU_METHODS:
            raise ValueError(f"Cannot generate {method}")
        if method == SHARED and not experiment.config["shared_attention_candidate"]:
            raise ValueError("Shared attention candidate is disabled")
        jobs = [job for job in experiment.samples(methods=(method,)) if not job.validate_generation()]
        if not jobs:
            continue
        config = {**experiment.config["common"], "share_attention": method == SHARED}
        engine = DieLineDiffV3(experiment.root / "cache/models", experiment.geometry, config,
                               device=device, lut_path=experiment.lut)
        ring_size = experiment.ring_size(method)
        for job in jobs:
            atlas = job.atlas
            atlas.path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            image = engine.generate(atlas.brief, atlas.seed, method="b2_regional" if method == "b2_regional" else "v3",
                                    ring_size=ring_size).convert("RGB")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
            expected = tuple(engine.last_run["atlas_size_px"])
            if image.size != expected:
                raise ValueError(f"Atlas {image.size} does not match geometry {expected}")
            temp = atlas.path.with_suffix(f".{os.getpid()}.tmp.png")
            image.save(temp)
            temp.replace(atlas.path)
            write_json(atlas.metadata, {
                "method": method, "brief_id": atlas.brief["id"], "seed": atlas.seed,
                "elapsed_seconds": seconds, "peak_memory_mib": peak,
                "gpu": torch.cuda.get_device_name(), "raw_sha256": file_hash(atlas.path),
                "details": engine.last_run,
            })
            print(f"{method} {atlas.identifier}: {seconds:.2f}s {peak:.0f} MiB", flush=True)
        del engine
        torch.cuda.empty_cache()


def score_input(job):
    atlas = job.atlas
    return {
        "atlas_sha256": file_hash(atlas.path),
        "run_sha256": file_hash(atlas.metadata),
        "geometry_sha256": file_hash(atlas.geometry / "topology.json"),
    }


def measure(experiment, jobs, device):
    import numpy as np

    from .methods.sdxl import prompt_for
    from .metrics.e01b import E01bEvaluator, preflight_atlas
    from .metrics.e02_physical import E02PhysicalEvaluator
    from .metrics.e02_repetition import DinoRepetition

    pending = []
    for job in jobs:
        atlas = job.atlas
        if not atlas.path.is_file() or not atlas.metadata.is_file() or (job.generation and not job.validate_generation()):
            raise FileNotFoundError(f"Missing or invalid atlas of {atlas.method}/{atlas.identifier}")
        if (job.score.is_file() and job.preflight.is_file()
                and read_json(job.score)["input"] == score_input(job)):
            continue
        pending.append(job)
    if not pending:
        return

    repetition = DinoRepetition(experiment.dino_model, experiment.metric_config["m4"], device=device)
    geometries = {"legacy": experiment.legacy, "new": experiment.geometry}
    topologies = {path: read_json(path / "topology.json") for path in geometries.values()}
    with E01bEvaluator(experiment.legacy, experiment.geometry, experiment.lut, device=device) as evaluator:
        evaluator.physical = {
            key: E02PhysicalEvaluator(path, device=device, learned=evaluator.evaluators[key].learned,
                                      config=experiment.metric_config["physical"])
            for key, path in geometries.items()
        }
        for block in range(0, len(pending), 6):
            batch = pending[block:block + 6]
            atlases = []
            for job in batch:
                with Image.open(job.atlas.path) as source:
                    atlases.append(source.convert("RGB"))
            additions = repetition.measure_batch([(rgb, topologies[job.atlas.geometry])
                                                for job, rgb in zip(batch, atlases, strict=True)])
            for job, rgb, extra in zip(batch, atlases, additions, strict=True):
                atlas = job.atlas
                meta = read_json(atlas.metadata)
                key = "legacy" if atlas.geometry == experiment.legacy else "new"
                record = evaluator.measure(rgb, atlas.brief, geometry=key, prompt=prompt_for(atlas.brief),
                                           elapsed_seconds=meta["elapsed_seconds"], peak_memory_mib=meta["peak_memory_mib"])
                for variant in VARIANTS:
                    record[variant].update(extra)
                preflight = preflight_atlas(np.asarray(rgb), evaluator.evaluators[key].labels, atlas.brief["main_color"])
                job.preflight.parent.mkdir(parents=True, exist_ok=True)
                temp = job.preflight.with_suffix(f".{os.getpid()}.tmp.png")
                Image.fromarray(preflight).save(temp)
                temp.replace(job.preflight)
                write_json(job.score, {"input": score_input(job), "variants": record})
                print(f"Measured {atlas.method} {atlas.identifier}", flush=True)


def grouped(jobs):
    groups = {}
    for job in jobs:
        if not job.score.is_file():
            raise FileNotFoundError(f"Missing metrics {job.score}")
        record = read_json(job.score)
        if record["input"] != score_input(job):
            raise ValueError(f"Stale measurement {job.score}")
        groups.setdefault(job.atlas.method, []).append((job, record["variants"]))
    return groups


def average(samples, metric, variant="preflight"):
    values = [item[1][variant].get(metric) for item in samples]
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def compare_candidates(groups, names):
    """Select by six predeclared, equally weighted directional ranks."""
    if any(len(groups[name]) != 10 for name in names):
        raise ValueError("Ten registered, measured briefs are required per candidate.")
    metrics = {
        method: {key: average(groups[method], key) for key in CALIBRATED_METRICS}
        for method in names
    }
    # The first three scores reward quality; the last three reward lower seam/repetition.
    ranks = {m: 0.0 for m in names}
    votes = {}
    for i, key in enumerate(CALIBRATED_METRICS):
        direction = 1 if i < 3 else -1
        scores = {m: direction * metrics[m][key] for m in names}
        votes[key] = [m for m, n in scores.items() if n == max(scores.values())]
        for m in votes[key]:
            ranks[m] += 1 / len(votes[key])
    selected = min(names, key=lambda name: (-ranks[name], average(groups[name], "elapsed_seconds"), names.index(name)))
    return {"selected": selected, "means": metrics, "wins": votes, "scores": ranks}


def select_resolution(experiment):
    jobs = list(experiment.samples(methods=RING_METHODS))
    result = compare_candidates(grouped(jobs), RING_METHODS)
    result["scores_sha256"] = {j.atlas.method + "/" + j.atlas.identifier: file_hash(j.score) for j in jobs}
    write_json(experiment.selection_path, result)
    print(json.dumps(result, ensure_ascii=False))


def select_final(experiment):
    initial = experiment.selection()
    jobs = list(experiment.samples(methods=(initial, SHARED)))
    evaluated = all(job.score.is_file() for job in jobs)
    if evaluated:
        test = compare_candidates(grouped(jobs), (initial, SHARED))
        result = {"initial": initial, "shared": "measured", **test}
    else:
        shared = "disabled" if not experiment.config["shared_attention_candidate"] else "generation-unavailable"
        result = {"initial": initial, "shared": shared, "selected": initial}
    result["scores_sha256"] = {j.atlas.method + "/" + j.atlas.identifier: file_hash(j.score)
                               for j in jobs if j.score.is_file()}
    write_json(experiment.chosen_path, result)
    print(json.dumps(result, ensure_ascii=False))


def gate(experiment, groups, chosen, visual):
    baselines = BENCHMARK_BASELINES
    current = [chosen, *baselines]
    if any(len(groups[method]) != 10 for method in current):
        raise ValueError("Gate samples must be the full set of ten registered briefs.")
    config = experiment.config["gate"]

    def value(method, key):
        values = [scores["preflight"].get(key) for _, scores in groups[method]]
        if len(values) != 10 or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(f"E02 gate requires ten finite {key} values for {method}")
        return sum(values) / 10

    checks = {}
    for tag, keys, toward_higher in (
        ("seven_folded_seams", ("non_native_7_sdr", "non_native_7_lpips"), False),
        ("bleed", ("bleed_fill_pct", "bleed_fill_v2_pct"), True),
    ):
        entries = {}
        for key in keys:
            measured = value(chosen, key)
            comparison = {method: value(method, key) for method in ("b2_regional", "b2_post")}
            better = all(measured > n if toward_higher else measured < n for n in comparison.values())
            entries[key] = {"v3": measured, "baselines": comparison, "passed": better}
        checks[tag] = {"passed": all(e["passed"] for e in entries.values()), "metrics": entries}

    stripe_means = {m: value(m, "stripe_index") for m in (chosen, *baselines)}
    bottom = min(baselines, key=stripe_means.__getitem__)
    ceiling = stripe_means[bottom] * config["stripe_factor"]
    clips = {m: value(m, "clip_score") for m in (chosen, "b2")}
    clip_floor = clips["b2"] - config["clip_margin"]
    secs = value(chosen, "elapsed_seconds")
    peak = value(chosen, "peak_memory_mib")
    fingerprint = stable_hash({
        "method": chosen,
        "samples": {job.atlas.brief["id"]: [file_hash(job.atlas.path), file_hash(job.score), file_hash(job.preflight)]
                    for job, _ in groups[chosen]},
    })
    review_matches = isinstance(visual, dict) and visual.get("fingerprint") == fingerprint
    reviews = visual.get("samples", {}) if review_matches else {}
    planned = {job.atlas.brief["id"] for job, _ in groups[chosen]}
    valid = (set(reviews) == planned and all(isinstance(entry, dict)
             and isinstance(entry.get("visible_artifact"), bool) and isinstance(entry.get("notes"), str)
             and entry["notes"].strip()
             for entry in reviews.values()))
    visual_ok = valid and all(not v["visible_artifact"] for v in reviews.values())
    checks["fold_stripe"] = {"passed": stripe_means[chosen] <= ceiling and visual_ok,
                             "objective_passed": stripe_means[chosen] <= ceiling, "baseline": bottom,
                             "threshold": ceiling, "means": stripe_means, "visual_passed": visual_ok,
                             "ready_for_visual_review": True}
    checks["clip"] = {"passed": clips[chosen] >= clip_floor, "means": clips, "minimum": clip_floor}
    checks["runtime"] = {"passed": secs <= config["seconds"] and peak <= config["peak_memory_mib"],
                          "mean_seconds": secs, "mean_peak_mib": peak,
                          "highest_sample_peak_mib": max(s["preflight"]["peak_memory_mib"] for _, s in groups[chosen]),
                          "seconds_limit": config["seconds"], "peak_mib_limit": config["peak_memory_mib"]}
    return {"selected": chosen, "n": 10, "fingerprint": fingerprint, "checks": checks,
            "visual_review": visual if valid else None, "passed": all(item["passed"] for item in checks.values())}


def summarize(experiment):
    chosen = read_json(experiment.chosen_path)["selected"]
    available = all(job.score.is_file() for job in experiment.samples(methods=(SHARED,)))
    jobs = list(experiment.samples(include_shared=available))
    groups = grouped(jobs)
    rows = [
        {"method": job.atlas.method, "variant": variant, "brief_id": job.atlas.brief["id"],
         "seed": job.atlas.seed, "continuous": job.atlas.brief["continuous"], **scores[variant]}
        for values in groups.values() for job, scores in values for variant in VARIANTS
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    results = experiment.output / "results.csv"
    temporary = results.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results)
    metrics = fields[5:]
    variants = {
        v: {meth: {"n": len(entries), **{key: average(entries, key, v) for key in metrics}}
            for meth, entries in groups.items()} for v in VARIANTS
    }
    cross_panel = {
        v: {meth: {"n": len(entries), **{key: average(entries, key, v) for key in metrics}}
            for meth, group in groups.items()
            if (entries := [pair for pair in group if pair[0].atlas.brief["continuous"]])}
        for v in VARIANTS
    }
    write_json(experiment.output / "statistics.json", {"variants": variants, "cross_panel": cross_panel})
    review_path = experiment.output / "visual-review.json"
    review = read_json(review_path) if review_path.is_file() else None
    from .e02_visual import render_examples, render_native_creases

    result = gate(experiment, groups, chosen, review)
    write_json(experiment.output / "gate.json", result)
    render_examples(experiment, chosen)
    render_native_creases(experiment, chosen)
    print(json.dumps(result, ensure_ascii=False))


def record_review(experiment, path):
    current = read_json(experiment.output / "gate.json")
    document = read_json(path)
    visual = document.get("visual_review", document) if isinstance(document, dict) else None
    names = {brief["id"] for brief in experiment.briefs[:10]}
    samples = visual.get("samples", {}) if isinstance(visual, dict) else {}
    if (set(samples) != names or visual.get("fingerprint") != current["fingerprint"]
            or any(not isinstance(item, dict) or not isinstance(item.get("visible_artifact"), bool)
                   or not isinstance(item.get("notes"), str) or not item["notes"].strip()
                   for item in samples.values())):
        raise ValueError("Record ten explicit native-fold observations for the current gate fingerprint.")
    write_json(experiment.output / "visual-review.json", visual)
    summarize(experiment)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("paths", "generate", "measure", "select-resolution", "select-final",
                                       "summarize", "review"))
    cli.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--review-file", type=Path)
    cli.add_argument("--methods", nargs="+", choices=(*GPU_METHODS, "b2", "b2_post", "b3", "b4", "v1", "v2"))
    args = cli.parse_args()
    if args.root is None:
        cli.error("source code/scripts/e01_env.sh or provide --root")
    if args.action == "review" and args.review_file is None:
        cli.error("review requires --review-file")
    experiment = Experiment(args.root, args.config)
    if args.action == "paths":
        print(json.dumps({"run": str(experiment.output), "config": str(args.config), "metrics": str(METRICS_CONFIG)}))
    elif args.action == "generate":
        generate(experiment, args.methods or CORE_NEW, args.device)
    elif args.action == "measure":
        measure(experiment, list(experiment.samples(methods=args.methods)), args.device)
    elif args.action == "select-resolution":
        select_resolution(experiment)
    elif args.action == "select-final":
        select_final(experiment)
    elif args.action == "review":
        record_review(experiment, args.review_file)
    else:
        summarize(experiment)


if __name__ == "__main__":
    main()
