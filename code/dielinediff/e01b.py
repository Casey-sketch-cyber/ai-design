"""Run the E01b pilot, registered tuning, and frozen holdout without regenerating E01."""

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .e01 import read_json, validate_briefs, write_json


LEGACY_METHODS = {"b2": "b2", "b3": "b3", "b4": "b4", "v1": "ours"}
NEW_METHODS = ("v2", "no_sync", "no_projection")
METHODS = tuple(LEGACY_METHODS) + NEW_METHODS
LEARNED = ("lpips", "clip", "hps", "image_reward")
SEEDS = (42, 1234, 2026)
TUNABLE = {"window_stride", "top_band", "guidance", "projection_strength"}
GEOMETRY_FILES = ("topology.json", "region_map.png", "bleed_owner.png", "box.obj")
PHASES = {
    "pilot": (5, (*LEGACY_METHODS, "v2")),
    "tune": (10, ("v2",)),
    "full": (50, METHODS),
}
CLIP_TOLERANCE = 0.05
PHYSICAL_PROTOCOL = "e01b-v2-crease"
STRIPE_TOLERANCE_FACTOR = 1.20


def file_hash(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def config_payload(config):
    return {key: value for key, value in config.items() if key != "frozen"}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("The E01b configuration must be a YAML mapping")
    for key in ("steps", "seeds", "pixels_per_mm", "window_stride", "top_band",
                "guidance", "projection_strength"):
        if key not in config:
            raise ValueError(f"Missing E01b configuration value {key}")
    if tuple(config["seeds"]) != SEEDS or config["steps"] != 30:
        raise ValueError("E01b reuses the E01 seeds [42, 1234, 2026] and 30 sampling steps")
    if config["pixels_per_mm"] != 6.4:
        raise ValueError("E01b uses the fixed 6.4 px/mm pre-press geometry")
    stride = config["window_stride"]
    if type(stride) is not int or not 8 <= stride <= 512 or stride % 8:
        raise ValueError("Use an 8-pixel multiple up to 512 for the E01b wraparound stride")
    if type(config["top_band"]) is not int or config["top_band"] < 16:
        raise ValueError("The E01b top/bottom neighbor band must be at least 16 latent cells")


@dataclass(frozen=True)
class Job:
    method: str
    brief: dict
    seed: int
    image: Path
    measurement: Path
    physical: bool = False

    @property
    def identifier(self):
        return f"{self.brief['id']}-{self.seed}"

    @property
    def metadata(self):
        return self.image.with_suffix(".run.json")

    @property
    def is_legacy(self):
        return self.method in LEGACY_METHODS


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        e01b_root = self.root / "outputs" / "e01b"
        self.geometry = (args.geometry or e01b_root / "geometry").resolve()
        self.legacy_geometry = (args.legacy_geometry or self.root / "outputs" / "geometry").resolve()
        self.legacy_outputs = (args.legacy_outputs or self.root / "outputs" / "e01"
                               / "full-52dd11c30fc1").resolve()
        if self.geometry == self.legacy_geometry:
            raise ValueError("Use a separate geometry directory for E01b")
        self.lut = (args.lut or self.root / "cache" / "icc" / "iso-coated-v2-lut.npy").resolve()
        self.briefs_path = args.briefs.resolve()
        self.config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        self.pilot_config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
        validate_config(self.config)
        validate_config(self.pilot_config)
        if self.config.get("protocol") != self.pilot_config.get("protocol"):
            raise ValueError("Current and pilot configurations must use the same experiment protocol")
        self.physical = self.config.get("protocol") == PHYSICAL_PROTOCOL
        self.base = e01b_root / "crease-fix" if self.physical else e01b_root
        self.briefs = [json.loads(line) for line in self.briefs_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        validate_briefs(self.briefs)
        if [row["id"] for row in self.briefs] != [f"E01-{n:03d}" for n in range(1, 51)]:
            raise ValueError("The shared E01/E01b brief order must be E01-001 through E01-050")
        if any(type(row["continuous"]) is not bool for row in self.briefs):
            raise ValueError("The continuous flag must be a boolean")

    def prepare_geometry(self):
        existed = self.geometry.exists()
        if all((self.geometry / name).is_file() for name in GEOMETRY_FILES):
            return
        if existed and any(self.geometry.iterdir()):
            raise ValueError(f"Incomplete geometry at {self.geometry}; choose a new --geometry directory")
        subprocess.run([sys.executable, "-m", "dielinediff.dieline", "--output", str(self.geometry),
                        "--pixels-per-mm", str(self.config["pixels_per_mm"])], check=True)

    def identity(self, config):
        inputs = {
            "config": config_payload(config),
            "briefs_sha256": file_hash(self.briefs_path),
            "geometry": {file: file_hash(self.geometry / file) for file in GEOMETRY_FILES},
            "legacy_geometry": {file: file_hash(self.legacy_geometry / file) for file in GEOMETRY_FILES},
            "legacy_outputs": str(self.legacy_outputs),
            "lut_sha256": file_hash(self.lut),
        }
        return self.base / "runs" / stable_hash(inputs)[:12], inputs

    def prepare(self):
        self.prepare_geometry()
        topology = read_json(self.geometry / "topology.json")
        image = topology["image"]
        front = topology["faces"]["F"]["rect"]
        if (image["pixels_per_mm"] != self.config["pixels_per_mm"]
                or image["width"] % 8 or image["height"] % 8 or front[3] - front[1] != 1024):
            raise ValueError("E01b geometry requires a 1024 px box height and an 8-pixel padded canvas")
        self.run, inputs = self.identity(self.config)
        self.pilot_run, _ = self.identity(self.pilot_config)
        self.shared = self.base / "legacy-metrics" / stable_hash({
            key: value for key, value in inputs.items() if key not in ("config", "geometry")
        })[:12]
        self.run.mkdir(parents=True, exist_ok=True)
        metadata = self.run / "inputs.json"
        if metadata.exists() and read_json(metadata) != inputs:
            raise ValueError(f"Conflicting E01b contents at {self.run}")
        if not metadata.exists():
            write_json(metadata, inputs)

    def jobs(self, phase, *, run=None, methods=None):
        run = self.run if run is None else run
        count, default_methods = PHASES[phase]
        methods = default_methods if methods is None else methods
        seeds = SEEDS if phase == "full" else SEEDS[:1]
        for method in methods:
            for brief in self.briefs[:count]:
                for seed in seeds:
                    name = f"{brief['id']}-{seed}"
                    if method in LEGACY_METHODS:
                        image = self.legacy_outputs / LEGACY_METHODS[method] / f"{name}.png"
                        score = self.shared / method / f"{name}.metrics.json"
                    else:
                        image = run / method / f"{name}.png"
                        score = image.with_suffix(".metrics.json")
                    yield Job(method, brief, seed, image, score, self.physical)

    def check_legacy(self, jobs):
        missing = [str(job.image) for job in jobs if job.is_legacy
                   and not (job.image.is_file() and job.metadata.is_file())]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} reused E01 artifacts, beginning with {missing[:3]}")


def coverage_summary(record, steps):
    values = {key: int(record[key]) for key in ("covered", "required", "steps_checked", "windows")}
    weight = float(record["minimum_weight"])
    sizes = [tuple(map(int, size)) for size in record["window_sizes_px"]]
    if len(sizes) != values["windows"] or any(len(size) != 2 or min(size) < 512 for size in sizes):
        raise ValueError(f"E01b requires a short side of at least 512 px for each window: {sizes!r}")
    if (values["required"] <= 0 or values["covered"] != values["required"]
            or values["steps_checked"] != steps or values["windows"] <= 0
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError(f"E01b latent coverage failed: {record!r}")
    return {**values, "minimum_weight": weight, "window_sizes_px": [list(size) for size in sizes]}


def complete_image(job, steps):
    if not job.image.is_file() or not job.metadata.is_file():
        return False
    if not job.is_legacy:
        coverage_summary(read_json(job.metadata)["coverage"], steps)
    return True


def generate(experiment, phase, device):
    """Generate only the requested phase's v2 images; preserve completed outputs."""
    import torch
    from .methods.ours_tf_v2 import DieLineDiffV2

    config = experiment.config
    jobs = [job for job in experiment.jobs(phase) if not job.is_legacy]
    pending = [job for job in jobs if not complete_image(job, config["steps"])]
    if not pending and (experiment.run / "coverage.png").is_file():
        return
    hardware = torch.device(device)
    if hardware.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E01b generation needs the NVIDIA GPU")
    torch.cuda.set_device(hardware.index if hardware.index is not None else 0)
    topology = read_json(experiment.geometry / "topology.json")
    engine = DieLineDiffV2(experiment.root / "cache" / "models",
                          {"directory": experiment.geometry, "topology": topology},
                          config, device, experiment.lut)
    heatmap = experiment.run / "coverage.png"
    if not heatmap.is_file():
        temporary_heatmap = heatmap.with_suffix(f".{os.getpid()}.tmp.png")
        engine.save_coverage(temporary_heatmap, synchronized=True)
        temporary_heatmap.replace(heatmap)
    for job in pending:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = engine.generate(job.brief, job.seed, synchronized=job.method != "no_sync",
                                project=job.method != "no_projection")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**20
        coverage = coverage_summary(engine.last_coverage, config["steps"])
        expected = topology["image"]["width"], topology["image"]["height"]
        if image.size != expected:
            raise ValueError(f"Generated atlas has size {image.size}, expected {expected}")
        job.image.parent.mkdir(parents=True, exist_ok=True)
        temp = job.image.with_suffix(f".{os.getpid()}.tmp.png")
        image.convert("RGB").save(temp)
        temp.replace(job.image)
        write_json(job.metadata, {"elapsed_seconds": elapsed, "peak_memory_mib": peak,
                                  "gpu": torch.cuda.get_device_name(), "coverage": coverage})
        print(f"{job.method} {job.identifier} {elapsed:.2f} s {peak:.0f} MiB", flush=True)


@lru_cache(maxsize=1)
def physical_implementation_hash():
    from .metrics import physical

    return file_hash(Path(physical.__file__))


def measurement_input(job):
    record = {"atlas_sha256": file_hash(job.image), "run_sha256": file_hash(job.metadata),
              "learned": list(LEARNED)}
    if job.physical:
        record["physical_implementation_sha256"] = physical_implementation_hash()
    return record


def score_for(job, verify=True):
    if not job.measurement.is_file() or not job.image.is_file() or not job.metadata.is_file():
        return None
    saved = read_json(job.measurement)
    if verify and saved["input"] != measurement_input(job):
        return None
    result = saved["variants"]
    if set(result) != {"raw", "preflight"}:
        raise ValueError(f"Invalid evaluation variants in {job.measurement}")
    return result


def csv_rows(jobs, scores):
    for job in jobs:
        result = scores.get(job.measurement)
        if result is None:
            continue
        for variant, metrics in result.items():
            n = int(job.brief["id"].split("-")[-1])
            yield {"method": job.method, "variant": variant, "brief_id": job.brief["id"],
                   "seed": job.seed, "split": "holdout" if n > 10 else "tune",
                   "continuous": job.brief["continuous"], **metrics}


def save_results(experiment, phase, jobs, scores):
    from .e01b_summary import METRIC_KEYS, NATIVE_METRIC_KEYS

    metric_keys = METRIC_KEYS if experiment.physical else NATIVE_METRIC_KEYS
    rows = list(csv_rows(jobs, scores))
    path = experiment.run / phase / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("method", "variant", "brief_id", "seed", "split", "continuous")
    extra = sorted({key for row in rows for key in row} - set(fields + metric_keys))
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + metric_keys + tuple(extra), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    print(f"{len(scores)}/{len(jobs)} measured images at {path}", flush=True)
    return rows


def measure(experiment, phase, device, follow_until):
    from .methods.sdxl import prompt_for
    from .metrics.e01b import E01bEvaluator

    jobs = list(experiment.jobs(phase))
    experiment.check_legacy(jobs)
    scores = {}
    pending = []
    for job in jobs:
        cached = score_for(job)
        if cached is None:
            pending.append(job)
        else:
            scores[job.measurement] = cached
    save_results(experiment, phase, jobs, scores)
    if not pending:
        return
    with E01bEvaluator(experiment.legacy_geometry, experiment.geometry, experiment.lut,
                      device=device, learned=LEARNED, physical=experiment.physical) as evaluator:
        while pending:
            waiting = []
            for job in pending:
                if not complete_image(job, experiment.config["steps"]):
                    waiting.append(job)
                    continue
                performance = read_json(job.metadata)
                current_input = measurement_input(job)
                metrics = evaluator.measure(job.image, job.brief,
                                            geometry="legacy" if job.is_legacy else "new",
                                            prompt=prompt_for(job.brief),
                                            elapsed_seconds=performance["elapsed_seconds"],
                                            peak_memory_mib=performance["peak_memory_mib"])
                job.measurement.parent.mkdir(parents=True, exist_ok=True)
                write_json(job.measurement, {"input": current_input, "variants": metrics})
                scores[job.measurement] = metrics
                print(f"Measured {job.method} {job.identifier}", flush=True)
            if len(waiting) != len(pending):
                save_results(experiment, phase, jobs, scores)
            if follow_until is None or follow_until.is_file():
                break
            pending = waiting
            if pending:
                time.sleep(5)


def pilot_gate(experiment):
    methods = ("b2", "b3", "b4", "v2") if experiment.physical else ("b2", "v2")
    pilot_jobs = list(experiment.jobs("pilot", run=experiment.pilot_run, methods=methods))
    clip_samples = {"b2": {}, "v2": {}}
    stripe_samples = {method: {} for method in methods} if experiment.physical else {}
    coverage = {}
    evidence = {}
    missing = []
    for job in pilot_jobs:
        if not complete_image(job, experiment.pilot_config["steps"]):
            missing.append(f"{job.method}/{job.identifier}")
            continue
        brief_id = job.brief["id"]
        if not job.is_legacy:
            coverage[brief_id] = read_json(job.metadata)["coverage"]
        score = score_for(job)
        if score is None:
            missing.append(f"{job.method}/{job.identifier} metrics")
            continue
        raw = score["raw"]
        if job.method in clip_samples:
            clip = raw.get("clip_score")
            if clip is None or not math.isfinite(clip):
                missing.append(f"{job.method}/{job.identifier} CLIP")
            else:
                clip_samples[job.method][brief_id] = clip
        if experiment.physical:
            stripe = raw.get("stripe_index")
            if stripe is None or not math.isfinite(stripe):
                missing.append(f"{job.method}/{job.identifier} stripe index")
            else:
                stripe_samples[job.method][brief_id] = stripe
        evidence[f"{job.method}/{job.identifier}"] = [file_hash(job.image),
                                                      file_hash(job.metadata), file_hash(job.measurement)]

    def means_for(samples):
        return {method: sum(values.values()) / len(values) if len(values) == 5 else None
                for method, values in samples.items()}

    means = means_for(clip_samples)
    difference = means["v2"] - means["b2"] if all(value is not None for value in means.values()) else None
    heatmap = experiment.pilot_run / "coverage.png"
    coverage_ok = len(coverage) == 5 and heatmap.is_file()
    clip_ok = difference is not None and abs(difference) <= CLIP_TOLERANCE
    stripe_report = {}
    if experiment.physical:
        stripe_means = means_for(stripe_samples)
        best_method = threshold = delta = None
        if all(stripe_means[method] is not None for method in ("b2", "b3", "b4")):
            best_method = min(("b2", "b3", "b4"), key=stripe_means.__getitem__)
            threshold = STRIPE_TOLERANCE_FACTOR * stripe_means[best_method]
            if stripe_means["v2"] is not None:
                delta = stripe_means["v2"] - threshold
        stripe_report = {
            "stripe_samples": stripe_samples, "stripe_means": stripe_means,
            "stripe_baseline_method": best_method,
            "stripe_baseline_mean": stripe_means[best_method] if best_method is not None else None,
            "stripe_tolerance_factor": STRIPE_TOLERANCE_FACTOR, "stripe_threshold": threshold,
            "stripe_difference_from_threshold": delta, "stripe_passed": delta is not None and delta <= 0,
        }
    material = {"pilot_key": experiment.pilot_run.name, "samples": evidence,
                "coverage_heatmap": file_hash(heatmap) if heatmap.is_file() else None}
    fingerprint = stable_hash(material) if not missing and coverage_ok else None
    review_path = experiment.pilot_run / "visual-review.json"
    review = read_json(review_path) if review_path.is_file() else None
    visual_ok = review_matches(review, fingerprint, set(coverage)) if fingerprint else False
    return {
        "ready_for_visual_review": fingerprint is not None,
        "passed": not missing and coverage_ok and clip_ok and stripe_report.get("stripe_passed", True) and visual_ok,
        "fingerprint": fingerprint, "coverage_passed": coverage_ok, "coverage": coverage,
        "clip_samples": clip_samples, "clip_means": means, "clip_difference": difference,
        "clip_tolerance": CLIP_TOLERANCE, "clip_passed": clip_ok, "missing": missing,
        "visual_review_passed": visual_ok, "visual_review": review, **stripe_report,
    }


def review_matches(review, fingerprint, brief_ids):
    if not isinstance(review, dict) or review.get("fingerprint") != fingerprint:
        return False
    reviewer = review.get("reviewer")
    samples = review.get("samples")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(samples, dict):
        return False
    if set(samples) != brief_ids:
        return False
    return all(isinstance(item, dict) and item.get("edge_noise") is False
               and isinstance(item.get("notes"), str) and item["notes"].strip()
               for item in samples.values())


def record_review(experiment, source):
    gate = pilot_gate(experiment)
    if not gate["ready_for_visual_review"]:
        raise ValueError("The five pilot images and their objective evidence must be available for review")
    review = read_json(source)
    if not isinstance(review, dict) or review.get("fingerprint") != gate["fingerprint"]:
        raise ValueError("The visual review must refer to the current pilot fingerprint")
    samples = review.get("samples")
    reviewer = review.get("reviewer")
    if (not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(samples, dict)
            or set(samples) != set(gate["coverage"])
            or any(not isinstance(record, dict) or type(record.get("edge_noise")) is not bool
                   or not isinstance(record.get("notes"), str) or not record["notes"].strip()
                   for record in samples.values())):
        raise ValueError("Record the reviewer, edge-noise finding and observations for all five images")
    write_json(experiment.pilot_run / "visual-review.json", review)


def check_stage(experiment, phase):
    current = config_payload(experiment.config)
    reference = config_payload(experiment.pilot_config)
    if phase == "pilot":
        if current != reference or experiment.config.get("frozen", False):
            raise ValueError("Pilot uses the predeclared pilot configuration")
        return
    modified = {key for key in current.keys() | reference.keys() if current.get(key) != reference.get(key)}
    if modified - TUNABLE:
        raise ValueError(f"Only the four declared parameters may change: {sorted(modified - TUNABLE)}")
    gate = pilot_gate(experiment)
    if not gate["passed"]:
        raise ValueError("Pilot gate failed or awaits visual review; tuning and full generation are disabled")
    if phase == "full":
        if experiment.config.get("frozen") is not True:
            raise ValueError("Use an explicitly frozen configuration for the full experiment")
        actual = list(experiment.jobs("tune"))
        tuning_dir = experiment.run / "tune"
        record_path = tuning_dir / "tuning-record.json"
        tuning_csv = tuning_dir / "results.csv"
        if not (record_path.is_file() and tuning_csv.is_file() and all(
                complete_image(job, experiment.config["steps"]) and score_for(job) is not None
                for job in actual)):
            raise ValueError("The exact frozen configuration needs ten measured tuning briefs and a tuning record")
        record = read_json(record_path)
        if (record["configuration_key"] != experiment.run.name
                or record["pilot_fingerprint"] != gate["fingerprint"]
                or record["results_sha256"] != file_hash(tuning_csv)):
            raise ValueError("The tuning record no longer matches the reviewed pilot or measured tuning data")


def summarize(experiment, phase):
    from .e01b_summary import build_summary, format_tables, render_edge_reviews, render_examples

    jobs = list(experiment.jobs(phase))
    scores = {}
    for job in jobs:
        score = score_for(job)
        if score is not None:
            scores[job.measurement] = score
    rows = save_results(experiment, phase, jobs, scores)
    result = build_summary(rows, experiment.briefs, PHASES[phase][1],
                           SEEDS if phase == "full" else SEEDS[:1], phase,
                           physical=experiment.physical)
    destination = experiment.run / phase
    write_json(destination / "statistics.json", result)
    (destination / "TABLES.md").write_text(format_tables(result), encoding="utf-8")
    if phase != "tune":
        render_examples(experiment, phase)
    if phase == "pilot":
        render_edge_reviews(experiment)
        gate = pilot_gate(experiment)
        write_json(experiment.pilot_run / "pilot-gate.json", gate)
        if gate["coverage_passed"]:
            seconds = [read_json(job.metadata)["elapsed_seconds"] for job in jobs if job.method == "v2"]
            average = sum(seconds) / len(seconds)
            write_json(destination / "runtime-estimate.json",
                       {"mean_v2_seconds_per_box": average, "planning_boxes": 450,
                        "estimated_generation_hours_at_v2_rate": 450 * average / 3600})
        objective = gate["coverage_passed"] and gate["clip_passed"] and gate.get("stripe_passed", True)
        if gate["passed"]:
            print("The previously reviewed pilot passed.", flush=True)
        elif objective and gate["visual_review"] is None:
            print("Objective pilot checks passed. Record an actual visual review before tuning.", flush=True)
        else:
            raise ValueError(f"Pilot checks did not pass. See {experiment.pilot_run / 'pilot-gate.json'}")
    if len(scores) != len(jobs):
        raise ValueError(f"Incomplete {phase} evaluation. See {destination / 'results.csv'}")
    if phase == "tune":
        record = {"briefs": [row["id"] for row in experiment.briefs[:10]], "seed": SEEDS[0],
                  "parameters": {key: experiment.config[key] for key in sorted(TUNABLE)},
                  "configuration_key": experiment.run.name,
                  "pilot_fingerprint": pilot_gate(experiment)["fingerprint"],
                  "results_sha256": file_hash(destination / "results.csv"),
                  "raw": result["variants"]["raw"]["overall"]["metrics"],
                  "preflight": result["variants"]["preflight"]["overall"]["metrics"]}
        write_json(destination / "tuning-record.json", record)
    print(f"E01b {phase} reports at {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("paths", "prepare", "check", "generate",
                                            "measure", "summarize", "review"))
    parser.add_argument("--phase", choices=tuple(PHASES), default="pilot")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "DIELINEDIFF_ROOT", "/root/autodl-tmp/ai-design")))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pilot-config", type=Path, default=Path("code/configs/E01b.yaml"))
    parser.add_argument("--briefs", type=Path, default=Path("data/briefs/E01_briefs.jsonl"))
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--legacy-geometry", type=Path)
    parser.add_argument("--legacy-outputs", type=Path)
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--follow-until", type=Path)
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    if args.config is None:
        args.config = Path("code/configs/E01b.frozen.yaml") if args.phase == "full" else args.pilot_config

    experiment = Experiment(args)
    experiment.prepare()
    if args.action == "paths":
        print(json.dumps({"run": str(experiment.run), "pilot_run": str(experiment.pilot_run),
                          "geometry": str(experiment.geometry),
                          "legacy_geometry": str(experiment.legacy_geometry),
                          "legacy_outputs": str(experiment.legacy_outputs),
                          "legacy_measurements": str(experiment.shared),
                          "phase_results": str(experiment.run / args.phase / "results.csv")}))
        return
    if args.action == "prepare":
        experiment.check_legacy(list(experiment.jobs(args.phase)))
        return
    if args.action == "review":
        if args.review_file is None:
            parser.error("review requires --review-file")
        record_review(experiment, args.review_file)
        gate = pilot_gate(experiment)
        write_json(experiment.pilot_run / "pilot-gate.json", gate)
        if not gate["passed"]:
            raise ValueError("Visual findings recorded. The complete pilot gate did not pass.")
        print(f"Recorded visual approval for {gate['fingerprint']}", flush=True)
        return
    check_stage(experiment, args.phase)
    if args.action == "check":
        experiment.check_legacy(list(experiment.jobs(args.phase)))
    elif args.action == "generate":
        generate(experiment, args.phase, args.device or "cuda")
    elif args.action == "measure":
        measure(experiment, args.phase, args.device or "cpu", args.follow_until)
    else:
        summarize(experiment, args.phase)


if __name__ == "__main__":
    main()
