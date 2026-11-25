"""Run the registered Task 08 cap-only selection on the ten frozen-ring cases."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02 import BENCHMARK_BASELINES, DEFAULT_CONFIG as TASK07_CONFIG, Sample, grouped
from .e02_capfix_eval import RING_REFERENCE, SELECTION_POLICY, evaluate, runtime_budget
from .e02_capfix_eval import select_addition, select_prompts, verify_ring, write_results
from .e02_capfix_ring import METHODS_DIR, TASK07_RING, bytes_digest, prepare, protect_ring, rgb_pixels
from .e02_freeze import verify
from .e02_history import read_briefs
from .e02_sources import SourceAtlas, sources


CONFIG = Path("code/configs/E02_capfix.yaml")
STAGES = ("c1", "c2", "c3", "c4")
PROMPTS = {
    "c1a": {"prompt_mode": "plain", "cap_scale": 1, "inpaint": False, "repaint": False},
    "c1b": {"prompt_mode": "emblem", "cap_scale": 1, "inpaint": False, "repaint": False},
}


class Experiment:
    def __init__(self, root, configuration):
        self.root = root.resolve()
        config = yaml.safe_load(configuration.read_text(encoding="utf-8"))
        if (config["protocol"], config["reference_run"], config["seed"]) != ("e02-v3-capfix", "c6bb770094d3", 42):
            raise ValueError("CapFix uses the same reviewed Task 07 ten briefs and seed 42.")
        old_config = yaml.safe_load(TASK07_CONFIG.read_text(encoding="utf-8"))
        self.config = {**config, "gate": old_config["gate"]}
        self.engine_config = {**old_config["common"], "capfix_prompts": config["capfix_prompts"],
                              "repaint_index": config["optional_repaint"]["step_index"]}
        self.metric_config, self.dino_model, freeze = verify(self.root)
        self.briefs = read_briefs()[:10]
        self.legacy = self.root / "outputs/geometry"
        self.geometry = self.root / "outputs/e01b/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        self.reference = self.root / "outputs/e02/step1" / config["reference_run"]
        self.baseline_config = read_json(self.reference / "inputs.json")
        if (self.baseline_config["generation_config"] != old_config
                or tuple(old_config["ring_sizes"][RING_REFERENCE]) != TASK07_RING
                or self.baseline_config["briefs"] != stable_hash(self.briefs)
                or self.baseline_config["metrics"] != stable_hash(freeze)
                or self.baseline_config["lut_sha256"] != file_hash(self.lut)
                or any(file_hash(self.geometry / name) != digest
                       for name, digest in self.baseline_config["geometry"].items())):
            raise ValueError("The Task 07 archive is inconsistent with the frozen evaluation inputs.")
        proof = protect_ring(self.baseline_config)
        ring_identity = {**proof, "reference_run": config["reference_run"],
                         "archive_input_sha256": file_hash(self.reference / "inputs.json")}
        phase = self.root / "outputs/e02/capfix"
        self.rings = phase / "rings" / stable_hash(ring_identity)[:12]
        self.rings.mkdir(parents=True, exist_ok=True)
        write_json(self.rings / "inputs.json", ring_identity)
        identity = {
            "config": self.config, "ring": ring_identity, "metrics": stable_hash(freeze),
            "selection": SELECTION_POLICY,
            "engines": {
                name: file_hash(METHODS_DIR / name)
                for name in ("v3_capfix.py", "v3_capfix_geometry.py", "v3_inpainting.py")
            },
        }
        self.output = phase / "runs" / stable_hash(identity)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "inputs.json", identity)
        self.selection = self.output / "selection.json"

    def _sample(self, atlas, directory):
        base = directory / "evaluation" / atlas.method
        return Sample(atlas, base / f"{atlas.identifier}.metrics.json",
                      base / f"{atlas.identifier}.preflight.png")

    def samples(self, *, methods):
        needed = set(methods)
        for item in sources(self.root, self.briefs, trial_only=True):
            if item.method in needed and item.method in BENCHMARK_BASELINES:
                yield self._sample(item, self.reference)
        for method in methods:
            if method in BENCHMARK_BASELINES and method != "b2_regional":
                continue
            for brief in self.briefs:
                archived = method in ("b2_regional", RING_REFERENCE)
                directory = self.reference if archived else self.output
                graph = directory / method / f"{brief['id']}-42.png"
                yield self._sample(SourceAtlas(method, brief, 42, graph, self.geometry), directory)

    def archive(self, brief):
        return self.reference / RING_REFERENCE / f"{brief['id']}-42.png"

    def result_groups(self, candidates):
        return grouped(list(self.samples(methods=(*BENCHMARK_BASELINES, RING_REFERENCE, *candidates))))


def _hardware(device):
    import torch

    gpu = torch.device(device)
    if gpu.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Task 08 generation requires a CUDA device.")
    torch.cuda.set_device(gpu.index or 0)
    return gpu


def _engine(experiment, device):
    from .methods.v3_capfix import CapFixEngine

    return CapFixEngine(experiment.root / "cache/models", experiment.geometry,
                        experiment.engine_config, device=device, lut_path=experiment.lut)


def prepare_rings(experiment, device):
    import torch

    completed = all((experiment.rings / f"{brief['id']}-42.json").is_file() for brief in experiment.briefs)
    if completed:
        return
    gpu = _hardware(device)
    engine = _engine(experiment, gpu)
    for brief in experiment.briefs:
        prepared, record, _, _ = prepare(engine, experiment.rings, brief, experiment.archive(brief))
        print(f"ring {brief['id']}-42 {record['ring_elapsed_seconds']:.2f}s, "
              f"{record['ring_peak_memory_mib']:.0f} MiB, {record['ring_supported_pixels']} identical pixels",
              flush=True)
        del prepared
    del engine
    torch.cuda.empty_cache()


def generate(experiment, name, parameters, device):
    import torch

    candidate = experiment.output / name
    candidate.mkdir(parents=True, exist_ok=True)
    plan = candidate / "candidate.json"
    if plan.is_file() and read_json(plan) != parameters:
        raise ValueError(f"Stored candidate differs from the scheduled change: {name}")
    write_json(plan, parameters)
    samples = list(experiment.samples(methods=(name,)))
    pending = [job for job in samples if not job.validate_generation()]
    if not pending:
        return samples
    model_identity = None
    if parameters["inpaint"]:
        from .methods.v3_inpainting import MODEL_DIRECTORY, WEIGHT_FILE, verify_inpaint_model

        models = experiment.root / "cache/models"
        if not (models / MODEL_DIRECTORY / WEIGHT_FILE).is_file():
            subprocess.run([sys.executable, "code/scripts/prepare_E02_inpainting.py",
                            "--models-dir", str(models)], check=True)
        model_identity = {**verify_inpaint_model(models),
                          "implementation_sha256": file_hash(METHODS_DIR / "v3_inpainting.py")}
    gpu = _hardware(device)
    engine = _engine(experiment, gpu)
    if model_identity is not None:
        # Exclude one-time checkpoint loading, as in the archived Task 07 timings.
        engine._open_inpaint()
        torch.cuda.synchronize(gpu)
    for job in pending:
        atlas = job.atlas
        prepared, record, body, supported = prepare(engine, experiment.rings, atlas.brief,
                                                    experiment.archive(atlas.brief))
        torch.cuda.synchronize(gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
        started = time.perf_counter()
        image, details = engine.generate_caps(atlas.brief, prepared, parameters)
        torch.cuda.synchronize(gpu)
        duration = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(gpu) / 2**20
        output = rgb_pixels(image)
        if (bytes_digest(output[body]) != record["body_rgb_sha256"]
                or bytes_digest(output[supported]) != record["ring_supported_rgb_sha256"]):
            raise RuntimeError(f"Cap generation modified a Task 07 ring pixel: {atlas.identifier}")
        if bytes_digest(prepared.frozen_latent.detach().cpu().numpy()) != record["latent_sha256"]:
            raise RuntimeError(f"Cap generation modified the frozen ring latent: {atlas.identifier}")
        if bytes_digest(prepared.rng_state.detach().cpu().numpy()) != record["rng_state_sha256"]:
            raise RuntimeError(f"Cap generation modified the stored RNG state: {atlas.identifier}")
        total = record["ring_elapsed_seconds"] + duration
        overall_peak = max(record["ring_peak_memory_mib"], peak)
        temporary = atlas.path.with_suffix(f".{os.getpid()}.tmp.png")
        image.save(temporary)
        temporary.replace(atlas.path)
        write_json(atlas.metadata, {
            "method": name, "brief_id": atlas.brief["id"], "seed": atlas.seed,
            "raw_sha256": file_hash(atlas.path), "gpu": torch.cuda.get_device_name(gpu),
            "elapsed_seconds": total, "peak_memory_mib": overall_peak,
            "cap_elapsed_seconds": duration, "cap_peak_memory_mib": peak,
            "ring": record, "candidate": parameters, "inpaint_model": model_identity, "details": details,
        })
        print(f"{name} {atlas.identifier}: {total:.2f}s ({duration:.2f}s caps) {overall_peak:.0f} MiB", flush=True)
        del prepared
    del engine
    torch.cuda.empty_cache()
    return samples


def _verify_during_selection(experiment, known):
    references = list(experiment.samples(methods=(RING_REFERENCE,)))
    proof = {RING_REFERENCE: [(item, None) for item in references]}
    for name, samples in known.items():
        proof[name] = [(item, None) for item in samples]
    verify_ring(proof)


def _run_candidate(experiment, name, parameters, device, known):
    jobs = generate(experiment, name, parameters, device)
    known[name] = jobs
    _verify_during_selection(experiment, known)
    evaluate(experiment, jobs, device)
    return parameters


def _bind_stage(record, groups):
    record["scores_sha256"] = {
        f"{name}/{item.atlas.identifier}": file_hash(item.score)
        for name in (record["reference"], record["challenger"])
        for item, _ in groups[name]
    }
    return record


def _output(experiment, ledger, visual=None):
    groups = experiment.result_groups(ledger["candidates"])
    result = write_results(experiment, groups, ledger["selected"], visual)
    if ledger["complete"]:
        from .e02_capfix_visual import render_capfix
        render_capfix(experiment, ledger["selected"])
    display = {key: check["passed"] for key, check in result["checks"].items()}
    print(json.dumps({"selected": result["selected"], "passed": result["passed"], "checks": display}))
    return result


def run(experiment, device, through):
    known = {}
    definitions = {}
    for name, config in PROMPTS.items():
        definitions[name] = _run_candidate(experiment, name, config, device, known)
    groups = experiment.result_groups(definitions)
    choice = _bind_stage(select_prompts(groups, "c1a", "c1b", experiment.config["gate"]), groups)
    if choice["selected"] is None:
        write_json(experiment.output / "unqualified-prompts.json", choice)
        raise RuntimeError("Neither Prompt C1 candidate qualified under the RTX 3090 budget.")
    ledger = {"stages": [choice], "candidates": definitions, "selected": choice["selected"], "complete": False}
    write_json(experiment.selection, ledger)
    if through == "c1":
        _output(experiment, ledger)
        return

    keys = {"c2": ("inpaint", True), "c3": ("cap_scale", 2), "c4": ("repaint", True)}
    for stage in STAGES[1:STAGES.index(through) + 1]:
        incumbent = ledger["selected"]
        if stage == "c4":
            limit = experiment.config["optional_repaint"]
            budget = runtime_budget(groups[incumbent], experiment.config["gate"])
            if not limit["enabled"] or budget["highest_sample_seconds"] > limit["maximum_parent_seconds"]:
                ledger["optional_repaint"] = {"tested": False, "parent": incumbent,
                                              "highest_parent_seconds": budget["highest_sample_seconds"],
                                              "maximum_parent_seconds": limit["maximum_parent_seconds"]}
                break
        field, value = keys[stage]
        candidate = {**definitions[incumbent], field: value}
        definitions[stage] = _run_candidate(experiment, stage, candidate, device, known)
        groups = experiment.result_groups(definitions)
        choice = _bind_stage(select_addition(groups, incumbent, stage, experiment.config["gate"],
                                             stage=stage.upper()), groups)
        ledger["stages"].append(choice)
        ledger["selected"] = choice["selected"]
        write_json(experiment.selection, ledger)
        print(f"{stage.upper()} compared with {incumbent}, selected {choice['selected']} "
              f"({choice['reason']}, net {choice['challenger_net_votes']})", flush=True)
    ledger["complete"] = through in ("c3", "c4")
    write_json(experiment.selection, ledger)
    review_path = experiment.output / "visual-review.json"
    visual = read_json(review_path) if review_path.is_file() else None
    _output(experiment, ledger, visual)


def record_review(experiment, path):
    ledger = read_json(experiment.selection)
    if not ledger["complete"]:
        raise ValueError("Complete the required cap candidates before recording the original-pixel review.")
    document = read_json(path)
    visual = document.get("visual_review", document) if isinstance(document, dict) else {}
    if not isinstance(visual, dict):
        raise ValueError("Expected a visual review object.")
    current = read_json(experiment.output / "gate.json")
    samples = visual.get("samples", {})
    names = {brief["id"] for brief in experiment.briefs}
    if (not isinstance(samples, dict) or set(samples) != names
            or visual.get("fingerprint") != current["fingerprint"]
            or any(not isinstance(item, dict) or not isinstance(item.get("visible_artifact"), bool)
                   or not isinstance(item.get("notes"), str) or not item["notes"].strip()
                   for item in samples.values())):
        raise ValueError("Record all ten capfold observations for the current gate fingerprint.")
    result = _output(experiment, ledger, visual)
    if result["visual_review"] is None:
        raise ValueError("The review fingerprint did not match the rebuilt evidence.")
    write_json(experiment.output / "visual-review.json", visual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("paths", "rings", "run", "review"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--through", choices=STAGES, default="c4")
    parser.add_argument("--review-file", type=Path)
    arguments = parser.parse_args()
    if arguments.root is None or arguments.action == "review" and arguments.review_file is None:
        parser.error("root is required; review also needs --review-file")
    experiment = Experiment(arguments.root, arguments.config)
    if arguments.action == "paths":
        print(json.dumps({"run": str(experiment.output), "rings": str(experiment.rings),
                          "reference": str(experiment.reference)}))
    elif arguments.action == "rings":
        prepare_rings(experiment, arguments.device)
    elif arguments.action == "review":
        record_review(experiment, arguments.review_file)
    else:
        run(experiment, arguments.device, arguments.through)


if __name__ == "__main__":
    main()
