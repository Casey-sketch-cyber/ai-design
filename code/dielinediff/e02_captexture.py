"""Run Task 13 on the registered cap-only cohorts and original E02 gate."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02 import BENCHMARK_BASELINES, DEFAULT_CONFIG as TASK07_CONFIG, Sample, grouped
from .e02_capfix import CONFIG as CAPFIX_CONFIG
from .e02_capfix_ring import METHODS_DIR, protect_ring
from .e02_capgeom import C3, _engine, trial
from .e02_freeze import verify
from .e02_history import read_briefs
from .e02_sources import SourceAtlas, sources


CONFIG = Path("code/configs/E02_captexture.yaml")
DOCS = Path("docs/experiments/E02")
PROTOCOL = "e02-v3-captexture"
G1 = {**C3, "tight": True}
ARCHIVED = ("g1", "g2_075", "g2_085")
HIGH = ("g2_092", "g2_095")
G4 = ("g4_f4_full", "g4_f4_70", "g4_f8_full", "g4_f8_70")
CANDIDATES = (*HIGH, "g5", *G4)
VALIDATION = "seed_1234"
METHOD_LABELS = {
    "g1": "G1", "g2_092": "G2 (0.92)", "g2_095": "G2 (0.95)", "g5": "G5",
    "g4_f4_full": "G4 / factor 4 / full",
    "g4_f4_70": "G4 / factor 4 / 70%",
    "g4_f8_full": "G4 / factor 8 / full",
    "g4_f8_70": "G4 / factor 8 / 70%",
}


class Experiment:
    def __init__(self, root):
        self.root = root.resolve()
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        if (config["protocol"], config["selection_seed"], config["validation_seed"],
                config["spatial_factors"], config["guided_fractions"], config["high_strengths"],
                config["texture_injection_strength"]) != (PROTOCOL, 42, 1234, [4, 8], [1.0, 0.70], [0.92, 0.95], 0.85):
            raise ValueError("Task 13 uses the registered ten-case cohorts and fixed candidate grid.")
        archived_inputs = read_json(DOCS / "capgeom-inputs.json")
        archived_config = archived_inputs["config"]
        old = yaml.safe_load(TASK07_CONFIG.read_text(encoding="utf-8"))
        capfix = yaml.safe_load(CAPFIX_CONFIG.read_text(encoding="utf-8"))
        self.config = {**config, "gate": old["gate"]}
        self.engine_config = {**old["common"], "capfix_prompts": capfix["capfix_prompts"]}
        self.metric_config, self.dino_model, freeze = verify(self.root)
        self.all_briefs = read_briefs()
        self.briefs = self.all_briefs[:10]
        self.legacy = self.root / "outputs/geometry"
        self.geometry = self.root / "outputs/e01b/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        e02 = self.root / "outputs/e02"
        self.original = e02 / "step1" / archived_config["task07_reference"]
        self.reference = e02 / "capgeom/runs" / config["task10_reference"]
        self.original_rings = e02 / "capfix/rings" / archived_config["archived_rings"]
        self.validation_ring_dir = self.reference / "ring-validation"
        legacy = read_json(self.original / "inputs.json")
        decisions = read_json(self.reference / "selection.json")
        proof = protect_ring(legacy)
        if (archived_inputs != read_json(self.reference / "inputs.json")
                or archived_inputs["metrics"] != stable_hash(freeze) or archived_inputs["ring"] != proof
                or file_hash(self.original / "inputs.json") != archived_inputs["task07_sha256"]
                or not decisions["complete"] or decisions["selected"] != "g1"
                or decisions["candidates"]["g1"] != G1
                or any(file_hash(self.geometry / name) != h for name, h in legacy["geometry"].items())
                or file_hash(self.lut) != legacy["lut_sha256"]
                or any(file_hash(METHODS_DIR / name) != value
                       for name, value in archived_inputs["implementation"].items())):
            raise ValueError("Task 13 could not confirm the reviewed ring and source identities.")
        for brief in self.briefs:
            for seed, directory in ((42, self.original_rings), (1234, self.validation_ring_dir)):
                for extension in ("npz", "json"):
                    ring = directory / f"{brief['id']}-{seed}.{extension}"
                    if not ring.is_file():
                        raise FileNotFoundError(ring)
        self.archived_scores = decisions["scores_sha256"]
        self.seed_1234_reference = read_json(DOCS / "capgeom-ring-proof.json")["seed_1234"]
        identity = {"config": self.config, "metrics": stable_hash(freeze), "ring": proof,
                    "task10_inputs_sha256": file_hash(self.reference / "inputs.json"),
                    "task10_selection_sha256": file_hash(self.reference / "selection.json"),
                    "registered_candidates": self.parameters()}
        self.output = e02 / "captexture/runs" / stable_hash(identity)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "inputs.json", identity)
        self.selection = self.output / "selection.json"

    def parameters(self):
        return {
            "g1": G1,
            **{name: {**G1, "strength": strength}
               for name, strength in zip(HIGH, self.config["high_strengths"], strict=True)},
            "g5": {**G1, "strength": self.config["texture_injection_strength"], "inject_texture": True},
            **{name: {**G1, "ilvr_factor": factor, "ilvr_fraction": fraction}
               for name, (factor, fraction) in zip(G4, (
                   (factor, fraction) for factor in self.config["spatial_factors"]
                   for fraction in self.config["guided_fractions"]
               ), strict=True)},
        }

    @staticmethod
    def _sample(atlas, run, stored_method=None):
        directory = run / "evaluation" / (stored_method or atlas.method)
        key = atlas.identifier
        return Sample(atlas, directory / f"{key}.metrics.json", directory / f"{key}.preflight.png")

    def samples(self, *, methods, seed=42):
        if seed == 42:
            for atlas in sources(self.root, self.all_briefs, trial_only=True):
                if atlas.method in methods and atlas.method in BENCHMARK_BASELINES:
                    job = self._sample(atlas, self.original)
                    if self.archived_scores[f"{atlas.method}/{atlas.identifier}"] != file_hash(job.score):
                        raise ValueError(f"The archived baseline score has changed: {job.score}")
                    yield job
        for method in methods:
            if method in BENCHMARK_BASELINES and method != "b2_regional":
                continue
            for brief in self.briefs:
                key = f"{brief['id']}-{seed}"
                stored = method
                if method == "b2_regional":
                    run = self.original
                elif method in ARCHIVED:
                    run = self.reference
                    if seed != 42:
                        if method != "g1" or seed != 1234:
                            raise ValueError("Only the reviewed G1 is registered as a seed 1234 reference.")
                        stored = VALIDATION
                else:
                    run = self.output
                path = run / stored / f"{key}.png"
                job = self._sample(SourceAtlas(method, brief, seed, path, self.geometry), run, stored)
                if seed == 42 and method in (*ARCHIVED, "b2_regional") and (
                    self.archived_scores[f"{method}/{key}"] != file_hash(job.score)
                ):
                    raise ValueError(f"The archived score has changed: {job.score}")
                if seed == 1234 and method == "g1":
                    reference = self.seed_1234_reference[brief["id"]]["atlas_sha256"][VALIDATION]
                    if file_hash(path) != reference:
                        raise ValueError(f"The archived G1 seed 1234 image has changed: {path}")
                yield job

    def groups42(self, completed):
        methods = (*BENCHMARK_BASELINES, *ARCHIVED, *completed)
        return grouped(list(self.samples(methods=methods)))

    def groups1234(self):
        return grouped(list(self.samples(methods=("g1", VALIDATION), seed=1234)))


def _texture_engine(experiment, device):
    from .methods.v3_captexture import CapTextureEngine

    return CapTextureEngine(experiment.root / "cache/models", experiment.geometry, experiment.engine_config,
                            device=device, lut_path=experiment.lut)


def run_trial(experiment, name, parameters, device, *, seed=42, source=None):
    extension = parameters.get("ilvr_factor") or parameters.get("inject_texture")
    if extension:
        files = ["v3_captexture.py", "v3_capgeom_init.py"]
        files.append("v3_captexture_ilvr.py" if parameters.get("ilvr_factor") else "v3_captexture_inject.py")
        identity = {file: file_hash(METHODS_DIR / file) for file in files}
    else:
        identity = None
    trial(experiment, name, parameters, device, seed=seed, source=source,
          implementation=identity, engine_factory=_texture_engine if extension else _engine)


def _visuals(experiment, chosen):
    from .e02_capfix_visual import render_capfix

    render_capfix(experiment, chosen, original_method="g2_085",
                  original_title="Task 10 G2 0.85", current_title=METHOD_LABELS[chosen])


def publish(experiment, ledger, visual=None):
    from .e02_captexture_eval import write_results

    result = write_results(experiment, experiment.groups42(ledger["completed"]),
                           experiment.groups1234(), ledger["selected"], visual, selection=ledger["decision"])
    _visuals(experiment, ledger["selected"])
    print(json.dumps({"selected": result["selected"], "passed": result["passed"],
                      "checks": {name: check["passed"] for name, check in result["checks"].items()}}), flush=True)
    return result


def request_tie_review(experiment, groups, candidates):
    from .e02_capfix_eval import gate
    from .e02_visual import render_native_creases

    folder = experiment.output / "tie-review"
    fingerprints = {}
    for method in candidates:
        fingerprints[method] = gate(experiment, groups, method, None)["fingerprint"]
        crops = SimpleNamespace(output=folder / method, geometry=experiment.geometry, samples=experiment.samples)
        render_native_creases(crops, method)
    path = folder / "request.json"
    write_json(path, {"seed": 42, "candidates": fingerprints})
    print(f"Primary metrics are tied; inspect native crops and supply --selection-review: {path}", flush=True)


def run(experiment, device, stage, review_path=None):
    parameters = experiment.parameters()
    for name in (HIGH if stage == "high" else CANDIDATES):
        run_trial(experiment, name, parameters[name], device)
    if stage == "high":
        return

    from .e02_captexture_eval import select_candidates, textures

    groups = experiment.groups42(CANDIDATES)
    physical = textures(groups, seed=42)
    saved_tie = experiment.output / "tie-visual-review.json"
    if review_path is None:
        registered = DOCS / "captexture-tie-review.json"
        review_path = next((path for path in (saved_tie, registered) if path.is_file()), None)
    tie_review = read_json(review_path) if review_path is not None else None
    decision = select_candidates(groups, CANDIDATES, experiment.config["gate"], texture=physical,
                                 visual_reviews=tie_review)
    chosen = decision["selected"]
    ledger = {"selected": chosen, "candidates": parameters, "completed": list(CANDIDATES),
              "decision": decision, "complete": False,
              "scores_42_sha256": {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                    for method, entries in groups.items() for job, _ in entries}}
    write_json(experiment.selection, ledger)
    if decision["requires_tie_visual_review"]:
        request_tie_review(experiment, groups, decision["primary_ties"])
        return
    if decision["native_artifact_counts"] is not None:
        write_json(saved_tie, tie_review)
    print(f"Task 13 seed 42 selected {chosen}", flush=True)
    run_trial(experiment, VALIDATION, parameters[chosen], device, seed=1234, source=chosen)
    ledger["scores_1234_sha256"] = {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                    for method, entries in experiment.groups1234().items() for job, _ in entries}
    ledger["complete"] = True
    write_json(experiment.selection, ledger)
    saved = experiment.output / "visual-review.json"
    visual = read_json(saved) if saved.is_file() else (tie_review or {}).get(chosen)
    result = publish(experiment, ledger, visual)
    if result["visual_review"] is not None:
        write_json(saved, result["visual_review"])


def review(experiment, path):
    ledger = read_json(experiment.selection)
    if not ledger["complete"]:
        raise ValueError("Finish the registered validation before publishing a gate.")
    visual = read_json(path)
    gate = read_json(experiment.output / "gate.json")
    observations = visual.get("samples", {})
    if (visual.get("fingerprint") != gate["fingerprint"]
            or set(observations) != {brief["id"] for brief in experiment.briefs}
            or any(not isinstance(row.get("visible_artifact"), bool)
                   or not isinstance(row.get("notes"), str) or not row["notes"].strip()
                   for row in observations.values())):
        raise ValueError("Provide the ten explicit observations bound to the current gate.")
    result = publish(experiment, ledger, visual)
    if not result["visual_review"]:
        raise ValueError("Visual review no longer matches the evaluated image fingerprints.")
    write_json(experiment.output / "visual-review.json", visual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("paths", "run", "review"), nargs="?", default="run")
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("high", "all"), default="all")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--selection-review", type=Path)
    args = parser.parse_args()
    if args.root is None or args.action == "review" and args.review_file is None:
        parser.error("The AutoDL data root is required. Review also requests --review-file.")
    experiment = Experiment(args.root)
    if args.action == "paths":
        print(json.dumps({"run": str(experiment.output), "reference": str(experiment.reference)}))
    elif args.action == "run":
        run(experiment, args.device, args.stage, args.selection_review)
    else:
        review(experiment, args.review_file)


if __name__ == "__main__":
    main()
