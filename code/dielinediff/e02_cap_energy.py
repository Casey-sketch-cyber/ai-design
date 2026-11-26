"""Task 18 compares cap-energy priors with a fixed, previously reviewed R0 ring."""

import argparse
import os
from functools import partial
from pathlib import Path

import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02 import DEFAULT_CONFIG as TASK07_CONFIG
from .e02_capfix import CONFIG as CAPFIX_CONFIG
from .e02_capfix_ring import protect_ring
from .e02_captexture import DOCS, G1
from .e02_freeze import verify
from .e02_history import read_briefs


CONFIG = Path("code/configs/E02_cap_energy.yaml")
SEEDS = (42, 1234)
CAPS = ("e0", "e1", "e2", "e3", "e4")
TASK17_METHOD = "u1_sigma28"


class Experiment:
    def __init__(self, root, config=CONFIG):
        self.root = Path(root).resolve()
        self.config = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (self.config["protocol"] != "e02-v3-cap-energy" or self.config["seeds"] != list(SEEDS)
                or self.config["ring"] != "r0" or tuple(self.config["candidates"]) != CAPS):
            raise ValueError("Task 18 requires fixed R0, both registered seeds, and the five cap recipes")
        self.base_experiment_config = yaml.safe_load(TASK07_CONFIG.read_text(encoding="utf-8"))
        self.config["gate"] = self.base_experiment_config["gate"]
        capfix = yaml.safe_load(CAPFIX_CONFIG.read_text(encoding="utf-8"))
        self.engine_config = {**self.base_experiment_config["common"], "capfix_prompts": capfix["capfix_prompts"],
                              "energy_match": self.config["energy_match"], "e4": self.config["e4"]}
        self.metric_config, self.dino_model, frozen = verify(self.root)
        self.geometry = self.root / "outputs/e01b/geometry"
        self.legacy = self.root / "outputs/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        self.all_briefs = read_briefs()
        self.briefs = self.all_briefs[:10]
        reference10 = read_json(DOCS / "capgeom-inputs.json")
        reference17 = read_json(DOCS / "ring-m2m3-inputs.json")
        e02 = self.root / "outputs/e02"
        self.original = e02 / "step1" / reference10["config"]["task07_reference"]
        self.previous15 = e02 / "uniform-ring/runs" / reference17["config"]["task15_reference"]
        self.previous17 = e02 / "ring-m2m3/runs" / self.config["task17_reference"]
        initial = read_json(self.original / "inputs.json")
        old = self.previous17 / "selection.json"
        decision = read_json(old)
        if (reference17 != read_json(self.previous17 / "inputs.json")
                or stable_hash(reference17)[:12] != self.config["task17_reference"]
                or decision != read_json(DOCS / "ring-m2m3-selection.json")
                or decision["ring"] != "r0" or decision["cap"]["selected"] != TASK17_METHOD
                or not decision["complete"] or not decision["visual_review_complete"]
                or initial["generation_config"] != self.base_experiment_config
                or initial["metrics"] != stable_hash(frozen)
                or initial["briefs"] != stable_hash(self.briefs)
                or file_hash(self.original / "inputs.json") != reference17["task07_inputs_sha256"]
                or file_hash(self.lut) != initial["lut_sha256"]
                or any(file_hash(self.geometry / name) != h for name, h in initial["geometry"].items())):
            raise ValueError("Task 18 requires the reviewed Task 17 cap and immutable original inputs")
        self.r0 = {42: e02 / "capfix/rings" / reference10["config"]["archived_rings"],
                   1234: e02 / "capgeom/runs" / stable_hash(reference10)[:12] / "ring-validation"}
        self.original_rings, self.validation_ring_dir = (self.r0[seed] for seed in SEEDS)
        for seed, directory in self.r0.items():
            for brief in self.briefs:
                key = f"{brief['id']}-{seed}"
                if file_hash(directory / f"{key}.npz") != reference17["original_ring_caches"][key]:
                    raise ValueError(f"The original R0 ring has changed: {key}")
        identity = {"config": self.config, "metric_lock": stable_hash(frozen), "ring": protect_ring(initial),
                    "r0_caches": reference17["original_ring_caches"],
                    "original_inputs_sha256": file_hash(self.original / "inputs.json"),
                    "task17_inputs_sha256": file_hash(self.previous17 / "inputs.json"),
                    "task17_selection_sha256": file_hash(old)}
        self.output = e02 / "cap-energy/runs" / stable_hash(identity)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "inputs.json", identity)
        self.samples = partial(sample_caps, self)

    def parameters(self, method):
        return {**G1, **self.config["candidates"][method],
                "texture_sigma_px_at_scale_1": self.config["energy_match"]["sigma_px_at_scale_1"]}


def sample_caps(experiment, *, methods, seed=42):
    from .e02_cap_energy_sources import sample_caps as locate
    yield from locate(experiment, methods=methods, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnose", "all", "candidates", *CAPS[1:]), default="all")
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cap-reviews", type=Path)
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--selection-review", type=Path)
    args = parser.parse_args()
    experiment = Experiment(args.root)
    print(f"Task 18: {experiment.output}", flush=True)
    from .e02_cap_energy_curves import run_initial_diagnosis
    run_initial_diagnosis(experiment)
    if args.stage == "diagnose":
        return
    if args.stage in CAPS[1:]:
        from .e02_cap_energy_generate import run_candidate
        run_candidate(experiment, args.stage, args.device)
        return
    from .e02_cap_energy_full import run
    run(experiment, args.device, args.stage, args.cap_reviews, args.visual, args.selection_review)


if __name__ == "__main__":
    main()
