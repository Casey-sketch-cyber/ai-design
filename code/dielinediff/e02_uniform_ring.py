"""Run the registered two-seed Task 15 experiment on the AutoDL data disk."""

import argparse
import os
from pathlib import Path

import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02 import DEFAULT_CONFIG as TASK07_CONFIG
from .e02_capfix import CONFIG as CAPFIX_CONFIG, _hardware
from .e02_capfix_ring import protect_ring
from .e02_captexture import DOCS, G1
from .e02_freeze import verify
from .e02_history import read_briefs
from .e02_uniform_ring_rings import generate_r1


CONFIG = Path("code/configs/E02_uniform_ring.yaml")
SEEDS = (42, 1234)
CAPS = ("g5", "u1", "u2")


class Experiment:
    def __init__(self, root, config=CONFIG):
        self.root = Path(root).resolve()
        self.config = yaml.safe_load(config.read_text(encoding="utf-8"))
        if self.config["protocol"] != "e02-v3-uniform-ring" or self.config["seeds"] != list(SEEDS):
            raise ValueError("Task 15 requires the registered two-seed configuration")
        self.negative_suffix = self.config["negative_suffix"]
        self.base_experiment_config = yaml.safe_load(TASK07_CONFIG.read_text(encoding="utf-8"))
        self.config["gate"] = self.base_experiment_config["gate"]
        capfix = yaml.safe_load(CAPFIX_CONFIG.read_text(encoding="utf-8"))
        self.engine_config = {**self.base_experiment_config["common"], "capfix_prompts": capfix["capfix_prompts"]}
        self.metric_config, self.dino_model, metric_lock = verify(self.root)
        self.all_briefs = read_briefs()
        self.briefs = self.all_briefs[:10]
        self.geometry = self.root / "outputs/e01b/geometry"
        self.legacy = self.root / "outputs/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        archive = read_json(DOCS / "capgeom-inputs.json")
        e02 = self.root / "outputs/e02"
        self.original = e02 / "step1" / archive["config"]["task07_reference"]
        self.previous = e02 / "captexture/runs" / self.config["task13_reference"]
        self.r0 = dict(zip(SEEDS, (e02 / "capfix/rings" / archive["config"]["archived_rings"],
                                  e02 / "capgeom/runs" / stable_hash(archive)[:12] / "ring-validation"), strict=True))
        previous_selection = read_json(self.previous / "selection.json")
        previous_inputs = read_json(self.previous / "inputs.json")
        original = read_json(self.original / "inputs.json")
        if (previous_inputs != read_json(DOCS / "captexture-inputs.json")
                or not previous_selection["complete"] or previous_selection["selected"] != "g5"
                or self.parameters("g5") != previous_inputs["registered_candidates"]["g5"]
                or original["generation_config"] != self.base_experiment_config
                or original["metrics"] != stable_hash(metric_lock)
                or original["briefs"] != stable_hash(self.briefs)
                or file_hash(self.original / "inputs.json") != archive["task07_sha256"]
                or file_hash(self.lut) != original["lut_sha256"]
                or any(file_hash(self.geometry / name) != signature for name, signature in original["geometry"].items())):
            raise ValueError("Task 15 source identities no longer match the reviewed archives")
        ring_archive = {}
        for seed, directory in self.r0.items():
            for brief in self.briefs:
                key = f"{brief['id']}-{seed}"
                proof = read_json(directory / f"{key}.json")
                if file_hash(directory / f"{key}.npz") != proof["cache_sha256"]:
                    raise ValueError(f"Original ring cache changed for {key}")
                ring_archive[key] = proof["cache_sha256"]
        payload = {"config": self.config, "frozen_metrics": stable_hash(metric_lock),
                   "frozen_ring_sampler": protect_ring(original), "original_ring_caches": ring_archive,
                   "task07_inputs_sha256": file_hash(self.original / "inputs.json"),
                   "task13_inputs_sha256": file_hash(self.previous / "inputs.json"),
                   "task13_selection_sha256": file_hash(self.previous / "selection.json")}
        self.output = e02 / "uniform-ring/runs" / stable_hash(payload)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        self.r1 = self.output / "r1-rings"
        write_json(self.output / "inputs.json", payload)
        self.use_ring("r0")

    def parameters(self, method):
        return {**G1, "strength": self.config["cap_strengths"][method],
                "inject_texture" if method == "g5" else "uniform_texture": True}

    def use_ring(self, name):
        self.ring_name = name
        self.original_rings = self.r1 if name == "r1" else self.r0[42]
        self.validation_ring_dir = self.r1 if name == "r1" else self.r0[1234]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ring", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--selection-review", type=Path)
    args = parser.parse_args()
    experiment = Experiment(args.root)
    print(f"Task 15: {experiment.output}", flush=True)
    generate_r1(experiment, _hardware(args.device))
    from .e02_uniform_ring_ring_eval import measure_rings
    decision = measure_rings(experiment, args.device)
    if args.stage == "all":
        from .e02_uniform_ring_full import run
        run(experiment, args.device, decision, args.visual, args.selection_review)


if __name__ == "__main__":
    main()
