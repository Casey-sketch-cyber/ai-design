"""Run the registered Task 17 ring ablations, followed by the locked cap comparison."""

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


CONFIG = Path("code/configs/E02_ring_m2m3.yaml")
SEEDS = (42, 1234)
RINGS = ("r0", "m3b", "m3", "m2", "m2_m3")
CAPS = ("u1", "u1_sigma28")


class Experiment:
    def __init__(self, root, config=CONFIG):
        self.root = Path(root).resolve()
        self.config = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (self.config["protocol"] != "e02-v3-ring-m2m3" or self.config["seeds"] != list(SEEDS)
                or self.config["ring_candidate_order"] != list(RINGS)):
            raise ValueError("Task 17 requires the registered two-seed experiment and five ring recipes")
        if self.config["m3"] != {"roll_stride": 37, "extra_buffer_latents_per_side": 8}:
            raise ValueError("Task 17 uses the registered M3 phase and canvas expansion")
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
        step10 = read_json(DOCS / "capgeom-inputs.json")
        step15 = read_json(DOCS / "uniform-ring-inputs.json")
        e02 = self.root / "outputs/e02"
        self.original = e02 / "step1" / step10["config"]["task07_reference"]
        self.previous = e02 / "captexture/runs" / step15["config"]["task13_reference"]
        self.previous15 = e02 / "uniform-ring/runs" / self.config["task15_reference"]
        original = read_json(self.original / "inputs.json")
        selection = read_json(self.previous15 / "selection.json")
        if (step15 != read_json(self.previous15 / "inputs.json")
                or selection != read_json(DOCS / "uniform-ring-selection.json")
                or stable_hash(step15)[:12] != self.config["task15_reference"]
                or selection["ring"] != "r0" or selection["cap"]["selected"] != "u1"
                or not selection["complete"] or not selection["visual_review_complete"]
                or original["generation_config"] != self.base_experiment_config
                or original["metrics"] != stable_hash(metric_lock)
                or original["briefs"] != stable_hash(self.briefs)
                or file_hash(self.original / "inputs.json") != step15["task07_inputs_sha256"]
                or file_hash(self.lut) != original["lut_sha256"]
                or any(file_hash(self.geometry / name) != value for name, value in original["geometry"].items())):
            raise ValueError("Task 17 sources differ from the reviewed archives or frozen metric registration")
        self.r0 = {42: e02 / "capfix/rings" / step10["config"]["archived_rings"],
                   1234: e02 / "capgeom/runs" / stable_hash(step10)[:12] / "ring-validation"}
        for seed, run in self.r0.items():
            for brief in self.briefs:
                key = f"{brief['id']}-{seed}"
                if file_hash(run / f"{key}.npz") != step15["original_ring_caches"][key]:
                    raise ValueError(f"Original ring bytes changed: {key}")
        payload = {"config": self.config, "frozen_metrics": stable_hash(metric_lock),
                   "frozen_ring_sampler": protect_ring(original),
                   "original_ring_caches": step15["original_ring_caches"],
                   "task07_inputs_sha256": file_hash(self.original / "inputs.json"),
                   "task15_inputs_sha256": file_hash(self.previous15 / "inputs.json"),
                   "task15_selection_sha256": file_hash(self.previous15 / "selection.json")}
        self.output = e02 / "ring-m2m3/runs" / stable_hash(payload)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        self.ring_paths = {"r0": self.r0}
        for method in RINGS[1:]:
            self.ring_paths[method] = dict.fromkeys(SEEDS, self.output / "rings" / method)
        write_json(self.output / "inputs.json", payload)
        self.use_ring("r0")

    def use_ring(self, method):
        self.ring_name = method
        self.original_rings = self.ring_paths[method][42]
        self.validation_ring_dir = self.ring_paths[method][1234]

    def parameters(self, cap):
        settings = self.config["caps"]
        params = {**G1, "strength": settings["strength"], "uniform_texture": True}
        if cap != "u1":
            params["texture_sigma_px_at_scale_1"] = settings["sigma_px_at_scale_2"][cap] / 2
        return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "rings", *RINGS[1:]), default="all")
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--selection-review", type=Path)
    args = parser.parse_args()
    experiment = Experiment(args.root)
    print(f"Task 17: {experiment.output}", flush=True)
    from .e02_ring_m2m3_rings import generate_rings
    ring_only = args.stage in RINGS[1:]
    generate_rings(experiment, _hardware(args.device), (args.stage,) if ring_only else RINGS[1:])
    if ring_only:
        return
    from .e02_ring_m2m3_ring_eval import measure_rings
    decision = measure_rings(experiment, args.device)
    if args.stage == "all":
        from .e02_ring_m2m3_full import run
        run(experiment, args.device, decision, args.visual, args.selection_review)


if __name__ == "__main__":
    main()
