"""Task 20: joint ring and cap generation and frozen E02 gates."""

import argparse
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import yaml

from .e01 import read_json
from .e02_cap_energy import Experiment as CapEnergyExperiment
from .e02_cap_energy_sources import EnergySample, archived_e0
from .e02_sources import SourceAtlas
from .e02_uniform_ring_full import _sample


CONFIG = Path("code/configs/E02_joint_cap.yaml")
SEEDS = (42, 1234)
CONTROL = "j0"
CAP_REFINEMENTS = ("j1", "j2", "j3_g020", "j3_g030")
JOINT_RECIPES = ("j4", "j4b", "j5")
RECIPES = (CONTROL, *CAP_REFINEMENTS, *JOINT_RECIPES)


class Experiment(CapEnergyExperiment):
    def __init__(self, root, config=CONFIG):
        options = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (options["protocol"] != "e02-v3-joint-cap" or options["seeds"] != list(SEEDS)
                or options["ring"] != "r0"):
            raise ValueError("Task 20 needs the registered seeds, R0 and E02 protocol")
        super().__init__(root, namespace="joint-cap", identity_extra={"joint_cap": options})
        self.options = options
        self.engine_config = {**self.engine_config, "joint_cap": options}
        self.samples = partial(samples, self)

    def graphics(self, seed):
        return self.output if seed == 42 else self.output / "validation"

    def stage1(self, recipe, seed):
        return self.output / "stage1" / recipe / f"seed-{seed}"


def samples(experiment, *, methods, seed=42):
    for method in methods:
        if method not in RECIPES:
            raise ValueError(f"Unknown Task 20 recipe: {method}")
        if method == CONTROL:
            for job in archived_e0(experiment, seed, alias=False):
                yield replace(job, atlas=replace(job.atlas, method=method))
            continue
        directory = experiment.graphics(seed)
        for brief in experiment.briefs:
            ident = f"{brief['id']}-{seed}"
            path = directory / method / f"{ident}.png"
            atlas = SourceAtlas(method, brief, seed, path, experiment.geometry)
            job = _sample(atlas, directory)
            yield EnergySample(atlas, job.score, job.preflight) if method in CAP_REFINEMENTS else job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="all",
                        choices=("paths", "all", "j1", "j2", "j3", "j4", "j4b", "j5", "publish"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cap-reviews", type=Path)
    parser.add_argument("--banner-reviews", type=Path)
    parser.add_argument("--fold-reviews", type=Path)
    parser.add_argument("--tie-reviews", type=Path)
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root or DIELINEDIFF_ROOT must point to AutoDL data")
    experiment = Experiment(args.root, args.config)
    print(f"Task 20: {experiment.output}", flush=True)
    if args.action != "paths":
        from .e02_joint_cap_workflow import run
        run(experiment, args)


if __name__ == "__main__":
    main()
