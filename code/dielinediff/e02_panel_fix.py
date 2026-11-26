"""Task 22: run panel cleanup and the dark-band repair on the frozen ring."""

import argparse
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import yaml

from .e01 import read_json
from .e01b import file_hash, stable_hash
from .e02_cap_energy import Experiment as FrozenExperiment
from .e02_cap_energy_sources import EnergySample, archived_e0
from .e02_captexture import DOCS
from .e02_sources import SourceAtlas
from .e02_uniform_ring_full import _sample


CONFIG = Path("code/configs/E02_panel_fix.yaml")
CONTROL = "k0"
ORIGINAL = "j0"
RECIPES = (CONTROL, "k1", "k2", "k3", "k4")
SEEDS = (42, 1234)
FOCUS = ("E01-003", "E01-004", "E01-005", "E01-009", "E01-010")
OLD_RUN = "j3_g030"


class Experiment(FrozenExperiment):
    def __init__(self, root, config=CONFIG):
        options = yaml.safe_load(config.read_text(encoding="utf-8"))
        if (options["protocol"] != "e02-v3-panel-fix" or options["seeds"] != list(SEEDS)
                or options["ring"] != "r0"):
            raise ValueError("Task 22 needs the preregistered panel experiment configuration")
        previous = read_json(DOCS / "joint-cap-inputs.json")
        suite = Path(root).resolve() / "outputs/e02/joint-cap/runs" / stable_hash(previous)[:12]
        if (previous != read_json(suite / "inputs.json")
                or read_json(suite / "selection.json") != read_json(DOCS / "joint-cap-selection.json")):
            raise ValueError("The reviewed Task 20 archive changed")
        super().__init__(root, namespace="panel-fix", identity_extra={
            "panel_fix": options,
            "task20_inputs_sha256": file_hash(suite / "inputs.json"),
            "task20_selection_sha256": file_hash(suite / "selection.json"),
        })
        self.options = options
        self.previous20 = suite
        self.engine_config = {**self.engine_config, "panel_fix": options,
                              "joint_cap": previous["joint_cap"]}
        self.samples = partial(samples, self)

    def graphics(self, seed):
        return self.output if seed == 42 else self.output / "validation"

    def stage1(self, seed):
        return self.output / "outpaint" / f"seed-{seed}"


def samples(experiment, *, methods, seed=42):
    for method in methods:
        if method == ORIGINAL:
            for sample in archived_e0(experiment, seed, alias=False):
                yield replace(sample, atlas=replace(sample.atlas, method=ORIGINAL))
            continue
        if method not in RECIPES:
            raise ValueError(f"Unknown Task 22 recipe: {method}")
        if method == CONTROL and seed != 42:
            raise ValueError("The K0 control was archived at seed 42")
        directory, source = (experiment.previous20, OLD_RUN) if method == CONTROL else (experiment.graphics(seed), method)
        for brief in experiment.briefs:
            key = f"{brief['id']}-{seed}"
            atlas = SourceAtlas(source, brief, seed, directory / source / f"{key}.png", experiment.geometry)
            job = _sample(atlas, directory)
            job = EnergySample(atlas, job.score, job.preflight)
            if method == CONTROL:
                recorded = read_json(DOCS / "joint-cap-content-review.json")[OLD_RUN]["samples"][brief["id"]]
                if not job.validate_generation() or recorded["raw_sha256"] != file_hash(atlas.path):
                    raise ValueError(f"The reviewed K0 archive changed for {key}")
                job = replace(job, atlas=replace(atlas, method=method))
            yield job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="all",
                        choices=("paths", "diagnose", "inpaint", "outpaint", "all"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--cap-reviews", type=Path)
    parser.add_argument("--fold-reviews", type=Path)
    parser.add_argument("--tie-reviews", type=Path)
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root or DIELINEDIFF_ROOT must refer to the AutoDL data disk")
    experiment = Experiment(args.root, args.config)
    print(f"Task 22: {experiment.output}", flush=True)
    if args.action != "paths":
        from .e02_panel_fix_workflow import run
        run(experiment, args)


if __name__ == "__main__":
    main()
