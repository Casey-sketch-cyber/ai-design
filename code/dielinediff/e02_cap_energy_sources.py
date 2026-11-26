"""Locate immutable E02 baselines and the archived E0 and live Task 18 caps."""

from dataclasses import replace

from .e01 import read_json
from .e01b import file_hash
from .e02 import BENCHMARK_BASELINES, Sample
from .e02_cap_energy import CAPS, TASK17_METHOD
from .e02_ring_m2m3_full import _archived
from .e02_sources import SourceAtlas
from .e02_uniform_ring_full import _sample


def initial_paths(atlas):
    return {face: atlas.path.with_name(f"{atlas.path.stem}.init-{face}.npy") for face in ("T", "Btm")}


class EnergySample(Sample):
    """A generated cap is reusable only when both exact pre-SDEdit inputs survive."""

    def validate_generation(self):
        if not super().validate_generation():
            return False
        initial = read_json(self.atlas.metadata).get("details", {}).get("initial_images", {})
        return all(path.is_file() and initial.get(face, {}).get("sha256") == file_hash(path)
                   for face, path in initial_paths(self.atlas).items())


def archived_e0(experiment, seed, *, alias=True):
    selection = read_json(experiment.previous17 / "selection.json")
    scores = selection[f"scores_{seed}_sha256"]
    run = experiment.previous17 if seed == 42 else experiment.previous17 / "validation"
    for brief in experiment.briefs:
        key = f"{brief['id']}-{seed}"
        atlas = SourceAtlas(TASK17_METHOD, brief, seed, run / TASK17_METHOD / f"{key}.png", experiment.geometry)
        sample = _sample(atlas, run)
        if scores[f"{TASK17_METHOD}/{key}"] != file_hash(sample.score) or not sample.validate_generation():
            raise ValueError(f"The reviewed E0 image or its score changed for {key}")
        yield replace(sample, atlas=replace(atlas, method="e0")) if alias else sample


def original_baselines(experiment, seed):
    selected = read_json(experiment.previous17 / "selection.json")[f"scores_{seed}_sha256"]
    for sample in _archived(experiment, methods=BENCHMARK_BASELINES, seed=seed):
        atlas = sample.atlas
        if file_hash(sample.score) != selected[f"{atlas.method}/{atlas.identifier}"]:
            raise ValueError(f"Task 17 baseline no longer matches {atlas.method}/{atlas.identifier}")
        yield sample


def sample_caps(experiment, *, methods, seed=42):
    for method in methods:
        if method not in CAPS:
            raise ValueError(f"Unregistered Task 18 cap {method}")
        if method == "e0" and not getattr(experiment, "is_validation", False):
            yield from archived_e0(experiment, seed)
            continue
        for brief in experiment.briefs:
            key = f"{brief['id']}-{seed}"
            path = experiment.output / method / f"{key}.png"
            atlas = SourceAtlas(method, brief, seed, path, experiment.geometry)
            ordinary = _sample(atlas, experiment.output)
            yield EnergySample(atlas, ordinary.score, ordinary.preflight)
