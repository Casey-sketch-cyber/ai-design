"""Immutable E02 historical sources and working directories on the data disk."""

from dataclasses import dataclass
from pathlib import Path

from .e01b import SEEDS


LEGACY_TAG = "full-52dd11c30fc1"
V2_RUN = "b598a9650a8a"
POST_TAG = "553a10692ea7"
LEGACY_METHODS = {"b2": "b2", "b3": "b3", "b4": "b4", "v1": "ours"}
REFERENCE_METHODS = ("b2", "b2_post", "b3", "b4", "v1", "v2")


@dataclass(frozen=True)
class SourceAtlas:
    method: str
    brief: dict
    seed: int
    path: Path
    geometry: Path

    @property
    def metadata(self):
        return self.path.with_suffix(".run.json")

    @property
    def identifier(self):
        return f"{self.brief['id']}-{self.seed}"


def sources(root, briefs, *, trial_only=False):
    legacy_geo = root / "outputs/geometry"
    e01b = root / "outputs/e01b"
    v2_geo = e01b / "geometry"
    legacy = root / "outputs/e01" / LEGACY_TAG
    bottom = e01b / "bottom-frame"
    for method in REFERENCE_METHODS:
        if method == "v2":
            directory = bottom / "runs" / V2_RUN / "v2"
            geometry = v2_geo
            subset, seeds = briefs[:5], SEEDS[:1]
        else:
            if method == "b2_post":
                directory = bottom / "baselines" / POST_TAG / "b2_post"
            else:
                directory = legacy / LEGACY_METHODS[method]
            geometry = legacy_geo
            subset, seeds = (briefs[:10], SEEDS[:1]) if trial_only else (briefs, SEEDS)
        for brief in subset:
            for seed in seeds:
                yield SourceAtlas(method, brief, seed, directory / f"{brief['id']}-{seed}.png", geometry)
