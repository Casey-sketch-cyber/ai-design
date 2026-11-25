"""Read the reviewed archives for the CPU stripe diagnostic."""

import csv
from dataclasses import dataclass
from pathlib import Path

from .e01 import read_json
from .e01b import file_hash
from .e02_history import read_briefs
from .e02_sources import SourceAtlas, sources


TASK07_RUN = "c6bb770094d3"
TASK08_RUN = "437980f564bb"
TASK10_RUN = "7b2fcb2c668b"
TASK07_METHODS = ("b2_regional", "v3_1280", "v3_1472", "v3_shared")
TASK08_METHODS = ("c1a", "c1b", "c2", "c3", "c4")
TASK10_METHODS = ("g1", "g2_075", "g2_085", "g3")
TRIAL_BRIEFS = frozenset(f"E01-{i:03d}" for i in range(1, 11))
EARLY_ARCHIVES = {
    "v2_task03": "outputs/e01b/runs/4b57f6e33a53/v2",
    "v2_task04": "outputs/e01b/crease-fix/runs/bc079eabca00/v2",
}


@dataclass(frozen=True)
class DiagnosticAtlas(SourceAtlas):
    cohort: str
    score: Path | None
    expected_v1: float | None
    raw_sha256: str

    @property
    def brief_id(self):
        return self.brief["id"]


def _frozen_history(root, manifest):
    history = manifest["history_evidence"]
    archived = {}
    for relative, expected in history["history_result_csv_sha256"].items():
        if not Path(relative).parent.name.startswith("physical-"):
            continue
        path = root / relative
        if file_hash(path) != expected:
            raise ValueError(f"Archived physical metrics have changed: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["method"], row["brief_id"], int(row["seed"])
                if key in archived:
                    raise ValueError(f"Repeated key in frozen history: {key}")
                archived[key] = row
    if len(archived) != history["physical_count"]:
        raise ValueError("Frozen history does not match its recorded count")
    return archived


def _archive_jobs(root, briefs, run, cohort, methods, seeds, frozen_method=None):
    shape = root / "outputs/e01b/geometry"
    geometry_sha256 = file_hash(shape / "topology.json")
    for method, seed in zip(methods, seeds, strict=True):
        for brief in briefs[:10]:
            identifier = f"{brief['id']}-{seed}"
            graph = run / method / f"{identifier}.png"
            score = run / "evaluation" / method / f"{identifier}.metrics.json"
            score_record = read_json(score)
            raw_hash = score_record["input"]["atlas_sha256"]
            job = DiagnosticAtlas(frozen_method or method, brief, seed, graph, shape, cohort, score,
                                  score_record["variants"]["preflight"]["stripe_index"], raw_hash)
            metadata = read_json(job.metadata)
            if (score_record["input"]["geometry_sha256"] != geometry_sha256
                    or score_record["input"]["run_sha256"] != file_hash(job.metadata)
                    or (metadata["method"], metadata["brief_id"], metadata["seed"], metadata["raw_sha256"])
                    != (method, job.brief_id, seed, raw_hash)):
                raise ValueError(f"Inconsistent archived run for {cohort}/{method}/{identifier}")
            yield job


def registered_atlases(root, manifest):
    briefs = read_briefs()
    archive = _frozen_history(root, manifest)
    for atlas in sources(root, briefs):
        key = atlas.method, atlas.brief["id"], atlas.seed
        row = archive.pop(key)
        yield DiagnosticAtlas(atlas.method, atlas.brief, atlas.seed, atlas.path, atlas.geometry, "history", None,
                              float(row["stripe_index"]), row["atlas_sha256"])
    if archive:
        raise ValueError("Unregistered items in the frozen physical history")
    e02 = root / "outputs/e02"
    yield from _archive_jobs(root, briefs, e02 / "step1" / TASK07_RUN,
                             "task07", TASK07_METHODS, (42,) * len(TASK07_METHODS))
    yield from _archive_jobs(root, briefs, e02 / "capfix/runs" / TASK08_RUN,
                             "task08", TASK08_METHODS, (42,) * len(TASK08_METHODS))
    run = e02 / "capgeom/runs" / TASK10_RUN
    yield from _archive_jobs(root, briefs, run, "task10", TASK10_METHODS, (42,) * len(TASK10_METHODS))
    yield from _archive_jobs(root, briefs, run, "task10", ("seed_1234",), (1234,), frozen_method="g1")
    for method, relative in EARLY_ARCHIVES.items():
        for brief in briefs[:5]:
            graph = root / relative / f"{brief['id']}-42.png"
            score = graph.with_suffix(".metrics.json")
            saved = read_json(score)
            if saved["input"]["run_sha256"] != file_hash(graph.with_suffix(".run.json")):
                raise ValueError(f"Archived historical run metadata changed: {graph}")
            frozen = (saved["input"].get("physical_implementation_sha256")
                      == manifest["files"]["code/dielinediff/metrics/physical.py"])
            value = saved["variants"]["preflight"]["stripe_index"] if frozen else None
            yield DiagnosticAtlas(method, brief, 42, graph, root / "outputs/e01b/geometry", "failure_control",
                                  score, value, saved["input"]["atlas_sha256"])


def registered_synthetic_source(atlas):
    """Use historical texture references within the same first ten brief identities."""
    return (atlas.cohort == "history" and (
        atlas.method == "b2_post" and atlas.brief_id in TRIAL_BRIEFS
        or atlas.method == "b3" and atlas.seed == 42 and atlas.brief_id in ("E01-001", "E01-002")
    ))
