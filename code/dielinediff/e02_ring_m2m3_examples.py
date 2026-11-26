"""Export the registered Task 17 historical controls and native final folds."""

import shutil

from .e01 import write_json
from .e01b import file_hash
from .e02_capgeom_eval import _cohort
from .e02_ring_m2m3 import SEEDS
from .e02_ring_m2m3_full import _archived
from .e02_ring_m2m3_ring_eval import fold_normal_profile
from .e02_uniform_ring_examples import _comparison, _evidence
from .e02_captexture_eval import _write_csv
from .metrics.e02_physical import E02PhysicalEvaluator


COMPARISONS = ("E01-001", "E01-003", "E01-007")
FOLDS = ("E01-003", "E01-007", "E01-009", "E01-010")
LABELS = {"u1": "U1", "u1_sigma28": "U1 sigma 28"}


def export_examples(experiment, groups, chosen):
    root = experiment.output / "examples"
    topology = file_hash(experiment.geometry / "topology.json")
    manifest = {"geometry_sha256": topology, "selected_cap": chosen, "selected_ring": experiment.ring_name,
                "columns": ["B2-regional", "Task 15 U1 + R0", f"Task 17 {LABELS[chosen]} + {experiment.ring_name.upper()}"],
                "seeds": {}}
    physical = E02PhysicalEvaluator(experiment.geometry, device="cpu")
    normal_rows = []
    for seed in SEEDS:
        old = {job.atlas.brief["id"]: job for job in _archived(experiment, methods=("u1",), seed=seed)}
        original = _cohort(groups[seed]["b2_regional"], seed=seed)
        final = _cohort(groups[seed][chosen], seed=seed)
        for brief, (job, _) in final.items():
            rgb = physical._native_atlas(job.atlas.path)
            faces = physical._physical_faces(rgb)
            normal_rows.extend({"method": chosen, "seed": seed, "brief_id": brief, **row}
                               for row in fold_normal_profile(physical, faces))
        destination = root / f"seed-{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        seed_manifest = {"comparisons": {}, "native_folds": {}}
        for brief in COMPARISONS:
            jobs = original[brief][0], old[brief], final[brief][0]
            sources = [_evidence(job, topology) for job in jobs]
            preview = _comparison(jobs, brief, seed, chosen, destination, titles=manifest["columns"])
            seed_manifest["comparisons"][brief] = {"file": preview.relative_to(root).as_posix(), "source": sources}
        for brief in FOLDS:
            name = f"{brief}-folds.png"
            source = experiment.output / f"seed-{seed}" / "native-folds" / name
            target = destination / name
            shutil.copy2(source, target)
            seed_manifest["native_folds"][brief] = {"file": target.relative_to(root).as_posix(),
                                                      "file_sha256": file_hash(target),
                                                      "raw_sha256": file_hash(final[brief][0].atlas.path)}
        manifest["seeds"][str(seed)] = seed_manifest
    write_json(root / "manifest.json", manifest)
    _write_csv(experiment.output / "final-normal-profile.csv", normal_rows)
