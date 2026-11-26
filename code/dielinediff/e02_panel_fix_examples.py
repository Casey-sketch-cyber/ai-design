"""Publish real Task 22 atlas comparisons and the requested native lid cuts."""

import shutil

from .e01 import write_json
from .e01b import file_hash
from .e02_capgeom_eval import _cohort
from .e02_captexture_eval import _write_csv
from .e02_joint_cap_examples import _lid_plate
from .e02_panel_fix import CONTROL, FOCUS, ORIGINAL, SEEDS
from .e02_ring_m2m3_ring_eval import fold_normal_profile
from .e02_uniform_ring_examples import _comparison, _evidence
from .metrics.e02_physical import E02PhysicalEvaluator


def publish_examples(experiment, groups, chosen, selection):
    evaluator = E02PhysicalEvaluator(experiment.geometry, device="cpu")
    rectangles = {face: tuple(evaluator.native_topology["faces"][face]["rect"]) for face in ("T", "Btm")}
    topo = file_hash(experiment.geometry / "topology.json")
    scores = selection["methods"]
    illustrative = selection["selected"] or min(
        (name for name in scores if name != CONTROL),
        key=lambda name: (-scores[name]["win_count"], scores[name]["means"]["stripe_index"]))
    columns = ("B2-regional", "K0 / Task 20", illustrative.upper() + (" (rejected)" if chosen == ORIGINAL else ""))
    base = experiment.output / "examples"
    manifest = {"selected": selection["selected"], "illustrative": illustrative,
                "seed_42_columns": columns, "geometry_sha256": topo, "seeds": {}}
    normals = []
    for seed in SEEDS:
        folder = base / f"seed-{seed}"
        folder.mkdir(exist_ok=True, parents=True)
        entries = _cohort(groups[seed][chosen], seed=seed)
        record = {"comparisons": {}, "native_folds": {}}
        for brief, (job, _) in entries.items():
            physical = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            normals.extend({"seed": seed, "method": chosen, "brief_id": brief, **row}
                           for row in fold_normal_profile(evaluator, physical))
        if seed == 42:
            rows = [_cohort(groups[seed][mode], seed=seed) for mode in ("b2_regional", CONTROL, illustrative)]
            for brief in FOCUS:
                jobs = tuple(block[brief][0] for block in rows)
                folds = _comparison(jobs, brief, seed, illustrative, folder, titles=columns)
                plate = folder / f"{brief}-lids.png"
                _lid_plate(plate, jobs, columns, rectangles)
                record["comparisons"][brief] = {
                    "file": folds.relative_to(base).as_posix(), "lids": plate.relative_to(base).as_posix(),
                    "lids_sha256": file_hash(plate), "sources": [_evidence(job, topo) for job in jobs],
                }
        for brief in FOCUS:
            name = f"{brief}-folds.png"
            path = folder / name
            shutil.copy2(experiment.output / f"seed-{seed}" / "native-folds" / name, path)
            record["native_folds"][brief] = {"file": path.relative_to(base).as_posix(), "sha256": file_hash(path)}
        manifest["seeds"][str(seed)] = record
    write_json(base / "manifest.json", manifest)
    _write_csv(experiment.output / "normal-profile.csv", normals)
