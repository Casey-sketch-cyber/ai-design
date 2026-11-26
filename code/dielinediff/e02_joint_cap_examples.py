"""Publish source-bound previews and native lid pixels for Task 20."""

import shutil
from PIL import Image, ImageDraw, ImageFont

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_cap_energy_visual import FOCUS
from .e02_capgeom_eval import _cohort
from .e02_captexture_eval import _write_csv
from .e02_joint_cap import CONTROL, SEEDS
from .e02_ring_m2m3_ring_eval import fold_normal_profile
from .e02_uniform_ring_examples import _comparison, _evidence
from .metrics.e02_physical import E02PhysicalEvaluator


GROUPS = ("E01-003", "E01-007", "E01-009")


def _lid_plate(path, jobs, names, rects):
    lids = []
    for job in jobs:
        with Image.open(job.atlas.path) as source:
            lids.append({face: source.crop(rect).convert("RGB") for face, rect in rects.items()})
    w = max(image.width for pair in lids for image in pair.values())
    h = max(image.height for pair in lids for image in pair.values())
    board = Image.new("RGB", (len(lids) * (w + 12) + 12, 2 * (h + 30) + 12), "white")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=16)
    for index, (name, pair) in enumerate(zip(names, lids, strict=True)):
        x = 12 + index * (w + 12)
        for row, face in enumerate(("T", "Btm")):
            y = 12 + row * (h + 30)
            draw.text((x, y), f"{name} | {face} | Native RGB", font=font, fill="black")
            board.paste(pair[face], (x, y + 20))
    board.save(path, optimize=True)


def publish_examples(experiment, groups, chosen, selection):
    output = experiment.output / "examples"
    geometry = file_hash(experiment.geometry / "topology.json")
    evaluator = E02PhysicalEvaluator(experiment.geometry, device="cpu")
    rects = {face: tuple(evaluator.native_topology["faces"][face]["rect"]) for face in ("T", "Btm")}
    eligible = selection["selected"] is not None
    measured = selection["methods"]
    exemplar = chosen if eligible else min(
        (name for name in measured if name != CONTROL),
        key=lambda method: (-measured[method]["win_count"], measured[method]["means"]["stripe_index"]))
    third_title = exemplar.upper() if eligible else f"{exemplar.upper()} (rejected)"
    titles = ("B2-regional", "J0 / Task 17", third_title)
    manifest = {"comparative_candidate": exemplar, "selected": selection["selected"],
                "comparative_titles": titles, "seeds": {}, "geometry_sha256": geometry}
    normal = []
    for seed in SEEDS:
        sub = output / f"seed-{seed}"
        sub.mkdir(parents=True, exist_ok=True)
        baseline = _cohort(groups[seed]["b2_regional"], seed=seed)
        if CONTROL in groups[seed]:
            control = _cohort(groups[seed][CONTROL], seed=seed)
        else:
            from .e02_cap_energy_sources import archived_e0
            control = {job.atlas.brief["id"]: (job, read_json(job.score)) for job in archived_e0(experiment, seed)}
        final = _cohort(groups[seed][chosen], seed=seed)
        record = {"comparisons": {}, "native_folds": {}}
        for brief, (job, _) in final.items():
            faces = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            normal.extend({"method": chosen, "seed": seed, "brief_id": brief, **row}
                          for row in fold_normal_profile(evaluator, faces))
        if seed == 42 or eligible:
            art = _cohort(groups[seed][exemplar], seed=seed)
            for brief in GROUPS:
                jobs = (baseline[brief][0], control[brief][0], art[brief][0])
                compare = _comparison(jobs, brief, seed, exemplar, sub, titles=titles)
                capplate = sub / f"{brief}-lids.png"
                _lid_plate(capplate, jobs, titles, rects)
                record["comparisons"][brief] = {"file": compare.relative_to(output).as_posix(),
                    "lids": capplate.relative_to(output).as_posix(), "lids_sha256": file_hash(capplate),
                    "sources": [_evidence(job, geometry) for job in jobs]}
        for brief in FOCUS:
            name = f"{brief}-folds.png"
            dst = sub / name
            shutil.copy2(experiment.output / f"seed-{seed}" / "native-folds" / name, dst)
            record["native_folds"][brief] = {"file": dst.relative_to(output).as_posix(), "sha256": file_hash(dst)}
        manifest["seeds"][str(seed)] = record
    write_json(output / "manifest.json", manifest)
    _write_csv(experiment.output / "normal-profile.csv", normal)
