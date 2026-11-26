"""Export and verify original-pixel lid reviews and final proof images."""

import shutil

from PIL import Image, ImageDraw, ImageFont

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_cap_energy import CAPS, SEEDS
from .e02_cap_energy_sources import archived_e0
from .e02_capgeom_eval import _cohort
from .e02_ring_m2m3_ring_eval import fold_normal_profile
from .e02_uniform_ring_examples import _comparison, _evidence
from .e02_captexture_eval import _write_csv
from .metrics.e02_physical import E02PhysicalEvaluator


FOCUS = ("E01-003", "E01-007", "E01-009")
COMPARISONS = ("E01-001", "E01-003", "E01-007")
CONTENT_FIELDS = ("collage", "ghosting", "mirror_repeat")


def export_content(experiment, method, *, seed=42):
    folder = experiment.output / "cap-content" / method / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    topology = read_json(experiment.geometry / "topology.json")
    rectangles = {face: tuple(topology["faces"][face]["rect"]) for face in ("T", "Btm")}
    records = {}
    font = ImageFont.load_default(size=16)
    for name in experiment.samples(methods=(method,), seed=seed):
        atlas = name.atlas
        with Image.open(atlas.path) as picture:
            lids = {cap: picture.crop(rect).convert("RGB") for cap, rect in rectangles.items()}
        width = max(tile.width for tile in lids.values()) + 24
        height = sum(tile.height + 33 for tile in lids.values()) + 40
        output = Image.new("RGB", (width, height), "white")
        pen = ImageDraw.Draw(output)
        pen.text((12, 9), f"{atlas.identifier} | {method.upper()} | original RGB pixels", font=font, fill="black")
        y = 43
        for cap, tile in lids.items():
            pen.text((12, y), "Top" if cap == "T" else "Bottom", font=font, fill="black")
            output.paste(tile, (12, y + 21))
            y += tile.height + 33
        path = folder / f"{atlas.identifier}.png"
        output.save(path, optimize=True)
        records[atlas.brief["id"]] = {"raw_sha256": file_hash(atlas.path), "review_view": path.name}
    if len(records) != 10:
        raise ValueError("The cap appearance review needs the original ten registered briefs")
    fingerprint = stable_hash({brief: info["raw_sha256"] for brief, info in records.items()})
    write_json(folder / "request.json", {"method": method, "seed": seed, "fingerprint": fingerprint, "samples": records})
    return {"method": method, "seed": seed, "fingerprint": fingerprint, "samples": records}


def verify_content(experiment, path):
    content = read_json(path)
    if set(content) != set(CAPS):
        raise ValueError("Provide one review for each of E0–E4")
    guards = {}
    for method in CAPS:
        item = content[method]
        request = read_json(experiment.output / "cap-content" / method / "seed-42/request.json")
        actual = {job.atlas.brief["id"]: file_hash(job.atlas.path)
                  for job in experiment.samples(methods=(method,), seed=42)}
        samples = item.get("samples")
        if (item.get("fingerprint") != request["fingerprint"] or stable_hash(actual) != request["fingerprint"]
                or not isinstance(samples, dict) or set(samples) != set(actual)):
            raise ValueError(f"Stale or incomplete cap review: {method}")
        flagged = []
        for brief, comment in samples.items():
            flags = tuple(comment.get(field) for field in CONTENT_FIELDS)
            if (any(not isinstance(value, bool) for value in flags) or not comment.get("notes", "").strip()
                    or comment.get("raw_sha256") != actual[brief]):
                raise ValueError(f"Invalid cap review entry: {method}/{brief}")
            if any(flags):
                flagged.append({"brief_id": brief, **{field: comment[field] for field in CONTENT_FIELDS}})
        guards[method] = {"passed": not flagged, "flagged": flagged, "reviewed": len(samples),
                          "fingerprint": request["fingerprint"]}
    return guards, content


def export_examples(experiment, groups, chosen, *, reference_only=False):
    destination = experiment.output / "examples"
    topology = file_hash(experiment.geometry / "topology.json")
    columns = ("B2-regional", "Task 17 E0", "Task 18 E3 (rejected)" if reference_only else f"Task 18 {chosen.upper()}")
    manifest = {"selected": None if reference_only else chosen, "reference_only": reference_only,
                "columns": list(columns), "geometry_sha256": topology, "seeds": {}}
    normal_rows = []
    evaluator = E02PhysicalEvaluator(experiment.geometry, device="cpu")
    for seed in SEEDS:
        old = {job.atlas.brief["id"]: job for job in archived_e0(experiment, seed)}
        baseline = _cohort(groups[seed]["b2_regional"], seed=seed)
        final = _cohort(groups[seed][chosen], seed=seed)
        folder = destination / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        files = {"comparisons": {}, "native_folds": {}}
        for brief, (job, _) in final.items():
            faces = evaluator._physical_faces(evaluator._native_atlas(job.atlas.path))
            normal_rows.extend({"method": chosen, "seed": seed, "brief_id": brief, **row}
                               for row in fold_normal_profile(evaluator, faces))
        if reference_only and seed == 42:
            examples = _cohort(groups[seed]["e3"], seed=seed)
        else:
            examples = final
        comparisons = () if reference_only and seed == 1234 else COMPARISONS
        for brief in comparisons:
            jobs = baseline[brief][0], old[brief], examples[brief][0]
            inputs = [_evidence(job, topology) for job in jobs]
            preview = _comparison(jobs, brief, seed, chosen, folder, titles=columns)
            files["comparisons"][brief] = {"file": preview.relative_to(destination).as_posix(), "sources": inputs}
        for brief in FOCUS:
            filename = f"{brief}-folds.png"
            src = experiment.output / f"seed-{seed}" / "native-folds" / filename
            dst = folder / filename
            shutil.copy2(src, dst)
            files["native_folds"][brief] = {"file": dst.relative_to(destination).as_posix(),
                                            "file_sha256": file_hash(dst)}
        manifest["seeds"][str(seed)] = files
    write_json(destination / "manifest.json", manifest)
    _write_csv(experiment.output / "normal-profile.csv", normal_rows)
