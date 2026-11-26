"""Measure cap texture along millimetre bands without changing the frozen gates."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_capfix_ring import SIDE_LABELS, TASK07_RING
from .e02_capgeom_eval import _crease_rows
from .metrics import _oriented_strip
from .metrics.physical import PIXELS_PER_MM, STRIPE_ENERGY_FLOOR, PhysicalEvaluator
from .methods import v3_capgeom_geometry, v3_capgeom_init, v3_ring_midscale_cap
from .methods.v3_capfix_geometry import cap_at_original_scale
from .methods.v3_geometry import build_cap_layout, build_ring_layout


CAP_FOLDS = ("F-T", "B-Btm")
CAP_FACES = ("T", "Btm")
STAGES = ("ring", "initial", "final")
MM_BINS = 20
TASK17 = "c5b232622150"
FOCUS = {
    (42, "E01-003"): ("B-Btm",),
    (42, "E01-007"): ("B-Btm",),
    (42, "E01-009"): ("F-T",),
    (1234, "E01-005"): ("B-Btm", "F-T", "F-R"),
}


def tangential_energy(face, edge):
    """Bin the frozen v1 squared tangential RGB difference at 4 px/mm."""
    step = int(PIXELS_PER_MM)
    strip = _oriented_strip(face, edge, MM_BINS * step).astype(np.float32) / 255
    if strip.shape[0] < 2 or strip.shape[1] != MM_BINS * step:
        raise ValueError(f"Texture face too small for {MM_BINS} mm on {edge}")
    squared = np.square(np.diff(strip, axis=0))
    return squared.reshape(len(squared), MM_BINS, step, 3).mean(axis=(0, 2, 3))


def _hash_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


class CapEnergyCurves:
    def __init__(self, geometry, *, cap_band):
        self.geometry = Path(geometry)
        self.evaluator = PhysicalEvaluator(self.geometry, device="cpu")
        self.topology = self.evaluator.native_topology
        with Image.open(self.geometry / "region_map.png") as image:
            self.labels = np.array(image)
        with Image.open(self.geometry / "bleed_owner.png") as image:
            self.owners = np.array(image)
        self.ring = build_ring_layout(self.topology, *TASK07_RING)
        self.cap_band = cap_band
        self.side_mask = np.isin(self.labels, SIDE_LABELS)
        self.caps = {face: build_cap_layout(self.topology, self.ring, face, band_latent=cap_band)
                     for face in CAP_FACES}
        self.folds = {f"{s['a']['face']}-{s['b']['face']}": (s["a"], s["b"]) for s in self.evaluator.creases}
        if not set(CAP_FOLDS) <= self.folds.keys():
            raise ValueError("The frozen cap folds differ from the registered geometry")

    def inspect(self, raw_path, ring_path, *, initial_paths=None):
        raw_path, ring_path = Path(raw_path), Path(ring_path)
        record_path = raw_path.with_suffix(".run.json")
        record = read_json(record_path)
        raw_hash = file_hash(raw_path)
        ring_hash = file_hash(ring_path)
        ring_proof_path = ring_path.with_suffix(".json")
        ring_proof = read_json(ring_proof_path)
        if (raw_hash != record["raw_sha256"]
                or ring_hash != record["ring"]["cache_sha256"] or ring_hash != ring_proof["cache_sha256"]
                or (record["brief_id"], record["seed"]) != (ring_proof["brief_id"], ring_proof["seed"])):
            raise ValueError(f"Raw atlas and frozen R0 archive differ for {raw_path}")
        with np.load(ring_path, allow_pickle=False) as data:
            ring_rgb = data["rgb"]
        if _hash_array(ring_rgb) != record["ring"]["ring_rgb_sha256"]:
            raise ValueError(f"Incorrect ring RGB bytes from {ring_path}")
        ring_image = Image.fromarray(ring_rgb)
        final = self.evaluator._native_atlas(raw_path)
        caps = {}
        texture_sha = {}
        candidate = record["candidate"]
        if not candidate["tight"]:
            raise ValueError("Expected tight U1 cap registration")
        for face in CAP_FACES:
            solver = record["details"]["caps"][face]["sdedit"]
            layout = v3_capgeom_geometry.build_capgeom_layout(
                self.caps[face], self.ring, self.cap_band, int(candidate["cap_scale"]))
            if [layout.width_px, layout.height_px] != record["details"]["caps"][face]["size_px"]:
                raise ValueError(f"Cap bounds differ from the registered run for {raw_path}/{face}")
            rgb = np.asarray(layout.read_frozen_ring_rgb(ring_image))
            mask = layout.known_mask.repeat(8, 0).repeat(8, 1)
            if initial_paths is None:
                if record["method"] != "u1_sigma28" or candidate["texture_sigma_px_at_scale_1"] != 14.0:
                    raise ValueError(f"No archived U1 sigma 28 reconstruction source at {raw_path}")
                harmonic, proof = v3_capgeom_init.harmonic_fill(rgb, mask)
                initial, measured = v3_ring_midscale_cap.inject_texture_at_scale(
                    harmonic, rgb, layout, sigma_px_at_scale_1=candidate["texture_sigma_px_at_scale_1"])
                for field in ("applied_center_residual_rms", "applied_edge_residual_rms",
                              "applied_core_residual_rms"):
                    # Use the already stored Task 17 proof as an independent reconstruction check.
                    np.testing.assert_allclose(measured[field], solver["texture_injection"][field],
                                               rtol=5e-6, atol=1e-8)
                texture_sha[face] = {"float32_rgb_sha256": _hash_array(initial),
                                     "harmonic_iterations": proof["channel_iterations"]}
            else:
                source = Path(initial_paths[face])
                initial = np.load(source, allow_pickle=False)
                texture_sha[face] = {"float32_rgb_sha256": _hash_array(initial),
                                     "file_sha256": file_hash(source), "path": str(source)}
                documented = solver.get("initial_sha256")
                if documented is not None and texture_sha[face]["file_sha256"] != documented:
                    raise ValueError(f"Recorded initial file differs for {raw_path}/{face}")
            if (initial.shape != rgb.shape or initial.dtype != np.float32 or not np.isfinite(initial).all()
                    or initial.min() < 0 or initial.max() > 1
                    or not np.array_equal(initial[mask], rgb[mask].astype(np.float32) / 255)):
                raise ValueError(f"Invalid pre-SDEdit chart source or a moved frozen strip at {raw_path}/{face}")
            rgb8 = Image.fromarray(np.rint(initial * 255).astype(np.uint8))
            caps[face] = cap_at_original_scale(layout, rgb8)
        projected = v3_capgeom_geometry.compose_capgeom_atlas(
            self.topology, self.labels, self.owners, self.ring, ring_image, caps)
        projected = self.evaluator._native_atlas(projected)
        if (not np.array_equal(final[self.side_mask], projected[self.side_mask])
                or _hash_array(projected[self.side_mask]) != record["ring"]["body_rgb_sha256"]):
            raise ValueError(f"Cap diagnostic changed frozen native side pixels for {raw_path}")
        init_faces = self.evaluator._physical_faces(projected)
        final_faces = self.evaluator._physical_faces(final)
        folded = {row["fold"]: row for row in _crease_rows(self.evaluator, final_faces)}
        return init_faces, final_faces, folded, {
            "raw_path": str(raw_path), "raw_sha256": raw_hash, "run_sha256": file_hash(record_path),
            "ring_path": str(ring_path), "ring_sha256": ring_hash, "ring_proof_sha256": file_hash(ring_proof_path),
            "projected_initial_rgb_sha256": _hash_array(projected), "initial_caps": texture_sha,
        }

    def profiles(self, initial_faces, final_faces, fold):
        result = {}
        for endpoint in self.folds[fold]:
            face, edge = endpoint["face"], endpoint["edge"]
            if face in CAP_FACES:
                result["initial"] = tangential_energy(initial_faces[face], edge)
                result["final"] = tangential_energy(final_faces[face], edge)
            elif fold in CAP_FOLDS:
                result["ring"] = tangential_energy(final_faces[face], edge)
            else:
                result[f"ring:{face}"] = tangential_energy(final_faces[face], edge)
        return result


def _curve_summary(values):
    near = float(values[:3].mean())
    reference = float(values[9:12].mean())
    # Descriptive marker. The threshold is independent of the frozen v1 gate.
    sustained = next((i for i in range(3, MM_BINS - 2)
                      if near > 0 and (values[i:i + 3] < 0.8 * near).all()), None)
    return {"energy_0_3mm": near, "energy_9_12mm": reference,
            "reference_over_near": reference / near if near > 0 else None,
            "v1_side_reconstruction": 1 + max(near - reference, 0) / (reference + STRIPE_ENERGY_FLOOR),
            "first_3mm_below_80pct_near_mm": sustained}


def _records(method, seed, brief, fold, profiles):
    for stage, values in profiles.items():
        for mm, value in enumerate(values):
            yield {"method": method, "seed": seed, "brief_id": brief, "fold": fold, "stage": stage,
                   "start_mm": mm, "end_mm": mm + 1, "energy": float(value)}


def _write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ring": ("#607280", "--", "Frozen ring"), "initial": ("#C16731", "-", "Before SDEdit"),
        "final": ("#256584", "-", "After SDEdit"), "ring:F": ("#607280", "--", "Front ring"),
        "ring:R": ("#856C66", "-", "Right ring"), "ring:L": ("#856C66", "-", "Left ring"),
        "ring:B": ("#607280", "--", "Back ring"),
    }
    figure, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(10, 2.85 * ((len(panels) + 1) // 2)),
                               squeeze=False, layout="constrained")
    for axis, panel in zip(axes.flat, panels):
        title, profile = panel
        for stage, values in profile.items():
            color, line, label = styles[stage]
            axis.plot(np.arange(MM_BINS) + 0.5, np.asarray(values) * 1e3,
                      color=color, linestyle=line, linewidth=1.6, label=label)
        axis.axvspan(0, 3, color="#A4BAC0", alpha=0.17)
        axis.axvspan(9, 12, color="#A4BAC0", alpha=0.17)
        axis.set(title=title, xlim=(0, MM_BINS), xlabel="Inward distance (mm)",
                 ylabel="Tangential energy (×10⁻³)")
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, loc="upper right", fontsize=8)
        axis.grid(axis="y", alpha=0.17)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagnose(root, output):
    root, output = Path(root), Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/mpl"))
    e02 = root / "outputs/e02"
    archive = e02 / "ring-m2m3/runs" / TASK17
    settings = read_json(e02 / "step1/c6bb770094d3/inputs.json")["generation_config"]["common"]
    rings = {42: e02 / "capfix/rings/0979bc4c77ef",
             1234: e02 / "capgeom/runs/7b2fcb2c668b/ring-validation"}
    geometry = root / "outputs/e01b/geometry"
    evaluator = CapEnergyCurves(geometry, cap_band=int(settings.get("cap_band", 16)))
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"source_task17_inputs": file_hash(archive / "inputs.json"),
                "geometry_sha256": file_hash(geometry / "topology.json"),
                "physical_source_sha256": file_hash(Path(__file__).with_name("metrics") / "physical.py"),
                "implementation": {Path(module.__file__).name: file_hash(Path(module.__file__))
                                   for module in (v3_capgeom_geometry, v3_capgeom_init, v3_ring_midscale_cap)},
                "cases": {}}
    panels, rows = [], []
    for (seed, brief), folds in FOCUS.items():
        ident = f"{brief}-{seed}"
        data = archive if seed == 42 else archive / "validation"
        raw = data / "u1_sigma28" / f"{ident}.png"
        initial, final, folded, proof = evaluator.inspect(raw, rings[seed] / f"{ident}.npz")
        score_path = data / "evaluation/u1_sigma28" / f"{ident}.metrics.json"
        scored = read_json(score_path)
        if scored["input"]["atlas_sha256"] != proof["raw_sha256"]:
            raise ValueError(f"E0 score is stale for {ident}")
        proof["score_sha256"] = file_hash(score_path)
        peak = max(folded.values(), key=lambda row: row["fold_stripe_index"])
        np.testing.assert_allclose(peak["fold_stripe_index"], scored["variants"]["raw"]["stripe_index"],
                                   rtol=1e-10, atol=1e-10)
        case = {"source": proof, "whole_box_peak": peak, "folds": {}}
        for fold in folds:
            profiles = evaluator.profiles(initial, final, fold)
            case["folds"][fold] = {"v1": folded[fold], "phases": {stage: _curve_summary(value)
                                                                    for stage, value in profiles.items()}}
            rows.extend(_records("e0", seed, brief, fold, profiles))
            panels.append((f"{brief} / {seed} / {fold}", profiles))
        evidence["cases"][ident] = case
        print(f"Task 18 E0 reconstructed {ident}", flush=True)
    _write_rows(output / "diagnostic-curves.csv", rows)
    _plot(output / "diagnostic-curves.png", panels)
    evidence["artifact_sha256"] = {f"diagnostic-curves.{suffix}": file_hash(output / f"diagnostic-curves.{suffix}")
                                   for suffix in ("csv", "png")}
    write_json(output / "diagnostic-curves.json", evidence)
    return evidence


def run_initial_diagnosis(experiment):
    """Publish the hash-checked CPU evidence before launching the cap run."""
    root = Path(experiment.root)
    destination = Path(experiment.output) / "initial-diagnosis"
    cached = root / "tmp/task18-cap-energy-diagnostic"
    names = ("diagnostic-curves.csv", "diagnostic-curves.png", "diagnostic-curves.json")
    for source in (destination, cached):
        if any(not (source / name).is_file() for name in names):
            continue
        evidence = read_json(source / names[2])
        if (evidence["source_task17_inputs"] != file_hash(
                root / "outputs/e02/ring-m2m3/runs" / TASK17 / "inputs.json")
                or evidence["geometry_sha256"] != file_hash(root / "outputs/e01b/geometry/topology.json")
                or evidence["physical_source_sha256"] != file_hash(Path(__file__).with_name("metrics") / "physical.py")
                or set(evidence["cases"]) != {f"{brief}-{seed}" for seed, brief in FOCUS}):
            continue
        code = (v3_capgeom_geometry, v3_capgeom_init, v3_ring_midscale_cap)
        if any(evidence["implementation"].get(Path(module.__file__).name) != file_hash(Path(module.__file__))
               for module in code):
            continue
        for ident, case in evidence["cases"].items():
            proof = case["source"]
            path = Path(proof["raw_path"])
            ring = Path(proof["ring_path"])
            score = path.parent.parent / "evaluation/u1_sigma28" / f"{ident}.metrics.json"
            inputs = {"raw": path, "run": path.with_suffix(".run.json"), "score": score,
                      "ring": ring, "ring_proof": ring.with_suffix(".json")}
            if any(proof[f"{key}_sha256"] != file_hash(file) for key, file in inputs.items()):
                break
        else:
            hashes = {name: file_hash(source / name) for name in names[:2]}
            if "artifact_sha256" in evidence and evidence["artifact_sha256"] != hashes:
                continue
            with Image.open(source / names[1]) as image:
                image.verify()
            with (source / names[0]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            count = sum(len(fold["phases"]) * MM_BINS for case in evidence["cases"].values()
                        for fold in case["folds"].values())
            if len(rows) != count:
                continue
            evidence["artifact_sha256"] = hashes
            if source != destination:
                destination.mkdir(parents=True, exist_ok=True)
                for name in names[:2]:
                    shutil.copyfile(source / name, destination / name)
            write_json(destination / names[2], evidence)
            return evidence
    return diagnose(root, destination)


def _candidate_case(analyzer, job, ring_path, method, seed):
    raw_path = job.atlas.path
    metadata = read_json(raw_path.with_suffix(".run.json"))
    archive = method.lower() in ("e0", "j0") and metadata["method"] == "u1_sigma28"
    if metadata["seed"] != seed or (not archive and metadata["method"] != method):
        raise ValueError(f"Unexpected candidate source for {method}/{job.atlas.identifier}")
    initial_paths = None
    if not archive:
        initial_paths = {}
        for face in CAP_FACES:
            proof = metadata["details"]["initial_images"][face]
            path = raw_path.with_suffix(f".init-{face}.npy")
            if (proof["filename"] != path.name or proof["dtype"] != "float32"
                    or proof["sha256"] != file_hash(path)):
                raise ValueError(f"Initial-image provenance mismatch for {raw_path}/{face}")
            initial_paths[face] = path
    initial, final, folded, proof = analyzer.inspect(raw_path, ring_path, initial_paths=initial_paths)
    if initial_paths is not None:
        for face, path in initial_paths.items():
            shape = list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
            if shape != metadata["details"]["initial_images"][face]["shape"]:
                raise ValueError(f"Initial-image shape differs for {raw_path}/{face}")
    return _summarize_case(analyzer, job, initial, final, folded, proof)


def _summarize_case(analyzer, job, initial, final, folded, proof):
    from .e02 import score_input
    from .e02_ring_m2m3_ring_eval import fold_normal_profile

    score = read_json(job.score)
    if score["input"] != score_input(job):
        raise ValueError(f"Stale frozen score for {job.atlas.identifier}")
    peak = max(folded.values(), key=lambda row: row["fold_stripe_index"])
    np.testing.assert_allclose(peak["fold_stripe_index"], score["variants"]["preflight"]["stripe_index"],
                               rtol=1e-10, atol=1e-10)
    proof["score_sha256"] = file_hash(job.score)
    stats = {"source": proof, "whole_box_peak": peak, "folds": folded,
             "fold_normal_profiles": fold_normal_profile(analyzer.evaluator, final)}
    profiles = {fold: analyzer.profiles(initial, final, fold) for fold in CAP_FOLDS}
    if peak["fold"] not in profiles:
        profiles[peak["fold"]] = analyzer.profiles(initial, final, peak["fold"])
    stats["phase_summary"] = {fold: {stage: _curve_summary(array) for stage, array in phases.items()}
                              for fold, phases in profiles.items()}
    return stats, profiles


def _candidate_cache(path, folder, analyzer, method, seed, jobs, ring_dir, stage1_source):
    if not path.is_file():
        return None
    saved = read_json(path)
    if (saved.get("implementation_sha256") != file_hash(Path(__file__))
            or saved.get("geometry_sha256") != file_hash(analyzer.geometry / "topology.json")
            or saved["method"] != method or saved["seed"] != seed
            or set(saved["cases"]) != {job.atlas.brief["id"] for job in jobs}
            or any(not (folder / name).is_file() or digest != file_hash(folder / name)
                   for name, digest in saved["artifact_sha256"].items())):
        return None
    for job in jobs:
        brief = job.atlas.brief["id"]
        proof = saved["cases"][brief]["source"]
        raw_path = job.atlas.path
        paths = {"raw": raw_path, "run": raw_path.with_suffix(".run.json"), "score": job.score}
        if stage1_source:
            prior = Path(proof["stage1_path"])
            paths.update(stage1=prior, stage1_run=prior.with_suffix(".run.json"))
        else:
            ring = Path(ring_dir) / f"{brief}-{seed}.npz"
            paths.update(ring=ring, ring_proof=ring.with_suffix(".json"))
        if any(not file.is_file() or proof[f"{key}_sha256"] != file_hash(file) for key, file in paths.items()):
            return None
        for face, init in proof.get("initial_caps", {}).items():
            source = raw_path.with_suffix(f".init-{face}.npy")
            if "file_sha256" in init and (not source.is_file()
                    or init["file_sha256"] != file_hash(source)):
                return None
    return saved


def _staged_case(analyzer, job, method, seed):
    raw = job.atlas.path
    run = raw.with_suffix(".run.json")
    metadata = read_json(run)
    details = metadata["details"]
    before = Path(details["stage1_raw_path"])
    prior = before.with_suffix(".run.json")
    before_hash = file_hash(before)
    raw_hash = file_hash(raw)
    if (metadata["method"] != method or metadata["seed"] != seed or raw_hash != metadata["raw_sha256"]
            or before_hash != details["stage1_raw_sha256"] or file_hash(prior) != details["stage1_run_sha256"]
            or before_hash != read_json(prior)["raw_sha256"]):
        raise ValueError(f"Stage 1 and final atlas provenance mismatch for {raw}")
    after = analyzer.evaluator._native_atlas(raw)
    sources = analyzer.evaluator._native_atlas(before)
    if not np.array_equal(after[analyzer.side_mask], sources[analyzer.side_mask]):
        raise ValueError(f"Stage 2 altered the shared ring for {raw}")
    initial = analyzer.evaluator._physical_faces(sources)
    final = analyzer.evaluator._physical_faces(after)
    folded = {row["fold"]: row for row in _crease_rows(analyzer.evaluator, final)}
    proof = {"raw_path": str(raw), "raw_sha256": raw_hash, "run_sha256": file_hash(run),
             "stage1_path": str(before), "stage1_sha256": before_hash, "stage1_run_sha256": file_hash(prior)}
    return _summarize_case(analyzer, job, initial, final, folded, proof)


def write_candidate_curves(experiment, method, *, seed=42, output=None, stage1_source=False):
    """Export ten paired cap curves, cohort means and two worst frozen v1 cases."""
    root = Path(experiment.root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/mpl"))
    folder = Path(output) if output is not None else Path(experiment.output) / "cap-energy-curves"
    if seed not in (42, 1234) or not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("Candidate curves need an approved seed and a project data-disk path")
    folder.mkdir(parents=True, exist_ok=True)
    ring_dir = experiment.original_rings if seed == 42 else experiment.validation_ring_dir
    analyzer = CapEnergyCurves(experiment.geometry, cap_band=int(experiment.engine_config.get("cap_band", 16)))
    jobs = tuple(experiment.samples(methods=(method,), seed=seed))
    expected = [f"E01-{i:03d}" for i in range(1, 11)]
    if sorted(job.atlas.brief["id"] for job in jobs) != expected:
        raise ValueError(f"{method}/{seed} curves require exactly the registered first ten briefs")
    prefix = f"{method}-{seed}"
    metadata = folder / f"{prefix}.json"
    cached = _candidate_cache(metadata, folder, analyzer, method, seed, jobs, ring_dir, stage1_source)
    if cached is not None:
        return cached
    rows, profiles, cases = [], {}, {}
    for job in jobs:
        brief = job.atlas.brief["id"]
        if stage1_source:
            cases[brief], profiles[brief] = _staged_case(analyzer, job, method, seed)
        else:
            cases[brief], profiles[brief] = _candidate_case(
                analyzer, job, Path(ring_dir) / f"{brief}-{seed}.npz", method, seed)
        for fold, stages in profiles[brief].items():
            rows.extend(_records(method, seed, brief, fold, stages))
    means = {fold: {stage: np.mean([profiles[brief][fold][stage] for brief in expected], axis=0)
                    for stage in STAGES} for fold in CAP_FOLDS}
    for fold, stages in means.items():
        rows.extend(_records(method, seed, "MEAN", fold, stages))
    worst = sorted(cases, key=lambda brief: (-cases[brief]["whole_box_peak"]["fold_stripe_index"], brief))[:2]
    summary = {"method": method, "seed": seed, "n": len(jobs),
               "implementation_sha256": file_hash(Path(__file__)),
               "geometry_sha256": file_hash(Path(experiment.geometry) / "topology.json"),
               "worst_two_by_frozen_whole_box_stripe_v1": worst, "cases": cases,
               "mean_curve_summary": {fold: {stage: _curve_summary(values) for stage, values in stages.items()}
                                      for fold, stages in means.items()}}
    panels = [(f"Mean of ten / {method.upper()} / {fold}", phases) for fold, phases in means.items()]
    for brief in worst:
        for fold, phases in profiles[brief].items():
            panels.append((f"{brief} / {method.upper()} / {fold}", phases))
    _write_rows(folder / f"{prefix}.csv", rows)
    _plot(folder / f"{prefix}.png", panels)
    summary["artifact_sha256"] = {f"{prefix}.{suffix}": file_hash(folder / f"{prefix}.{suffix}")
                                  for suffix in ("csv", "png")}
    write_json(metadata, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Diagnostic output must stay on the project data disk")
    diagnose(args.root, args.output)


if __name__ == "__main__":
    main()
