"""Generate the two-seed matched B2-regional-M2 supplementary baseline."""

import os
from pathlib import Path
from time import perf_counter

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_capfix_ring import protect_ring
from .e02_sources import SourceAtlas
from .e02_uniform_ring_baseline import ORIGINAL, SEEDS, TASK07_RUN, _matches_record, _pin_models, _verify_original
from .methods.v3_ring_m2 import (
    M2_REGIONAL_SETTINGS, M2RegionalFlat, m2_prompt_plan, m2_schedule, m2_tokenizer_proof,
)


METHOD = "b2_regional_m2"


def baseline_atlases(exp):
    """Enumerate the registered first ten briefs for each registered seed."""
    if [brief["id"] for brief in exp.briefs] != [f"E01-{i:03d}" for i in range(1, 11)]:
        raise ValueError("B2-regional-M2 uses E01-001 through E01-010")
    output = exp.output / METHOD
    frozen = (exp.root / "outputs/e02/step1", exp.root / "outputs/e02/uniform-ring")
    if not output.resolve().is_relative_to(exp.output.resolve()) or any(
            output.resolve().is_relative_to(path.resolve()) for path in frozen):
        raise ValueError("B2-regional-M2 must be written outside the archived runs")
    return tuple(
        SourceAtlas(METHOD, brief, seed, output / f"{brief['id']}-{seed}.png", exp.geometry)
        for seed in SEEDS for brief in exp.briefs
    )


def _previous_parent(exp):
    return exp.root / "outputs/e02/uniform-ring/runs" / exp.config["task15_reference"]


def _paired_sources(exp, atlases, geometry_sha):
    task07 = exp.root / "outputs/e02/step1" / TASK07_RUN
    task15 = _previous_parent(exp)
    archived_inputs = read_json(task15 / "baseline-inputs.json")
    baseline_signature = stable_hash(archived_inputs)
    sources = {}
    for atlas in atlases:
        prior = task07 if atlas.seed == 42 else task15
        source = SourceAtlas(ORIGINAL, atlas.brief, atlas.seed,
                             prior / ORIGINAL / atlas.path.name, atlas.geometry)
        if atlas.seed == 42:
            _verify_original(source, task07, geometry_sha)
        record = read_json(source.metadata)
        score = prior / "evaluation" / ORIGINAL / f"{source.identifier}.metrics.json"
        metadata_sha = file_hash(source.metadata)
        expected = {"atlas_sha256": record["raw_sha256"], "run_sha256": metadata_sha,
                    "geometry_sha256": geometry_sha}
        if (not _matches_record(source, record) or read_json(score)["input"] != expected
                or (atlas.seed != 42 and record["registration_sha256"] != baseline_signature)):
            raise ValueError(f"Prior B2-regional source or evaluation changed: {source.identifier}")
        sources[source.identifier] = {"raw_sha256": record["raw_sha256"], "run_sha256": metadata_sha,
                                      "score_sha256": file_hash(score)}
    return sources


def _registration(exp, atlases):
    if exp.config["m2"] != M2_REGIONAL_SETTINGS:
        raise ValueError("M2 must use the preregistered shared prompt, schedule and feather width")
    models = exp.root / "cache/models"
    original = exp.root / "outputs/e02/step1" / TASK07_RUN / "inputs.json"
    frozen = read_json(original)
    prior15 = _previous_parent(exp)
    source15 = read_json(prior15 / "inputs.json")
    if (stable_hash(frozen)[:12] != TASK07_RUN or stable_hash(source15)[:12] != prior15.name
            or source15["task07_inputs_sha256"] != file_hash(original)
            or stable_hash(list(exp.briefs)) != frozen["briefs"]):
        raise ValueError("M2 baseline does not match the registered Task 07/15 source and brief cohort")
    settings = frozen["generation_config"]["common"]
    geometry = {name: file_hash(exp.geometry / name) for name in frozen["geometry"]}
    lut = file_hash(exp.lut)
    if geometry != frozen["geometry"] or lut != frozen["lut_sha256"]:
        raise ValueError("B2-regional-M2 requires the original atlas geometry and LUT")
    source_proof = protect_ring(frozen)
    model_proof = _pin_models(models)
    prior_baseline = read_json(prior15 / "baseline-inputs.json")
    if (prior_baseline["settings"] != settings or prior_baseline["models"] != model_proof
            or prior_baseline["geometry"] != geometry or prior_baseline["lut_sha256"] != lut):
        raise ValueError("B2-regional-M2 must use the same scheduler, models and geometry as the paired baseline")
    from transformers import CLIPTokenizer

    tokenizers = tuple(CLIPTokenizer.from_pretrained(models / "sdxl" / name, local_files_only=True)
                       for name in ("tokenizer", "tokenizer_2"))
    prompts = {}
    for brief in exp.briefs:
        plan = m2_prompt_plan(brief, exp.config["m2"])
        prompts[brief["id"]] = {"positive_sha256": stable_hash(list(plan.values())),
                                "tokenizers": m2_tokenizer_proof(tokenizers, plan)}
    return {
        "task07_manifest_sha256": file_hash(original), "task15_manifest_sha256": file_hash(prior15 / "inputs.json"),
        "settings": settings, "m2": exp.config["m2"], "m2_weights": list(m2_schedule(settings["steps"], exp.config["m2"])),
        "geometry": geometry, "lut_sha256": lut, "models": model_proof,
        "paired_source": _paired_sources(exp, atlases, geometry["topology.json"]),
        "generation_source": source_proof, "prompts": prompts,
        "implementation_sha256": {str(path.name): file_hash(path)
                                  for path in (Path(__file__), Path(__file__).parent / "methods/v3_ring_m2.py")},
    }


def _pending(atlases, registration):
    for atlas in atlases:
        if not atlas.path.is_file() or not atlas.metadata.is_file():
            yield atlas
            continue
        record = read_json(atlas.metadata)
        if (not _matches_record(atlas, record) or record["registration_sha256"] != registration):
            raise ValueError(f"Existing B2-regional-M2 atlas has changed: {atlas.path}")


def generate_m2_baseline(exp, device):
    """Return all twenty registered atlas references, generating only missing cases."""
    atlases = baseline_atlases(exp)
    evidence = _registration(exp, atlases)
    registration = stable_hash(evidence)
    target = exp.output / "m2-baseline-inputs.json"
    if target.is_file() and read_json(target) != evidence:
        raise ValueError("Cannot resume B2-regional-M2 with different registered inputs")
    if not target.is_file():
        write_json(target, evidence)
    pending = list(_pending(atlases, registration))
    if not pending:
        return atlases

    import torch

    hardware = torch.device(device)
    if hardware.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("B2-regional-M2 generation requires a CUDA GPU")
    torch.cuda.set_device(hardware.index or 0)
    settings = {**evidence["settings"], "m2": evidence["m2"]}
    engine = M2RegionalFlat(exp.root / "cache/models", exp.geometry, settings,
                            device=hardware, lut_path=exp.lut)
    for atlas in pending:
        torch.cuda.synchronize(hardware)
        torch.cuda.reset_peak_memory_stats(hardware)
        start = perf_counter()
        image = engine.generate(atlas.brief, atlas.seed, method=ORIGINAL).convert("RGB")
        torch.cuda.synchronize(hardware)
        elapsed = perf_counter() - start
        peak = torch.cuda.max_memory_allocated(hardware) / 2**20
        details = engine.last_run
        prompt = details["m2"]
        expected = evidence["prompts"][atlas.brief["id"]]
        if ({key: prompt[key] for key in expected} != expected
                or details["m2_weights"] != evidence["m2_weights"]
                or image.size != tuple(details["atlas_size_px"])):
            raise ValueError(f"B2-regional-M2 did not match the registered conditioning: {atlas.identifier}")
        atlas.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = atlas.path.with_suffix(f".{os.getpid()}.tmp.png")
        image.save(temporary)
        temporary.replace(atlas.path)
        write_json(atlas.metadata, {
            "method": METHOD, "brief_id": atlas.brief["id"], "seed": atlas.seed,
            "elapsed_seconds": elapsed, "peak_memory_mib": peak,
            "gpu": torch.cuda.get_device_name(hardware), "raw_sha256": file_hash(atlas.path),
            "registration_sha256": registration, "details": details,
        })
        print(f"{METHOD} {atlas.identifier}: {elapsed:.2f}s {peak:.0f} MiB", flush=True)
    return atlases
