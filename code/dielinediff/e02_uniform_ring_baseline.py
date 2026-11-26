"""Registered regional baselines for Task 15, with only the R1 negative prompt varied."""

import ast
from hashlib import file_digest
import os
from pathlib import Path
import time

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_capfix_ring import protect_ring
from .e02_sources import SourceAtlas
from .methods.sdxl import NEGATIVE
from .methods.v3 import DieLineDiffV3


ORIGINAL = "b2_regional"
R1 = "b2_regional_r1"
TASK07_RUN = "c6bb770094d3"
SEEDS = (42, 1234)


def regional_negative(suffix):
    if not isinstance(suffix, str) or not suffix or suffix != suffix.strip() or suffix.startswith(","):
        raise ValueError("R1 needs the registered nonempty negative-prompt suffix")
    return f"{NEGATIVE}, {suffix}"


def regional_conditions_with_negative(engine, brief, negative):
    """Call the original conditioning code, changing its encoder's negative argument only."""
    encoder = engine.pipe.encode_prompt
    had_override = "encode_prompt" in vars(engine.pipe)
    proof = {}

    def encode_original_regions(*args, **kwargs):
        prompts = kwargs["prompt"]
        original_negative = kwargs["negative_prompt"]
        if original_negative != [NEGATIVE] * len(prompts):
            raise ValueError("The inherited regional negative-prompt contract changed")
        kwargs["negative_prompt"] = [negative] * len(prompts)
        proof.update(positive_sha256=stable_hash(prompts), positive_count=len(prompts),
                     negative_prompt=negative, negative_sha256=stable_hash(negative))
        return encoder(*args, **kwargs)

    engine.pipe.encode_prompt = encode_original_regions
    try:
        conditions = DieLineDiffV3._conditions(engine, brief)
    finally:
        if had_override:
            engine.pipe.encode_prompt = encoder
        else:
            del engine.pipe.encode_prompt
    if not proof:
        raise RuntimeError("Regional conditioning never reached its registered encoders")
    return conditions, proof


def tokenizer_proof(tokenizers, negative):
    """Describe the full input IDs before truncation for the two actual CLIP tokenizers."""
    if len(tokenizers) != 2:
        raise ValueError("SDXL regional conditioning expects two text tokenizers")
    proof = {}
    for index, tokenizer in enumerate(tokenizers, 1):
        ids = tokenizer(negative, add_special_tokens=True, truncation=False)["input_ids"]
        maximum = int(tokenizer.model_max_length)
        if len(ids) > maximum:
            raise ValueError(f"R1 negative exceeds CLIP tokenizer {index}: {len(ids)}/{maximum} tokens")
        proof[f"tokenizer_{index}"] = {
            "tokens": len(ids), "capacity": maximum, "input_ids_sha256": stable_hash(ids),
        }
    return proof


class RegisteredRegionalBaseline(DieLineDiffV3):
    negative_prompt = NEGATIVE

    def _conditions(self, brief):
        result, self.last_prompt_proof = regional_conditions_with_negative(self, brief, self.negative_prompt)
        return result


def _task07(root):
    return Path(root) / "outputs/e02/step1" / TASK07_RUN


def baseline_atlases(exp):
    """List both methods, reading original seed 42 directly from the immutable archive."""
    trial = tuple(exp.briefs[:10])
    if [brief["id"] for brief in trial] != [f"E01-{i:03d}" for i in range(1, 11)]:
        raise ValueError("Uniform-ring baselines require registered briefs E01-001 through E01-010")
    archive = _task07(exp.root)
    if exp.output.resolve().is_relative_to(archive.resolve()):
        raise ValueError("Task 15 cannot write in the Task 07 archive")
    atlases = []
    for method in (ORIGINAL, R1):
        for seed in SEEDS:
            archived = method == ORIGINAL and seed == 42
            directory = (archive if archived else exp.output) / method
            if not archived and not directory.resolve().is_relative_to(exp.output.resolve()):
                raise ValueError("Task 15 baseline destination escaped its run")
            atlases.extend(SourceAtlas(method, brief, seed, directory / f"{brief['id']}-{seed}.png",
                                       exp.geometry)
                           for brief in trial)
    return tuple(atlases)


def _pin_models(models):
    """Check the published Task 07 revisions and exact FP16 payloads from the E01 entry point."""
    script = Path(__file__).resolve().parents[2] / "code/scripts/fetch_E01_models.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    pins = {
        node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("REPOS", "WEIGHTS")
    }
    proof = {}
    for component, (repository, revision) in pins["REPOS"].items():
        marker = (models / f"{component}.revision").read_text(encoding="utf-8").strip()
        if marker != revision:
            raise ValueError(f"The registered {component} revision has changed")
        proof[component] = {"repository": repository, "revision": revision, "weights": {}}
    for component, filename, size, checksum in pins["WEIGHTS"]:
        file = models / component / filename
        if file.stat().st_size != size:
            raise ValueError(f"The registered model weight size changed: {file}")
        with file.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != checksum:
            raise ValueError(f"The registered model weight SHA-256 changed: {file}")
        proof[component]["weights"][filename] = actual
    return proof


def _matches_record(atlas, metadata):
    return (metadata["method"] == atlas.method and metadata["brief_id"] == atlas.brief["id"]
            and metadata["seed"] == atlas.seed and metadata["raw_sha256"] == file_hash(atlas.path))


def _verify_original(atlas, archive, geometry_sha):
    record = read_json(atlas.metadata)
    score = read_json(archive / "evaluation" / ORIGINAL / f"{atlas.identifier}.metrics.json")
    expected = {"atlas_sha256": record["raw_sha256"], "run_sha256": file_hash(atlas.metadata),
                "geometry_sha256": geometry_sha}
    if not _matches_record(atlas, record) or score["input"] != expected:
        raise ValueError(f"Task 07 B2-regional archive does not match its registered score: {atlas.identifier}")


def _registration(exp, atlases):
    archive = _task07(exp.root)
    reference = read_json(archive / "inputs.json")
    if stable_hash(reference)[:12] != TASK07_RUN:
        raise ValueError("Task 07 registration manifest has changed")
    if stable_hash(exp.briefs[:10]) != reference["briefs"]:
        raise ValueError("Task 15 briefs differ from the registered Task 07 first ten")
    settings = reference["generation_config"]["common"]
    if exp.base_experiment_config.get("common", exp.base_experiment_config) != settings:
        raise ValueError("Task 15 B2-regional settings differ from Task 07")
    geometry = {name: file_hash(exp.geometry / name) for name in reference["geometry"]}
    if geometry != reference["geometry"] or file_hash(exp.lut) != reference["lut_sha256"]:
        raise ValueError("Task 15 B2-regional must use the Task 07 atlas geometry and LUT")
    source_proof = protect_ring(reference)
    for atlas in atlases:
        if atlas.method == ORIGINAL and atlas.seed == 42:
            _verify_original(atlas, archive, geometry["topology.json"])

    from transformers import CLIPTokenizer

    models = exp.root / "cache/models"
    tokenizers = tuple(CLIPTokenizer.from_pretrained(models / "sdxl" / directory, local_files_only=True)
                       for directory in ("tokenizer", "tokenizer_2"))
    negatives = {ORIGINAL: NEGATIVE, R1: regional_negative(exp.negative_suffix)}
    tokenizer_checks = {method: tokenizer_proof(tokenizers, negative) for method, negative in negatives.items()}
    return settings, negatives, {
        "task07_run": TASK07_RUN, "task07_manifest_sha256": file_hash(archive / "inputs.json"),
        "baseline_implementation_sha256": file_hash(Path(__file__)),
        "settings": settings, "geometry": geometry, "lut_sha256": reference["lut_sha256"],
        "models": _pin_models(models), "generation_source": source_proof,
        "negative_prompts": negatives, "tokenizer_checks": tokenizer_checks,
    }


def _completed(atlas, registration):
    if not atlas.path.is_file() or not atlas.metadata.is_file():
        return False
    record = read_json(atlas.metadata)
    if not _matches_record(atlas, record) or record["registration_sha256"] != registration:
        raise ValueError(f"Existing Task 15 atlas has a different registration or raw SHA: {atlas.path}")
    return True


def generate_baselines(exp, device):
    """Generate 30 new atlases using the original regional path, returning all 40 references."""
    atlas_list = baseline_atlases(exp)
    settings, negatives, evidence = _registration(exp, atlas_list)
    registration = stable_hash(evidence)
    path = exp.output / "baseline-inputs.json"
    if path.is_file() and read_json(path) != evidence:
        raise ValueError("Existing Task 15 baseline run has different registered inputs")
    exp.output.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        write_json(path, evidence)
    pending = [atlas for atlas in atlas_list if (atlas.method != ORIGINAL or atlas.seed != 42)
               and not _completed(atlas, registration)]
    if not pending:
        return atlas_list

    import torch

    hardware = torch.device(device)
    if hardware.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Regional image generation requires the registered NVIDIA GPU")
    torch.cuda.set_device(hardware.index or 0)
    engine = RegisteredRegionalBaseline(exp.root / "cache/models", exp.geometry, settings,
                                        device=hardware, lut_path=exp.lut)
    actual_tokenizers = (engine.pipe.tokenizer, engine.pipe.tokenizer_2)
    for method, negative in negatives.items():
        if tokenizer_proof(actual_tokenizers, negative) != evidence["tokenizer_checks"][method]:
            raise ValueError(f"Loaded SDXL tokenization differs from the registered {method} precheck")
    for atlas in pending:
        engine.negative_prompt = negatives[atlas.method]
        torch.cuda.synchronize(hardware)
        torch.cuda.reset_peak_memory_stats(hardware)
        started = time.perf_counter()
        image = engine.generate(atlas.brief, atlas.seed, method=ORIGINAL).convert("RGB")
        torch.cuda.synchronize(hardware)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(hardware) / 2**20
        if image.size != tuple(engine.last_run["atlas_size_px"]):
            raise ValueError(f"Unexpected Task 15 B2-regional atlas size: {image.size}")
        atlas.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = atlas.path.with_suffix(f".{os.getpid()}.tmp.png")
        image.save(temporary)
        temporary.replace(atlas.path)
        write_json(atlas.metadata, {
            "method": atlas.method, "brief_id": atlas.brief["id"], "seed": atlas.seed,
            "elapsed_seconds": elapsed, "peak_memory_mib": peak,
            "gpu": torch.cuda.get_device_name(hardware), "raw_sha256": file_hash(atlas.path),
            "registration_sha256": registration,
            "details": {**engine.last_run, **engine.last_prompt_proof,
                        "negative_tokens": evidence["tokenizer_checks"][atlas.method]},
        })
        print(f"{atlas.method} {atlas.identifier}: {elapsed:.2f}s {peak:.0f} MiB", flush=True)
    return atlas_list
