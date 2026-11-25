"""Verify and cache Task 07's exact stage-one ring for the cap-only experiment."""

import ast
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash


TASK07_COMMIT = "b18ed8e775bf072fbcd3b32b9f381abdfbdb9728"
METHODS_DIR = Path("code/dielinediff/methods")
TASK07_RING = (1280, 1024)
SIDE_LABELS = (1, 2, 3, 4)


def ast_source(path, revision=None):
    if revision is None:
        source = path.read_text(encoding="utf-8")
    else:
        source = subprocess.check_output(["git", "show", f"{revision}:{path.as_posix()}"], encoding="utf-8")
    return ast.parse(source)


def method_members(tree):
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DieLineDiffV3")
    return engine, {node.name: node for node in engine.body if isinstance(node, ast.FunctionDef)}


def canonical(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def protect_ring(reference):
    """Bind the stage-one source blocks to the reviewed Task 07 revision."""
    original_hashes = reference["generation_implementation"]
    v3 = METHODS_DIR / "v3.py"
    archived_blob = subprocess.check_output(["git", "show", f"{TASK07_COMMIT}:{v3.as_posix()}"])
    if sha256(archived_blob).hexdigest() != original_hashes[v3.name]:
        raise ValueError("The archive does not identify the fixed Task 07 ring implementation")
    checked = {}
    for name, previous_hash in original_hashes.items():
        if name == v3.name:
            continue
        actual = file_hash(METHODS_DIR / name)
        if actual != previous_hash:
            raise ValueError(f"Task 07 dependency was changed: {name}")
        checked[name] = actual
    archived = ast.parse(archived_blob)
    current = ast_source(v3)
    old_engine, old = method_members(archived)
    current_engine, new = method_members(current)
    old_module = [node for node in archived.body if node is not old_engine]
    new_module = [node for node in current.body if node is not current_engine]
    if canonical(old_module) != canonical(new_module):
        raise ValueError("Task 07's v3 module-level inference helpers changed")
    for key in old:
        if key != "_generate_ring" and (key not in new or canonical([old[key]]) != canonical([new[key]])):
            raise ValueError(f"Task 07 inference method changed: {key}")
    ring_block = []
    for node in old["_generate_ring"].body:
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "generated_caps" for target in node.targets)):
            break
        ring_block.append(node)
    source = new["_prepare_ring"].body
    extracted = [node for node in source if not isinstance(node, ast.Return)
                 and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                          and isinstance(node.value.value, str))]
    if canonical(ring_block) != canonical(extracted):
        raise ValueError("The extracted stage-one ring no longer matches Task 07.")
    return {"commit": TASK07_COMMIT, "ring_block_ast_sha256": sha256(canonical(ring_block).encode()).hexdigest(),
            "unchanged_source_files": checked}


def rgb_pixels(image):
    return np.asarray(image.convert("RGB"))


def bytes_digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def archived_rgb(path):
    meta = read_json(path.with_suffix(".run.json"))
    if meta["method"] != "v3_1280" or meta["seed"] != 42 or meta["raw_sha256"] != file_hash(path):
        raise ValueError(f"Invalid Task 07 source image: {path}")
    with Image.open(path) as image:
        return rgb_pixels(image)


def verify_projection(engine, prepared, archive):
    atlas, support = engine.geometry.compose_atlas(
        engine.topology, engine.region_map, engine.bleed_owner, prepared.layout, prepared.image, {},
        return_support=True,
    )
    ring_support = np.isin(support, SIDE_LABELS)
    body = np.isin(engine.region_map, SIDE_LABELS)
    output = rgb_pixels(atlas)
    if (not ring_support.any() or not np.all(ring_support[body])
            or not np.array_equal(output[ring_support], archive[ring_support])):
        raise RuntimeError("Stage-one pixels differ from the reviewed Task 07 atlas")
    return body, ring_support, {
        "side_pixels": int(body.sum()),
        "ring_supported_pixels": int(ring_support.sum()),
        "body_rgb_sha256": bytes_digest(output[body]),
        "task07_body_rgb_sha256": bytes_digest(archive[body]),
        "ring_supported_rgb_sha256": bytes_digest(output[ring_support]),
        "task07_ring_supported_rgb_sha256": bytes_digest(archive[ring_support]),
        "matches_task07_body": True,
    }


def _load_ring(engine, path):
    import torch
    from .methods.v3_capfix import PreparedRing

    with np.load(path, allow_pickle=False) as data:
        latent, rgb, state = data["latent"], data["rgb"], data["rng_state"]
    layout, _, _ = engine._ring_layout(TASK07_RING)
    return PreparedRing(layout, torch.as_tensor(latent, device=engine.device),
                        Image.fromarray(rgb), torch.as_tensor(state, dtype=torch.uint8), None)


def prepare(engine, directory, brief, archive_path):
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"{brief['id']}-42"
    alpha_device = "cpu" if brief["id"] == "E01-001" else str(engine.device)
    path = prefix.with_suffix(".npz")
    meta_path = prefix.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        record = read_json(meta_path)
        if record["cache_sha256"] != file_hash(path):
            raise ValueError(f"Frozen ring cache mismatch: {path}")
        cached = _load_ring(engine, path)
        fields = record
    else:
        # Task 07's first cap moved DDIM alpha coefficients to CUDA in add_noise.
        # Match that original per-brief state when reconstructing only stage one.
        scheduler = engine.pipe.scheduler
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(alpha_device)
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        start = time.perf_counter()
        cached = engine.prepare_ring(brief, 42, ring_size=TASK07_RING)
        torch.cuda.synchronize(engine.device)
        seconds = time.perf_counter() - start
        fields = {"ring_elapsed_seconds": seconds,
                  "ring_peak_memory_mib": torch.cuda.max_memory_allocated(engine.device) / 2**20}
    archive = archived_rgb(archive_path)
    body, supported, proof = verify_projection(engine, cached, archive)
    latent = cached.frozen_latent.detach().cpu().numpy()
    ring_rgb = rgb_pixels(cached.image)
    state = cached.rng_state.detach().cpu().numpy()
    values = {**proof, "latent_sha256": bytes_digest(latent), "ring_rgb_sha256": bytes_digest(ring_rgb),
              "rng_state_sha256": bytes_digest(state), "archive_sha256": file_hash(archive_path)}
    if "cache_sha256" in fields:
        if any(fields.get(key) != value for key, value in values.items()):
            raise ValueError(f"Frozen ring metadata mismatch: {path}")
    else:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as stream:
            np.savez(stream, latent=latent, rgb=ring_rgb, rng_state=state)
        tmp.replace(path)
        fields.update(values, cache_sha256=file_hash(path), brief_id=brief["id"], seed=42)
        write_json(meta_path, fields)
    if fields.get("scheduler_alphas_device") != alpha_device:
        if "scheduler_alphas_device" in fields:
            raise ValueError(f"Unexpected Task 07 scheduler state in {meta_path}")
        fields["scheduler_alphas_device"] = alpha_device
        write_json(meta_path, fields)
    return cached, fields, body, supported
