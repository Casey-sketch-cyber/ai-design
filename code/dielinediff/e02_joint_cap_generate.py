"""Archive Task 20 RTX 3090 generation independently from frozen scoring."""

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_cap_energy_sources import archived_e0
from .e02_capfix_ring import METHODS_DIR, bytes_digest, rgb_pixels
from .e02_capgeom import _check_generation, _hardware, ring, trial
from .e02_capgeom_eval import evaluate
from .methods.v3_capenergy_engine import capture_initial_caps


def control_parameters(experiment):
    source = next(archived_e0(experiment, 42, alias=False))
    return read_json(source.atlas.metadata)["candidate"]


def generate_refinement(experiment, method, parameters, device, *, seed=42):
    from .methods.v3_joint_cap_engine import JointCapEngine

    factory = lambda exp, hardware: JointCapEngine(
        exp.root / "cache/models", exp.geometry, exp.engine_config, device=hardware, lut_path=exp.lut)
    files = ["v3_joint_cap_engine.py", "v3_joint_cap_prompts.py", "v3_ring_midscale_cap.py",
             "v3_capgeom_init.py", "v3_capenergy_engine.py"]
    if parameters.get("cap_backend") == "base_blend" or parameters.get("guidance_gamma"):
        files.extend(("v3_joint_cap_blend.py", "v3_joint_cap_refine.py"))
    if parameters.get("guidance_gamma"):
        files.append("v3_joint_cap_guidance.py")
    hashes = {name: file_hash(METHODS_DIR / name) for name in files}
    run = _for_seed(experiment, seed)
    trial(run, method, parameters, device, seed=seed, source=method if seed == 1234 else None,
          engine_factory=factory, implementation=hashes, capture=capture_initial_caps)


def _for_seed(experiment, seed):
    from copy import copy
    from functools import partial
    from .e02_joint_cap import samples

    run = copy(experiment)
    run.output = experiment.graphics(seed)
    # Sample lookup still binds canonical seed layouts through the original experiment.
    run.samples = partial(samples, experiment)
    return run


def _png(image, path):
    temporary = path.with_suffix(f".{os.getpid()}.tmp.png")
    image.save(temporary)
    temporary.replace(path)


def _with_timer(device, call):
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize(device)
    return (result, time.perf_counter() - start, torch.cuda.max_memory_allocated(device) / (2 ** 20))


def _files(path):
    return {"ring": path.with_suffix(".ring.png"), "T": path.with_suffix(".cap-T.png"),
            "Btm": path.with_suffix(".cap-Btm.png")}


def _read_art(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _stage_plan(experiment, recipe, files):
    setting = "joint_outpaint" if recipe == "j4b" else "joint_stage1"
    return {"recipe": recipe, "options": experiment.options[setting],
            "implementation": {key: file_hash(METHODS_DIR / key) for key in files}}


def _stage_cached(path, plan):
    record = path.with_suffix(".run.json")
    if not record.is_file():
        return None
    evidence = read_json(record)
    files = {"atlas": path, **_files(path)}
    if "latent" in evidence["source_sha256"]:
        files["latent"] = path.with_suffix(".ring.npy")
    if evidence["plan"] != plan:
        raise ValueError(f"Stage 1 settings changed for {path}")
    if any(not file.is_file() or evidence["source_sha256"][key] != file_hash(file)
           for key, file in files.items()):
        return None
    return evidence

def _joint_engine(experiment, hardware, *, refine=False, guide=False):
    from .methods.v3_joint_cap_engine import JointCapEngine
    from .methods.v3_joint_cap_ring import JointRingMixin

    if refine:
        from .methods.v3_joint_cap_stage2 import JointStageTwoMixin

        class Engine(JointStageTwoMixin, JointRingMixin, JointCapEngine):
            pass
    elif guide:
        from .methods.v3_joint_cap_j5 import GuidedJointRingMixin

        class Engine(GuidedJointRingMixin, JointRingMixin, JointCapEngine):
            pass
    else:
        class Engine(JointRingMixin, JointCapEngine):
            pass
    return Engine(experiment.root / "cache/models", experiment.geometry, experiment.engine_config,
                  device=hardware, lut_path=experiment.lut)


def generate_joint_stage1(experiment, recipe, device, *, seed=42, source_recipe=None, guidance=None):
    import torch
    from .methods.v3_joint_cap_ring import compose_joint_stage1

    base = source_recipe or recipe
    if (base not in ("coarse_047", "coarse_055", "redilation", "j4b")
            or (guidance is not None) != (source_recipe is not None)
            or (source_recipe and recipe != f"j5-{base}")):
        raise ValueError(f"Unknown Stage 1 recipe {recipe}")
    frozen_ring = base == "j4b"
    folder = experiment.stage1(recipe, seed)
    folder.mkdir(parents=True, exist_ok=True)
    files = ["v3_joint_cap_ring.py", "v3_joint_cap_geometry.py", "v3_joint_cap_prompts.py",
             "v3_joint_cap_blend.py", "v3_circular.py", "v3_ring_m2.py", "v3.py"]
    if guidance:
        files.extend(("v3_joint_cap_j5.py", "v3_joint_cap_guidance.py", "v3_joint_cap_stage2.py"))
    status = _stage_plan(experiment, base, files)
    if guidance:
        status.update(recipe=recipe, base_recipe=base, guidance=guidance)
    todo = [(brief, folder / f"{brief['id']}-{seed}.png") for brief in experiment.briefs]
    pending = []
    for brief, path in todo:
        cached = _stage_cached(path, status)
        if cached is None:
            pending.append((brief, path))
        elif guidance and not frozen_ring:
            parent_path = experiment.stage1(base, seed) / path.name
            if file_hash(parent_path) != cached["guidance_parent_atlas_sha256"]:
                raise ValueError(f"Guidance parent atlas changed for {path.stem}")
    if not pending:
        return
    hardware = _hardware(device)
    engine = _joint_engine(experiment, hardware, guide=guidance is not None)
    engine._base_unet()
    topology = read_json(experiment.geometry / "topology.json")
    with Image.open(experiment.geometry / "region_map.png") as image:
        labels = np.array(image)
    with Image.open(experiment.geometry / "bleed_owner.png") as image:
        owners = np.array(image)
    for brief, path in pending:
        def generate():
            if frozen_ring:
                if guidance:
                    result = engine.guided_outpaint(brief, seed, frozen, options=status["options"], settings=guidance)
                else:
                    result = engine.generate_frozen_outpaint(brief, seed, frozen, options=status["options"])
                if rgb_pixels(result.ring_image).tobytes() != rgb_pixels(frozen.image).tobytes():
                    raise ValueError(f"Frozen-ring outpaint changed R0 for {path.stem}")
            else:
                mode = "single_pass_redilation" if base == "redilation" else "coarse_to_fine"
                options = {**status["options"], "mode": mode,
                           "subject_center_fraction": status["options"]["central_fraction"]}
                if base == "coarse_055":
                    options["stage1b_strength"] = options["stage1b_extra_strength"]
                if guidance:
                    result = engine.guided_joint(brief, seed, stage1_parent, options=options, settings=guidance)
                else:
                    result = engine.generate_joint_stage1(brief, seed, options=options)
            atlas = compose_joint_stage1(topology, labels, owners, result)
            return result, atlas

        frozen, reference, _, _ = ring(experiment, engine, brief, seed) if frozen_ring else (None,) * 4
        stage1_parent, parent_cost, parent_peak = None, 0, 0
        reference_parent = None
        anchor_sha = reference["latent_sha256"] if reference else None
        if guidance and not frozen_ring:
            path_parent = experiment.stage1(base, seed) / path.name
            stage1_parent, previous = _load_stage1(topology, path_parent)
            reference_parent = previous["raw_sha256"]
            anchor_sha = previous["source_sha256"]["latent"]
            parent_cost = previous["elapsed_seconds"]
            parent_peak = previous["peak_memory_mib"]
        (result, atlas), seconds, peak = _with_timer(hardware, generate)
        parts = _files(path)
        outputs = {"atlas": path, **parts}
        for key, art in (("ring", result.ring_image), *result.caps.items(), ("atlas", atlas)):
            _png(art, outputs[key])
        ring_latent = result.ring_latent
        if ring_latent is not None:
            latent = path.with_suffix(".ring.npy")
            temporary = latent.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, ring_latent.detach().cpu().numpy(), allow_pickle=False)
            temporary.replace(latent)
            outputs["latent"] = latent
        elapsed = seconds + parent_cost + (reference["ring_elapsed_seconds"] if reference else 0)
        peak = max(peak, parent_peak, reference["ring_peak_memory_mib"] if reference else 0)
        hashes = {key: file_hash(p) for key, p in outputs.items()}
        write_json(path.with_suffix(".run.json"), {
            "recipe": recipe, "brief_id": brief["id"], "seed": seed, "gpu": torch.cuda.get_device_name(hardware),
            "plan": status, "raw_sha256": hashes["atlas"], "source_sha256": hashes,
            "elapsed_seconds": elapsed, "stage1_seconds": seconds, "peak_memory_mib": peak,
            "peak_allocated_mib": peak, "metadata": result.proof, "ring": reference if frozen_ring else None,
            **({"guidance_reference_latent_sha256": anchor_sha,
                "guidance_parent_atlas_sha256": reference_parent, "guidance_parent_seconds": parent_cost}
               if guidance else {}),
        })
        print(f"{recipe} stage 1 {path.stem}: {elapsed:.2f}s, {peak:.0f} MiB", flush=True)
    del engine
    torch.cuda.empty_cache()


def _load_stage1(topology, path):
    import torch
    from .methods.v3_joint_cap_ring import JointStageOne, ring_layout_from_metadata

    record = read_json(path.with_suffix(".run.json"))
    inputs = {"atlas": path, **_files(path)}
    latent_path = path.with_suffix(".ring.npy")
    if "latent" in record["source_sha256"]:
        inputs["latent"] = latent_path
    if any(file_hash(file) != record["source_sha256"][key] for key, file in inputs.items()):
        raise ValueError(f"Stage 1 source has changed for {path}")
    proof = record["metadata"]
    frozen = _read_art(inputs["ring"])
    caps = {key: _read_art(inputs[key]) for key in ("T", "Btm")}
    latent = torch.from_numpy(np.load(latent_path, allow_pickle=False)) if "latent" in inputs else None
    stage = JointStageOne(ring_layout_from_metadata(topology, proof), frozen, caps, proof, latent)
    return stage, record


def generate_joint_caps(experiment, method, recipe, options, device, *, seed=42, guidance=None):
    import torch

    run = _for_seed(experiment, seed)
    jobs = list(run.samples(methods=(method,), seed=seed))
    files = ["v3_joint_cap_stage2.py", "v3_joint_cap_blend.py", "v3_joint_cap_prompts.py",
             "v3_joint_cap_geometry.py", "v3_capgeom_geometry.py"]
    if guidance:
        files.append("v3_joint_cap_guidance.py")
    plan = {"recipe": recipe, "options": options, "guidance": guidance,
            "implementation": {key: file_hash(METHODS_DIR / key) for key in files}}
    for job in jobs:
        if not job.validate_generation():
            continue
        meta = read_json(job.atlas.metadata)
        initial = Path(meta["details"]["stage1_raw_path"])
        prior = read_json(initial.with_suffix(".run.json"))
        if (meta["candidate"] != plan or file_hash(initial) != prior["raw_sha256"]
                or meta["details"]["stage1_raw_sha256"] != prior["raw_sha256"]):
            raise ValueError(f"Cached Stage 2 plan changed for {job.atlas.identifier}")
    pending = [job for job in jobs if not job.validate_generation()]
    if pending:
        hardware = _hardware(device)
        engine = _joint_engine(experiment, hardware, refine=True)
        if options["base"]:
            engine._base_unet()
        else:
            engine._open_inpaint()
        topology = read_json(experiment.geometry / "topology.json")
        for job in pending:
            item = job.atlas
            item.path.parent.mkdir(parents=True, exist_ok=True)
            path = experiment.stage1(recipe, seed) / f"{item.identifier}.png"
            start, previous = _load_stage1(topology, path)
            (atlas, details), seconds, peak = _with_timer(hardware, lambda: engine.joint_atlas(
                item.brief, seed, start, options=options))
            if previous["ring"]:
                ring_data = previous["ring"]
                _, original, body, support = ring(experiment, engine, item.brief, seed)
                pixels = rgb_pixels(atlas)
                if (original["cache_sha256"] != ring_data["cache_sha256"]
                        or bytes_digest(pixels[body]) != original["body_rgb_sha256"]
                        or bytes_digest(pixels[support]) != original["ring_supported_rgb_sha256"]):
                    raise ValueError(f"Stage 2 modified an R0 ring pixel in {item.identifier}")
            elapsed = seconds + previous["elapsed_seconds"]
            peak = max(peak, previous["peak_memory_mib"])
            _png(atlas, item.path)
            write_json(item.metadata, {
                "method": method, "brief_id": item.brief["id"], "seed": seed,
                "raw_sha256": file_hash(item.path), "elapsed_seconds": elapsed, "cap_elapsed_seconds": seconds,
                "peak_memory_mib": peak, "gpu": torch.cuda.get_device_name(hardware), "candidate": plan,
                "ring": previous["ring"], "validation_of": method if seed == 1234 else None,
                "details": {**details, "stage1_raw_path": str(path), "stage1_raw_sha256": previous["raw_sha256"],
                            "stage1_run_sha256": file_hash(path.with_suffix(".run.json"))},
            })
            print(f"{method} {item.identifier}: {elapsed:.2f}s (Stage 2 {seconds:.2f}s), {peak:.0f} MiB", flush=True)
        del engine
        torch.cuda.empty_cache()
    if recipe in ("j4b", "j5-j4b"):
        _check_generation(run, jobs, seed=seed)
    evaluate(run, jobs, device)
