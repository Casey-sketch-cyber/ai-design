"""Generate and score the Task 22 configurations with verified R0 and shared outpaint."""

from copy import copy
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_capfix_ring import METHODS_DIR, bytes_digest, rgb_pixels
from .e02_capgeom import _check_generation, _hardware, ring, trial
from .e02_capgeom_eval import evaluate
from .e02_joint_cap_generate import _files, _load_stage1, _png, _stage_cached, _with_timer
from .e02_panel_fix import CONTROL
from .methods.v3_capenergy_engine import capture_initial_caps
from .methods.v3_inpainting import verify_inpaint_model


def _evaluation(experiment, seed):
    run = copy(experiment)
    run.output = experiment.graphics(seed)
    return run


def control_parameters(experiment):
    control = next(experiment.samples(methods=(CONTROL,)))
    base = read_json(control.atlas.metadata)["candidate"]
    if base["guidance_gamma"] != 0.30 or not base["inpaint"] or base["cap_scale"] != 2:
        raise ValueError("Unexpected archived J3 γ=.30 control")
    return base


def generate_inpaint(experiment, method, parameters, device, *, seed=42):
    from .methods.v3_panel_fix_engine import PanelFixEngine

    def engine(exp, hardware):
        return PanelFixEngine(exp.root / "cache/models", exp.geometry, exp.engine_config,
                              device=hardware, lut_path=exp.lut)

    names = ("v3_panel_fix_engine.py", "v3_joint_cap_engine.py", "v3_joint_cap_refine.py",
             "v3_joint_cap_prompts.py", "v3_joint_cap_blend.py", "v3_joint_cap_guidance.py",
             "v3_ring_midscale_cap.py", "v3_capgeom_init.py")
    hashes = {name: file_hash(METHODS_DIR / name) for name in names}
    run = _evaluation(experiment, seed)
    trial(run, method, parameters, device, seed=seed, source=method if seed == 1234 else None,
          engine_factory=engine, implementation=hashes, capture=capture_initial_caps)


def _engine(experiment, device, *, synthesis):
    from .methods.v3_panel_fix_engine import PanelFixEngine

    if synthesis:
        from .methods.v3_panel_fix_outpaint import PanelFixOutpaintMixin

        class Engine(PanelFixOutpaintMixin, PanelFixEngine):
            pass
    else:
        from .methods.v3_panel_fix_stage2 import PanelFixStageTwoMixin

        class Engine(PanelFixStageTwoMixin, PanelFixEngine):
            pass
    return Engine(experiment.root / "cache/models", experiment.geometry, experiment.engine_config,
                  device=device, lut_path=experiment.lut)


def generate_outpaint(experiment, repair, device, *, seed):
    import torch

    from .methods.v3_joint_cap_geometry import compose_joint_stage1
    from .methods.v3_panel_fix_stage2 import export_panel_maps

    folder = experiment.stage1(seed)
    folder.mkdir(parents=True, exist_ok=True)
    settings = {**experiment.options["outpaint"], **repair}
    files = ("v3_panel_fix_outpaint.py", "v3_panel_fix_engine.py", "v3_joint_cap_geometry.py",
             "v3_joint_cap_ring.py", "v3_joint_cap_blend.py", "v3_circular.py")
    plan = {"options": settings, "prompts": experiment.options["cap_prompt"],
            "implementation": {name: file_hash(METHODS_DIR / name) for name in files}}
    todo = [(brief, folder / f"{brief['id']}-{seed}.png") for brief in experiment.briefs]
    pending = [(brief, path) for brief, path in todo if _stage_cached(path, plan) is None]
    if not pending:
        return

    hardware = _hardware(device)
    engine = _engine(experiment, hardware, synthesis=True)
    engine._base_unet()
    for brief, path in pending:
        source, frozen, _, _ = ring(experiment, engine, brief, seed)

        def generate():
            result = engine.generate_panel_outpaint(brief, seed, source, options=settings)
            output = compose_joint_stage1(engine.topology, engine.region_map, engine.bleed_owner, result)
            return result, output

        (result, output), seconds, peak = _with_timer(hardware, generate)
        if (rgb_pixels(result.ring_image).tobytes() != rgb_pixels(source.image).tobytes()
                or bytes_digest(result.ring_latent.detach().cpu().numpy()) != frozen["latent_sha256"]):
            raise ValueError(f"Outpaint changed R0 in {path.stem}")
        pixels = {"atlas": path, **_files(path)}
        for name, image in (("atlas", output), ("ring", result.ring_image), *result.caps.items()):
            _png(image, pixels[name])
        latent = path.with_suffix(".ring.npy")
        temporary = latent.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, result.ring_latent.detach().cpu().numpy(), allow_pickle=False)
        temporary.replace(latent)
        hashes = {name: file_hash(item) for name, item in {**pixels, "latent": latent}.items()}
        elapsed = float(frozen["ring_elapsed_seconds"]) + seconds
        highest = max(peak, frozen["ring_peak_memory_mib"])
        write_json(path.with_suffix(".run.json"), {
            "recipe": "panel-fix-outpaint", "brief_id": brief["id"], "seed": seed, "plan": plan,
            "raw_sha256": hashes["atlas"], "source_sha256": hashes, "metadata": result.proof,
            "ring": frozen, "stage1_seconds": seconds, "elapsed_seconds": elapsed, "peak_memory_mib": highest,
            "gpu": torch.cuda.get_device_name(hardware),
        })
        if brief["id"] == "E01-001":
            export_panel_maps(engine.topology, result, experiment.options["stage2"], experiment.output / "maps",
                              band=int(engine.config.get("cap_band", 16)))
        print(f"repaired outpaint {path.stem}: {elapsed:.2f} s with R0, {highest:.0f} MiB", flush=True)
    del engine
    torch.cuda.empty_cache()


def generate_stage2(experiment, method, device, *, seed):
    import torch

    gamma = float(experiment.options["guidance_gamma"]["k4"]) if method == "k4" else 0.0
    options = experiment.options["stage2"]
    files = ("v3_panel_fix_stage2.py", "v3_panel_fix_engine.py", "v3_joint_cap_geometry.py",
             "v3_joint_cap_stage2.py", "v3_joint_cap_blend.py", "v3_joint_cap_guidance.py",
             "v3_ring_midscale_cap.py", "v3_capgeom_init.py")
    parameters = {"tight": True, "cap_scale": int(options["cap_scale"]), "strength": options["edge_strength"],
                  "inpaint": True, "guidance_gamma": gamma, "stage2": options,
                  "implementation": {name: file_hash(METHODS_DIR / name) for name in files}}
    run = _evaluation(experiment, seed)
    jobs = list(experiment.samples(methods=(method,), seed=seed))
    pending = []
    for job in jobs:
        if job.validate_generation():
            data = read_json(job.atlas.metadata)
            source = experiment.stage1(seed) / f"{job.atlas.identifier}.png"
            evidence = data["details"]
            if (data["candidate"] != parameters or file_hash(source) != evidence["stage1_raw_sha256"]
                    or file_hash(source.with_suffix(".run.json")) != evidence["stage1_run_sha256"]):
                raise ValueError(f"The seam repair inputs changed for {job.atlas.identifier}")
        else:
            pending.append(job)

    if pending:
        hardware = _hardware(device)
        engine = _engine(experiment, hardware, synthesis=False)
        identity = verify_inpaint_model(experiment.root / "cache/models")
        engine._open_inpaint()
        for job in pending:
            atlas = job.atlas
            atlas.path.parent.mkdir(parents=True, exist_ok=True)
            source = experiment.stage1(seed) / f"{atlas.identifier}.png"
            stage, proof = _load_stage1(engine.topology, source)
            _, ring_proof, body, supported = ring(experiment, engine, atlas.brief, seed)
            (image, details), seconds, peak = _with_timer(
                hardware, lambda: engine.panel_atlas(atlas.brief, seed, stage, options=options, gamma=gamma))
            pixels = rgb_pixels(image)
            if (ring_proof["cache_sha256"] != proof["ring"]["cache_sha256"]
                    or bytes_digest(pixels[body]) != ring_proof["body_rgb_sha256"]
                    or bytes_digest(pixels[supported]) != ring_proof["ring_supported_rgb_sha256"]):
                raise ValueError(f"Seam repair moved an immutable ring pixel in {atlas.identifier}")
            _png(image, atlas.path)
            files = capture_initial_caps(engine, atlas)
            elapsed = seconds + proof["elapsed_seconds"]
            peak = max(peak, proof["peak_memory_mib"])
            write_json(atlas.metadata, {
                "method": method, "brief_id": atlas.brief["id"], "seed": seed, "raw_sha256": file_hash(atlas.path),
                "candidate": parameters, "ring": ring_proof, "elapsed_seconds": elapsed, "peak_memory_mib": peak,
                "cap_elapsed_seconds": seconds, "gpu": torch.cuda.get_device_name(hardware),
                "inpaint_model": identity, "validation_of": method if seed == 1234 else None,
                "details": {**details, "initial_images": files, "stage1_raw_path": str(source),
                            "stage1_raw_sha256": proof["raw_sha256"],
                            "stage1_run_sha256": file_hash(source.with_suffix(".run.json"))},
            })
            print(f"{method} {atlas.identifier}: {elapsed:.2f} s ({seconds:.2f} s seam repair), {peak:.0f} MiB",
                  flush=True)
        del engine
        torch.cuda.empty_cache()
    _check_generation(run, jobs, seed=seed)
    evaluate(run, jobs, device)
