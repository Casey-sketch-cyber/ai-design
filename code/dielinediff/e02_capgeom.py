"""Execute Task 10 on the registered ten-case geometry experiment."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02 import BENCHMARK_BASELINES, DEFAULT_CONFIG as TASK07_CONFIG, Sample, grouped
from .e02_capfix import CONFIG as CAPFIX_CONFIG, _hardware
from .e02_capfix_ring import METHODS_DIR, SIDE_LABELS, TASK07_RING, archived_rgb, bytes_digest, protect_ring, rgb_pixels
from .e02_capgeom_eval import compare_step, evaluate, select_strength, write_results
from .e02_freeze import verify
from .e02_history import read_briefs
from .e02_sources import SourceAtlas, sources


CONFIG = Path("code/configs/E02_capgeom.yaml")
REFERENCE = "c3"
PROTOCOL = "e02-v3-capgeom"
ASCENT = ("g1", "g2", "g3", "all")
C3 = {"prompt_mode": "plain", "cap_scale": 2, "inpaint": True, "repaint": False,
      "tight": False, "strength": None, "aligned_attention": False}


class Experiment:
    def __init__(self, root, path):
        self.root = root.resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (config["protocol"] != PROTOCOL or config["selection_seed"] != 42
                or config["validation_seed"] != 1234 or config["strengths"] != [0.75, 0.85]):
            raise ValueError("The task is registered to seed 42, the two SDEdit strengths and seed 1234 validation.")
        old_config = yaml.safe_load(TASK07_CONFIG.read_text(encoding="utf-8"))
        capfix = yaml.safe_load(CAPFIX_CONFIG.read_text(encoding="utf-8"))
        self.config = {**config, "gate": old_config["gate"]}
        self.engine_config = {**old_config["common"], "capfix_prompts": capfix["capfix_prompts"]}
        self.metric_config, self.dino_model, freeze = verify(self.root)
        self.all_briefs = read_briefs()
        self.briefs = self.all_briefs[:10]
        self.legacy = self.root / "outputs/geometry"
        self.geometry = self.root / "outputs/e01b/geometry"
        self.lut = self.root / "cache/icc/iso-coated-v2-lut.npy"
        e02 = self.root / "outputs/e02"
        self.original = e02 / "step1" / config["task07_reference"]
        self.reference = e02 / "capfix/runs" / config["capfix_reference"]
        self.original_rings = e02 / "capfix/rings" / config["archived_rings"]
        reference_data = read_json(self.original / "inputs.json")
        previous = read_json(self.reference / "selection.json")
        if (reference_data["metrics"] != stable_hash(freeze)
                or reference_data["briefs"] != stable_hash(self.briefs)
                or reference_data["generation_config"] != old_config
                or any(file_hash(self.geometry / file) != d for file, d in reference_data["geometry"].items())
                or reference_data["lut_sha256"] != file_hash(self.lut)
                or previous["selected"] != REFERENCE
                or previous["candidates"][REFERENCE] != {k: C3[k] for k in ("prompt_mode", "cap_scale", "inpaint", "repaint")}
                or not previous["complete"]):
            raise ValueError("The archived C3 run does not match Task 07 and the frozen metrics.")
        proof = protect_ring(reference_data)
        code = ("v3_capgeom.py", "v3_capgeom_geometry.py", "v3_capgeom_attention.py",
                "v3_capfix.py", "v3_capfix_geometry.py", "v3_inpainting.py")
        payload = {"config": self.config, "metrics": stable_hash(freeze), "ring": proof,
                   "task07_sha256": file_hash(self.original / "inputs.json"),
                   "capfix_sha256": file_hash(self.reference / "inputs.json"),
                   "implementation": {file: file_hash(METHODS_DIR / file) for file in code}}
        self.output = e02 / "capgeom/runs" / stable_hash(payload)[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "inputs.json", payload)
        self.selection = self.output / "selection.json"

    @staticmethod
    def _sample(atlas, run):
        evaluation = run / "evaluation" / atlas.method
        return Sample(atlas, evaluation / f"{atlas.identifier}.metrics.json",
                      evaluation / f"{atlas.identifier}.preflight.png")

    def samples(self, *, methods, seed=42):
        if seed == 42:
            for item in sources(self.root, self.all_briefs, trial_only=True):
                if item.method in methods and item.method in BENCHMARK_BASELINES:
                    yield self._sample(item, self.original)
        for method in methods:
            if method in BENCHMARK_BASELINES and method != "b2_regional":
                continue
            if method in (*BENCHMARK_BASELINES, REFERENCE) and seed != 42:
                raise ValueError("Reference scores are only used at the registered selection seed.")
            run = self.original if method == "b2_regional" else self.reference if method == REFERENCE else self.output
            for brief in self.briefs:
                path = run / method / f"{brief['id']}-{seed}.png"
                yield self._sample(SourceAtlas(method, brief, seed, path, self.geometry), run)

    def groups42(self, completed):
        return grouped(list(self.samples(methods=(*BENCHMARK_BASELINES, REFERENCE, *completed))))


def _engine(experiment, device):
    from .methods.v3_capgeom import CapGeomEngine

    return CapGeomEngine(experiment.root / "cache/models", experiment.geometry, experiment.engine_config,
                         device=device, lut_path=experiment.lut)


def _restore_alphas(engine, brief):
    device = "cpu" if brief["id"] == "E01-001" else engine.device
    scheduler = engine.pipe.scheduler
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return str(device)


def _measure_ring(engine, brief, seed, *, layer=None):
    import torch
    from .methods.v3_capgeom_attention import capture_aligned_bank

    torch.cuda.synchronize(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    start = time.perf_counter()
    network = engine._base_unet()
    alphas_device = _restore_alphas(engine, brief)
    bank = {} if layer else None
    context = capture_aligned_bank(network, bank, layer) if bank is not None else nullcontext()
    with context:
        prepared = engine.prepare_ring(brief, seed, ring_size=TASK07_RING)
    torch.cuda.synchronize(engine.device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(engine.device) / 2**20
    return prepared, {"ring_elapsed_seconds": seconds, "ring_peak_memory_mib": peak,
                      "scheduler_alphas_device": alphas_device}, bank


def _project(engine, prepared):
    atlas, support = engine.geometry.compose_atlas(
        engine.topology, engine.region_map, engine.bleed_owner,
        prepared.layout, prepared.image, {}, return_support=True,
    )
    body = np.isin(engine.region_map, SIDE_LABELS)
    side = np.isin(support, SIDE_LABELS)
    rgb = rgb_pixels(atlas)
    if not np.all(side[body]):
        raise ValueError("The frozen ring is missing registered side pixels.")
    return rgb, body, side


def _ring_values(engine, prepared):
    rgb, body, side = _project(engine, prepared)
    latent = prepared.frozen_latent.detach().cpu().numpy()
    state = prepared.rng_state.detach().cpu().numpy()
    pixels = rgb_pixels(prepared.image)
    values = {
        "side_pixels": int(body.sum()), "ring_supported_pixels": int(side.sum()),
        "body_rgb_sha256": bytes_digest(rgb[body]), "ring_supported_rgb_sha256": bytes_digest(rgb[side]),
        "latent_sha256": bytes_digest(latent), "ring_rgb_sha256": bytes_digest(pixels),
        "rng_state_sha256": bytes_digest(state),
    }
    return values, (latent, pixels, state), rgb, body, side


def _load(engine, path):
    import torch
    from .methods.v3_capfix import PreparedRing

    with np.load(path, allow_pickle=False) as data:
        latent, rgb, state = data["latent"], data["rgb"], data["rng_state"]
    layout, _, _ = engine._ring_layout(TASK07_RING)
    return PreparedRing(layout, torch.as_tensor(latent), Image.fromarray(rgb), torch.as_tensor(state), None)


def validation_rings(experiment):
    return getattr(experiment, "validation_ring_dir", experiment.output / "ring-validation")


def ring(experiment, engine, brief, seed):
    directory = experiment.original_rings if seed == 42 else validation_rings(experiment)
    directory.mkdir(parents=True, exist_ok=True)
    key = f"{brief['id']}-{seed}"
    path = directory / f"{key}.npz"
    sidecar = directory / f"{key}.json"
    if path.is_file() and sidecar.is_file():
        record = read_json(sidecar)
        if record["cache_sha256"] != file_hash(path):
            raise ValueError(f"Ring cache bytes changed for {key}")
        prepared = _load(engine, path)
    elif seed == 42:
        raise FileNotFoundError(f"Task 08 reviewed ring cache missing for {key}")
    else:
        prepared, record, _ = _measure_ring(engine, brief, seed)
    fields, contents, rgb, body, support = _ring_values(engine, prepared)
    if "cache_sha256" in record:
        if any(record.get(k) != value for k, value in fields.items()):
            raise ValueError(f"Ring cache values changed for {key}")
    else:
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, latent=contents[0], rgb=contents[1], rng_state=contents[2])
        temporary.replace(path)
        record.update(fields, cache_sha256=file_hash(path), brief_id=brief["id"], seed=seed)
        write_json(sidecar, record)
    if seed == 42:
        archive = archived_rgb(experiment.original / "v3_1280" / f"{key}.png")
        if not np.array_equal(rgb[support], archive[support]):
            raise ValueError(f"Task 07 ring and side bleed differ for {key}")
        record = {**record, "matches_task07_body": True, "matches_seed_ring": True}
    else:
        record = {**record, "matches_seed_ring": True}
    return prepared, record, body, support


def prepare_validation(experiment, device):
    import torch
    path = validation_rings(experiment)
    if all((path / f"{b['id']}-1234.npz").is_file() and (path / f"{b['id']}-1234.json").is_file()
           for b in experiment.briefs):
        return
    engine = _engine(experiment, _hardware(device))
    for brief in experiment.briefs:
        _, record, _, _ = ring(experiment, engine, brief, 1234)
        print(f"validation ring {brief['id']}-1234: {record['ring_elapsed_seconds']:.2f}s", flush=True)
    del engine
    torch.cuda.empty_cache()


def _style_record(engine, brief, seed, layer, frozen):
    """Capture K/V while regenerating the same frozen ring and check its exact identity."""
    prepared, cost, bank = _measure_ring(engine, brief, seed, layer=layer)
    if len(bank) != engine.config["steps"]:
        raise ValueError("A ring K/V pair must exist for all original DDIM steps.")
    fields, _, _, _, _ = _ring_values(engine, prepared)
    if any(frozen[k] != value for k, value in fields.items()):
        raise ValueError("Recording G3 attention altered the original ring or RNG.")
    total = sum(tensor.numel() * tensor.element_size() for pairs in bank.values() for tensor in pairs.values())
    recorded = {"layer": layer, "steps": list(bank), "cpu_bank_bytes": total, "cost": cost}
    return bank, recorded


def generate(experiment, candidate, parameters, device, *, seed=42, source=None, bound_g3=False,
             engine_factory=_engine, implementation=None):
    import torch
    from .methods.v3_capgeom_attention import inject_aligned_bank
    from .methods.v3_inpainting import verify_inpaint_model

    destination = experiment.output / candidate
    destination.mkdir(parents=True, exist_ok=True)
    plan = {"parameters": parameters, "seed": seed, "validation_of": source}
    if implementation is not None:
        plan["cap_implementation"] = implementation
    if parameters["strength"] is not None:
        plan["harmonic_implementation_sha256"] = file_hash(METHODS_DIR / "v3_capgeom_init.py")
    plan_path = destination / "candidate.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError(f"Cached cap geometry plan has changed: {candidate}")
    write_json(plan_path, plan)
    jobs = list(experiment.samples(methods=(candidate,), seed=seed))
    pending = [job for job in jobs if not job.validate_generation()]
    if bound_g3:
        for job in jobs:
            if job.validate_generation():
                cost = read_json(job.atlas.metadata)
                if over_budget(cost, experiment.config["gate"]):
                    return jobs, budget_loss(job, cost)
    if not pending:
        return jobs, None

    hardware = _hardware(device)
    engine = engine_factory(experiment, hardware)
    models = experiment.root / "cache/models"
    identity = verify_inpaint_model(models)
    engine._open_inpaint()
    torch.cuda.synchronize(hardware)
    for job in pending:
        item = job.atlas
        prepared, proof, body, support = ring(experiment, engine, item.brief, seed)
        used = []
        attention = {}
        layer = experiment.config["attention"]["layer"] if parameters["aligned_attention"] else None
        try:
            if layer:
                bank, attention = _style_record(engine, item.brief, seed, layer, proof)
                ring_timing = attention["cost"]
            else:
                ring_timing = proof

            torch.cuda.synchronize(hardware)
            torch.cuda.reset_peak_memory_stats(hardware)
            start = time.perf_counter()
            context = inject_aligned_bank(engine._open_inpaint(), bank, layer, used) if layer else nullcontext()
            with context:
                image, details = engine.generate_caps(item.brief, prepared, parameters)
            torch.cuda.synchronize(hardware)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(hardware) / 2**20
        except torch.cuda.OutOfMemoryError as error:
            if bound_g3:
                return jobs, {"reason": "cuda_out_of_memory", "brief_id": item.brief["id"], "message": str(error)}
            raise
        if layer:
            count = (int(engine.config["steps"] * parameters["strength"]) if parameters["strength"]
                     else engine.config["steps"])
            expected = attention["steps"][-count:]
            if used != expected * 2:
                raise ValueError(f"The two cap schedules did not both use exact ring steps: {used!r}")
            attention["cap_timesteps"] = used
            del bank
        pixels = rgb_pixels(image)
        if (bytes_digest(pixels[body]) != proof["body_rgb_sha256"]
                or bytes_digest(pixels[support]) != proof["ring_supported_rgb_sha256"]
                or bytes_digest(prepared.frozen_latent.detach().cpu().numpy()) != proof["latent_sha256"]
                or bytes_digest(prepared.rng_state.detach().cpu().numpy()) != proof["rng_state_sha256"]):
            raise RuntimeError(f"Generation modified a frozen ring pixel, latent or RNG: {item.identifier}")
        elapsed = ring_timing["ring_elapsed_seconds"] + seconds
        peak = max(peak, ring_timing["ring_peak_memory_mib"])
        temporary = item.path.with_suffix(f".{os.getpid()}.tmp.png")
        image.save(temporary)
        temporary.replace(item.path)
        metadata = {
            "method": candidate, "brief_id": item.brief["id"], "seed": seed,
            "raw_sha256": file_hash(item.path), "gpu": torch.cuda.get_device_name(hardware),
            "elapsed_seconds": elapsed, "peak_memory_mib": peak, "cap_elapsed_seconds": seconds,
            "ring": proof, "candidate": parameters, "validation_of": source, "inpaint_model": identity,
            "time_aligned_attention": attention, "details": details,
        }
        write_json(item.metadata, metadata)
        print(f"{candidate} {item.identifier}: {elapsed:.2f}s (caps {seconds:.2f}s), {peak:.0f} MiB", flush=True)
        del prepared
        if bound_g3 and over_budget(metadata, experiment.config["gate"]):
            return jobs, budget_loss(job, metadata)
    del engine
    torch.cuda.empty_cache()
    return jobs, None


def over_budget(record, limits):
    return record["elapsed_seconds"] > limits["seconds"] or record["peak_memory_mib"] > limits["peak_memory_mib"]


def budget_loss(job, cost):
    return {"reason": "exceeds_hardware_budget", "brief_id": job.atlas.brief["id"],
            "elapsed_seconds": cost["elapsed_seconds"], "peak_memory_mib": cost["peak_memory_mib"]}


def _check_generation(experiment, jobs, *, seed):
    region_path = experiment.geometry / "region_map.png"
    with Image.open(region_path) as source:
        mask = np.isin(np.asarray(source), SIDE_LABELS)
    for job in jobs:
        item = job.atlas
        expected = (experiment.original_rings if seed == 42 else validation_rings(experiment))
        source = read_json(expected / f"{item.identifier}.json")
        record = read_json(item.metadata)["ring"]
        with Image.open(item.path) as picture:
            actual = bytes_digest(rgb_pixels(picture)[mask])
        if (not job.validate_generation() or not record["matches_seed_ring"]
                or any(record[k] != source[k] for k in ("cache_sha256", "latent_sha256", "body_rgb_sha256",
                                                        "ring_rgb_sha256", "ring_supported_rgb_sha256"))
                or actual != record["body_rgb_sha256"]):
            raise ValueError(f"Subsequent ring verification failed for {item.identifier}.")


def trial(experiment, name, params, device, *, seed=42, source=None, engine_factory=_engine,
          implementation=None):
    jobs, rejected = generate(experiment, name, params, device, seed=seed, source=source,
                               bound_g3=name == "g3", engine_factory=engine_factory,
                               implementation=implementation)
    if rejected is not None:
        return rejected
    _check_generation(experiment, jobs, seed=seed)
    evaluate(experiment, jobs, device)
    return None


def _publish(experiment, selection, visual=None, banner=None):
    groups = experiment.groups42(selection["completed"])
    validation = (grouped(list(experiment.samples(methods=("seed_1234",), seed=1234)))
                  if selection["complete"] else None)
    result = write_results(experiment, groups, validation, selection["selected"], visual, banner)
    if selection["complete"]:
        from .e02_capfix_visual import render_capfix

        render_capfix(experiment, selection["selected"], original_method=REFERENCE,
                      original_title="Task 08 C3", current_title="Task 10 cap geometry")
    print(json.dumps({"selected": result["selected"], "passed": result["passed"],
                      "checks": {name: check["passed"] for name, check in result["checks"].items()}}))
    return result


def _write_selection(experiment, selection):
    groups = experiment.groups42(selection["completed"])
    selection["scores_sha256"] = {
        f"{method}/{item.atlas.identifier}": file_hash(item.score)
        for method, entries in groups.items() for item, _ in entries
    }
    write_json(experiment.selection, selection)


def run(experiment, device, through):
    params = {REFERENCE: dict(C3), "g1": {**C3, "tight": True}}
    trial(experiment, "g1", params["g1"], device)
    present = ["g1"]
    limits = experiment.config["gate"]
    groups = experiment.groups42(present)
    choice = compare_step(groups, REFERENCE, "g1", limits, stage="G1")
    ledger = {"stages": [choice], "candidates": params, "completed": present,
              "selected": choice["selected"], "complete": False}
    _write_selection(experiment, ledger)
    print(f"G1 versus C3 selected {ledger['selected']}", flush=True)
    if through == "g1":
        return
    incumbent = ledger["selected"]
    for name, strength in zip(("g2_075", "g2_085"), experiment.config["strengths"], strict=True):
        params[name] = {**params[incumbent], "strength": strength}
        trial(experiment, name, params[name], device)
        present.append(name)
    groups = experiment.groups42(present)
    choice = select_strength(groups, incumbent, "g2_075", "g2_085", limits)
    ledger["stages"].append(choice)
    ledger["selected"] = choice["selected"]
    _write_selection(experiment, ledger)
    print(f"G2 after {incumbent} selected {ledger['selected']}", flush=True)
    if through == "g2":
        return
    incumbent = ledger["selected"]
    params["g3"] = {**params[incumbent], "aligned_attention": True}
    rejected = trial(experiment, "g3", params["g3"], device)
    if rejected:
        choice = {"stage": "G3", "reference": incumbent, "challenger": "g3",
                  "selected": incumbent, "measured": False, **rejected}
    else:
        present.append("g3")
        groups = experiment.groups42(present)
        choice = {**compare_step(groups, incumbent, "g3", limits, stage="G3"), "measured": True}
    ledger["stages"].append(choice)
    ledger["selected"] = choice["selected"]
    _write_selection(experiment, ledger)
    print(f"G3 after {incumbent} selected {ledger['selected']} ({choice['reason']})", flush=True)
    if through == "g3":
        return
    chosen = ledger["selected"]
    prepare_validation(experiment, device)
    trial(experiment, "seed_1234", params[chosen], device, seed=1234, source=chosen)
    ledger["complete"] = True
    _write_selection(experiment, ledger)
    visual_path = experiment.output / "visual-review.json"
    banner_path = experiment.output / "banner-review.json"
    _publish(experiment, ledger, read_json(visual_path) if visual_path.exists() else None,
             read_json(banner_path) if banner_path.exists() else None)


def review(experiment, path):
    ledger = read_json(experiment.selection)
    if not ledger["complete"]:
        raise ValueError("Finish selection and seed 1234 validation before marking original-pixel evidence.")
    record = read_json(path)
    visual, banner = record["visual_review"], record["banner_review"]
    current = read_json(experiment.output / "gate.json")
    names = {brief["id"] for brief in experiment.briefs}
    for item, field in ((visual, "visible_artifact"), (banner, "stacked_banner_white_line")):
        rows = item.get("samples", {})
        if (item.get("fingerprint") != current["fingerprint"] or set(rows) != names or any(
                not isinstance(entry, dict) or not isinstance(entry.get(field), bool)
                or not isinstance(entry.get("notes"), str) or not entry["notes"].strip()
                for entry in rows.values())):
            raise ValueError(f"Expected ten explicit observations for the current gate: {field}.")
    result = _publish(experiment, ledger, visual, banner)
    if result.get("visual_review") is None:
        raise ValueError("The visual observations do not bind to the rebuilt gate.")
    write_json(experiment.output / "visual-review.json", visual)
    write_json(experiment.output / "banner-review.json", banner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("paths", "run", "review"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--through", choices=ASCENT, default="all")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    if args.root is None or args.action == "review" and args.review_file is None:
        parser.error("root is required; review additionally requires --review-file")
    experiment = Experiment(args.root, args.config)
    if args.action == "paths":
        print(json.dumps({"run": str(experiment.output), "reference": str(experiment.reference)}))
    elif args.action == "run":
        run(experiment, args.device, args.through)
    else:
        review(experiment, args.review_file)


if __name__ == "__main__":
    main()
