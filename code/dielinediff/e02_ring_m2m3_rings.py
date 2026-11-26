"""Hash-bound ring generation for the Task 17 ablations."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_capgeom import _restore_alphas, _ring_values
from .methods.v3 import regional_routing
from .methods.v3_capfix import CapFixEngine, PreparedRing
from .methods.v3_ring_m3 import extended_ring_layout


class ExpandedLayoutMixin:
    def _ring_layout(self, size):
        width, height = size
        if (width, height) == (1280, 1024):
            return super()._ring_layout(size)
        band = self.config.get("cap_band", 16)
        key = (width, height, band)
        if key not in self.layouts:
            ring = extended_ring_layout(self.topology, size)
            caps = {face: self.geometry.build_cap_layout(self.topology, ring, face, band_latent=band)
                    for face in ("T", "Btm")}
            self.layouts[key] = ring, regional_routing(ring.region_ids), caps
        return self.layouts[key]


class ExpandedRingEngine(ExpandedLayoutMixin, CapFixEngine):
    pass


def _ring_cache(experiment, engine, method, brief, seed):
    key = f"{brief['id']}-{seed}"
    directory = experiment.ring_paths[method][seed]
    path = directory / f"{key}.npz"
    proof = read_json(path.with_suffix(".json"))
    size = experiment.config["ring_candidates"][method]["size_px"]
    original = read_json(experiment.r0[seed] / f"{key}.json")
    if (proof["cache_sha256"] != file_hash(path) or proof["brief_id"] != brief["id"] or proof["seed"] != seed
            or (method != "r0" and (proof["ring_size_px"] != size or proof["method"] != method
                                     or proof["r0_cache_sha256"] != original["cache_sha256"]))):
        raise ValueError(f"Task 17 ring provenance failed for {method}/{key}")
    with np.load(path, allow_pickle=False) as data:
        latent, rgb, state = data["latent"], data["rgb"], data["rng_state"]
    layout, _, _ = engine._ring_layout(size)
    if tuple(latent.shape[-2:]) != layout.latent_shape or rgb.shape != (size[1], size[0], 3):
        raise ValueError(f"Task 17 cached canvas dimensions differ for {method}/{key}")
    prepared = PreparedRing(layout, torch.as_tensor(latent), Image.fromarray(rgb), torch.as_tensor(state), None)
    actual, _, _, body, support = _ring_values(engine, prepared)
    if any(proof[field] != value for field, value in actual.items()):
        raise ValueError(f"Task 17 ring cache values changed for {method}/{key}")
    return prepared, {**proof, "matches_seed_ring": True}, body, support


def selected_ring(experiment, engine, brief, seed):
    return _ring_cache(experiment, engine, experiment.ring_name, brief, seed)


def _registration(experiment, method):
    directory = experiment.ring_paths[method][42]
    directory.mkdir(parents=True, exist_ok=True)
    recipe = experiment.config["ring_candidates"][method]
    paths = [Path(__file__).with_name("methods") / "v3_ring_m3.py"]
    if recipe["roll"] or recipe["m2"]:
        paths.append(paths[0].with_name("v3_ring_roll.py"))
    if recipe["m2"]:
        paths.append(paths[0].with_name("v3_ring_m2.py"))
    identity = {"run_inputs_sha256": file_hash(experiment.output / "inputs.json"),
                "recipe": recipe, "seed_set": experiment.config["seeds"],
                "implementation": {path.name: file_hash(path) for path in paths}}
    record = directory / "registration.json"
    if record.exists() and read_json(record) != identity:
        raise ValueError(f"The registered implementation for {method} has changed")
    if not record.exists():
        write_json(record, identity)
    return stable_hash(identity)


def _engine(experiment, method, device):
    recipe = experiment.config["ring_candidates"][method]
    config = {**experiment.engine_config, "m2": experiment.config["m2"],
              "m3": experiment.config["m3"], "ring_roll": recipe["roll"]}
    engine_type = ExpandedRingEngine
    if recipe["m2"]:
        from .methods.v3_ring_m2 import M2RingMixin
        from .methods.v3_ring_roll import M3RingMixin

        class M2Engine(M3RingMixin, M2RingMixin, ExpandedRingEngine):
            pass

        engine_type = M2Engine
    elif recipe["roll"]:
        from .methods.v3_attention import region_attention
        from .methods.v3_circular import circular_ring_codec
        from .methods.v3_ring_roll import M3RingMixin, PhasedRegionRouting

        class RollingRingEngine(M3RingMixin, ExpandedRingEngine):
            def _prepare_ring(self, contexts, pooled, generator, size):
                ring, routing, caps = self._ring_layout(size)
                router = PhasedRegionRouting(routing)
                unet = self.pipe.unet
                with circular_ring_codec(unet, self.pipe.vae):
                    with region_attention((unet,), router, contexts), self._ring_phase(
                            rolling=True, set_phase=router.set_phase) as phase:
                        latent = self._denoise(size, generator, contexts[0], pooled)
                    image = self._rgb(latent)
                self.ring_phase_shifts = phase.shifts
                return ring, caps, None, latent, image

        engine_type = RollingRingEngine
    return engine_type(experiment.root / "cache/models", experiment.geometry, config,
                       device=device, lut_path=experiment.lut)


def _prepare(engine, brief, seed, size):
    torch.cuda.synchronize(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    start = time.perf_counter()
    engine._base_unet()
    alphas_device = _restore_alphas(engine, brief)
    prepared = engine.prepare_ring(brief, seed, ring_size=size)
    torch.cuda.synchronize(engine.device)
    return prepared, {"ring_elapsed_seconds": time.perf_counter() - start,
                      "ring_peak_memory_mib": torch.cuda.max_memory_allocated(engine.device) / 2**20,
                      "scheduler_alphas_device": alphas_device}


def generate_rings(experiment, device, methods):
    for method in methods:
        registration = _registration(experiment, method)
        directory = experiment.ring_paths[method][42]
        size = experiment.config["ring_candidates"][method]["size_px"]
        engine = _engine(experiment, method, device)
        for seed in experiment.config["seeds"]:
            for brief in experiment.briefs:
                key = f"{brief['id']}-{seed}"
                cache = directory / f"{key}.npz"
                sidecar = cache.with_suffix(".json")
                if cache.is_file() and sidecar.is_file():
                    _, proof, _, _ = _ring_cache(experiment, engine, method, brief, seed)
                    if proof["registration_sha256"] != registration:
                        raise ValueError(f"Task 17 cached registration differs for {method}/{key}")
                    continue
                prepared, cost = _prepare(engine, brief, seed, size)
                fields, (latent, rgb, state), _, _, _ = _ring_values(engine, prepared)
                original = read_json(experiment.r0[seed] / f"{key}.json")
                if (size == [1280, 1024] and fields["rng_state_sha256"] != original["rng_state_sha256"]):
                    raise ValueError(f"{method}/{key} changed the same-size seeded random stream")
                if method != "m3b" and size == [1280, 1152]:
                    counterpart = read_json(experiment.ring_paths["m3b"][seed] / f"{key}.json")
                    if fields["rng_state_sha256"] != counterpart["rng_state_sha256"]:
                        raise ValueError(f"{method}/{key} changed the expanded same-size random stream")
                temporary = cache.with_suffix(f".{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, latent=latent, rgb=rgb, rng_state=state)
                temporary.replace(cache)
                proof = {**cost, **fields, "method": method, "brief_id": brief["id"], "seed": seed,
                         "cache_sha256": file_hash(cache), "registration_sha256": registration,
                         "r0_cache_sha256": original["cache_sha256"], "ring_size_px": size,
                         "body_rows_latent": list(prepared.layout.body_rows), "gpu": torch.cuda.get_device_name(device)}
                optional = getattr(engine, "m2_prompt_proof", None)
                if optional is not None:
                    proof["m2_prompts"] = optional
                shifts = getattr(engine, "ring_phase_shifts", None)
                if shifts is not None:
                    proof["unet_phase_shifts"] = shifts
                write_json(sidecar, proof)
                print(f"{method} {key}: {cost['ring_elapsed_seconds']:.2f}s, {cost['ring_peak_memory_mib']:.0f} MiB", flush=True)
        del engine
        torch.cuda.empty_cache()
