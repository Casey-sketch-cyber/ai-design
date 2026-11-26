"""Generate and bind the paired R1 rings without altering the archived sampler."""

import os

import numpy as np
import torch

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_capgeom import _load, _measure_ring, _ring_values, ring
from .e02_uniform_ring_baseline import regional_conditions_with_negative, regional_negative, tokenizer_proof
from .methods.v3_capfix import CapFixEngine


class R1Engine(CapFixEngine):
    def _conditions(self, brief):
        conditions, self.prompt_proof = regional_conditions_with_negative(self, brief, self.negative_prompt)
        return conditions


def r1_ring(experiment, engine, brief, seed):
    key = f"{brief['id']}-{seed}"
    path = experiment.r1 / f"{key}.npz"
    sidecar = path.with_suffix(".json")
    proof = read_json(sidecar)
    original = read_json(experiment.r0[seed] / f"{key}.json")
    negative = regional_negative(experiment.negative_suffix)
    if (proof["cache_sha256"] != file_hash(path) or proof["r0_cache_sha256"] != original["cache_sha256"]
            or proof["prompt"]["negative_prompt"] != negative
            or proof["rng_state_sha256"] != original["rng_state_sha256"]):
        raise ValueError(f"R1 provenance failed for {key}")
    prepared = _load(engine, path)
    actual, _, _, body, support = _ring_values(engine, prepared)
    if any(proof[field] != value for field, value in actual.items()):
        raise ValueError(f"R1 ring values changed for {key}")
    return prepared, {**proof, "matches_seed_ring": True}, body, support


def generate_r1(experiment, device):
    experiment.r1.mkdir(parents=True, exist_ok=True)
    engine = R1Engine(experiment.root / "cache/models", experiment.geometry, experiment.engine_config,
                      device=device, lut_path=experiment.lut)
    engine.negative_prompt = regional_negative(experiment.negative_suffix)
    tokenizers = tokenizer_proof((engine.pipe.tokenizer, engine.pipe.tokenizer_2), engine.negative_prompt)
    for seed in experiment.r0:
        for brief in experiment.briefs:
            identifier = brief["id"]
            key = f"{identifier}-{seed}"
            path = experiment.r1 / f"{key}.npz"
            sidecar = path.with_suffix(".json")
            if path.is_file() and sidecar.is_file():
                r1_ring(experiment, engine, brief, seed)
                continue
            _, original, _, _ = ring(experiment, engine, brief, seed)
            prepared, cost, _ = _measure_ring(engine, brief, seed)
            fields, (latent, rgb, state), _, _, _ = _ring_values(engine, prepared)
            if fields["rng_state_sha256"] != original["rng_state_sha256"]:
                raise ValueError(f"R1 unexpectedly changed the sampler RNG state for {key}")
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez(stream, latent=latent, rgb=rgb, rng_state=state)
            temporary.replace(path)
            proof = {**cost, **fields, "brief_id": identifier, "seed": seed, "cache_sha256": file_hash(path),
                     "r0_cache_sha256": original["cache_sha256"], "prompt": engine.prompt_proof,
                     "negative_tokenizers": tokenizers}
            write_json(sidecar, proof)
            print(f"R1 {key}: {cost['ring_elapsed_seconds']:.2f}s, {cost['ring_peak_memory_mib']:.0f} MiB", flush=True)
    del engine
    torch.cuda.empty_cache()
