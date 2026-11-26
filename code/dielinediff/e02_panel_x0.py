"""Replay two archived panels and save the actual DDIM predicted-original samples."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from .e01 import read_json, write_json
from .e01b import file_hash, stable_hash
from .e02_capfix_ring import bytes_digest
from .e02_capgeom import _hardware, ring
from .e02_history import read_briefs
from .e02_ring_m2m3_rings import ExpandedLayoutMixin
from .methods.v3_circular import full_vae_canvas
from .methods.v3_inpainting import verify_inpaint_model
from .methods.v3_joint_cap_engine import JointCapEngine
from .methods.v3_ring_midscale_cap import MidscaleTextureEngine


CASES = {"j0": "E01-009", "j3_g030": "E01-004"}
SEED = 42


class DDIMTrace:
    """Observe scheduler outputs without rewriting the timestep, latent or RNG state."""

    def __init__(self, scheduler, full_steps, strength):
        self.scheduler = scheduler
        self.full_steps = full_steps
        self.executed_steps = int(full_steps * strength)
        self.steps = []
        self.frames = []
        self.final = None
        self._step = scheduler.step

    def step(self, epsilon, timestep, latent, *args, **kwargs):
        result = self._step(epsilon, timestep, latent, *args, **kwargs)
        length = self.executed_steps
        index = self.full_steps - length + len(self.steps)
        if int(timestep) != int(self.scheduler.timesteps[index]):
            raise RuntimeError("DDIM does not match the registered SDEdit suffix")
        self.steps.append(int(timestep))
        if len(self.steps) == 1 or (index + 1) % 3 == 0 or len(self.steps) == length:
            prediction = result[1] if isinstance(result, tuple) else result.pred_original_sample
            self.frames.append((index, int(timestep), prediction.detach().cpu().clone()))
        if len(self.steps) == length:
            previous = result[0] if isinstance(result, tuple) else result.prev_sample
            self.final = previous.detach().cpu().clone()
        return result


class PanelTraceMixin:
    def encode(self, rgb):
        trace = getattr(self, "_cap_trace", None)
        if trace is not None and not trace.steps:
            self._trace_initial = rgb.detach().cpu().clone()
        return super().encode(rgb)

    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        trace = DDIMTrace(self.pipe.scheduler, int(self.config["steps"]), candidate["strength"])
        self._cap_trace = trace
        self._trace_initial = None
        try:
            with patch.object(self.pipe.scheduler, "step", trace.step):
                generated = super()._candidate_cap(
                    layout, prepared, generator, network, text, pooled, routing, candidate)
        finally:
            self._cap_trace = None
        if trace.final is None or len(trace.steps) != generated[1]["executed_steps"]:
            raise RuntimeError("Traced scheduler steps do not cover the archived cap schedule")
        destination = self.diagnostic_directory / layout.face
        destination.mkdir(parents=True, exist_ok=True)
        full_shape = (layout.height_px, layout.width_px, 3)
        if self._trace_initial is None:
            raise RuntimeError("The SDEdit prior was never encoded before the first step")
        initial = self._trace_initial.squeeze(0).permute(1, 2, 0).contiguous().numpy()
        if initial.shape != full_shape or initial.dtype != np.float32:
            raise RuntimeError("Could not recover the exact pre-VAE cap prior")
        np.save(destination / "initial.npy", initial, allow_pickle=False)
        initial_rgb = Image.fromarray((initial.clip(0, 1) * 255).round().astype(np.uint8))
        initial_rgb.save(destination / "initial.png")
        Image.fromarray(layout.known_mask.repeat(8, 0).repeat(8, 1).astype(np.uint8) * 255).save(
            destination / "known-mask.png")
        layout.read_frozen_ring_rgb(prepared.image).save(destination / "known-rgb.png")
        sequence = []
        with full_vae_canvas(self.pipe.vae):
            for index, timestep, captured in trace.frames:
                path = destination / f"x0-step-{index + 1:02d}.png"
                self._rgb(captured.to(self.device)).save(path)
                sequence.append({"step": index + 1, "timestep": timestep,
                                 "file": path.name, "sha256": file_hash(path)})
            self._rgb(trace.final.to(self.device)).save(destination / "final-decode.png")
        write_json(destination / "trace.json", {
            "face": layout.face, "scale": layout.scale, "size_px": [layout.width_px, layout.height_px],
            "center_rect_px": [8 * coordinate for coordinate in layout.center_rect],
            "atlas_face_rect_px": self.topology["faces"][layout.face]["rect"],
            "known_latents": int(layout.known_mask.sum()), "candidate": candidate,
            "schedule": generated[1], "timesteps": trace.steps, "frames": sequence,
            "initial_rgb_sha256": bytes_digest(initial),
            "initial_npy_sha256": file_hash(destination / "initial.npy"),
            "final_decode_sha256": file_hash(destination / "final-decode.png"),
        })
        return generated


class J0Trace(PanelTraceMixin, ExpandedLayoutMixin, MidscaleTextureEngine):
    pass


class J3Trace(PanelTraceMixin, JointCapEngine):
    pass


def replay(root, destination, device):
    documents = Path("docs/experiments/E02")
    task10 = read_json(documents / "capgeom-inputs.json")
    task17 = read_json(documents / "ring-m2m3-inputs.json")
    task20 = read_json(documents / "joint-cap-inputs.json")
    e02 = root / "outputs/e02"
    common = yaml.safe_load(Path("code/configs/E02_step1.yaml").read_text())["common"]
    capfix = yaml.safe_load(Path("code/configs/E02_capfix.yaml").read_text())["capfix_prompts"]
    options = yaml.safe_load(Path("code/configs/E02_joint_cap.yaml").read_text())
    files = {
        "j0": e02 / "ring-m2m3/runs" / task20["config"]["task17_reference"] / "u1_sigma28",
        "j3_g030": e02 / "joint-cap/runs" / stable_hash(task20)[:12] / "j3_g030",
    }
    experiment = SimpleNamespace(
        root=root, output=destination, original=e02 / "step1" / task10["config"]["task07_reference"],
        original_rings=e02 / "capfix/rings" / task10["config"]["archived_rings"],
    )
    brief_set = {brief["id"]: brief for brief in read_briefs()[:10]}
    for method, brief_name in CASES.items():
        name = f"{brief_name}-{SEED}"
        archived = files[method] / f"{name}.png"
        record = read_json(archived.with_suffix(".run.json"))
        if (record["brief_id"], record["seed"], record["raw_sha256"]) != (brief_name, SEED, file_hash(archived)):
            raise ValueError(f"The archived output changed for {method}/{name}")
        ring_artifact = experiment.original_rings / f"{name}.npz"
        if file_hash(ring_artifact) != task17["original_ring_caches"][name]:
            raise ValueError(f"The archived R0 ring changed for {name}")
        kind = J0Trace if method == "j0" else J3Trace
        engine = kind(root / "cache/models", root / "outputs/e01b/geometry",
                      {**common, "capfix_prompts": capfix, "joint_cap": options},
                      device=device, lut_path=root / "cache/icc/iso-coated-v2-lut.npy")
        engine.diagnostic_directory = destination / method / brief_name
        model = verify_inpaint_model(root / "cache/models")
        if model != record["inpaint_model"]:
            raise ValueError(f"The archived inpainting model differs for {method}/{name}")
        prepared, ring_proof, *_ = ring(experiment, engine, brief_set[brief_name], SEED)
        if ring_proof["cache_sha256"] != record["ring"]["cache_sha256"]:
            raise ValueError(f"The archived ring differs for {method}/{name}")
        image, details = engine.generate_caps(brief_set[brief_name], prepared, record["candidate"])
        with Image.open(archived) as opened:
            expected = np.asarray(opened.convert("RGB"))
        actual = np.asarray(image)
        if actual.shape != expected.shape or not np.array_equal(expected, actual):
            raise RuntimeError(f"Trace did not reproduce the archived atlas byte for byte: {method}/{name}")
        write_json(engine.diagnostic_directory / "replay.json", {
            "method": method, "brief_id": brief_name, "seed": SEED,
            "archived_raw_sha256": record["raw_sha256"], "ring": ring_proof,
            "candidate": record["candidate"], "caps": details["caps"],
            "inpaint_model": model, "archived_atlas_rgb_exact": True,
            "replayed_atlas_rgb_sha256": bytes_digest(actual),
        })
        print(f"{method}/{name}: original RGB exact; full-size DDIM predictions saved", flush=True)
        del engine, prepared
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    replay(args.root, args.output, _hardware(args.device))


if __name__ == "__main__":
    main()
