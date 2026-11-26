"""Measure the M3 UNet phase change against the corresponding downsample phase."""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from .e02 import DEFAULT_CONFIG
from .e02_history import read_briefs
from .methods.v3 import regional_routing
from .methods.v3_attention import region_attention
from .methods.v3_capfix import CapFixEngine
from .methods.v3_circular import circular_ring_codec
from .methods.v3_ring_m3 import EXPANDED_RING_SIZE, extended_ring_layout
from .methods.v3_ring_roll import PhasedRegionRouting, predict_at_phase, ring_shift


def _difference(first, second):
    delta = first.float() - second.float()
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "mae": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }


@torch.inference_mode()
def check_phase(engine, brief, *, seed=42):
    """Compare the same early expanded-ring latent under four full SDXL forwards."""
    if torch.device(engine.device).type != "cuda":
        raise ValueError("The deployed SDXL phase check requires CUDA")
    layout = extended_ring_layout(engine.topology, EXPANDED_RING_SIZE)
    routing = PhasedRegionRouting(regional_routing(layout.region_ids))
    contexts, pooled = engine._conditions(brief)
    unet, vae, scheduler = engine.pipe.unet, engine.pipe.vae, engine.pipe.scheduler
    phase_stride = 2 ** sum(bool(block.downsamplers) for block in unet.down_blocks)
    latent_h, latent_w = layout.latent_shape
    rolling_shift = ring_shift(1, latent_w)
    reference_shift = rolling_shift % phase_stride
    same_phase_shift = rolling_shift - reference_shift
    config = engine.config
    condition = {
        "text_embeds": pooled,
        "time_ids": torch.tensor(
            (layout.height_px, layout.width_px, 0, 0, layout.height_px, layout.width_px),
            dtype=engine.dtype, device=engine.device,
        )[None].expand(2, -1),
    }
    generator = torch.Generator(device=engine.device).manual_seed(seed)
    if brief["id"] != "E01-001":
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(engine.device)
    scheduler.set_timesteps(config["steps"], device=engine.device)
    warm_t, measure_t = scheduler.timesteps[:2]
    latent = torch.randn(
        (1, 4, latent_h, latent_w), device=engine.device, dtype=engine.dtype, generator=generator
    ) * scheduler.init_noise_sigma
    with circular_ring_codec(unet, vae), region_attention((unet,), routing, contexts):
        warmed = engine._predict(latent, warm_t, contexts[0], condition, None)
        if config.get("projection_strength", 0) > 0 and round(
                int(config["steps"]) * (1 - float(config.get("projection_tail", 0.30)))) == 0:
            warmed = engine.color_constraint(latent, warmed, warm_t)
        latent = scheduler.step(warmed, warm_t, latent, return_dict=False)[0]
        predictions = {}
        for shift in (0, reference_shift, same_phase_shift, rolling_shift):
            routing.set_phase(1, shift)
            output = predict_at_phase(engine._predict, latent, shift, measure_t, contexts[0], condition, None)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite deployed epsilon at phase {shift}")
            predictions[shift] = output
    ordinary = _difference(predictions[reference_shift], predictions[0])
    rolled = _difference(predictions[rolling_shift], predictions[0])
    matching = _difference(predictions[rolling_shift], predictions[reference_shift])
    aligned = _difference(predictions[same_phase_shift], predictions[0])
    ratio = rolled["rmse"] / ordinary["rmse"] if ordinary["rmse"] else None
    return {
        "brief_id": brief["id"], "seed": seed, "step_index": 1, "timestep": int(measure_t),
        "gpu": torch.cuda.get_device_name(engine.device), "dtype": str(engine.dtype),
        "ring_size_px": list(EXPANDED_RING_SIZE), "latent_shape": list(latent.shape),
        "downsample_phase_stride": phase_stride, "m3_shift": rolling_shift,
        "matching_reference_shift": reference_shift, "aligned_control_shift": same_phase_shift,
        "difference_m3_vs_native": rolled,
        "difference_reference_phase_vs_native": ordinary,
        "difference_m3_vs_matching_phase": matching,
        "difference_aligned_control_vs_native": aligned,
        "m3_to_reference_phase_rmse_ratio": ratio,
        "same_order_as_downsample_phase": ratio is not None and math.isfinite(ratio) and 0.5 <= ratio <= 2,
        "restored_prediction_coordinates": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brief", default="E01-003")
    arguments = parser.parse_args()
    root, output = arguments.root.resolve(), arguments.output.resolve()
    if root not in output.parents:
        parser.error("Store phase evidence beneath the project data root")
    brief = next((item for item in read_briefs() if item["id"] == arguments.brief), None)
    if brief is None:
        parser.error("Unknown registered brief")
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["common"]
    engine = CapFixEngine(root / "cache/models", root / "outputs/e01b/geometry", config,
                          device="cuda", lut_path=root / "cache/icc/iso-coated-v2-lut.npy")
    result = check_phase(engine, brief)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
