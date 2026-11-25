"""Pinned SDXL 0.1 inpainting weights and native nine-channel cap conditioning."""

from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
MODEL_LICENSE = "CreativeML Open RAIL++-M"
MODEL_DIRECTORY = "sdxl-inpaint"
WEIGHT_FILE = "unet/diffusion_pytorch_model.fp16.safetensors"
WEIGHT_BYTES = 5_135_178_560
WEIGHT_SHA256 = "6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f"
UPSTREAM = {
    WEIGHT_FILE: (WEIGHT_FILE, WEIGHT_SHA256),
    "unet/config.json": ("unet/config.json", "aaedf41fbcb29d2d8725241413b8525bd97005dea3f86d8056199469032537e9"),
    "upstream/README.md": ("README.md", "80f91748ac64a5f0a72d7b133fc6a2ff5291a35464ea341960b8c23ea59affa4"),
    "upstream/vae-config.json": (
        "vae/config.json", "b822341e6125efedebffb8e3f35e7288aab06bb4c49853d50a16964bf104a727"),
    "upstream/scheduler-config.json": (
        "scheduler/scheduler_config.json", "a8e9a8517e8739938ea85819012bad195d41ccb3e11e2501cf196da64099b1f6"),
}
ORIGINAL_VAE = {
    "vae/config.json": "f18b16fa4381c90ab44a3d7abd2e90afd05a2c31b8ca69998bc93c6453aeb7b7",
    "vae/diffusion_pytorch_model.safetensors": "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",
}
VAE_FACTOR = 8


@lru_cache(maxsize=16)
def _file_sha256(path, _size, _mtime, _inode):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def file_sha256(path):
    stat = path.stat()
    return _file_sha256(path.resolve(), stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _verify_file(path, expected):
    try:
        actual = file_sha256(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing C2 asset {path}; run code/scripts/prepare_E02_inpainting.py") from None
    if actual != expected:
        raise ValueError(f"C2 asset SHA-256 mismatch for {path}: {actual}")


def _read_config(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_inpaint_model(models_dir):
    """Verify published C2 weights and the exact VAE reused from Task 07."""
    models = Path(models_dir)
    directory = models / MODEL_DIRECTORY
    checkpoint = directory / WEIGHT_FILE
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing C2 checkpoint {checkpoint}; run code/scripts/prepare_E02_inpainting.py")
    if checkpoint.stat().st_size != WEIGHT_BYTES:
        raise ValueError("SDXL inpainting checkpoint size does not match the pinned FP16 file")
    for saved, (_, checksum) in UPSTREAM.items():
        _verify_file(directory / saved, checksum)
    for name, checksum in ORIGINAL_VAE.items():
        _verify_file(models / name, checksum)
    unet = _read_config(directory / "unet/config.json")
    common_unet = _read_config(models / "sdxl/unet/config.json")
    text_keys = ("cross_attention_dim", "addition_embed_type", "addition_time_embed_dim",
                 "projection_class_embeddings_input_dim")
    if unet["in_channels"] != 9 or unet["out_channels"] != 4 or any(
            unet.get(key) != common_unet.get(key) for key in text_keys):
        raise ValueError("The inpainting checkpoint is incompatible with the Task 07 SDXL conditioning")
    vae = _read_config(models / "vae/config.json")
    upstream_vae = _read_config(directory / "upstream/vae-config.json")
    codec_keys = ("latent_channels", "scaling_factor", "shift_factor", "latents_mean",
                  "latents_std", "block_out_channels", "in_channels", "out_channels")
    if any(vae.get(key) != upstream_vae.get(key) for key in codec_keys):
        raise ValueError("The original VAE does not match the inpainting latent interface")
    if vae["latent_channels"] != 4 or 2 ** (len(vae["block_out_channels"]) - 1) != VAE_FACTOR:
        raise ValueError("The inpainting latent interface must use four channels and an eightfold VAE")

    scheduler = _read_config(models / "sdxl/scheduler/scheduler_config.json")
    upstream_scheduler = _read_config(directory / "upstream/scheduler-config.json")
    schedule_keys = ("prediction_type", "num_train_timesteps", "beta_schedule", "beta_start", "beta_end")
    if scheduler.get("prediction_type") != "epsilon" or any(
            scheduler.get(key) != upstream_scheduler.get(key) for key in schedule_keys):
        raise ValueError("The Task 07 DDIM schedule is incompatible with the inpainting training schedule")
    official = f"https://huggingface.co/{MODEL_ID}"
    return {
        "schema": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "model_card_url": f"{official}/blob/{MODEL_REVISION}/README.md",
        "license_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        "weight_url": f"{official}/resolve/{MODEL_REVISION}/{WEIGHT_FILE}",
        "weight_bytes": WEIGHT_BYTES,
        "weight_sha256": WEIGHT_SHA256,
        "unet_config_sha256": UPSTREAM["unet/config.json"][1],
        "upstream_files_sha256": {name: checksum for name, (_, checksum) in UPSTREAM.items()},
        "unet_channels": [unet["in_channels"], unet["out_channels"]],
        "reused_vae": {"model_id": "madebyollin/sdxl-vae-fp16-fix", "files_sha256": ORIGINAL_VAE,
                       "latent_channels": vae["latent_channels"], "latent_factor": VAE_FACTOR,
                       "scaling_factor": vae["scaling_factor"]},
        "scheduler": {"sampler": "Task 07 DDIM", **{key: scheduler[key] for key in schedule_keys}},
    }


class InpaintUNet:
    """Lazily load only the new UNet and reuse it across both cap predictions."""

    def __init__(self, models_dir, dtype):
        self.models = Path(models_dir)
        self.dtype = dtype
        self.unet = None
        self.evidence = None

    def load(self, device):
        if self.unet is None:
            self.evidence = verify_inpaint_model(self.models)
            self.unet = UNet2DConditionModel.from_pretrained(
                self.models / MODEL_DIRECTORY / "unet", torch_dtype=self.dtype,
                variant="fp16", local_files_only=True, use_safetensors=True,
            )
        return self.unet.to(device)


def prepare_inpaint_condition(cap_pixels, known_mask, encode):
    """Encode frozen known RGB as the native SDXL masked image and paint mask."""
    if cap_pixels.ndim != 4 or cap_pixels.shape[:2] != (1, 3) or not cap_pixels.is_floating_point():
        raise ValueError("C2 expects one BCHW RGB image normalized to [0,1]")
    source_mask = known_mask if torch.is_tensor(known_mask) else np.array(known_mask, copy=True)
    known = torch.as_tensor(source_mask, dtype=torch.bool, device=cap_pixels.device)
    if known.ndim != 2 or tuple(side * VAE_FACTOR for side in known.shape) != cap_pixels.shape[-2:]:
        raise ValueError("C2 requires a native latent-grid known mask matching the RGB canvas")
    known = known[None, None]
    pixels_known = known.repeat_interleave(VAE_FACTOR, -2).repeat_interleave(VAE_FACTOR, -1)
    # The shared encoder maps [0,1] to [-1,1]. RGB 0.5 therefore masks to normalized zero.
    masked_image = torch.where(pixels_known, cap_pixels, cap_pixels.new_tensor(0.5))
    masked_latents = encode(masked_image)
    if masked_latents.shape != (1, 4, *known.shape[-2:]):
        raise ValueError("C2 masked pixels must encode to four native SDXL channels")
    paint_mask = (~known).to(dtype=masked_latents.dtype, device=masked_latents.device)
    return paint_mask, masked_latents


def predict_inpaint(unet, scaled_latents, timestep, text, added_cond_kwargs,
                    paint_mask, masked_image_latents):
    """Predict epsilon from the native [scaled latent, paint mask, masked image] input."""
    batch = scaled_latents.shape[0]
    condition = tuple(value.to(scaled_latents).expand(batch, -1, -1, -1)
                      for value in (paint_mask, masked_image_latents))
    inputs = torch.cat((scaled_latents, *condition), dim=1)
    return unet(inputs, timestep, encoder_hidden_states=text,
                added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
