"""Frozen SDXL components and the common E01 prompt conventions."""

from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline
from PIL import ImageColor

STYLE = (
    "flat orthographic packaging print artwork, evenly lit, "
    "clean graphic design, edge-to-edge illustration"
)
NEGATIVE = "photo of a box, perspective, cast shadow, highlights, watermark, grid, mockup"


def colors(brief):
    """Return the swatch chosen before generation, in sRGB."""
    return ImageColor.getrgb(brief["main_color"])


def prompt_for(brief, face=None):
    descriptions = {
        "F": brief["front_prompt"],
        "B": brief["side_prompt"] + ", calm background for an information panel",
        "L": brief["side_prompt"],
        "R": brief["side_prompt"],
        "T": brief["top_prompt"],
        "Btm": brief["top_prompt"] + ", simple ornamental treatment",
    }
    continuity = ", wraparound artwork" if brief["continuous"] else ""
    if face is None:
        return f"{STYLE}{continuity}, {brief['front_prompt']}, {brief['main_color']}"
    return (f"{STYLE}{continuity}, centered graphic elements with clear margins, "
            f"{descriptions[face]}, color accent {brief['main_color']}")


def load_pipeline(models, *, controlnet=None, device="cuda"):
    models = Path(models)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        models / "vae", torch_dtype=dtype, local_files_only=True, use_safetensors=True
    )
    pipeline = StableDiffusionXLPipeline
    kwargs = {}
    if controlnet is not None:
        from diffusers import StableDiffusionXLControlNetPipeline

        pipeline = StableDiffusionXLControlNetPipeline
        kwargs["controlnet"] = controlnet
    pipe = pipeline.from_pretrained(
        models / "sdxl", vae=vae, variant="fp16", torch_dtype=dtype,
        local_files_only=True, use_safetensors=True, add_watermarker=False, **kwargs,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def padded_size(width, height):
    """Pad to complete VAE blocks and crop back after decoding."""
    return (width + 7) // 8 * 8, (height + 7) // 8 * 8
