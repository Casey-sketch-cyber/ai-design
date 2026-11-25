"""Generate the entire dieline in one SDXL + ControlNet pass."""

from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel
from PIL import Image

from .sdxl import NEGATIVE, load_pipeline, padded_size, prompt_for


def panel_edges(region_map, width, height):
    with Image.open(region_map) as image:
        labels = np.asarray(image)
    if labels.shape != (height, width):
        raise ValueError("Region map and topology dimensions differ")
    # Only boundaries belonging to actual panels and flaps are conditioning lines.
    solid = np.where(labels <= 9, labels, 0)
    edge_map = np.zeros(labels.shape, dtype=np.uint8)
    edge_map[:, 1:] |= (solid[:, 1:] != solid[:, :-1]).astype(np.uint8) * 255
    edge_map[1:, :] |= (solid[1:, :] != solid[:-1, :]).astype(np.uint8) * 255
    width_padded, height_padded = padded_size(width, height)
    control_image = Image.new("RGB", (width_padded, height_padded))
    control_image.paste(Image.fromarray(edge_map).convert("RGB"), (0, 0))
    return control_image


class ControlNetBaseline:
    def __init__(self, models, geometry, config, device):
        self.config = config
        self.device = device
        self.width = geometry["topology"]["image"]["width"]
        self.height = geometry["topology"]["image"]["height"]
        self.control_image = panel_edges(
            geometry["directory"] / "region_map.png", self.width, self.height
        )
        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            Path(models) / "controlnet", torch_dtype=dtype, variant="fp16",
            local_files_only=True, use_safetensors=True,
        )
        self.pipe = load_pipeline(models, controlnet=controlnet, device=device)

    @torch.inference_mode()
    def generate(self, brief, seed):
        width, height = self.control_image.size
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt=prompt_for(brief), negative_prompt=NEGATIVE,
            image=self.control_image, width=width, height=height,
            num_inference_steps=self.config["steps"],
            controlnet_conditioning_scale=self.config["controlnet_scale"],
            guidance_scale=self.config["guidance"],
            generator=generator,
        ).images[0]
        return image.crop((0, 0, self.width, self.height))
