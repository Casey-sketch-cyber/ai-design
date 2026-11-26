"""Render a coordinate-coded view of the M3b and pre-UNet M3 layouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dielinediff.methods.v3_geometry import build_ring_layout
from dielinediff.methods.v3_ring_m3 import extended_ring_layout
from dielinediff.methods.v3_ring_roll import predict_at_phase, ring_shift


def coordinate_map(layout, y_offset):
    y, x = np.indices(layout.latent_shape)
    return np.stack((x, y - y_offset + 8, layout.region_ids * 35), -1).astype(np.uint8)


def render(topology_path, output):
    topology = json.loads(topology_path.read_text())
    original = build_ring_layout(topology, 1280, 1024)
    expanded = extended_ring_layout(topology, (1280, 1152))
    original_map = coordinate_map(original, 0)
    expanded_map = coordinate_map(expanded, 8)
    model_input = []

    def capture(tensor):
        model_input.append(tensor)
        return tensor

    latent = torch.from_numpy(expanded_map).permute(2, 0, 1)[None]
    shift = ring_shift(1, expanded.latent_shape[1])
    restored = predict_at_phase(capture, latent, shift)
    torch.testing.assert_close(restored, latent, atol=0, rtol=0)
    restored_map = restored[0].permute(1, 2, 0).numpy()
    shifted_map = model_input[0][0].permute(1, 2, 0).numpy()
    panels = (
        ("R0: 160 x 128 latent cells", "11.43 mm buffer per side", original, original_map),
        ("M3b: 160 x 144 latent cells", "22.86 mm buffer per side", expanded, expanded_map),
        ("M3 at k=1: roll +37 cells", "Model input and regional labels", expanded, shifted_map),
        ("M3: unroll model output -37", "Projection and DDIM use native coordinates", expanded,
         restored_map),
    )
    page = Image.new("RGB", (704, 756), "white")
    draw = ImageDraw.Draw(page)
    title = ImageFont.load_default(size=23)
    label = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    draw.text((16, 10), "Ring coordinates before and after the M3 roll", font=title, fill="#12212b")
    for i, (heading, caption, layout, data) in enumerate(panels):
        x, y = 16 + 352 * (i % 2), 48 + 330 * (i // 2)
        draw.text((x, y), heading, font=label, fill="#12212b")
        draw.text((x, y + 22), caption, font=small, fill="#12212b")
        image = Image.fromarray(data)
        page.paste(image.resize((320, image.height * 2), Image.Resampling.NEAREST), (x, y + 42))
        for row in layout.body_rows:
            tick_y = y + 42 + 2 * row
            draw.line((x + 321, tick_y, x + 330, tick_y), fill="#12212b", width=2)
    draw.text((16, 728), "Red: native x   Green: shared native y   Blue: physical face ID   Ticks: body edges",
              font=label, fill="#12212b")
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    render(arguments.geometry / "topology.json", arguments.output)
