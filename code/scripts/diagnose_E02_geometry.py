"""Draw v3 coordinate-registration evidence with synthetic, coordinate-coded data."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from dielinediff.dieline.geometry import REGION_LABELS
from dielinediff.methods.v3_geometry import (
    CAPS,
    SIDES,
    build_cap_layout,
    build_ring_layout,
    compose_atlas,
    compose_support_counts,
)


SCALES = ((1280, 1024), (1472, 1152))
INK = (26, 39, 56)


def _font(size):
    return ImageFont.load_default(size=size)


def _preview(image, size):
    image.thumbnail(size, Image.Resampling.NEAREST)
    return image


def _ring_view(ring):
    h, w = ring.latent_shape
    yy, xx = np.indices((h, w))
    data = np.stack((xx / (w - 1), yy / (h - 1), ring.region_ids / 6))
    return torch.as_tensor(data, dtype=torch.float32)[None]


def _cap_view(cap, coded):
    gathered = cap.read_frozen_ring(coded)[0].permute(1, 2, 0).numpy()
    picture = np.full((*cap.latent_shape, 3), (222, 225, 232), dtype=np.uint8)
    x0, y0, x1, y1 = cap.center_rect
    picture[y0:y1, x0:x1] = (251, 251, 247)
    picture[y0 - cap.bleed_cells["top"]:y0, x0:x1] = (137, 222, 177)
    picture[y1:y1 + cap.bleed_cells["bottom"], x0:x1] = (137, 222, 177)
    picture[y0:y1, x0 - cap.bleed_cells["left"]:x0] = (137, 222, 177)
    picture[y0:y1, x1:x1 + cap.bleed_cells["right"]] = (137, 222, 177)
    picture[cap.known_mask] = np.rint(gathered[cap.known_mask] * 255).astype(np.uint8)
    return Image.fromarray(picture)


def registration_figure(topology, labels, owners, output):
    page = Image.new("RGB", (1215, 675), (248, 250, 252))
    draw = ImageDraw.Draw(page)
    draw.text((15, 12), "E02 coordinate-coded registration", font=_font(23), fill=INK)
    draw.text((15, 43), "Frozen strip colors encode ring x, y, and face ID. "
                        "White is cap core; green is free cap bleed; gray corners are discarded.",
              font=_font(15), fill=INK)
    palette = np.array([(245, 246, 249), (101, 156, 180), (89, 145, 168),
                        (155, 202, 218), (136, 187, 204), (111, 183, 125),
                        (86, 161, 110)], dtype=np.uint8)
    support_report = {}
    for index, (width, height) in enumerate(SCALES):
        y = 83 + index * 285
        ring = build_ring_layout(topology, width, height)
        coded = _ring_view(ring)
        ring_image = np.rint(coded[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        draw.text((15, y), f"Ring {width} x {height} px", font=_font(17), fill=INK)
        ring_preview = Image.fromarray(ring_image)
        ring_preview = ring_preview.resize((298, round(298 * height / width)), Image.Resampling.NEAREST)
        page.paste(ring_preview, (15, y + 24))
        for boundary in ring.body_rows:
            row = y + 24 + round(boundary / ring.latent_shape[0] * ring_preview.height)
            draw.line((15, row, 312, row), fill=(245, 242, 241), width=1)
        for side in SIDES:
            left, _, right, _ = ring.face_rects[side]
            center = 15 + round((left + right) / 2 / ring.latent_shape[1] * ring_preview.width)
            draw.text((center, y + 30), side, anchor="mt", font=_font(13), fill=(250, 250, 250))
        draw.text((15, y + 266), f"Buffer {ring.buffer_mm[0]:.2f} mm per side",
                  font=_font(14), fill=INK)

        caps = {}
        for c, face in enumerate(CAPS):
            x = 347 + c * 275
            cap = build_cap_layout(topology, ring, face)
            canvas = _cap_view(cap, coded)
            preview = canvas.resize((232, round(232 * cap.height_px / cap.width_px)), Image.Resampling.NEAREST)
            page.paste(preview, (x, y + 24))
            draw.text((x, y), f"{face} cap {cap.width_px} x {cap.height_px} px", font=_font(17), fill=INK)
            natives = [edge for edge, (_, _, _, halo) in cap.edge_sources.items() if halo == 0]
            edge_order = " / ".join(side for side, _, _, _ in cap.edge_sources.values())
            draw.text((x, y + 220), edge_order + " clockwise from top", font=_font(14), fill=INK)
            draw.text((x, y + 242), "Native fold at " + ", ".join(natives), font=_font(14), fill=INK)
            caps[face] = (cap, np.asarray(canvas.resize((cap.width_px, cap.height_px), Image.Resampling.NEAREST)))

        ring_native = Image.fromarray(ring_image).resize((width, height), Image.Resampling.NEAREST)
        _, support = compose_atlas(topology, labels, owners, ring, ring_native, caps, return_support=True)
        counts = compose_support_counts(labels, owners, support)
        support_report[f"{width}x{height}"] = {"buffer_mm": ring.buffer_mm, **counts}
        atlas = _preview(Image.fromarray(palette[support]), (272, 231))
        page.paste(atlas, (897, y + 24))
        draw.text((897, y), "Registered source support", font=_font(17), fill=INK)
        draw.text((897, y + 246), "Gray has no registered source", font=_font(14), fill=INK)
        bleed = counts["regions"]["Bleed"]
        draw.text((897, y + 266), f"Bleed {bleed['supported']:,} / {bleed['total']:,} pixels",
                  font=_font(14), fill=INK)

    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, optimize=True)
    output.with_suffix(".json").write_text(json.dumps(support_report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = json.loads((args.geometry / "topology.json").read_text(encoding="utf-8"))
    with Image.open(args.geometry / "region_map.png") as labels, Image.open(args.geometry / "bleed_owner.png") as owners:
        registration_figure(topology, np.asarray(labels), np.asarray(owners), args.output)


if __name__ == "__main__":
    main()
