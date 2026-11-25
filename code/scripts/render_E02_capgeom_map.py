"""Render C3/G1 cap coordinates on the same grid from the production gather map."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from diagnose_E02_geometry import INK, _cap_view, _font, _ring_view
from dielinediff.methods.v3_capfix_geometry import build_capfix_layout
from dielinediff.methods.v3_capgeom_geometry import build_capgeom_layout
from dielinediff.methods.v3_geometry import CAPS, EDGES, build_cap_layout, build_ring_layout


CAP_NAMES = {"T": "Top cap (T)", "Btm": "Bottom cap (Btm)"}


def render(topology, output):
    ring = build_ring_layout(topology, 1280, 1024)
    coded = _ring_view(ring)
    page = Image.new("RGB", (884, 854), (248, 250, 252))
    draw = ImageDraw.Draw(page)
    columns = (34, 474)
    draw.text((34, 19), "How the frozen side strips meet each cap", font=_font(27), fill=INK)
    draw.text((34, 56), "Frozen strip colors encode the source ring coordinates (x, y) and face ID.",
              font=_font(15), fill=INK)
    for x, color, label in ((34, (251, 251, 247), "Generated core"),
                            (224, (137, 222, 177), "Free gap in C3"),
                            (415, (222, 225, 232), "Discarded corners and padding")):
        draw.rectangle((x, 84, x + 16, 99), fill=color, outline=(136, 145, 159))
        draw.text((x + 23, 83), label, font=_font(15), fill=INK)
    for x, label in zip(columns, ("Task 08 C3 (G0)", "Task 10 G1"), strict=True):
        draw.text((x, 126), label, font=_font(20), fill=INK)

    for row, face in enumerate(CAPS):
        y = 162 + 320 * row
        original = build_cap_layout(topology, ring, face, band_latent=16)
        old = build_capfix_layout(original, ring, 16, 2)
        attached = build_capgeom_layout(original, ring, 16, 2)
        for col, (x, layout) in enumerate(zip(columns, (old, attached), strict=True)):
            raster = _cap_view(layout, coded)
            raster = raster.resize((2 * raster.width, 2 * raster.height), Image.Resampling.NEAREST)
            page.paste(raster, (x, y))
            x0, y0, x1, y1 = layout.center_rect
            midpoint_x, midpoint_y = x + x0 + x1, y + y0 + y1
            draw.text((midpoint_x, midpoint_y - 10), CAP_NAMES[face], font=_font(15), anchor="mm", fill=INK)
            draw.text((midpoint_x, midpoint_y + 10), "784 x 512 px", font=_font(14), anchor="mm", fill=INK)
            distances = " / ".join(str(layout.bleed_cells[edge]) for edge in EDGES)
            draw.text((x + 12, y + raster.height + 6), f"Gap (top / right / bottom / left)  {distances}",
                      font=_font(13), fill=INK)
            if col == 1:
                for edge, (side, _, reverse, _) in original.edge_sources.items():
                    center = {
                        "top": (midpoint_x, y + 2 * y0 + 14),
                        "right": (x + 2 * x1 - 28, midpoint_y),
                        "bottom": (midpoint_x, y + 2 * y1 - 14),
                        "left": (x + 2 * x0 + 29, midpoint_y),
                    }[edge]
                    draw.text(center, side + (" rev." if reverse else ""), anchor="mm",
                              font=_font(13), fill=INK)

    draw.text((34, 802), "Each latent cell is 8 px and shown at 2x. Rev. marks reversed edge order.",
              font=_font(14), fill=INK)
    draw.text((34, 827), "Atlas export samples owned bleed within 3 mm of non-native cuts from the matching frozen side.",
              font=_font(14), fill=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = json.loads((args.geometry / "topology.json").read_text(encoding="utf-8"))
    render(topology, args.output)


if __name__ == "__main__":
    main()
