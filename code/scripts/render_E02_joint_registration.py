"""Draw the Task 20 physical Stage 1 registration and actual Stage 2 strengths."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dielinediff.methods.v3_geometry import CAPS, build_ring_layout
from dielinediff.methods.v3_joint_cap_geometry import (
    NATIVE, JointStageOne, build_joint_ring_layout, build_outpaint_layout,
)
from dielinediff.methods.v3_joint_cap_stage2 import STAGE2_OPTIONS, _layout, stage2_strength_map


INK = "#172934"
MUTED = "#45606a"
COLORS = {"L": "#cfdae1", "F": "#b9cee2", "R": "#d9e3d5", "B": "#d8c8db",
          "T": "#70a5ce", "Btm": "#bf8ab4"}
CONTEXT = (177, 210, 210)
TITLE = ImageFont.load_default(size=23)
LABEL = ImageFont.load_default(size=16)
SMALL = ImageFont.load_default(size=14)


def _scaled(rect, origin, ratio):
    return tuple(round(v * ratio + origin[i % 2]) for i, v in enumerate(rect))


def _ring_plot(draw, info, at, ratio):
    r = info.ring
    x, y, x1, y1 = _scaled((0, 0, r.width_px, r.height_px), at, ratio)
    draw.rectangle((x, y, x1, y1), fill="#f1f3f4", outline=INK)
    for face, rect in r.face_rects.items():
        l, t, rr, b = _scaled([8 * v for v in rect], at, ratio)
        draw.rectangle((l, t, rr, b), fill=COLORS[face], outline=MUTED)
        draw.text(((l + rr) // 2, (t + b) // 2), face, anchor="mm", font=LABEL, fill=INK)
    for face, rect in info.cap_rects_px.items():
        l, t, rr, b = _scaled(rect, at, ratio)
        draw.rectangle((l, t, rr, b), fill=COLORS[face], outline=INK)
        draw.text(((l + rr) // 2, (t + b) // 2), face, anchor="mm", font=LABEL, fill=INK)
        edge = b if NATIVE[face][2] == "bottom" else t
        draw.line((l, edge, rr, edge), fill=INK, width=4)


def _strip_plot(draw, ring, info, face, at, ratio):
    x, y, x1, y1 = _scaled((0, 0, *info.size_px), at, ratio)
    draw.rectangle((x, y, x1, y1), fill="#f1f3f4", outline=INK)
    left, top, right, bottom = info.cap_rect_px
    known_start, known_stop = (v * 8 for v in info.known_rows_latent)
    if face == "T":
        known_y = known_start
    else:
        known_y = known_stop
    for side, (l, _, r, _) in ring.face_rects.items():
        a, b, c, d = _scaled((8 * l, known_start, 8 * r, known_stop), at, ratio)
        draw.rectangle((a, b, c, d), fill=COLORS[side], outline=MUTED)
        draw.text(((a + c) // 2, (b + d) // 2), side, anchor="mm", font=LABEL, fill=INK)
    l, t, r, b = _scaled((left, top, right, bottom), at, ratio)
    draw.rectangle((l, t, r, b), fill=COLORS[face], outline=INK)
    draw.text(((l + r) // 2, (t + b) // 2), face, anchor="mm", font=LABEL, fill=INK)
    edge = _scaled((left, known_y, right, known_y), at, ratio)
    draw.line(edge, fill=INK, width=4)
    col = (left + right) // 2
    mark = col, (known_y - 17 if face == "T" else known_y + 17)
    draw.text(_scaled(mark, at, ratio), "native", anchor="mm", font=SMALL, fill=INK)


def render_registration(topology, destination):
    joint = build_joint_ring_layout(topology)
    r0 = build_ring_layout(topology, 1280, 1024)
    page = Image.new("RGB", (1120, 740), "white")
    draw = ImageDraw.Draw(page)
    draw.text((18, 10), "Native Stage 1 registration", font=TITLE, fill=INK)
    draw.text((18, 46), "J4   joint ring and caps, 1280 x 1536", font=LABEL, fill=INK)
    _ring_plot(draw, joint, (18, 70), 0.39)
    draw.text((18, 680), "Cyclic X seam joins B right to L left", font=SMALL, fill=MUTED)
    draw.text((18, 704), "Gray regions are temporary context; dark strokes are the native folds.", font=SMALL, fill=MUTED)
    for i, face in enumerate(CAPS):
        a = 560
        y = 46 + 326 * i
        label = "top extension" if face == "T" else "bottom extension"
        draw.text((a, y), f"J4b   {label}, 1280 x 640", font=LABEL, fill=INK)
        _strip_plot(draw, r0, build_outpaint_layout(topology, r0, face), face, (a, y + 24), 0.42)
    draw.text((560, 680), "The full colored L/F/R/B row block is frozen R0 context.", font=SMALL, fill=MUTED)
    page.save(destination, optimize=True)


def _field_colors(field):
    low = np.asarray((221, 231, 234))
    high = np.asarray((205, 100, 34))
    a = (field / STAGE2_OPTIONS["edge_strength"])[..., None]
    return np.rint(low * (1 - a) + high * a).astype(np.uint8)


def render_strength(topology, destination):
    ring = build_joint_ring_layout(topology).ring
    stage_one = JointStageOne(ring, Image.new("RGB", (1, 1)), {}, {})
    page = Image.new("RGB", (1112, 511), "white")
    draw = ImageDraw.Draw(page)
    draw.text((18, 12), "Stage 2 editing strength on the latent grid", font=TITLE, fill=INK)
    for i, face in enumerate(CAPS):
        x, y = 18 + 549 * i, 70
        layout = _layout(topology, stage_one, face, 16, 1)
        field = stage2_strength_map(layout, topology["box_mm"], settings=STAGE2_OPTIONS)
        canvas = np.full((*field.shape, 3), 245, dtype=np.uint8)
        canvas[layout.known_mask] = CONTEXT
        l, t, r, b = layout.center_rect
        canvas[t:b, l:r] = _field_colors(field[t:b, l:r])
        image = Image.fromarray(canvas)
        page.paste(image.resize((image.width * 5, image.height * 5), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y - 24), f"{face}  /  native edge: {NATIVE[face][2]}", font=LABEL, fill=INK)
        l, t, r, b = [a * 5 + (x if j % 2 == 0 else y) for j, a in enumerate((l, t, r, b))]
        draw.rectangle((l, t, r, b), outline=MUTED)
        ny = b if NATIVE[face][2] == "bottom" else t
        draw.line((l, ny, r, ny), fill=INK, width=4)
    x = 18
    for color, label in ((CONTEXT, "Frozen ring context"),
                         (tuple(_field_colors(np.asarray(0)).tolist()), "0.00  native collar"),
                         (tuple(_field_colors(np.asarray(0.30)).tolist()), "0.30  interior"),
                         (tuple(_field_colors(np.asarray(0.70)).tolist()), "0.70  non-native cuts")):
        draw.rectangle((x, 448, x + 20, 467), fill=color, outline=MUTED)
        draw.text((x + 27, 449), label, font=LABEL, fill=INK)
        x += 245
    draw.text((18, 481), "First 15 mm are locked. Editing ramps in over the next 3 mm.", font=SMALL, fill=MUTED)
    page.save(destination, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    topology = json.loads((args.geometry / "topology.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    render_registration(topology, args.output / "joint-cap-registration.png")
    render_strength(topology, args.output / "joint-cap-stage2-strength.png")
