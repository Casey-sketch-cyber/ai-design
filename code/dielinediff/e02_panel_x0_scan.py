"""Measure native panel-boundary profiles in the registered DDIM replays."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import deltaE_ciede2000, rgb2lab

from .e01 import read_json, write_json
from .e01b import file_hash


# Coordinates lie in the full 2x input canvas, after lifting the registered
# observed corners from the printed cap. Exclude corner neighborhoods so that
# the strips measure the specified single straight edge.
SEGMENTS = {
    "j0_009_btm_top": ("j0", "E01-009", "y", 356, 620, 1013),
    "j0_009_btm_right": ("j0", "E01-009", "x", 1025, 380, 589),
    "j3_004_btm_left": ("j3_g030", "E01-004", "x", 388, 318, 456),
}


def load_rgb(path):
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"))


def profile(lab, axis, coordinate, tangent_start, tangent_stop, width=4):
    """Compare adjacent native strips and their tangential color consistency."""
    normal = lab if axis == "y" else lab.swapaxes(0, 1)
    tangent = slice(tangent_start, tangent_stop)
    before = normal[coordinate - width:coordinate, tangent].mean(axis=0)
    after = normal[coordinate:coordinate + width, tangent].mean(axis=0)
    signed = after[:, 0] - before[:, 0]
    distance = deltaE_ciede2000(before, after)
    return {
        "delta_l_mean": float(signed.mean()), "delta_l_median": float(np.median(signed)),
        "delta_e_mean": float(distance.mean()), "delta_e_median": float(np.median(distance)),
        "delta_e_p90": float(np.quantile(distance, 0.9)),
        "same_sign_fraction": float(np.mean(np.sign(signed) == np.sign(signed.mean()))),
    }


def _source_frames(diagnostic, method, brief):
    base = diagnostic / method / brief
    replay = read_json(base / "replay.json")
    if not replay["archived_atlas_rgb_exact"]:
        raise ValueError("Panel profiles require byte-exact archival replay")
    folder = base / "Btm"
    registration = read_json(folder / "trace.json")
    initial = folder / "initial.npy"
    final = folder / "final-decode.png"
    if (file_hash(initial) != registration["initial_npy_sha256"]
            or file_hash(final) != registration["final_decode_sha256"]):
        raise ValueError(f"The initial or final trace changed: {brief}")
    rgb = np.load(initial, allow_pickle=False)
    if rgb.shape != (registration["size_px"][1], registration["size_px"][0], 3):
        raise ValueError(f"The traced RGB prior has no valid shape: {brief}")
    yield "initial", None, None, rgb
    for evidence in registration["frames"]:
        path = folder / evidence["file"]
        if file_hash(path) != evidence["sha256"]:
            raise ValueError(f"DDIM frame changed after replay: {path}")
        yield "x0", evidence["step"], evidence["timestep"], load_rgb(path)
    yield "final", None, None, load_rgb(final)


def _tile(rgb, axis, coordinate, begin, end):
    if axis == "y":
        center = (begin + end) // 2
        return (Image.fromarray(rgb).crop((center - 72, coordinate - 43, center + 72, coordinate + 43)),
                43)
    center = (begin + end) // 2
    cropped = Image.fromarray(rgb).crop((coordinate - 43, center - 72, coordinate + 43, center + 72))
    return cropped.transpose(Image.Transpose.ROTATE_270), 43


def scan(diagnostic, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, strips, sources = [], {}, {}
    for key, (method, brief, axis, coordinate, tangent_start, tangent_stop) in SEGMENTS.items():
        label = f"{method}/{brief}/Btm"
        sources[label] = file_hash(diagnostic / method / brief / "replay.json")
        strips[key] = []
        for stage, step, timestep, rgb in _source_frames(diagnostic, method, brief):
            normal_tangent = rgb.shape[0:2] if axis == "y" else rgb.shape[0:2][::-1]
            if not (4 <= coordinate < normal_tangent[0] - 4 and
                    0 <= tangent_start < tangent_stop <= normal_tangent[1]):
                raise ValueError(f"Panel profile outside the native 2x canvas: {key}")
            normal = slice(coordinate - 4, coordinate + 4)
            tangent = slice(tangent_start, tangent_stop)
            patch = rgb[normal, tangent] if axis == "y" else rgb[tangent, normal]
            lab = rgb2lab(patch)
            row = {
                "segment": key, "method": method, "brief_id": brief, "face": "Btm", "stage": stage,
                "step": step, "timestep": timestep, "normal_axis": axis, "normal": coordinate,
                "tangent_start": tangent_start, "tangent_stop": tangent_stop, "strip_width_px": 4,
                **profile(lab, axis, 4, 0, tangent_stop - tangent_start),
            }
            rows.append(row)
            if stage in ("initial", "final") or step in (6, 12, 15, 18, 24, 30):
                pixels = (rgb.clip(0, 1) * 255).round().astype(np.uint8) if rgb.dtype.kind == "f" else rgb
                tile, marker = _tile(pixels, axis, coordinate, tangent_start, tangent_stop)
                strips[key].append((stage if step is None else f"x0 / {step}", row["delta_l_mean"], tile, marker))

    csv_path = output / "panel-x0-profiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure = Image.new("RGB", (8 * 154 + 12, 3 * 142 + 10), "white")
    draw = ImageDraw.Draw(figure)
    regular, title = ImageFont.load_default(size=14), ImageFont.load_default(size=16)
    for n, (key, sequence) in enumerate(strips.items()):
        row = 7 + n * 142
        draw.text((5, row), key.replace("_", " "), font=title, fill="#20354d")
        for col, (label, luminance, tile, marker) in enumerate(sequence):
            x = 5 + col * 154
            top = row + 45
            figure.paste(tile, (x + 3, top))
            draw.text((x + 3, row + 25), f"{label}   dL* {luminance:+.2f}", font=regular, fill="#20354d")
            draw.line((x, top + marker, x + 2, top + marker), fill="#cc4218")
            draw.line((x + 147, top + marker, x + 153, top + marker), fill="#cc4218")
    figure_path = output / "panel-x0-contact.png"
    figure.save(figure_path)
    proof = {"replays": sources, "profiles_sha256": file_hash(csv_path),
             "contact_sha256": file_hash(figure_path), "rows": len(rows),
             "vertical_edge_tiles_rotated_90_degrees": True}
    write_json(output / "panel-x0-proof.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(scan(args.trace, args.output), flush=True)


if __name__ == "__main__":
    main()
