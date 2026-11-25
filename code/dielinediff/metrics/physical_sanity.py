"""Check the preregistered crease statistic on synthetic control images."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import FACES
from .physical import PhysicalEvaluator


def run_controls(geometry_dir, atlas):
    evaluator = PhysicalEvaluator(geometry_dir)
    native = evaluator._native_atlas(atlas)
    flat = np.full_like(native, [152, 158, 162])
    gradual = flat.copy()
    design_border = flat.copy()
    for name in FACES:
        x0, y0, x1, y1 = evaluator.native_topology["faces"][name]["rect"]
        ramp = np.rint(np.linspace(106, 173, x1 - x0)).astype(np.uint8)
        gradual[y0:y1, x0:x1] = ramp[None, :, None]
        middle = (x0 + x1) // 2
        design_border[y0:y1, x0:middle] = (98, 149, 125)
        design_border[y0:y1, middle:x1] = (208, 121, 81)
    damaged = native.copy()
    x0, y0, x1, _ = evaluator.native_topology["faces"]["Btm"]["rect"]
    height = round(3 * evaluator.native_topology["image"]["pixels_per_mm"])
    on = (np.arange(x1 - x0) // 2 % 2) == 0
    damaged[y0:y0 + height, x0:x1] = np.where(
        on[:, None], [237, 49, 192], [35, 201, 89]
    ).astype(np.uint8)
    scores = {}
    for name, image in (
        ("flat", flat), ("gradual", gradual), ("natural_boundary", design_border),
        ("selfcheck", native), ("selfcheck_with_stripe", damaged),
    ):
        result = evaluator.measure(image, lpips=False)
        scores[name] = {key: result[key] for key in ("stripe_index", "back_bottom_stripe_index")}
    clean = scores["selfcheck"]["back_bottom_stripe_index"]
    corrupted = scores["selfcheck_with_stripe"]["back_bottom_stripe_index"]
    checks = {
        "flat_does_not_trigger": scores["flat"]["stripe_index"] == 1,
        "gradual_does_not_trigger": scores["gradual"]["stripe_index"] < 1.05,
        "natural_boundary_does_not_trigger": scores["natural_boundary"]["stripe_index"] < 1.05,
        "injected_stripe_increases_index": corrupted > 1.5 * clean,
    }
    return {"controls": scores, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_controls(args.geometry, args.atlas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return not all(result["checks"].values())


if __name__ == "__main__":
    raise SystemExit(main())
