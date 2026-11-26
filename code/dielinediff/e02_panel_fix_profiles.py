"""Measure the old J4b dark track on the repaired original cap pixels."""

from statistics import mean

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_captexture_eval import _write_csv
from .methods.v3_joint_cap_stage2 import _native_y


def _measure(image, face, depth_mm):
    pixels = np.asarray(image.convert("RGB"))
    luminance = rgb2lab(pixels)[..., 0]
    distance = _native_y(face, pixels.shape[0], float(depth_mm))
    coverage = np.all(pixels < 55, axis=-1).mean(axis=1)
    curve = []
    for band in range(20):
        select = (distance >= band) & (distance < band + 1)
        if not select.any():
            raise ValueError("The raw cap cannot resolve a physical 1 mm band")
        curve.append({"near_mm": band, "far_mm": band + 1, "mean_Lstar": float(luminance[select].mean()),
                      "max_dark_row_fraction": float(coverage[select].max())})
    found = (distance < 9) & (coverage >= 0.8)
    return {"dark_track": bool(found.any()), "dark_track_rows": int(found.sum()),
            "dark_depth_mm": sorted(float(value) for value in distance[found]), "profile": curve}


def write_panel_profiles(exp, method, *, seed):
    geometry = read_json(exp.geometry / "topology.json")
    output = exp.output / "dark-profiles"
    output.mkdir(exist_ok=True)
    cases, rows = {}, []
    depth = geometry["box_mm"]["width"]
    for job in exp.samples(methods=(method,), seed=seed):
        atlas = job.atlas
        proof = read_json(atlas.metadata)
        if file_hash(atlas.path) != proof["raw_sha256"]:
            raise ValueError(f"Incorrect raw repair atlas for {atlas.identifier}")
        start_path = exp.stage1(seed) / f"{atlas.identifier}.png"
        source = read_json(start_path.with_suffix(".run.json"))
        results = {}
        with Image.open(atlas.path) as painted, Image.open(start_path) as before:
            for face in ("T", "Btm"):
                rect = tuple(geometry["faces"][face]["rect"])
                initial, final = (image.crop(rect).convert("RGB") for image in (before, painted))
                native = _native_y(face, initial.height, float(depth))
                pixels, result = np.asarray(initial), np.asarray(final)
                locked = native <= 15.0
                if not np.array_equal(pixels[locked], result[locked]):
                    raise RuntimeError(f"Stage 2 changed the protected native RGB in {atlas.identifier}/{face}")
                raw_stage1 = start_path.with_suffix(f".cap-{face}.png")
                if file_hash(raw_stage1) != source["source_sha256"][face]:
                    raise ValueError(f"Incorrect direct cap crop for {atlas.identifier}/{face}")
                with Image.open(raw_stage1) as direct:
                    stages = {"outpaint_raw": _measure(direct, face, depth),
                              "stage1_atlas": _measure(initial, face, depth), "final_atlas": _measure(final, face, depth)}
                results[face] = {"protected_atlas_pixels": int(locked.sum()) * initial.width, "stages": stages}
                for label, record in stages.items():
                    rows.extend({"brief_id": atlas.brief["id"], "face": face, "stage": label, **row}
                                for row in record["profile"])
        cases[atlas.brief["id"]] = {"raw_sha256": proof["raw_sha256"], "stage1_sha256": source["raw_sha256"],
                                     "caps": results}
    summary = {}
    for face in ("T", "Btm"):
        caps = [item["caps"][face]["stages"]["final_atlas"] for item in cases.values()]
        summary[face] = {"dark_track_count": sum(item["dark_track"] for item in caps),
                         "mean_Lstar_3_6mm": mean(row["mean_Lstar"] for item in caps
                                                  for row in item["profile"][3:6])}
    name = f"{method}-{seed}"
    write_json(output / f"{name}.json", {"cases": cases, "summary": summary,
                                        "definition": {"rgb_each_below": 55, "min_row_fraction": 0.8, "max_depth_mm": 9}})
    _write_csv(output / f"{name}.csv", rows)
