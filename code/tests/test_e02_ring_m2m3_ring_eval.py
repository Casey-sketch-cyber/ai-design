from types import SimpleNamespace

import numpy as np
import pytest

from dielinediff.e01 import write_json
from dielinediff.e02_ring_m2m3_ring_eval import (
    BRIEFS,
    RULES,
    SEEDS,
    _cached_score,
    _normal_fields,
    decide_ring,
    fold_normal_profile,
    normal_profiles,
    task15,
)
from dielinediff.dieline.geometry import build_geometry, write_geometry


ORDER = ("r0", "m3b", "m3", "m2", "m2_m3")
MARGINS = {
    "wrap_lb_delta_e00_increase": 1.0, "wrap_lb_lpips_increase": 0.02,
    "clip_score_decrease": 0.02, "image_reward_decrease": 0.10,
    "m4_repeat_score_increase": 0.03, "side_bleed_fill_pct_decrease": 3.0,
    "side_bleed_fill_v2_pct_decrease": 3.0,
}


def _rows():
    out = []
    floor = {"r0": 1.3, "m3b": 1.22, "m3": 1.23, "m2": 1.21, "m2_m3": 1.15}
    for method in ORDER:
        for seed in SEEDS:
            for brief in BRIEFS:
                row = {"method": method, "brief_id": brief, "seed": seed, **dict.fromkeys(task15.FIELDS, 1.0)}
                row.update(ring_only_box_floor=floor[method], side_bleed_fill_pct=60.0,
                           side_bleed_fill_v2_pct=60.0, m4_repeat_score=0.5)
                if method == "m2_m3" and seed == 1234:
                    row["side_bleed_fill_v2_pct"] = 56.9
                if method == "m2" and seed == 42:
                    row["m4_repeat_score"] = 0.531
                out.append(row)
    return out


def test_all_guards_apply_independently_to_each_seed_and_rank_paired_floor():
    scores = _rows()
    selected = decide_ring(scores, MARGINS, ORDER)
    assert selected["selected"] == "m3b"
    assert selected["eligible"] == ["m3b", "m3"]
    assert len(RULES) == 7
    assert not selected["methods"]["m2_m3"]["seeds"]["1234"]["quality_guards"]["side_bleed_fill_v2_pct"]["passed"]
    assert not selected["methods"]["m2"]["seeds"]["42"]["quality_guards"]["m4_repeat_score"]["passed"]
    for row in scores:
        if row["method"] in selected["eligible"] and row["seed"] == 1234:
            row["ring_only_box_floor"] = 1.3
    selected = decide_ring(scores, MARGINS, ORDER)
    assert selected["selected"] == "r0"
    assert not selected["methods"]["m3b"]["seeds"]["1234"]["stripe_floor"]["passed"]


def test_normal_fields_preserve_sign_and_scale():
    profile = np.r_[np.full(12, 40), np.full(24, 50)]
    score = _normal_fields(profile, 4)
    assert score == {"mean_Lstar_0_3mm": 40.0, "mean_Lstar_3_9mm": 50.0,
                     "near_minus_reference_Lstar": -10.0, "normal_peak_Lstar_per_mm": 40.0,
                     "normal_peak_inward_mm": 3.0}


def test_static_ring_profiles_keep_cap_absent_not_white(tmp_path):
    write_geometry(build_geometry(pixels_per_mm=4), tmp_path)
    obj = task15.E02PhysicalEvaluator(tmp_path, device="cpu")
    rgb = np.full((*obj.native_shape, 3), 100, np.uint8)
    profile = normal_profiles(obj, rgb, method="r0", seed=42, brief_id="E01-001")
    assert len(profile) == 8
    assert sum(r["across_profile_peak_Lstar_per_mm"] == 0.0 for r in profile) == 6
    assert all(r["across_profile_peak_Lstar_per_mm"] is None
               for r in profile if r["other_face"] in ("T", "Btm"))
    assert all(r["near_minus_reference_Lstar"] == 0.0 for r in profile)

    faces = obj._physical_faces(rgb)
    faces["T"] = np.full_like(faces["T"], 120)
    both = fold_normal_profile(obj, faces)
    assert len(both) == 10 and all(row["across_profile_peak_Lstar_per_mm"] is not None for row in both)
    front_top = [r for r in both if r["fold"] == "F-T"]
    assert len(front_top) == 2 and all(r["across_boundary_jump_Lstar"] > 0 for r in front_top)


def test_score_resume_does_not_load_model_for_matching_case(tmp_path):
    checkpoint = tmp_path / "ring-evaluation" / "r0" / "E01-001-42.json"
    checkpoint.parent.mkdir(parents=True)
    stamp = {"cache_sha256": "test", "method": "r0"}
    saved = {"input": stamp, "score": {}, "folds": [], "normal_profiles": []}
    write_json(checkpoint, saved)
    case = ("r0", 42, {"id": "E01-001"}, None, None, None)
    loaded = _cached_score(SimpleNamespace(output=tmp_path), case, (1280, 1024), stamp, None, {}, [])
    assert loaded == saved
