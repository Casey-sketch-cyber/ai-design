from types import SimpleNamespace

import numpy as np
import pytest

from dielinediff.e02 import BENCHMARK_BASELINES, Sample
from dielinediff.e02_captexture_eval import TEXTURE_FIELDS
from dielinediff.e02_sources import SourceAtlas
from dielinediff.e02_uniform_ring_eval import (
    EXTRA_TEXTURE_FIELDS,
    _energy_at_half_size,
    gate_for_seed,
    select_candidates,
)


LIMITS = {"seconds": 60, "peak_memory_mib": 20480, "stripe_factor": 1.2, "clip_margin": 0.05}
SCORE = {
    "non_native_7_delta_e00": 10.0, "wrap_lb_delta_e00": 4.0,
    "non_native_7_sdr": 3.1, "non_native_7_lpips": 0.35, "wrap_lb_lpips": 0.2,
    "stripe_index": 1.3, "clip_score": 0.7, "image_reward": -1.7,
    "bleed_fill_pct": 57.0, "bleed_fill_v2_pct": 52.0, "elapsed_seconds": 22.0,
    "peak_memory_mib": 10000,
}
IMPROVED = {
    "non_native_7_delta_e00": 8.0, "non_native_7_sdr": 2.6, "non_native_7_lpips": 0.3,
    "bleed_fill_pct": 60.0, "bleed_fill_v2_pct": 55.0,
}


def _group(root, method, *, seed=42, updates=None, over_cap=False):
    values, data = [], {}
    for i in range(1, 11):
        brief = {"id": f"E01-{i:03d}", "continuous": i % 2 == 1}
        atlas = SourceAtlas(method, brief, seed, root / f"{method}-{seed}.png", root)
        item = Sample(atlas, root / f"{method}-{seed}.json", root / f"{method}-{seed}-preflight.png")
        score = {**SCORE, **(updates or {})}
        values.append((item, {"raw": score.copy(), "preflight": score}))
        texture = dict.fromkeys((*TEXTURE_FIELDS, *EXTRA_TEXTURE_FIELDS), 0.001)
        if i == 10 and over_cap:
            texture["bottom_interior_texture_energy"] = 0.0021
        data[brief["id"]] = texture
    return values, data


def _fixtures(root, choices):
    groups, texture = {}, {}
    for k, args in choices.items():
        groups[k], texture[k] = _group(root, k, **args)
    return groups, texture


def _review(experiment, groups, method, seed, count=0):
    for _, (job, _) in enumerate(groups[method]):
        for path in (job.atlas.path, job.score, job.preflight):
            path.write_bytes(b"record fingerprint fixture")
    fingerprint = gate_for_seed(experiment, groups, method, seed=seed)["fingerprint"]
    return {"fingerprint": fingerprint, "samples": {
        f"E01-{i:03d}": {"visible_artifact": i <= count, "notes": "Reviewed."}
        for i in range(1, 11)
    }}


def test_per_box_noise_and_reward_guards_can_retain_same_ring_g5(tmp_path):
    groups, texture = _fixtures(tmp_path, {
        "b2_regional": {"updates": {"image_reward": -1.6}},
        "b2_regional_r1": {"updates": {"image_reward": -4.0}},
        "g5": {},
        "u1": {"updates": IMPROVED, "over_cap": True},
        "u2": {"updates": {**IMPROVED, "image_reward": -1.71}},
    })
    result = select_candidates(groups, ("u1", "u2"), LIMITS, texture=texture)
    assert result["selected"] == "g5" and result["eligible"] == ["g5"]
    assert result["methods"]["u1"]["win_count"] == 6
    assert result["methods"]["u1"]["hard_constraints"]["maximum_texture"]["over_limit"] == [
        {"brief_id": "E01-010", "cap": "bottom", "value": 0.0021,
         "maximum": 0.002, "cap_to_ring_ratio": pytest.approx(2.1)}
    ]
    assert result["methods"]["u2"]["hard_constraints"]["image_reward"]["minimum"] == pytest.approx(-1.7)
    assert not result["methods"]["u2"]["hard_constraints"]["image_reward"]["passed"]


def test_v1_precedes_visual_and_exact_ties_require_bound_reviews(tmp_path):
    arms = ("u1", "u2")
    choices = {name: {} for name in (*BENCHMARK_BASELINES, "g5")}
    choices.update(u1={"updates": {**IMPROVED, "stripe_index": 1.2}},
                   u2={"updates": {**IMPROVED, "stripe_index": 1.4}})
    groups, texture = _fixtures(tmp_path, choices)
    experiment = SimpleNamespace(config={"gate": LIMITS})
    reviews = {name: _review(experiment, groups, name, 42, count) for name, count in zip(arms, (1, 0), strict=True)}
    got = select_candidates(groups, arms, LIMITS, texture=texture, visual_reviews=reviews)
    assert got["selected"] == "u1" and got["reason"] == "lowest_frozen_stripe_v1"
    assert not got["requires_tie_visual_review"]

    for _, data in groups["u2"]:
        data["preflight"]["stripe_index"] = 1.2
    got = select_candidates(groups, arms, LIMITS, texture=texture, visual_reviews={"u2": reviews["u2"]})
    assert got["selected"] == "u1" and got["requires_tie_visual_review"]
    assert got["stripe_ties"] == list(arms)
    got = select_candidates(groups, arms, LIMITS, texture=texture, visual_reviews=reviews)
    assert got["selected"] == "u2" and got["reason"] == "reviewed_exact_stripe_tie"
    assert not got["requires_tie_visual_review"]


def test_gate_uses_seed_local_original_controls_and_rejects_mixed_seeds(tmp_path):
    def choices(seed, b4):
        data = {name: {"seed": seed, "updates": {
            "stripe_index": 1.6, "non_native_7_sdr": 4.0, "non_native_7_lpips": 0.4,
            "bleed_fill_pct": 50.0, "bleed_fill_v2_pct": 45.0,
        }} for name in BENCHMARK_BASELINES}
        data["b4"]["updates"]["stripe_index"] = b4
        data["u2"] = {"seed": seed}
        data["b2_regional_r1"] = {"seed": seed, "updates": {"stripe_index": 1.0}}
        return _fixtures(tmp_path, data)[0]

    experiment = SimpleNamespace(config={"gate": LIMITS})
    first, validation = choices(42, 1.1), choices(1234, 1.02)
    review_first = _review(experiment, first, "u2", 42)
    review_validation = _review(experiment, validation, "u2", 1234)
    source = gate_for_seed(experiment, first, "u2", seed=42, visual=review_first)
    follow = gate_for_seed(experiment, validation, "u2", seed=1234, visual=review_validation)
    assert source["passed"] and not follow["passed"]
    assert source["checks"]["fold_stripe"]["threshold"] == pytest.approx(1.32)
    assert follow["checks"]["fold_stripe"]["threshold"] == pytest.approx(1.224)
    assert follow["checks"]["fold_stripe"]["visual_passed"]
    with pytest.raises(ValueError, match="seed 1234"):
        gate_for_seed(experiment, {**validation, "b2": first["b2"]}, "u2", seed=1234)


def test_two_by_two_average_suppresses_one_pixel_checkerboard():
    y, x = np.indices((96, 96))
    checker = np.repeat((255 * ((x + y) % 2)).astype(np.uint8)[..., None], 3, axis=2)
    broad = np.repeat(((x // 4) % 2 * 255).astype(np.uint8)[..., None], 3, axis=2)
    assert _energy_at_half_size(checker) == 0
    assert _energy_at_half_size(broad) > 0


def test_reviewed_cap_content_is_a_hard_constraint_before_e0_ranking(tmp_path):
    groups, texture = _fixtures(tmp_path, {
        "b2_regional": {},
        "e0": {},
        "e1": {"updates": IMPROVED},
    })
    appearance = {
        "e0": {"passed": True, "flagged": [], "reviewed": 10},
        "e1": {"passed": False, "flagged": [{"brief_id": "E01-003", "ghosting": True}], "reviewed": 10},
    }
    result = select_candidates(groups, ("e0", "e1"), LIMITS, texture=texture,
                               incumbent_method="e0", content_guards=appearance)
    assert result["selected"] == "e0"
    assert result["methods"]["e1"]["win_count"] == 6
    assert not result["methods"]["e1"]["eligible"]
    assert result["methods"]["e1"]["hard_constraints"]["cap_content"] == appearance["e1"]
