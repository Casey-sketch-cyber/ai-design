from types import SimpleNamespace

import pytest

from dielinediff.e02 import BENCHMARK_BASELINES, Sample
from dielinediff.e02_capfix_eval import gate
from dielinediff.e02_captexture_eval import (
    CLIP_MINIMUM,
    ENERGIES,
    TEXTURE_FIELDS,
    _validation_summary,
    select_candidates,
    write_results,
)
from dielinediff.e02_sources import SourceAtlas


GATE = {"stripe_factor": 1.2, "clip_margin": 0.05, "seconds": 60, "peak_memory_mib": 20480}
BASE = {
    "non_native_7_delta_e00": 13.0, "wrap_lb_delta_e00": 4.0,
    "non_native_7_sdr": 4.0,
    "non_native_7_lpips": 0.4, "wrap_lb_lpips": 0.2,
    "stripe_index": 1.5, "clip_score": 0.7, "image_reward": 0.1, "hps_v2": 0.2,
    "bleed_fill_pct": 57.0, "bleed_fill_v2_pct": 52.0,
    "elapsed_seconds": 22.0, "peak_memory_mib": 10000,
}
BETTER = {
    "non_native_7_delta_e00": 11.0, "non_native_7_sdr": 2.0, "non_native_7_lpips": 0.3,
    "bleed_fill_pct": 60.0, "bleed_fill_v2_pct": 55.0,
}


def _group(root, method, *, seed=42, changes=None, exceptional=None, energy=(0.002, 0.002)):
    rows, texture = [], {}
    for n in range(1, 11):
        brief = {"id": f"E01-{n:03d}", "continuous": n % 2 == 1}
        atlas = SourceAtlas(method, brief, seed, root / "raw.png", root)
        job = Sample(atlas, root / "score.json", root / "preflight.png")
        score = {**BASE, **(changes or {})}
        if n == 10 and exceptional is not None:
            score.update(exceptional)
        rows.append((job, {"raw": score.copy(), "preflight": score}))
        texture[brief["id"]] = {**dict.fromkeys(TEXTURE_FIELDS, 0.0004), **dict(zip(ENERGIES, energy, strict=True))}
    return rows, texture


def _inputs(root, specs):
    groups, texture = {}, {}
    for method, definition in specs.items():
        groups[method], texture[method] = _group(root, method, **definition)
    return groups, texture


def test_constraints_apply_per_box_and_independently_to_each_cap(tmp_path):
    groups, texture = _inputs(tmp_path, {
        "b2_regional": {"energy": (0.004, 0.006)},
        "g1": {"energy": (0.004, 0.006)},
        "repeat": {"changes": BETTER, "energy": (0.015, 0.0029)},
        "too_slow": {"changes": BETTER, "energy": (0.007, 0.009),
                     "exceptional": {"elapsed_seconds": 60.01}},
        "at_limit": {"changes": {**BETTER, "clip_score": CLIP_MINIMUM},
                     "exceptional": {"elapsed_seconds": 60.0, "peak_memory_mib": 20480},
                     "energy": (0.002, 0.003)},
    })
    result = select_candidates(groups, ("repeat", "too_slow", "at_limit"), GATE, texture=texture)
    assert result["selected"] == "at_limit"
    assert result["eligible"] == ["at_limit"]
    assert result["texture_minimum"] == dict(zip(ENERGIES, (0.002, 0.003), strict=True))
    wrong_bottom = result["methods"]["repeat"]["hard_constraints"]["texture"]
    assert wrong_bottom["top"]["passed"] and not wrong_bottom["bottom"]["passed"]
    assert result["methods"]["too_slow"]["hard_constraints"]["hardware"]["highest_sample_seconds"] == 60.01
    fallback = select_candidates(groups, ("repeat", "too_slow"), GATE, texture=texture)
    assert fallback["selected"] == "g1"
    assert fallback["reason"] == "fallback_no_new_candidate_passes_hard_constraints"
    assert not fallback["requires_tie_visual_review"]


def test_rank_uses_six_g1_wins_then_frozen_stripe_then_registered_order(tmp_path):
    groups, texture = _inputs(tmp_path, {
        "b2_regional": {}, "g1": {},
        "lowest_stripe_only": {"changes": {"non_native_7_sdr": 2.0, "stripe_index": 1.01}},
        "higher_stripe_six": {"changes": {**BETTER, "stripe_index": 2.0}},
        "tied_first": {"changes": {**BETTER, "stripe_index": 1.7}},
        "tied_second": {"changes": {**BETTER, "stripe_index": 1.7}},
    })
    first = ("lowest_stripe_only", "higher_stripe_six", "tied_first", "tied_second")
    result = select_candidates(groups, first, GATE, texture=texture)
    assert result["selected"] == "tied_first"
    assert result["methods"]["lowest_stripe_only"]["win_count"] == 1
    assert result["methods"]["tied_first"]["win_count"] == 6
    assert result["stripe_ties"] == ["tied_first", "tied_second"]
    assert result["primary_ties"] == ["higher_stripe_six", "tied_first", "tied_second"]
    assert result["requires_tie_visual_review"]
    assert result["reason"] == "awaiting_tie_visual_review"
    result = select_candidates(groups, ("higher_stripe_six", "tied_second"), GATE, texture=texture)
    assert result["selected"] == "tied_second"
    assert result["requires_tie_visual_review"]


def test_seed_validation_pairs_selected_and_archived_g1_separately(tmp_path):
    selected = "g4_f4_70"
    primary, t42 = _inputs(tmp_path, {selected: {"energy": (0.004, 0.003)}})
    validated, t1234 = _inputs(tmp_path, {
        "g1": {"seed": 1234, "energy": (0.002, 0.003)},
        "seed_1234": {"seed": 1234, "energy": (0.005, 0.005)},
    })
    out = _validation_summary(primary, validated, t42, t1234, selected)
    top = ENERGIES[0]
    assert out["selected_source_group"] == "seed_1234"
    assert out["variants"]["preflight"][top]["mean_paired_delta_1234_minus_42"] == pytest.approx(0.001)
    assert out["versus_g1_seed_1234"]["preflight"][top]["mean_paired_difference"] == pytest.approx(0.003)
    with pytest.raises(ValueError, match="chosen configuration and archived G1"):
        _validation_summary(primary, {"seed_1234": validated["seed_1234"]}, t42, t1234, selected)


def test_primary_ties_require_all_bound_native_reviews_before_validation(tmp_path):
    candidates = ("low_stripe", "middle_stripe", "high_stripe")
    specs = {method: {} for method in (*BENCHMARK_BASELINES, "g1")}
    specs.update(low_stripe={"changes": {**BETTER, "stripe_index": 1.4}},
                 middle_stripe={"changes": {**BETTER, "stripe_index": 1.5}},
                 high_stripe={"changes": {**BETTER, "stripe_index": 1.6}})
    groups, texture = _inputs(tmp_path, specs)
    experiment = SimpleNamespace(config={"gate": GATE}, output=tmp_path)
    for sample, _ in groups["low_stripe"]:
        for file in (sample.atlas.path, sample.score, sample.preflight):
            file.write_bytes(b"evidence fingerprint fixture")

    def review(method, count):
        fingerprint = gate(experiment, groups, method)["fingerprint"]
        return {"fingerprint": fingerprint, "samples": {
            f"E01-{n:03d}": {"visible_artifact": n <= count, "notes": "Native fold reviewed."}
            for n in range(1, 11)
        }}

    reviews = {m: review(m, count) for m, count in zip(candidates, (5, 1, 0), strict=True)}
    partial = {m: record for m, record in reviews.items() if m != "middle_stripe"}
    provisional = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=partial)
    assert provisional["selected"] == "low_stripe"
    assert provisional["requires_tie_visual_review"]
    assert provisional["primary_ties"] == list(candidates)
    with pytest.raises(ValueError, match="Resolve all primary ties"):
        write_results(experiment, groups, {"g1": groups["g1"]}, "low_stripe", selection=provisional)

    invalid = {**reviews, "middle_stripe": {**reviews["middle_stripe"], "fingerprint": "stale"}}
    provisional = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=invalid)
    assert provisional["requires_tie_visual_review"] and provisional["selected"] == "low_stripe"
    invalid["middle_stripe"]["samples"] = None
    assert select_candidates(groups, candidates, GATE, texture=texture,
                             visual_reviews=invalid)["requires_tie_visual_review"]

    selected = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=reviews)
    assert selected["selected"] == "high_stripe"
    assert not selected["requires_tie_visual_review"]
    assert selected["reason"] == "reviewed_primary_tie"
    assert selected["native_artifact_counts"] == {"low_stripe": 5, "middle_stripe": 1, "high_stripe": 0}

    reviews["high_stripe"] = review("high_stripe", 1)
    selected = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=reviews)
    assert selected["selected"] == "low_stripe"

    for _, record in groups["middle_stripe"]:
        record["preflight"]["stripe_index"] = 1.4
    selected = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=reviews)
    assert selected["selected"] == "middle_stripe"

    reviews["low_stripe"] = review("low_stripe", 1)
    selected = select_candidates(groups, candidates, GATE, texture=texture, visual_reviews=reviews)
    assert selected["selected"] == "low_stripe"
