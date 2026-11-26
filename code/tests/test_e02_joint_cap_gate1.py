"""Boundary and provenance checks for Task 20 Stage 1 selection."""

import copy

import pytest

from dielinediff.dieline.geometry import BoxSpec, build_geometry
from dielinediff.e02_joint_cap_gate1 import BRIEFS, SEEDS, _ring_seams, decide_gate1


def _rows():
    return [{"brief_id": brief, "seed": seed,
             "ring_delta_e00": 5.0 if seed == 42 else 7.0,
             "ring_lpips": .2 if seed == 42 else .31,
             "ring_only_box_floor": 1.2 if seed == 42 else 1.3,
             "m4_repeat_score": .88 if seed == 42 else .8,
             "cap_ft_stripe": 1.15, "cap_bbtm_stripe": 1.15,
             "elapsed_seconds": 60.0, "peak_allocated_mib": 20480.0}
            for seed in SEEDS for brief in BRIEFS]


def _banner():
    return {str(seed): {brief: {"stacked_banner_white_line": brief == "E01-001"}
                        for brief in BRIEFS} for seed in SEEDS}


def test_all_literal_boundaries_are_inclusive_and_seed_matched():
    original = _rows()
    candidate = copy.deepcopy(original)
    for row in candidate:
        row["ring_delta_e00"] *= 1.05
        row["ring_lpips"] *= 1.05
        row["m4_repeat_score"] += .02
    assert decide_gate1(original, candidate, _banner(), _banner())["passed"]
    waiting = decide_gate1(original, candidate, _banner())
    assert waiting["objective_passed"] and not waiting["complete"] and not waiting["passed"]
    check = waiting["seeds"]["1234"]["checks"]["ring_delta_e00"]
    assert check["r0"] == 7.0 and check["limit"] == pytest.approx(7.35)


def test_single_cap_outlier_and_banner_regression_are_not_averaged_away():
    original = _rows()
    candidate = copy.deepcopy(original)
    candidate[0]["cap_bbtm_stripe"] += .001
    for row in candidate[1:10]:
        row["cap_bbtm_stripe"] = 1.0
    labels = _banner()
    labels["1234"]["E01-002"]["stacked_banner_white_line"] = True
    decision = decide_gate1(original, candidate, _banner(), labels)
    assert decision["complete"] and not decision["passed"]
    assert decision["seeds"]["42"]["checks"]["cap_bbtm_stripe"]["failed_briefs"] == ["E01-001"]
    assert decision["seeds"]["1234"]["checks"]["banner_count"]["value"] == 2
    assert not decision["seeds"]["1234"]["checks"]["banner_count"]["passed"]


def test_four_side_adjacencies_are_distinct_from_cap_folds_and_missing_cases_fail():
    topology = build_geometry(BoxSpec(), pixels_per_mm=4).topology
    assert set(_ring_seams(topology)) == {"F-L", "F-R", "B-R", "B-L"}
    rows = _rows()
    with pytest.raises(ValueError, match="exactly ten"):
        decide_gate1(rows, rows[:-1], _banner(), _banner())
