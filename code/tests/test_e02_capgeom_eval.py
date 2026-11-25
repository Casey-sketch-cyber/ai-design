import numpy as np
import pytest

from dielinediff.dieline.geometry import build_geometry, write_geometry
from dielinediff.e02 import Sample
from dielinediff.e02_capgeom_eval import (
    _crease_rows,
    banner_summary,
    compare_step,
    cross_seed,
    long_tail,
    select_strength,
)
from dielinediff.e02_sources import SourceAtlas
from dielinediff.metrics.physical import PhysicalEvaluator, stripe_energy


GATE = {"seconds": 60, "peak_memory_mib": 20480}
SCORE = {
    "non_native_7_delta_e00": 16.0, "wrap_lb_delta_e00": 4.0,
    "non_native_7_sdr": 3.1, "non_native_7_lpips": 0.35, "wrap_lb_lpips": 0.2,
    "stripe_index": 1.0, "clip_score": 0.7, "image_reward": 0.1,
    "bleed_fill_pct": 57.0, "bleed_fill_v2_pct": 52.0, "hps_v2": 0.2,
    "elapsed_seconds": 22.0, "peak_memory_mib": 10000,
}


def _group(root, method, seed=42, patch=None):
    rows = []
    for i in range(1, 11):
        brief = {"id": f"E01-{i:03d}", "continuous": i % 2 == 1}
        atlas = SourceAtlas(method, brief, seed, root / f"{method}-{i}.png", root)
        score = {**SCORE, **(patch(i) if patch is not None else {})}
        sample = Sample(atlas, root / "score.json", root / "preflight.png")
        rows.append((sample, {"preflight": score, "raw": score.copy()}))
    return rows


def test_two_strengths_use_same_parent_and_preserve_task08_tiebreak(tmp_path):
    first = lambda _: {"non_native_7_delta_e00": 12.0}
    groups = {
        "c3": _group(tmp_path, "c3"),
        "g2_075": _group(tmp_path, "g2_075", patch=first),
        "g2_085": _group(tmp_path, "g2_085", patch=first),
    }
    result = select_strength(groups, "c3", "g2_075", "g2_085", GATE)
    assert result["selected"] == "g2_075"
    assert all(row["reference"] == "c3" for row in result["comparisons"].values())
    assert result["head_to_head"]["challenger_net_votes"] == 0

    groups["g2_085"] = _group(tmp_path, "g2_085", patch=lambda _: {
        "non_native_7_delta_e00": 10.0, "non_native_7_lpips": 0.34,
    })
    result = select_strength(groups, "c3", "g2_075", "g2_085", GATE)
    assert result["selected"] == "g2_085"
    assert result["head_to_head"]["selected"] == "g2_085"

    groups["g1"] = _group(tmp_path, "g1", patch=lambda _: {"stripe_index": 0.98})
    result = compare_step(groups, "c3", "g1", GATE, stage="G1")
    assert result["selected"] == "c3"
    assert result["reason"] == "no_cap_improvement"
    assert result["stage"] == "G1"


def test_tail_type_seven_and_cross_seed_are_descriptive(tmp_path):
    seed_42 = {"winner": _group(tmp_path, "winner", patch=lambda i: {
        "non_native_7_delta_e00": (6 * i + 4) / 7, "stripe_index": i / 10 + 1,
    })}
    seed_1234 = {"seed_1234": _group(tmp_path, "seed_1234", seed=1234, patch=lambda i: {
        "non_native_7_delta_e00": (6 * (i + 1) + 4) / 7, "stripe_index": i / 10 + 1.2,
    })}
    tail = long_tail(seed_42)["methods"]["winner"]
    assert tail["cap_6_delta_e00"]["p90"] == pytest.approx(9.1)
    assert tail["stripe_index"]["p90"] == pytest.approx(1.91)
    assert [row["brief_id"] for row in tail["cap_6_delta_e00"]["worst_two"]] == ["E01-010", "E01-009"]
    paired = cross_seed(seed_42, seed_1234, "winner")
    delta = paired["variants"]["preflight"]["cap_6_delta_e00"]
    assert delta["mean_paired_delta_1234_minus_42"] == pytest.approx(1)
    assert paired["seed_1234_tail"]["methods"]["winner"]["cap_6_delta_e00"]["p90"] == pytest.approx(10.1)
    with pytest.raises(ValueError, match="only the frozen"):
        cross_seed(seed_42, {**seed_1234, "competitor": seed_1234["seed_1234"]}, "winner")


def test_fold_contributions_and_nonadditive_per_box_floors(tmp_path):
    write_geometry(build_geometry(pixels_per_mm=4), tmp_path)
    evaluator = PhysicalEvaluator(tmp_path, device="cpu")
    faces = {}
    for name, face in evaluator.topology["faces"].items():
        x0, y0, x1, y1 = face["rect"]
        faces[name] = np.full((y1-y0, x1-x0, 3), 120, dtype=np.uint8)
    faces["Btm"][:12, 10:-10:2] = 220
    faces["F"][:12, 40:-40:2] = 190
    rows = _crease_rows(evaluator, faces)
    actual = max(row["fold_stripe_index"] for row in rows)
    assert len(rows) == 5
    assert actual == stripe_energy(faces, evaluator.creases)[0]
    assert all(row["fold_stripe_index"] == pytest.approx(1 + row["ring_excess"] + row["cap_excess"]) for row in rows)
    by_name = {row["fold"]: row for row in rows}
    assert by_name["F-T"]["ring_excess"] > 0 and by_name["F-T"]["cap_excess"] == 0
    assert by_name["B-Btm"]["cap_excess"] > 0 and by_name["B-Btm"]["ring_excess"] == 0
    ring_floor = max(row["ring_only_fold_floor"] for row in rows)
    cap_floor = max(row["cap_only_fold_floor"] for row in rows)
    assert cap_floor == actual > ring_floor > 1
    assert cap_floor + ring_floor - 1 > actual


def test_banner_groups_are_bound_to_reviewed_atlases(tmp_path):
    group = _group(tmp_path, "choice", patch=lambda i: {"stripe_index": 1 + i / 10, "image_reward": i})
    review = {"reviewer": "Codex", "fingerprint": "exact",
              "samples": {f"E01-{i:03d}": {"stacked_banner_white_line": i < 3, "notes": "looked"}
                          for i in range(1, 11)}}
    stripe = {"cases": {brief: {"ring_only_box_floor": 1.2} for brief in review["samples"]}}
    report = banner_summary(group, stripe, review, "exact")
    assert report["observational"]
    assert report["groups"]["present"]["n"] == 2
    assert report["groups"]["present"]["mean_image_reward"] == 1.5
    assert report["groups"]["absent"]["mean_image_reward"] == 6.5
    assert report["present_minus_absent"]["mean_image_reward"] == -5.0
    with pytest.raises(ValueError, match="current selected atlas"):
        banner_summary(group, stripe, review, "stale")
