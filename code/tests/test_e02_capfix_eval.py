from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from dielinediff.e02 import BENCHMARK_BASELINES, Sample
from dielinediff.e02_capfix_eval import (
    RING_REFERENCE,
    cap_six,
    gate,
    select_addition,
    select_prompts,
    verify_ring,
)
from dielinediff.e02_sources import SourceAtlas


GATE = {"stripe_factor": 1.2, "clip_margin": 0.05, "seconds": 60, "peak_memory_mib": 20480}
COMMON = {
    "non_native_7_delta_e00": 16.0, "wrap_lb_delta_e00": 4.0,
    "non_native_7_sdr": 3.1,
    "non_native_7_lpips": 0.35, "wrap_lb_lpips": 0.2,
    "stripe_index": 1.0, "clip_score": 0.7, "image_reward": 0.1,
    "bleed_fill_pct": 57.0, "bleed_fill_v2_pct": 52.0,
    "elapsed_seconds": 22.0, "peak_memory_mib": 10000,
}


def _group(directory, method, patches=None, tail=None):
    out = []
    for i in range(1, 11):
        brief = {"id": f"E01-{i:03d}", "continuous": i % 2 == 1}
        atlas = SourceAtlas(method, brief, 42, directory / f"{method}-{i}.png", directory)
        sample = Sample(atlas, directory / "score.json", directory / "preflight.png")
        scores = {**COMMON, **(patches or {}), **(tail or {} if i == 10 else {})}
        out.append((sample, {"preflight": scores, "raw": scores.copy()}))
    return out


def test_cap_six_only_excludes_wrap_lb():
    for suffix, six, wrap in (("delta_e00", [3, 8, 9, 10, 11, 13], 5),
                             ("lpips", [0.3, 0.4, 0.1, 0.6, 0.14, 0.21], 0.6)):
        result = cap_six({**COMMON, f"non_native_7_{suffix}": (sum(six) + wrap) / 7,
                          f"wrap_lb_{suffix}": wrap})
        assert result[f"cap_6_{suffix}"] == pytest.approx(sum(six) / 6)


def test_prompt_votes_tie_outlier_and_registered_split(tmp_path):
    groups = {"c1a": _group(tmp_path, "c1a"), "c1b": _group(tmp_path, "c1b", {"elapsed_seconds": 21.8})}
    choice = select_prompts(groups, "c1a", "c1b", GATE)
    assert choice["selected"] == "c1a"
    assert choice["challenger_net_votes"] == 0
    assert "elapsed_seconds" in choice["ties"]

    groups["c1b"] = _group(tmp_path, "c1b", {"non_native_7_delta_e00": 13.5})
    assert select_prompts(groups, "c1a", "c1b", GATE)["selected"] == "c1b"
    groups["c1b"] = _group(tmp_path, "c1b", {"non_native_7_delta_e00": 13.5}, {"elapsed_seconds": 61})
    choice = select_prompts(groups, "c1a", "c1b", GATE)
    assert choice["selected"] == "c1a"
    assert choice["hardware"]["c1b"]["over_limit"] == [
        {"brief_id": "E01-010", "elapsed_seconds": 61.0, "peak_memory_mib": 10000.0},
    ]
    groups["c1b"][-1] = groups["c1b"][0]
    with pytest.raises(ValueError, match="exactly E01-001"):
        select_prompts(groups, "c1a", "c1b", GATE)


def test_addition_requires_target_improvement_and_positive_net(tmp_path):
    groups = {"incumbent": _group(tmp_path, "incumbent")}
    groups["challenger"] = _group(tmp_path, "challenger", {"clip_score": 0.73, "image_reward": 0.2})
    decision = select_addition(groups, "incumbent", "challenger", GATE, stage="C2")
    assert decision["selected"] == "incumbent"
    assert decision["reason"] == "no_cap_improvement"

    groups["challenger"] = _group(tmp_path, "challenger", {"non_native_7_delta_e00": 15.5,
                                                          "non_native_7_sdr": 3.2, "stripe_index": 1.3})
    decision = select_addition(groups, "incumbent", "challenger", GATE, stage="C3")
    assert decision["selected"] == "incumbent"
    assert decision["reason"] == "no_net_improvement"

    groups["challenger"] = _group(tmp_path, "challenger", {"non_native_7_delta_e00": 15.5,
                                                          "non_native_7_lpips": 0.34, "elapsed_seconds": 30})
    decision = select_addition(groups, "incumbent", "challenger", GATE, stage="C4")
    assert decision["selected"] == "challenger"
    assert decision["challenger_net_votes"] == 2


def test_original_five_checks_and_all_box_hardware_gate(tmp_path):
    groups = {name: _group(tmp_path, name, {"non_native_7_sdr": 4.0, "non_native_7_lpips": 0.4,
                                          "bleed_fill_pct": 55.0, "bleed_fill_v2_pct": 50.0})
              for name in BENCHMARK_BASELINES}
    selected = "selected"
    groups[selected] = _group(tmp_path, selected, {"clip_score": 0.8})
    experiment = SimpleNamespace(config={"gate": GATE})
    for sample, _ in groups[selected]:
        for path in (sample.atlas.path, sample.score, sample.preflight):
            path.write_bytes(b"fake for fingerprint")
    initial = gate(experiment, groups, selected)
    assert not initial["passed"] and not initial["checks"]["fold_stripe"]["visual_passed"]
    visual = {"fingerprint": initial["fingerprint"], "samples": {
        job.atlas.brief["id"]: {"visible_artifact": False, "notes": "Native folds reviewed."}
        for job, _ in groups[selected]
    }}
    accepted = gate(experiment, groups, selected, visual)
    assert accepted["passed"]
    assert set(accepted["checks"]) == {"seven_folded_seams", "bleed", "fold_stripe", "clip", "runtime"}
    groups[selected][-1][1]["preflight"]["elapsed_seconds"] = 61
    groups[selected][-1][1]["preflight"]["peak_memory_mib"] = 20500
    rejected = gate(experiment, groups, selected, visual)
    assert not rejected["passed"] and not rejected["checks"]["runtime"]["passed"]
    assert rejected["checks"]["runtime"]["mean_seconds"] < GATE["seconds"]
    assert rejected["checks"]["runtime"]["highest_sample_seconds"] == 61


def test_ring_is_verified_independently_from_native_pixels(tmp_path):
    labels = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    Image.fromarray(labels).save(tmp_path / "region_map.png")
    base = np.full((*labels.shape, 3), 111, dtype=np.uint8)
    sides = np.isin(labels, [1, 2, 3, 4])
    actual_body = sha256(base[sides].tobytes()).hexdigest()
    groups = {name: _group(tmp_path, name) for name in (RING_REFERENCE, "c1a", "c1b")}
    for method, cases in groups.items():
        for sample, _ in cases:
            candidate = base.copy()
            if method != RING_REFERENCE:
                candidate[labels == 5] = 150
            atlas = sample.atlas
            Image.fromarray(candidate).save(atlas.path)
            atlas.metadata.write_text(json.dumps({
                "method": method, "brief_id": atlas.brief["id"], "seed": atlas.seed,
                "raw_sha256": sha256(atlas.path.read_bytes()).hexdigest(),
                "ring": {
                    "cache_sha256": "a" * 64, "latent_sha256": "b" * 64,
                    "ring_rgb_sha256": "c" * 64, "body_rgb_sha256": actual_body,
                    "task07_body_rgb_sha256": actual_body, "matches_task07_body": True,
                },
            }))
    assert set(verify_ring(groups)["methods"]) == {"c1a", "c1b"}
    atlas = groups["c1a"][0][0].atlas
    changed = base.copy()
    changed[labels == 1] = 222
    Image.fromarray(changed).save(atlas.path)
    record = json.loads(atlas.metadata.read_text())
    record["raw_sha256"] = sha256(atlas.path.read_bytes()).hexdigest()
    record["ring"]["body_rgb_sha256"] = record["ring"]["task07_body_rgb_sha256"] = sha256(
        changed[sides].tobytes()
    ).hexdigest()
    atlas.metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Side panels differ"):
        verify_ring(groups)
