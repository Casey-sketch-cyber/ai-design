"""Regression for Step 1's single-sided CLIP quality check."""

from types import SimpleNamespace

from dielinediff.e02 import BENCHMARK_BASELINES, gate


def test_high_clip_passes_equivalence_and_visual_is_bound(tmp_path):
    def by_method(method, clip):
        group = []
        for i in range(10):
            photo = tmp_path / f"{method}-{i}.png"
            photo.write_bytes(f"{method}-{i}".encode())
            v3 = method == "v3_1280"
            scores = {"non_native_7_sdr": 2 if v3 else 4,
                      "non_native_7_lpips": 0.2 if v3 else 0.4,
                      "bleed_fill_pct": 80 if v3 else 60,
                      "bleed_fill_v2_pct": 80 if v3 else 60,
                      "stripe_index": 1, "clip_score": clip, "elapsed_seconds": 48,
                      "peak_memory_mib": 10240}
            job = SimpleNamespace(atlas=SimpleNamespace(brief={"id": str(i)}, path=photo),
                                  score=photo, preflight=photo)
            group.append((job, {"preflight": scores}))
        return group

    experiment = SimpleNamespace(config={"gate": {"stripe_factor": 1.2, "clip_margin": 0.05,
                                                  "seconds": 60, "peak_memory_mib": 20480}})
    method = "v3_1280"
    groups = {name: by_method(name, 0.68) for name in BENCHMARK_BASELINES}
    groups[method] = by_method(method, 0.75)
    result = gate(experiment, groups, method, visual=None)
    assert result["checks"]["clip"]["passed"]
    assert not result["passed"]
    review = {"fingerprint": result["fingerprint"], "samples": {
        str(i): {"visible_artifact": False, "notes": "Clear at native scale"} for i in range(10)
    }}
    assert gate(experiment, groups, method, review)["passed"]
    groups[method][0][0].atlas.path.write_bytes(b"changed")
    assert not gate(experiment, groups, method, review)["passed"]
