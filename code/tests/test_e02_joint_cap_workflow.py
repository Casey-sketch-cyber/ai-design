"""Check Task 20 ranking fallbacks and source-bound stage caching."""

from dielinediff.e01 import write_json
from dielinediff.e01b import file_hash
from dielinediff.e02_joint_cap_generate import _files, _stage_cached
from dielinediff.e02_joint_cap_workflow import _parent


def _score(eligible, wins, stripe):
    return {"eligible": eligible, "win_count": wins, "means": {"stripe_index": stripe}}


def test_no_ineligible_candidate_is_promoted():
    value = {"methods": {"j0": _score(False, 0, 1.5), "j1": _score(False, 4, 1.2),
                         "j2": _score(False, 4, 1.3)}, "selected": None, "requires_tie_visual_review": False}
    assert _parent(value, ("j0", "j1"), fallbacks=("j0",)) == ("j0", "registered_control_for_diagnostic_only")
    assert _parent(value, ("j1", "j2")) == ("j1", "no_eligible_parent_diagnostic_only")
    assert value["selected"] is None


def test_eligible_parent_precedes_higher_win_count_of_rejected_method():
    value = {"methods": {"j1": _score(True, 2, 1.5), "j2": _score(False, 6, 1.2)},
             "selected": "j1", "requires_tie_visual_review": False}
    assert _parent(value, ("j1", "j2")) == ("j1", "eligible_parent")
    value["requires_tie_visual_review"] = True
    assert _parent(value, ("j1", "j2")) == (None, "exact_stripe_tie")


def test_stage1_reuse_checks_stored_latent(tmp_path):
    path = tmp_path / "E01-001-42.png"
    latent = path.with_suffix(".ring.npy")
    parts = {"atlas": path, **_files(path), "latent": latent}
    for label, part in parts.items():
        part.write_bytes(label.encode())
    plan = {"recipe": "coarse_047"}
    info = {"plan": plan, "source_sha256": {key: file_hash(p) for key, p in parts.items()}}
    write_json(path.with_suffix(".run.json"), info)
    assert _stage_cached(path, plan) == info
    latent.write_bytes(b"altered")
    assert _stage_cached(path, plan) is None
