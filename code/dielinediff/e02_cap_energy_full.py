"""Select Task 18 caps after the appearance review, then verify on seed 1234."""

from copy import copy
from functools import partial

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import grouped
from .e02_cap_energy import CAPS, sample_caps
from .e02_cap_energy_generate import run_candidate
from .e02_cap_energy_sources import archived_e0, original_baselines
from .e02_cap_energy_visual import export_content, export_examples, verify_content
from .e02_capgeom import _check_generation
from .e02_captexture import DOCS
from .e02_uniform_ring_eval import SELECTION_POLICY, gate_for_seed, select_candidates, textures, write_results
from .e02_uniform_ring_full import _native


RESULTS = "results-cap-energy.csv"


def _cached_review(path, output, local_name, repo_name):
    if path is not None:
        return path
    return next((x for x in (output / local_name, DOCS / repo_name) if x.is_file()), None)


def _save_results(experiment, groups, chosen, decision, physical, final_review_path):
    explicit = final_review_path is not None
    saved = experiment.output / "visual-review.json"
    review_path = _cached_review(final_review_path, experiment.output, saved.name, "cap-energy-visual-review.json")
    visual = read_json(review_path) if review_path else None
    gate = write_results(experiment, groups, chosen, visual=visual, selection=decision,
                         texture_by_seed={42: physical}, results_file=RESULTS)
    accepted = {seed: entry["visual_review"] for seed, entry in gate["seeds"].items()}
    reviewed = bool(accepted) and all(accepted.values())
    if reviewed:
        write_json(saved, accepted)
    elif explicit:
        raise ValueError("The original-pixel gate review must match both current seed fingerprints")
    print({seed: {key: row["passed"] for key, row in value["checks"].items()}
           for seed, value in gate["seeds"].items()}, flush=True)
    return reviewed


def _prepare(experiment, device):
    from .e02_cap_energy_curves import write_candidate_curves

    _check_generation(experiment, list(archived_e0(experiment, 42, alias=False)), seed=42)
    export_content(experiment, "e0")
    for method in CAPS[1:]:
        run_candidate(experiment, method, device)
        export_content(experiment, method)
        write_candidate_curves(experiment, method)
    write_candidate_curves(experiment, "e0")


def _write_tie_request(experiment, groups, selection):
    ties = selection["stripe_ties"]
    folder = experiment.output / "tie-review"
    folder.mkdir(exist_ok=True)
    fingerprints = {}
    for method in ties:
        fingerprints[method] = gate_for_seed(experiment, groups, method, seed=42)["fingerprint"]
        _native(experiment, groups[method], method, folder / method)
    write_json(folder / "request.json", fingerprints)
    print(f"Task 18 needs original-pixel tie review in {folder}", flush=True)


def _report_e0_when_none_qualifies(experiment, first, physical, ledger):
    from .e02_cap_energy_curves import write_candidate_curves

    historical_gate = read_json(experiment.previous17 / "gate.json")
    if historical_gate != read_json(DOCS / "gate-ring-m2m3.json"):
        raise ValueError("The E0 reference gate differs from the reviewed Task 17 record")
    groups = {42: first, 1234: grouped([*original_baselines(experiment, 1234),
                                        *archived_e0(experiment, 1234)])}
    reviews = {}
    for seed in (42, 1234):
        canonical = grouped([*original_baselines(experiment, seed), *archived_e0(experiment, seed, alias=False)])
        outcome = gate_for_seed(experiment, canonical, "u1_sigma28", seed=seed,
                                visual=historical_gate["seeds"][str(seed)]["visual_review"])
        if not outcome["visual_review"]:
            raise ValueError(f"The archived pixel review for E0 is stale at seed {seed}")
        current = gate_for_seed(experiment, groups[seed], "e0", seed=seed)
        reviews[str(seed)] = {**outcome["visual_review"], "fingerprint": current["fingerprint"]}
        _native(experiment, groups[seed]["e0"], "e0", experiment.output / f"seed-{seed}")
    write_candidate_curves(experiment, "e0", seed=1234)
    reference = write_results(experiment, groups, "e0", visual=reviews, texture_by_seed={42: physical},
                              results_file=RESULTS)
    write_json(experiment.output / "visual-review.json", reviews)
    write_json(experiment.output / "gate.json", {"selected": None, "complete": True, "passed": False,
                                               "seeds": {}, "reference_only": reference})
    summary_path = experiment.output / "evaluation-summary.json"
    summary = read_json(summary_path)
    write_json(summary_path, {**summary, "selection": ledger["cap"], "reference_only": "e0"})
    export_examples(experiment, groups, "e0", reference_only=True)
    ledger["scores_1234_sha256"] = {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, values in groups[1234].items() for job, _ in values}
    ledger.update(complete=True, visual_review_complete=True, reference_only="e0")
    write_json(experiment.output / "selection.json", ledger)
    print("Task 18: no eligible cap. Both archived E0 gates are reported as a reference.", flush=True)


def run(experiment, device, stage, cap_review_path=None, final_review_path=None, selection_review_path=None):
    from .e02_cap_energy_curves import write_candidate_curves

    _prepare(experiment, device)
    pending = experiment.output / "cap-content"
    if stage == "candidates":
        print(f"Task 18 candidate art and review requests: {pending}", flush=True)
        return
    appearance_path = _cached_review(cap_review_path, experiment.output, "cap-content-review.json",
                                     "cap-energy-cap-content-review.json")
    if appearance_path is None:
        print(f"Review the original-pixel lids in {pending} and provide --cap-reviews", flush=True)
        return
    guards, raw_reviews = verify_content(experiment, appearance_path)
    write_json(experiment.output / "cap-content-review.json", raw_reviews)
    first = grouped([*original_baselines(experiment, 42), *experiment.samples(methods=CAPS)])
    physical = textures(first)
    tie_reviews = read_json(selection_review_path) if selection_review_path else None
    selected = select_candidates(first, CAPS, experiment.config["gate"], texture=physical,
                                 visual_reviews=tie_reviews, incumbent_method="e0", content_guards=guards)
    ledger = {"ring": "r0", "cap": selected, "complete": False,
              "cap_content_review_sha256": file_hash(experiment.output / "cap-content-review.json"),
              "candidate_parameters": {method: experiment.parameters(method) for method in CAPS},
              "selection_policy": {**SELECTION_POLICY, "reference": "e0",
                                   "ranking": ["more_primary_wins_vs_e0", *SELECTION_POLICY["ranking"][1:]],
                                   "cap_content": "All ten must have no obvious collage, ghosting, or mirrored repetition."},
              "scores_42_sha256": {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, values in first.items() for job, _ in values}}
    write_json(experiment.output / "selection.json", ledger)
    if selected["requires_tie_visual_review"]:
        _write_tie_request(experiment, first, selected)
        return
    chosen = selected["selected"]
    if chosen is None:
        _report_e0_when_none_qualifies(experiment, first, physical, ledger)
        return

    print(f"Task 18 seed 42 selected {chosen}; verifying on seed 1234", flush=True)
    validation = copy(experiment)
    validation.output = experiment.output / "validation"
    validation.is_validation = True
    validation.samples = partial(sample_caps, validation)
    run_candidate(validation, chosen, device, seed=1234, source=chosen)
    last = grouped([*original_baselines(experiment, 1234), *validation.samples(methods=(chosen,), seed=1234)])
    groups = {42: first, 1234: last}
    for seed, scored in groups.items():
        _native(experiment, scored[chosen], chosen, experiment.output / f"seed-{seed}")
    export_content(validation, chosen, seed=1234)
    write_candidate_curves(validation, chosen, seed=1234, output=experiment.output / "cap-energy-curves")
    ledger["visual_review_complete"] = _save_results(experiment, groups, chosen, selected, physical, final_review_path)
    export_examples(experiment, groups, chosen)
    ledger["scores_1234_sha256"] = {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, values in last.items() for job, _ in values}
    ledger["complete"] = True
    write_json(experiment.output / "selection.json", ledger)
