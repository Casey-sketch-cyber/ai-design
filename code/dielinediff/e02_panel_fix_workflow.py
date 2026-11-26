"""Orchestrate the Task 22 diagnoses, the four trials, and the frozen gates."""

import shutil

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import grouped
from .e02_cap_energy_curves import write_candidate_curves
from .e02_cap_energy_sources import original_baselines
from .e02_cap_energy_visual import export_content, verify_content
from .e02_captexture import DOCS
from .e02_joint_cap_full import _archived_reviews
from .e02_panel_fix import CONTROL, OLD_RUN, ORIGINAL, RECIPES, SEEDS
from .e02_panel_fix_profiles import write_panel_profiles
from .e02_panel_fix_generate import control_parameters, generate_inpaint, generate_outpaint, generate_stage2
from .e02_uniform_ring_eval import gate_for_seed, select_candidates, write_results
from .e02_uniform_ring_full import _native


def _evidence_path(explicit, output, saved, published):
    if explicit is not None:
        return explicit
    return next((p for p in (output / saved, DOCS / published) if p.is_file()), None)


def _diagnosis(experiment, explicit):
    path = _evidence_path(explicit, experiment.output, "diagnosis.json", "panel-fix-diagnosis.json")
    if path is None:
        raise ValueError("Bind the two required Task 22 diagnostic decisions with --diagnosis")
    data = read_json(path)
    if data["task20_selection_sha256"] != file_hash(experiment.previous20 / "selection.json"):
        raise ValueError("Cannot use a diagnostic for a different original archive")
    if (data["panels"]["assessment"] not in ("semantic", "geometry", "mixed", "unresolved")
            or not data["panels"]["conclusion"].strip() or not data["dark_band"]["conclusion"].strip()):
        raise ValueError("Record the conclusions from both diagnostics before generating candidates")
    if data["dark_band"]["repaired"] and not data["dark_band"]["options"]:
        raise ValueError("Record the verified correction before using the repaired outpaint")
    write_json(experiment.output / "diagnosis.json", data)
    return data


def _archive_curves(experiment, method, seed):
    key = f"{method}-{seed}"
    dest = experiment.output / "cap-energy-curves"
    dest.mkdir(exist_ok=True)
    before = experiment.previous20 / "cap-energy-curves"
    record = read_json(before / f"{key}.json")
    if record != read_json(DOCS / "joint-cap-curves" / f"{key}.json"):
        raise ValueError(f"The reviewed curve archive changed for {key}")
    for file, digest in record["artifact_sha256"].items():
        if file_hash(before / file) != digest:
            raise ValueError(f"Changed curve evidence for {key}: {file}")
        shutil.copy2(before / file, dest / file)
    shutil.copy2(before / f"{key}.json", dest / f"{key}.json")


def _cohort(experiment, names, seed):
    return grouped([*original_baselines(experiment, seed), *experiment.samples(methods=names, seed=seed)])


def _accept_caps(experiment, candidates, supplied):
    path = _evidence_path(supplied, experiment.output, "cap-reviews.json", "panel-fix-cap-review.json")
    ledger = read_json(path) if path else {}
    old = read_json(DOCS / "joint-cap-content-review.json")[OLD_RUN]
    if CONTROL in ledger and ledger[CONTROL] != old:
        raise ValueError("The archived K0 appearance assessment changed")
    source = {CONTROL: old, **ledger}
    lacking = [key for key in candidates if key not in source]
    if lacking:
        print(f"Original-pixel review required for {', '.join(lacking)} in {experiment.output / 'cap-content'}",
              flush=True)
        return None
    guards, matched = verify_content(experiment, {name: source[name] for name in candidates}, methods=candidates)
    write_json(experiment.output / "cap-reviews.json", matched)
    return guards


def _inpaint(experiment, args, name, settings, *, seed):
    generate_inpaint(experiment, name, settings, args.device, seed=seed)
    export_content(experiment, name, seed=seed)
    write_candidate_curves(experiment, name, seed=seed)


def _outpaint(experiment, args, name, repair, *, seed):
    generate_outpaint(experiment, repair, args.device, seed=seed)
    generate_stage2(experiment, name, args.device, seed=seed)
    write_panel_profiles(experiment, name, seed=seed)
    export_content(experiment, name, seed=seed)
    write_candidate_curves(experiment, name, seed=seed)


def _present_gates(experiment, args, first, chosen, decision):
    report_method = chosen or ORIGINAL
    groups = {42: first, 1234: _cohort(experiment, (report_method,), 1234)}
    for seed, entries in groups.items():
        _native(experiment, entries[report_method], report_method, experiment.output / f"seed-{seed}")
    if chosen:
        path = _evidence_path(args.fold_reviews, experiment.output, "fold-reviews.json", "panel-fix-fold-review.json")
        reviews = read_json(path) if path else None
    else:
        reviews = _archived_reviews(experiment, groups)
    actual = write_results(experiment, groups, report_method, visual=reviews, selection=decision if chosen else None,
                           results_file="results-panel-fix.csv")
    bound = {seed: gate["visual_review"] for seed, gate in actual["seeds"].items()}
    complete = len(bound) == 2 and all(bound.values())
    if chosen:
        write_json(experiment.output / "fold-review-request.json",
                   {seed: record["fingerprint"] for seed, record in actual["seeds"].items()})
    else:
        write_json(experiment.output / "gate.json", {"selected": None, "passed": False, "complete": True,
                   "seeds": {}, "reference_only": actual})
    if complete:
        write_json(experiment.output / "fold-reviews.json", bound)
    from .e02_panel_fix_examples import publish_examples
    publish_examples(experiment, groups, report_method, decision)
    return complete, bool(chosen and actual["passed"])


def run(experiment, args):
    if args.action == "diagnose":
        from .e02_panel_fix_diagnostics import run_diagnostics
        run_diagnostics(experiment, args.device)
        return
    diagnosis = _diagnosis(experiment, args.diagnosis)
    if args.action != "outpaint":
        base = control_parameters(experiment)
        params = {name: {**base, "panel_prompt": True,
                         "guidance_gamma": experiment.options["guidance_gamma"][name],
                         "geometry_repair": diagnosis["panels"].get("geometry_repair")}
                  for name in ("k1", "k2")}
        export_content(experiment, CONTROL)
        _archive_curves(experiment, OLD_RUN, 42)
        for name, settings in params.items():
            _inpaint(experiment, args, name, settings, seed=42)
    if args.action == "inpaint":
        return

    repair = diagnosis["dark_band"]["options"] if diagnosis["dark_band"]["repaired"] else None
    if repair:
        for name in ("k3", "k4"):
            _outpaint(experiment, args, name, repair, seed=42)
    else:
        print("The registered J4b repair was inconclusive; no K3 or K4 cohort was started.", flush=True)
    if args.action == "outpaint":
        return

    candidates = RECIPES if repair else RECIPES[:3]
    guards = _accept_caps(experiment, candidates, args.cap_reviews)
    if guards is None:
        return
    first = _cohort(experiment, (*candidates, ORIGINAL), 42)
    ties = read_json(args.tie_reviews) if args.tie_reviews else None
    decision = select_candidates(first, candidates, experiment.config["gate"], incumbent_method=CONTROL,
                                 content_guards=guards, visual_reviews=ties)
    write_json(experiment.output / "selection-preview.json", decision)
    if decision["requires_tie_visual_review"]:
        folder = experiment.output / "tie-review"
        requests = {}
        for name in decision["stripe_ties"]:
            _native(experiment, first[name], name, folder / name)
            requests[name] = gate_for_seed(experiment, first, name, seed=42)["fingerprint"]
        write_json(folder / "request.json", requests)
        print(f"Exact stripe-v1 tie awaiting original-pixel review in {folder}", flush=True)
        return

    chosen = decision["selected"]
    if chosen in params:
        _inpaint(experiment, args, chosen, params[chosen], seed=1234)
    elif chosen in ("k3", "k4"):
        _outpaint(experiment, args, chosen, repair, seed=1234)
    elif chosen is None:
        for seed in SEEDS:
            _archive_curves(experiment, ORIGINAL, seed)
    else:
        raise ValueError("The archived K0 cannot pass its unchanged original-pixel guard")
    complete, passed = _present_gates(experiment, args, first, chosen, decision)
    write_json(experiment.output / "selection.json", {
        "cap": decision, "refinements": params, "outpaint_repair": repair,
        "complete": complete, "passed": passed, "reference_only": ORIGINAL if chosen is None else None})
    if complete:
        print(f"Task 22 locked {chosen or 'no recipe'}; Step 1 passed: {passed}", flush=True)
    else:
        print("Both native fold reviews must be bound to finish the cross-seed gate.", flush=True)
