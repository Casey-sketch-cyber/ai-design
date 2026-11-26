"""Run the registered candidates and leave review evidence at each decision point."""

from .e01 import read_json, write_json
from .e02 import grouped
from .e02_cap_energy_sources import archived_e0, original_baselines
from .e02_cap_energy_visual import export_content
from .e02_capgeom import _check_generation
from .e02_cap_energy_curves import write_candidate_curves
from .e02_joint_cap import CAP_REFINEMENTS, CONTROL
from .e02_joint_cap_generate import control_parameters, generate_refinement
from .e02_joint_cap_reviews import content_reviews
from .e02_uniform_ring_eval import select_candidates


def registered_parameters(experiment):
    original = {**control_parameters(experiment), "clean_prompt": False, "cap_backend": "inpaint",
                "guidance_gamma": 0.0}
    return original, {**original, "clean_prompt": True}


def _sample_groups(experiment, methods, seed=42):
    return grouped([*original_baselines(experiment, seed), *experiment.samples(methods=methods, seed=seed)])


def _record_selection(experiment, args, recipes, label):
    guards, _ = content_reviews(experiment, recipes, args.cap_reviews)
    if guards is None:
        return None
    groups = _sample_groups(experiment, recipes)
    visual = read_json(args.tie_reviews) if args.tie_reviews else None
    outcomes = select_candidates(groups, recipes, experiment.config["gate"], incumbent_method=CONTROL,
                                 content_guards=guards, visual_reviews=visual)
    write_json(experiment.output / f"selection-{label}.json", outcomes)
    if outcomes["requires_tie_visual_review"]:
        from .e02_uniform_ring_eval import gate_for_seed
        from .e02_uniform_ring_full import _native

        destination = experiment.output / "tie-review" / label
        fingerprints = {}
        for method in outcomes["stripe_ties"]:
            _native(experiment, groups[method], method, destination / method)
            fingerprints[method] = gate_for_seed(experiment, groups, method, seed=42)["fingerprint"]
        write_json(destination / "request.json", fingerprints)
        print(f"Original-pixel exact tie review required in {destination}", flush=True)
    return outcomes


def _produce_refinement(experiment, args, method, parameters, *, seed=42):
    generate_refinement(experiment, method, parameters, args.device, seed=seed)
    export_content(experiment, method, seed=seed)
    if seed == 42:
        write_candidate_curves(experiment, method)


def _parent(outcomes, allowed, fallbacks=None):
    if outcomes["requires_tie_visual_review"]:
        return None, "exact_stripe_tie"
    scores = outcomes["methods"]
    eligible = [name for name in allowed if scores[name]["eligible"]]
    if eligible:
        if outcomes["selected"] in eligible:
            return outcomes["selected"], "eligible_parent"
        return min(eligible, key=lambda name: (-scores[name]["win_count"], scores[name]["means"]["stripe_index"],
                                               allowed.index(name))), "eligible_parent"
    if fallbacks:
        return fallbacks[0], "registered_control_for_diagnostic_only"
    return min(allowed, key=lambda name: (-scores[name]["win_count"], scores[name]["means"]["stripe_index"],
                                          allowed.index(name))), "no_eligible_parent_diagnostic_only"


def _refinements(experiment, args, through):
    original, first = registered_parameters(experiment)
    params = {CONTROL: original, "j1": first}
    _check_generation(experiment, list(archived_e0(experiment, 42, alias=False)), seed=42)
    export_content(experiment, CONTROL)
    _produce_refinement(experiment, args, "j1", first)
    if through == "j1":
        return params

    observed = _record_selection(experiment, args, (CONTROL, "j1"), "before-j2")
    if observed is None:
        return None
    parent, reason = _parent(observed, (CONTROL, "j1"), fallbacks=(CONTROL,))
    if parent is None:
        return None
    params["j2"] = {**params[parent], "inpaint": False, "cap_backend": "base_blend"}
    write_json(experiment.output / "parent-j2.json", {"parent": parent, "reason": reason, "parameters": params["j2"]})
    _produce_refinement(experiment, args, "j2", params["j2"])
    if through == "j2":
        return params

    observed = _record_selection(experiment, args, (CONTROL, "j1", "j2"), "before-j3")
    if observed is None:
        return None
    parent, reason = _parent(observed, ("j1", "j2"))
    if parent is None:
        return None
    for method, gamma in zip(CAP_REFINEMENTS[2:], experiment.options["guidance"]["gamma"], strict=True):
        params[method] = {**params[parent], "guidance_gamma": gamma}
        _produce_refinement(experiment, args, method, params[method])
    write_json(experiment.output / "parent-j3.json", {"parent": parent, "reason": reason,
               "parameters": {method: params[method] for method in CAP_REFINEMENTS[2:]}})
    return params


def run(experiment, args):
    if args.action in ("j1", "j2", "j3"):
        _refinements(experiment, args, args.action)
        return
    from .e02_joint_cap_full import run_joint
    run_joint(experiment, args)
