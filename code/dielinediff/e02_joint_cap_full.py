"""Execute the joint-generation gates and the locked cross-seed evaluation."""

from .e01 import read_json, write_json
from .e02 import grouped
from .e02_cap_energy_curves import write_candidate_curves
from .e02_cap_energy_sources import archived_e0, original_baselines
from .e02_cap_energy_visual import export_content
from .e02_captexture import DOCS
from .e02_joint_cap import CAP_REFINEMENTS, CONTROL, SEEDS
from .e02_joint_cap_generate import generate_joint_caps, generate_joint_stage1
from .e02_joint_cap_reviews import content_reviews
from .e02_joint_cap_workflow import _parent, _produce_refinement, _record_selection, _refinements, _sample_groups
from .e02_uniform_ring_eval import gate_for_seed, write_results
from .e02_uniform_ring_full import _native


def _find(experiment, explicit, staged, published):
    choices = (explicit, experiment.output / staged, DOCS / published)
    return next((path for path in choices if path is not None and path.is_file()), None)


def _banner_reviews(experiment, args, recipe):
    def load(path):
        value = read_json(path)
        if "seeds" in value:
            return value
        return value.get(recipe)

    for path in (args.banner_reviews, experiment.output / "banner-reviews.json", DOCS / "joint-cap-banner-review.json"):
        if path is not None and path.is_file():
            review = load(path)
            if review is not None:
                bound = experiment.output / f"banner-review-{recipe}.json"
                write_json(bound, review)
                return bound
    return None


def _measure_gate1(experiment, args, recipe):
    from .e02_joint_cap_gate1 import score_gate1

    for seed in SEEDS:
        generate_joint_stage1(experiment, recipe, args.device, seed=seed)
    gate = score_gate1(experiment, recipe, device=args.device, banner_review=_banner_reviews(experiment, args, recipe))
    write_json(experiment.output / f"gate1-{recipe}.json", gate)
    return gate


def _j4_stage1(experiment, args):
    all_gates = {}
    chosen = None
    for recipe in experiment.options["joint_stage1"]["modes"]:
        result = _measure_gate1(experiment, args, recipe)
        all_gates[recipe] = result
        if not result["complete"]:
            write_json(experiment.output / "j4-stage1.json", {"attempted": list(all_gates),
                       "selected": None, "awaiting_review": recipe, "complete": False})
            return None, False
        if result["passed"]:
            chosen = recipe
            break
        if not result.get("retry_for_softness", False):
            break
    write_json(experiment.output / "j4-stage1.json", {"attempted": list(all_gates), "selected": chosen,
               "awaiting_review": None, "complete": True})
    return chosen, True


def _stage2_options(experiment):
    outcome = experiment.output / "selection-before-j3.json"
    if not outcome.is_file():
        return None
    choice = read_json(outcome)
    source = choice["methods"]["j2"]
    original = choice["methods"][CONTROL]
    better = source["eligible"] and (not original["eligible"] or source["win_count"] > 0
             or source["means"]["stripe_index"] < original["means"]["stripe_index"])
    return {**experiment.options["joint_stage2"], "base": better}


def _produce_joint(experiment, args, method, plan, *, seed=42):
    generate_joint_stage1(experiment, plan["recipe"], args.device, seed=seed,
                          source_recipe=plan.get("base_recipe"), guidance=plan.get("guidance"))
    generate_joint_caps(experiment, method, plan["recipe"], plan["options"], args.device, seed=seed,
                        guidance=plan.get("guidance"))
    export_content(experiment, method, seed=seed)
    write_candidate_curves(experiment, method, seed=seed, stage1_source=True)


def _archived_reviews(experiment, groups):
    gate = read_json(DOCS / "gate-ring-m2m3.json")
    if read_json(experiment.previous17 / "gate.json") != gate:
        raise ValueError("The reviewed Task 17 reference gate changed")
    accepted = {}
    for seed in SEEDS:
        canonical = grouped([*original_baselines(experiment, seed), *archived_e0(experiment, seed, alias=False)])
        old = gate_for_seed(experiment, canonical, "u1_sigma28", seed=seed,
                            visual=gate["seeds"][str(seed)]["visual_review"])
        if not old["visual_review"]:
            raise ValueError(f"E0 evidence no longer matches the archive at {seed}")
        current = gate_for_seed(experiment, groups[seed], CONTROL, seed=seed)
        accepted[str(seed)] = {**old["visual_review"], "fingerprint": current["fingerprint"]}
    return accepted


def _read_fold_reviews(experiment, args):
    path = _find(experiment, args.fold_reviews, "fold-reviews.json", "joint-cap-fold-review.json")
    return read_json(path) if path else None


def _deliver(experiment, args, refinements, joint, decision):
    recipes = (*refinements, *joint)
    guards, content = content_reviews(experiment, recipes, args.cap_reviews)
    if guards is None:
        return
    write_json(experiment.output / "cap-reviews.json", content)
    if decision["requires_tie_visual_review"]:
        print("Resolve the exact original-pixel stripe tie before locking a recipe.")
        return
    chosen = decision["selected"]
    visual_choice = chosen or CONTROL
    if chosen in joint:
        _produce_joint(experiment, args, chosen, joint[chosen], seed=1234)
    elif chosen is not None:
        _produce_refinement(experiment, args, chosen, refinements[chosen], seed=1234)
        write_candidate_curves(experiment, chosen, seed=1234)
    else:
        write_candidate_curves(experiment, CONTROL, seed=1234)

    first = _sample_groups(experiment, recipes)
    last = _sample_groups(experiment, (visual_choice,), seed=1234)
    groups = {42: first, 1234: last}
    for seed in SEEDS:
        _native(experiment, groups[seed][visual_choice], visual_choice, experiment.output / f"seed-{seed}")
    reviews = _read_fold_reviews(experiment, args) if chosen else _archived_reviews(experiment, groups)
    gate = write_results(experiment, groups, visual_choice, visual=reviews,
                         selection=decision if chosen is not None else None, results_file="results-joint-cap.csv")
    bound = {seed: details.get("visual_review") for seed, details in gate["seeds"].items()}
    if chosen is None:
        review_reference = {"selected": None, "complete": True, "passed": False, "seeds": {},
                            "reference_only": gate}
        write_json(experiment.output / "gate.json", review_reference)
    else:
        requests = {seed: details["fingerprint"] for seed, details in gate["seeds"].items()}
        write_json(experiment.output / "fold-review-request.json", requests)
    complete = bool(bound) and all(bound.values())
    if complete:
        write_json(experiment.output / "fold-reviews.json", bound)
    write_json(experiment.output / "selection.json", {"cap": decision, "refinements": refinements, "joint": joint,
               "complete": complete, "reference_only": CONTROL if chosen is None else None})
    from .e02_joint_cap_examples import publish_examples
    publish_examples(experiment, groups, visual_choice, decision)
    if complete:
        from .e02_joint_cap_rectangle import diagnose_rectangle

        diagnose_rectangle(experiment.root, experiment.output / "diagnosis")
    if not complete:
        print("Review the original-pixel five-fold requests on both seeds.")
    else:
        print(f"Task 20 locked {chosen if chosen is not None else 'no recipe'}; Step 1 passed: "
              f"{chosen is not None and gate['passed']}")


def _guidance_for_j5(experiment, outcomes):
    variants = ("j3_g020", "j3_g030")
    chosen, _ = _parent(outcomes, variants)
    if chosen is None:
        return None
    record = read_json(experiment.output / "parent-j3.json")
    parent = record["parent"]
    curve_dir = experiment.output / "cap-energy-curves"

    def near_reference(method):
        data = read_json(curve_dir / f"{method}-42.json")["mean_curve_summary"]
        return sum(data[fold]["final"]["energy_9_12mm"] for fold in ("F-T", "B-Btm")) / 2

    def stripe(method):
        return outcomes["methods"][method]["means"]["stripe_index"]

    directional = stripe(chosen) >= stripe(parent) or near_reference(chosen) <= near_reference(parent)
    return {**experiment.options["guidance"], "gamma": record["parameters"][chosen]["guidance_gamma"],
            "directional": directional, "reference": chosen}


def run_joint(experiment, args):
    if args.action == "j4":
        recipe, ready = _j4_stage1(experiment, args)
        options = _stage2_options(experiment)
        if ready and recipe and options:
            _produce_joint(experiment, args, "j4", {"recipe": recipe, "options": options})
        return
    if args.action == "j4b":
        generate_joint_stage1(experiment, "j4b", args.device)
        options = _stage2_options(experiment)
        if options:
            _produce_joint(experiment, args, "j4b", {"recipe": "j4b", "options": options})
        return

    refinements = _refinements(experiment, args, "j3")
    if refinements is None:
        return
    write_candidate_curves(experiment, CONTROL)
    options = _stage2_options(experiment)
    if options is None:
        raise ValueError("Missing registered J2-to-J0 comparison")
    generate_joint_stage1(experiment, "j4b", args.device)
    joint = {"j4b": {"recipe": "j4b", "options": options}}
    _produce_joint(experiment, args, "j4b", joint["j4b"])
    stage1, ready = _j4_stage1(experiment, args)
    if not ready:
        return
    if stage1:
        joint["j4"] = {"recipe": stage1, "options": options}
        _produce_joint(experiment, args, "j4", joint["j4"])
    recipes = (*refinements, *joint)
    outcome = _record_selection(experiment, args, recipes, "before-j5")
    if outcome is None:
        return
    preferred, reason = _parent(outcome, tuple(joint))
    if preferred is None:
        return
    seed42 = gate_for_seed(experiment, _sample_groups(experiment, (preferred,)), preferred, seed=42)
    threshold = seed42["checks"]["fold_stripe"]["threshold"]
    measured = outcome["methods"][preferred]["means"]["stripe_index"]
    plan = None
    if measured > threshold:
        guidance = _guidance_for_j5(experiment, outcome)
        if guidance is None:
            return
        inherited = joint[preferred]
        plan = {**inherited, "base_recipe": inherited["recipe"], "recipe": f"j5-{inherited['recipe']}",
                "guidance": guidance}
        _produce_joint(experiment, args, "j5", plan)
        joint["j5"] = plan
    write_json(experiment.output / "j5-plan.json", {"parent": preferred, "reason": reason, "value": measured,
                                                    "threshold": threshold, "plan": plan})
    if args.action == "j5":
        return
    decision = _record_selection(experiment, args, (*refinements, *joint), "final")
    if decision is not None:
        _deliver(experiment, args, refinements, joint, decision)
