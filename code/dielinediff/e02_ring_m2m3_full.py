"""Run the fair B2-M2 control and select and verify one cap on the chosen ring."""

from copy import copy
from functools import partial

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import BENCHMARK_BASELINES, grouped, measure
from .e02_capgeom import _check_generation, trial
from .e02_capfix_ring import METHODS_DIR
from .e02_captexture import DOCS
from .e02_ring_m2m3 import CAPS, SEEDS
from .e02_ring_m2m3_rings import ExpandedLayoutMixin, selected_ring
from .e02_sources import SourceAtlas, sources
from .e02_uniform_ring_eval import SELECTION_POLICY, gate_for_seed, select_candidates, textures, write_results
from .e02_uniform_ring_full import _native, _sample


SUPPLEMENT = "b2_regional_m2"
U1 = "u1"


def _archived(experiment, *, methods, seed):
    ledger = read_json(experiment.previous15 / "selection.json")
    scores = ledger[f"scores_{seed}_sha256"]
    legacy = set(methods) & (set(BENCHMARK_BASELINES) - {"b2_regional"})
    if legacy:
        source = sources(experiment.root, experiment.briefs, trial_only=seed == 42)
        for atlas in source:
            if atlas.seed == seed and atlas.method in legacy:
                run = experiment.original if seed == 42 else experiment.previous15
                candidate = _sample(atlas, run)
                if scores[f"{atlas.method}/{atlas.identifier}"] != file_hash(candidate.score):
                    raise ValueError(f"Task 15 original baseline score changed: {candidate.score}")
                yield candidate
    for method in ("b2_regional", U1):
        if method not in methods:
            continue
        run = experiment.original if seed == 42 else experiment.previous15
        if method == U1:
            run = experiment.previous15 if seed == 42 else experiment.previous15 / "validation"
        for brief in experiment.briefs:
            key = f"{brief['id']}-{seed}"
            atlas = SourceAtlas(method, brief, seed, run / method / f"{key}.png", experiment.geometry)
            candidate = _sample(atlas, run)
            if scores[f"{atlas.method}/{atlas.identifier}"] != file_hash(candidate.score):
                raise ValueError(f"Task 15 score changed: {candidate.score}")
            yield candidate


def _cap_samples(experiment, *, methods, seed=42):
    for method in methods:
        if method not in CAPS:
            raise ValueError(f"Unregistered cap candidate {method}")
        if experiment.ring_name == "r0" and seed == 42 and method == U1:
            yield from _archived(experiment, methods=(U1,), seed=42)
            continue
        for brief in experiment.briefs:
            key = f"{brief['id']}-{seed}"
            atlas = SourceAtlas(method, brief, seed, experiment.output / method / f"{key}.png", experiment.geometry)
            yield _sample(atlas, experiment.output)


def _candidate(experiment, method, device, *, seed, validation_of=None):
    if experiment.ring_name == "r0" and method == U1 and seed == 42:
        _check_generation(experiment, list(experiment.samples(methods=(method,), seed=seed)), seed=seed)
        return
    from .methods.v3_ring_midscale_cap import MidscaleTextureEngine

    class CapOnSelectedRing(ExpandedLayoutMixin, MidscaleTextureEngine):
        pass

    def factory(exp, hardware):
        return CapOnSelectedRing(exp.root / "cache/models", exp.geometry, exp.engine_config,
                                 device=hardware, lut_path=exp.lut)

    paths = ("v3_uniform_cap.py", "v3_uniform_texture.py", "v3_captexture.py", "v3_capgeom_init.py",
             "v3_ring_m3.py", "v3_ring_midscale_cap.py")
    code = {name: file_hash(METHODS_DIR / name) for name in paths}
    trial(experiment, method, experiment.parameters(method), device, seed=seed, source=validation_of,
          implementation={"source_sha256": code, "ring_recipe": experiment.ring_name},
          engine_factory=factory, ring_loader=selected_ring)


def _cap_selection(groups, experiment, selected_path):
    physical = textures(groups)
    review = read_json(selected_path) if selected_path is not None else None
    selected = select_candidates(groups, CAPS, experiment.config["gate"], texture=physical, visual_reviews=review,
                                 incumbent_method=U1)
    return selected, physical


def _publish(experiment, groups, chosen, *, ledger, texture, review_path):
    explicit = review_path is not None
    saved = experiment.output / "visual-review.json"
    if review_path is None:
        review_path = next((path for path in (saved, DOCS / "ring-m2m3-visual-review.json") if path.is_file()), None)
    review = read_json(review_path) if review_path is not None else None
    verdict = write_results(experiment, groups, chosen, visual=review, selection=ledger["cap"],
                            texture_by_seed={42: texture}, results_file="results-ring-m2m3.csv")
    verified = {seed: outcome["visual_review"] for seed, outcome in verdict["seeds"].items()}
    ledger["visual_review_complete"] = all(verified.values())
    if ledger["visual_review_complete"]:
        write_json(saved, verified)
    elif explicit:
        raise ValueError("The supplied original-pixel notes must bind both seeds to their actual sources")
    print({seed: {key: result["passed"] for key, result in outcome["checks"].items()}
           for seed, outcome in verdict["seeds"].items()}, flush=True)


def _phase_diagnostic(experiment, device):
    from pathlib import Path

    from .e02_ring_m2m3_rings import _engine
    from .e02_ring_phase import check_phase

    destination = experiment.output / "m3-phase.json"
    code = Path(__file__).parent
    sources = (code / "e02_ring_phase.py", METHODS_DIR / "v3_ring_roll.py", METHODS_DIR / "v3_ring_m3.py")
    identity = {"inputs_sha256": file_hash(experiment.output / "inputs.json"),
                "code": {item.name: file_hash(item) for item in sources}}
    if destination.is_file():
        cached = read_json(destination)
        if cached["inputs"] == identity:
            return
    engine = _engine(experiment, "m3b", device)
    brief = next(brief for brief in experiment.briefs if brief["id"] == "E01-003")
    proof = check_phase(engine, brief)
    write_json(destination, {"inputs": identity, **proof})
    print(f"M3 downsample-phase epsilon RMSE ratio: {proof['m3_to_reference_phase_rmse_ratio']:.6f}", flush=True)


def run(experiment, device, ring_decision, visual_path=None, selection_review_path=None):
    from .e02_ring_m2_baseline import generate_m2_baseline

    _phase_diagnostic(experiment, device)
    experiment.use_ring(ring_decision["selected"])
    experiment.samples = partial(_cap_samples, experiment)
    print(f"Task 17 selected {experiment.ring_name}; generating the matched B2-regional-M2 control", flush=True)
    supplemental = generate_m2_baseline(experiment, device)
    if any(atlas.method != SUPPLEMENT for atlas in supplemental):
        raise ValueError("The supplemental cohort must be B2-regional-M2")
    extras = [_sample(atlas, experiment.output) for atlas in supplemental]
    measure(experiment, extras, device)
    references = {seed: [*_archived(experiment, methods=BENCHMARK_BASELINES, seed=seed),
                         *(job for job in extras if job.atlas.seed == seed)] for seed in SEEDS}
    if any(len(references[seed]) != 60 for seed in SEEDS):
        raise ValueError("Each seed needs the five original ten-case controls plus B2-regional-M2")
    for method in CAPS:
        _candidate(experiment, method, device, seed=42)

    groups42 = grouped(references[42] + list(experiment.samples(methods=CAPS)))
    cap_selection, physical = _cap_selection(groups42, experiment, selection_review_path)
    policy = {**SELECTION_POLICY, "reference": U1,
              "ranking": ["more_primary_wins_vs_u1", *SELECTION_POLICY["ranking"][1:]]}
    ledger = {"ring": experiment.ring_name, "cap": cap_selection, "selection_policy": policy, "complete": False,
              "candidate_parameters": {method: experiment.parameters(method) for method in CAPS},
              "ring_decision_sha256": file_hash(experiment.output / "ring-selection.json"),
              "scores_42_sha256": {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, entries in groups42.items() for job, _ in entries}}
    path = experiment.output / "selection.json"
    write_json(path, ledger)
    chosen = cap_selection["selected"]
    if cap_selection["requires_tie_visual_review"]:
        folder = experiment.output / "tie-review"
        folder.mkdir(exist_ok=True)
        fingerprints = {}
        for method in cap_selection["stripe_ties"]:
            _native(experiment, groups42[method], method, folder / method)
            fingerprints[method] = gate_for_seed(experiment, groups42, method, seed=42)["fingerprint"]
        write_json(folder / "request.json", fingerprints)
        print(f"Task 17 requires original-pixel comparison at {folder}", flush=True)
        return
    if chosen is None:
        write_results(experiment, {42: groups42}, chosen, selection=cap_selection,
                      texture_by_seed={42: physical}, results_file="results-ring-m2m3.csv")
        print("Task 17: neither candidate passed the registered cap hard constraints", flush=True)
        return
    print(f"Task 17 seed 42 selected {chosen}; verifying on seed 1234", flush=True)
    validation = copy(experiment)
    validation.output = experiment.output / "validation"
    validation.samples = partial(_cap_samples, validation)
    _candidate(validation, chosen, device, seed=1234, validation_of=chosen)
    groups1234 = grouped(references[1234] + list(validation.samples(methods=(chosen,), seed=1234)))
    groups = {42: groups42, 1234: groups1234}
    for seed, scored in groups.items():
        _native(experiment, scored[chosen], chosen, experiment.output / f"seed-{seed}")
    _publish(experiment, groups, chosen, ledger=ledger, texture=physical, review_path=visual_path)
    ledger["scores_1234_sha256"] = {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, entries in groups1234.items() for job, _ in entries}
    from .e02_ring_m2m3_examples import export_examples
    export_examples(experiment, groups, chosen)
    ledger["complete"] = True
    write_json(path, ledger)
