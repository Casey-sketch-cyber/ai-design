"""Generate the matched baselines, select on seed 42, and verify on seed 1234."""

from copy import copy
from functools import partial
from types import SimpleNamespace

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import BENCHMARK_BASELINES, Sample, grouped, measure
from .e02_capgeom import _check_generation, ring, trial
from .e02_capfix_ring import METHODS_DIR
from .e02_captexture import DOCS
from .e02_sources import SourceAtlas, sources
from .e02_uniform_ring import CAPS
from .e02_uniform_ring_baseline import baseline_atlases, generate_baselines
from .e02_uniform_ring_eval import SELECTION_POLICY, gate_for_seed, select_candidates, textures, write_results
from .e02_uniform_ring_rings import r1_ring
from .e02_visual import render_native_creases


def _sample(atlas, run):
    evaluation = run / "evaluation" / atlas.method
    return Sample(atlas, evaluation / f"{atlas.identifier}.metrics.json", evaluation / f"{atlas.identifier}.preflight.png")


def _samples(experiment, *, methods, seed=42):
    reference = read_json(experiment.previous / "selection.json")["scores_42_sha256"] if seed == 42 else {}
    wanted = {brief["id"] for brief in experiment.briefs}
    legacy = set(methods) & (set(BENCHMARK_BASELINES) - {"b2_regional"})
    if legacy:
        for atlas in sources(experiment.root, experiment.all_briefs, trial_only=seed == 42):
            if atlas.seed == seed and atlas.method in legacy and atlas.brief["id"] in wanted:
                sample = _sample(atlas, experiment.original if seed == 42 else experiment.output)
                if seed == 42 and file_hash(sample.score) != reference[f"{atlas.method}/{atlas.identifier}"]:
                    raise ValueError(f"Archived baseline score changed: {sample.score}")
                yield sample
    if {"b2_regional", "b2_regional_r1"}.intersection(methods):
        for atlas in baseline_atlases(experiment):
            if atlas.seed == seed and atlas.method in methods:
                archived = seed == 42 and atlas.method == "b2_regional"
                sample = _sample(atlas, experiment.original if archived else experiment.output)
                if archived and file_hash(sample.score) != reference[f"{atlas.method}/{atlas.identifier}"]:
                    raise ValueError(f"Archived B2-regional score changed: {sample.score}")
                yield sample
    for method in methods:
        if method not in CAPS:
            continue
        archived = method == "g5" and seed == 42 and experiment.ring_name == "r0"
        run = experiment.previous if archived else experiment.output
        for brief in experiment.briefs:
            key = f"{brief['id']}-{seed}"
            atlas = SourceAtlas(method, brief, seed, run / method / f"{key}.png", experiment.geometry)
            sample = _sample(atlas, run)
            if archived and file_hash(sample.score) != reference[f"g5/{key}"]:
                raise ValueError(f"Archived G5 score changed: {sample.score}")
            yield sample


def _candidate(experiment, name, device, *, seed, source=None):
    from .methods.v3_uniform_cap import UniformTextureEngine
    from .methods.v3_uniform_texture import UNIFORM_TEXTURE_PARAMETERS

    if experiment.config["uniform_texture"] != UNIFORM_TEXTURE_PARAMETERS:
        raise ValueError("The registered uniform-texture configuration differs from the implementation")
    if name == "g5" and seed == 42 and experiment.ring_name == "r0":
        _check_generation(experiment, list(experiment.samples(methods=(name,), seed=seed)), seed=seed)
        return

    def engine_factory(exp, hardware):
        return UniformTextureEngine(exp.root / "cache/models", exp.geometry, exp.engine_config,
                                    device=hardware, lut_path=exp.lut)

    files = ("v3_captexture.py", "v3_capgeom_init.py")
    files += (("v3_captexture_inject.py",) if name == "g5" else ("v3_uniform_cap.py", "v3_uniform_texture.py"))
    implementation = {file: file_hash(METHODS_DIR / file) for file in files}
    trial(experiment, name, experiment.parameters(name), device, seed=seed, source=source,
          implementation=implementation, engine_factory=engine_factory,
          ring_loader=r1_ring if experiment.ring_name == "r1" else ring)


def _native(experiment, entries, chosen, destination):
    jobs = tuple(job for job, _ in entries)
    facade = SimpleNamespace(geometry=experiment.geometry, output=destination, samples=lambda **kw: jobs)
    render_native_creases(facade, chosen)


def run(experiment, device, ring_decision, visual_path=None, selection_review_path=None):
    experiment.use_ring(ring_decision["selected"])
    experiment.samples = partial(_samples, experiment)
    print(f"Task 15 selected {experiment.ring_name} rings; generating matched regional controls", flush=True)
    generate_baselines(experiment, device)
    original_methods = (*BENCHMARK_BASELINES, "b2_regional_r1")
    matched42 = list(experiment.samples(methods=original_methods))
    matched1234 = list(experiment.samples(methods=original_methods, seed=1234))
    measure(experiment, matched42 + matched1234, device)
    for method in CAPS:
        _candidate(experiment, method, device, seed=42)

    groups42 = grouped(matched42 + list(experiment.samples(methods=CAPS)))
    texture42 = textures(groups42)
    reviews = read_json(selection_review_path) if selection_review_path else None
    selected = select_candidates(groups42, CAPS, experiment.config["gate"], texture=texture42, visual_reviews=reviews)
    chosen = selected["selected"]
    ledger = {"ring": experiment.ring_name, "cap": selected, "complete": False,
              "ring_decision_sha256": file_hash(experiment.output / "ring-selection.json"),
              "candidate_parameters": {method: experiment.parameters(method) for method in CAPS},
              "selection_policy": SELECTION_POLICY,
              "scores_42_sha256": {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, entries in groups42.items() for job, _ in entries}}
    path = experiment.output / "selection.json"
    write_json(path, ledger)
    if selected["requires_tie_visual_review"]:
        folder = experiment.output / "tie-review"
        folder.mkdir(exist_ok=True)
        fingerprints = {}
        for method in selected["stripe_ties"]:
            fingerprints[method] = gate_for_seed(experiment, groups42, method, seed=42)["fingerprint"]
            _native(experiment, groups42[method], method, folder / method)
        write_json(folder / "request.json", fingerprints)
        print(f"Task 15 requires a native-pixel comparison at {folder}", flush=True)
        return
    if chosen is None:
        write_results(experiment, {42: groups42}, chosen, selection=selected, texture_by_seed={42: texture42})
        print("Task 15: none of G5, U1, and U2 meets the registered hard constraints", flush=True)
        return

    print(f"Task 15 seed 42 selected {chosen}; verifying the same settings at seed 1234", flush=True)
    validation = copy(experiment)
    validation.output = experiment.output / "validation"
    validation.samples = partial(_samples, validation)
    _candidate(validation, chosen, device, seed=1234, source=chosen)
    groups = {42: groups42, 1234: grouped(matched1234 + list(validation.samples(methods=(chosen,), seed=1234)))}
    for seed, scored in groups.items():
        _native(experiment, scored[chosen], chosen, experiment.output / f"seed-{seed}")
    explicit_review = visual_path is not None
    saved_review = experiment.output / "visual-review.json"
    if visual_path is None:
        visual_path = next((path for path in (saved_review, DOCS / "uniform-ring-visual-review.json")
                            if path.is_file()), None)
    visual = read_json(visual_path) if visual_path is not None else None
    verdict = write_results(experiment, groups, chosen, visual=visual, selection=selected, texture_by_seed={42: texture42})
    verified_reviews = {seed: gate["visual_review"] for seed, gate in verdict["seeds"].items()}
    ledger["visual_review_complete"] = all(verified_reviews.values())
    if ledger["visual_review_complete"]:
        write_json(saved_review, verified_reviews)
    elif explicit_review:
        raise ValueError("The supplied review must bind both seeds to their current atlas and scores")
    ledger["scores_1234_sha256"] = {f"{method}/{job.atlas.identifier}": file_hash(job.score)
                                   for method, entries in groups[1234].items() for job, _ in entries}
    ledger["complete"] = True
    write_json(path, ledger)
    from .e02_uniform_ring_examples import export_examples
    export_examples(experiment, groups, chosen, experiment.output / "examples")
    print({seed: {key: result["passed"] for key, result in gate["checks"].items()}
           for seed, gate in verdict["seeds"].items()}, flush=True)
