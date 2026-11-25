"""Apply Task 08 selection and acceptance to the unchanged E02 scores."""

import csv
from hashlib import sha256
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02 import BENCHMARK_BASELINES, VARIANTS, gate as e02_gate, measure as evaluate

BRIEF_IDS = frozenset(f"E01-{i:03d}" for i in range(1, 11))
RING_REFERENCE = "v3_1280"
CAP_FIELDS = ("cap_6_delta_e00", "cap_6_lpips")
TIME_TIE_SECONDS = 0.25
RING_HASHES = ("cache_sha256", "latent_sha256", "ring_rgb_sha256", "body_rgb_sha256", "task07_body_rgb_sha256")
DIRECTIONS = {
    "cap_6_delta_e00": -1,
    "cap_6_lpips": -1,
    "non_native_7_sdr": -1,
    "non_native_7_lpips": -1,
    "stripe_index": -1,
    "clip_score": 1,
    "image_reward": 1,
    "elapsed_seconds": -1,
}
SELECTION_POLICY = {
    "variant": "preflight",
    "direction": DIRECTIONS,
    "decision": "One vote per directional improvement in the paired mean; ties abstain.",
    "time_tie_seconds": TIME_TIE_SECONDS,
    "prompt_ties": ["cap_6_delta_e00", "cap_6_lpips", "elapsed_seconds", "c1a"],
    "addition": "Retain a budget-eligible challenger only if it wins more votes and improves at least one cap metric.",
    "budget": "Every registered box must be at or below both original E02 hardware limits.",
}


def _split(groups, method):
    entries = groups[method]
    identities = {(job.atlas.brief["id"], job.atlas.seed) for job, _ in entries}
    if len(entries) != len(BRIEF_IDS) or identities != {(brief, 42) for brief in BRIEF_IDS}:
        raise ValueError(f"{method} requires exactly E01-001 through E01-010 at seed 42.")
    return entries


def cap_six(scores):
    """Recover the six lid seams' mean from the seven equally weighted seams."""
    return {
        f"cap_6_{metric}": (
            7 * _strict_number(scores, f"non_native_7_{metric}") - _strict_number(scores, f"wrap_lb_{metric}")
        ) / 6
        for metric in ("delta_e00", "lpips")
    }


def _strict_number(scores, key):
    value = scores.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Expected a finite Task 08 score for {key}.")
    return float(value)


def _values(entries):
    return [
        (job.atlas.brief["id"], {**scores["preflight"], **cap_six(scores["preflight"])})
        for job, scores in entries
    ]


def runtime_budget(entries, gate_config):
    """Check each case against the existing E02 limits and retain exact outliers."""
    rows = [{
        "brief_id": job.atlas.brief["id"],
        "elapsed_seconds": _strict_number(scores["preflight"], "elapsed_seconds"),
        "peak_memory_mib": _strict_number(scores["preflight"], "peak_memory_mib"),
    } for job, scores in entries]
    seconds_limit = gate_config["seconds"]
    peak_limit = gate_config["peak_memory_mib"]
    outliers = [row for row in rows
                if row["elapsed_seconds"] > seconds_limit or row["peak_memory_mib"] > peak_limit]
    return {
        "passed": not outliers,
        "highest_sample_seconds": max(row["elapsed_seconds"] for row in rows),
        "highest_sample_peak_mib": max(row["peak_memory_mib"] for row in rows),
        "seconds_limit": seconds_limit,
        "peak_mib_limit": peak_limit,
        "over_limit": outliers,
    }


def _score_set(groups, method, gate_config):
    entries = _split(groups, method)
    rows = dict(_values(entries))
    means = {
        key: math.fsum(_strict_number(scores, key) for scores in rows.values()) / len(rows)
        for key in (*DIRECTIONS, "peak_memory_mib")
    }
    return means, runtime_budget(entries, gate_config)


def _comparison(groups, first, second, gate_config):
    means, budgets = {}, {}
    for method in (first, second):
        means[method], budgets[method] = _score_set(groups, method, gate_config)
    deltas = {metric: means[second][metric] - means[first][metric] for metric in DIRECTIONS}
    gains = {metric: DIRECTIONS[metric] * delta for metric, delta in deltas.items()}
    if abs(deltas["elapsed_seconds"]) < TIME_TIE_SECONDS:
        gains["elapsed_seconds"] = 0.0
    wins = [metric for metric, value in gains.items() if value > 0]
    losses = [metric for metric, value in gains.items() if value < 0]
    return {
        "n": len(BRIEF_IDS),
        "reference": first,
        "challenger": second,
        "means": means,
        "hardware": budgets,
        "challenger_minus_reference": deltas,
        "challenger_wins": wins,
        "challenger_losses": losses,
        "ties": [metric for metric, value in gains.items() if value == 0],
        "challenger_net_votes": len(wins) - len(losses),
    }


def select_prompts(groups, c1a, c1b, gate_config):
    """Choose between C1a and C1b before testing any additional modification."""
    record = _comparison(groups, c1a, c1b, gate_config)
    eligible = [name for name in (c1a, c1b) if record["hardware"][name]["passed"]]
    if len(eligible) < 2:
        selected = eligible[0] if eligible else None
        reason = "hardware_budget" if selected else "no_budget_eligible_prompt"
    elif net := record["challenger_net_votes"]:
        selected = c1b if net > 0 else c1a
        reason = "net_votes"
    else:
        reason = "cap_metrics_then_time_tiebreak"
        keys = CAP_FIELDS if "elapsed_seconds" in record["ties"] else (*CAP_FIELDS, "elapsed_seconds")
        selected = min(eligible, key=lambda name: (
            *(record["means"][name][key] for key in keys), name != c1a,
        ))
    return {"stage": "C1", **record, "selected": selected, "reason": reason}


def select_addition(groups, incumbent, challenger, gate_config, *, stage):
    """Test one change against the current cap settings and keep only net gain."""
    if stage not in {"C2", "C3", "C4"}:
        raise ValueError("Expected one of C2, C3 or C4.")
    record = _comparison(groups, incumbent, challenger, gate_config)
    if not record["hardware"][incumbent]["passed"]:
        raise ValueError("The current cap configuration exceeds the hardware budget.")
    if not record["hardware"][challenger]["passed"]:
        reason = "challenger_exceeds_budget"
    elif not set(CAP_FIELDS).intersection(record["challenger_wins"]):
        reason = "no_cap_improvement"
    elif record["challenger_net_votes"] <= 0:
        reason = "no_net_improvement"
    else:
        reason = "net_cap_improvement"
    selected = challenger if reason == "net_cap_improvement" else incumbent
    return {"stage": stage, **record, "selected": selected, "reason": reason}


def _body_sha256(path, mask):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Native side-panel mask does not match the atlas: {path}")
    return sha256(rgb[mask].tobytes()).hexdigest()


def verify_ring(groups, candidates=None):
    """Verify cached-ring provenance and compare the four native sides to Task 07."""
    references = {job.atlas.brief["id"]: job.atlas for job, _ in _split(groups, RING_REFERENCE)}
    if candidates is None:
        candidates = [name for name in groups if name not in (*BENCHMARK_BASELINES, RING_REFERENCE)]
    first = references[min(references)]
    with Image.open(first.geometry / "region_map.png") as image:
        mask = np.isin(np.asarray(image), [1, 2, 3, 4])
    if mask.ndim != 2 or not mask.any():
        raise ValueError("The Task 07 region map has no labeled side panels.")
    body = {
        brief: _body_sha256(atlas.path, mask)
        for brief, atlas in references.items()
    }
    initial = {}
    proof = {}
    for method in candidates:
        cases = {}
        for job, _ in _split(groups, method):
            atlas = job.atlas
            brief = atlas.brief["id"]
            record = read_json(atlas.metadata)
            ring = record.get("ring", {})
            if not isinstance(ring, dict) or any(
                not isinstance(ring.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", ring[key]) is None
                for key in RING_HASHES
            ):
                raise ValueError(f"Missing ring provenance for {method}/{atlas.identifier}.")
            if (atlas.geometry != references[brief].geometry or record.get("method") != method
                    or record.get("brief_id") != brief or record.get("seed") != atlas.seed
                    or record.get("raw_sha256") != file_hash(atlas.path)):
                raise ValueError(f"Stale ring provenance for {method}/{atlas.identifier}.")
            actual = _body_sha256(atlas.path, mask)
            if (ring.get("matches_task07_body") is not True
                    or actual != ring["body_rgb_sha256"]
                    or actual != ring["task07_body_rgb_sha256"]
                    or actual != body[brief]):
                raise ValueError(f"Side panels differ from the Task 07 atlas: {method}/{atlas.identifier}.")
            shared = tuple(ring[field] for field in RING_HASHES[:3])
            if shared != initial.setdefault(brief, shared):
                raise ValueError(f"Frozen ring changed between candidates for {atlas.identifier}.")
            cases[brief] = {**ring, "candidate_raw_sha256": record["raw_sha256"]}
        proof[method] = cases
    if not proof:
        raise ValueError("There are no candidate atlases with verified Task 07 ring provenance.")
    return {"reference": RING_REFERENCE, "n": len(BRIEF_IDS), "methods": proof}


def gate(experiment, groups, chosen, visual=None):
    """Reuse the five original checks and enforce Task 08's per-box hardware bound."""
    entries = _split(groups, chosen)
    for baseline in BENCHMARK_BASELINES:
        _split(groups, baseline)
    result = e02_gate(experiment, groups, chosen, visual)
    result["checks"]["runtime"].update(runtime_budget(entries, experiment.config["gate"]))
    result["passed"] = all(item["passed"] for item in result["checks"].values())
    return result


def write_results(experiment, groups, chosen, visual=None):
    """Write Task 08 rows for both archived controls and every tested candidate."""
    result = gate(experiment, groups, chosen, visual)
    result["ring_proof"] = verify_ring(groups)
    rows = []
    for method in groups:
        for job, measured in _split(groups, method):
            atlas = job.atlas
            for variant in VARIANTS:
                scores = measured[variant]
                rows.append({
                    "method": method, "variant": variant, "brief_id": atlas.brief["id"], "seed": atlas.seed,
                    "continuous": atlas.brief["continuous"], **scores, **cap_six(scores),
                })
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = Path(experiment.output)
    results = output / "results-capfix.csv"
    temporary = results.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results)
    write_json(output / "gate.json", result)
    return result
