import numpy as np

from dielinediff.dieline.geometry import build_geometry
from dielinediff.metrics.physical import PIXELS_PER_MM, _press_creases, stripe_energy
from dielinediff.stripe_diag import METRICS, collect_terms, compare_references, face_energy


def test_diagnostic_reproduces_v1_and_floors_only_the_reference():
    physical = build_geometry(pixels_per_mm=PIXELS_PER_MM)
    rng = np.random.default_rng(214)
    faces = {}
    for name, face in physical.topology["faces"].items():
        left, top, right, bottom = face["rect"]
        art = np.full((bottom - top, right - left, 3), 127, dtype=np.uint8)
        art[:12] = rng.integers(80, 175, art[:12].shape, dtype=np.uint8)
        faces[name] = art
    creases = tuple(_press_creases(physical))
    actual, _ = stripe_energy(faces, creases)
    interiors, terms = collect_terms(faces, creases)
    rows = compare_references(interiors, terms, 0.002)
    assert max(row["v1"] for row in rows) == actual
    assert face_energy(faces["T"]) == 0
    assert all(row["v2_b"] <= row["v1"] for row in rows)
    assert all(row[f"{side}_reference_energy_v2_b"]
               == max(row[f"{side}_reference_energy_v1"], 0.002)
               for row in rows for side in ("a", "b"))
    assert all(sum(row[f"controls_{metric}"] for row in rows) == 1 for metric in METRICS)
