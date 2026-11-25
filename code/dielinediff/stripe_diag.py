"""CPU-only measurements of strip energy on registered physical faces."""

import math

import numpy as np

from .metrics import FACES, _oriented_strip
from .metrics.physical import PIXELS_PER_MM, STRIPE_ENERGY_FLOOR, STRIPE_MM, STRIPE_REFERENCE_OFFSET_MM
from .metrics.physical import _stripe_side


METRICS = ("v1", "v2_a", "v2_b", "v2_c")
WIDTH = round(STRIPE_MM * PIXELS_PER_MM)
OFFSET = round(STRIPE_REFERENCE_OFFSET_MM * PIXELS_PER_MM)
PARALLEL_OFFSET = round(11 * PIXELS_PER_MM)
CALIBRATION_PERCENTILE = 0.10


def face_energy(face):
    """Mean of horizontal and vertical RGB first-difference energies, at least 9 mm inside."""
    if min(face.shape[:2]) <= 2 * OFFSET + 1:
        raise ValueError("A face is too small for the registered interior crop.")
    interior = face[OFFSET:-OFFSET, OFFSET:-OFFSET].astype(np.float32) / 255
    return float((np.square(np.diff(interior, axis=0)).mean()
                  + np.square(np.diff(interior, axis=1)).mean()) / 2)


def side_energy(face, edge):
    """Measure the crease and the two same-width parallel reference bands."""
    raw = _oriented_strip(face, edge, PARALLEL_OFFSET + WIDTH)
    if raw.shape[0] < 2 or raw.shape[1] < PARALLEL_OFFSET + WIDTH:
        raise ValueError(f"Not enough physical pixels for stripe measurements at {edge}.")

    def energy(band):
        return float(np.square(np.diff(band, axis=0)).mean())

    frozen = _oriented_strip(face, edge, OFFSET + WIDTH).astype(np.float32) / 255
    parallel = raw[:, PARALLEL_OFFSET:PARALLEL_OFFSET + WIDTH].astype(np.float32) / 255
    return {
        "crease_energy": energy(frozen[:, :WIDTH]),
        "reference_energy_v1": energy(frozen[:, OFFSET:OFFSET + WIDTH]),
        "reference_energy_v2_c": energy(parallel),
    }


def _ratio(crease, reference):
    excess = float(np.float32(crease) - np.float32(reference))
    return 1 + max(0.0, excess) / (reference + STRIPE_ENERGY_FLOOR)


def collect_terms(faces, creases):
    """Collect absolute energies once before the historical reference floor is known."""
    interiors = {name: face_energy(faces[name]) for name in FACES}
    rows = []
    for seam in creases:
        a, b = seam["a"], seam["b"]
        row = {"fold": f"{a['face']}-{b['face']}", "a_face": a["face"], "a_edge": a["edge"],
               "b_face": b["face"], "b_edge": b["edge"]}
        for name, side in (("a", a), ("b", b)):
            face, edge = faces[side["face"]], side["edge"]
            energy = side_energy(face, edge)
            reconstructed = _ratio(energy["crease_energy"], energy["reference_energy_v1"])
            if not math.isclose(reconstructed, _stripe_side(face, edge), abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"The frozen v1 energy did not reproduce for {row['fold']}/{name}.")
            row.update({f"{name}_{key}": value for key, value in energy.items()})
        rows.append(row)
    if len(rows) != 5:
        raise ValueError("The registered physical topology must have five native folds.")
    return interiors, rows


def compare_references(interiors, rows, reference_floor):
    """Substitute reference estimates without adding a candidate to any gate."""
    atlas_median = float(np.median(list(interiors.values())))
    if not math.isfinite(reference_floor) or reference_floor < 0:
        raise ValueError("The diagnostic reference floor must be nonnegative and finite.")
    evaluated = []
    for base in rows:
        row = dict(base)
        for side in ("a", "b"):
            crease = row[f"{side}_crease_energy"]
            original = row[f"{side}_reference_energy_v1"]
            refs = {
                "v1": original,
                "v2_a": atlas_median,
                "v2_b": max(reference_floor, original),
                "v2_c": row[f"{side}_reference_energy_v2_c"],
            }
            for metric, reference in refs.items():
                row[f"{side}_reference_energy_{metric}"] = reference
                row[f"{side}_{metric}"] = _ratio(crease, reference)
        for metric in METRICS:
            row[metric] = (row[f"a_{metric}"] + row[f"b_{metric}"]) / 2
        evaluated.append(row)
    for metric in METRICS:
        peak = max(evaluated, key=lambda row: row[metric])
        for row in evaluated:
            row[f"controls_{metric}"] = row is peak
    return evaluated


def measure_atlas(faces, creases, reference_floor):
    return compare_references(*collect_terms(faces, creases), reference_floor)
