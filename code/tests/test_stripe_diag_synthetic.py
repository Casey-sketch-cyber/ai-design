import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dielinediff.metrics.physical import _stripe_side
from dielinediff.stripe_diag_synthetic import (
    METRICS, _oriented, cases, run_synthetic,
)


TOP = {"a": {"face": "F", "edge": "top"}, "b": {"face": "T", "edge": "bottom"}, "reverse": False}
BOTTOM = {"a": {"face": "B", "edge": "bottom"}, "b": {"face": "Btm", "edge": "top"}, "reverse": False}


def texture():
    rows, columns = np.mgrid[:128, :80]
    tint = (rows / 4 + columns / 3).astype(np.uint8)
    body = np.stack((130 + tint, 155 + tint, 185 + tint), axis=-1).astype(np.uint8)
    return {"F": body, "B": body[::-1].copy(), "L": body[:, :64].copy(), "R": body[:, :64].copy(),
            "T": body[:64].copy(), "Btm": body[:64].copy()}


def mock_measure(faces, creases, floor):
    for seam in creases:
        a, b = seam["a"], seam["b"]
        ratios = [_stripe_side(faces[end["face"]], end["edge"]) for end in (a, b)]
        yield {
            "a_face": a["face"], "b_face": b["face"], "fold": a["face"] + "-" + b["face"],
            "a_crease_energy": 0.0, "b_crease_energy": 0.0,
            "a_reference_energy_v1": 0.0, "b_reference_energy_v1": 0.0,
            **{metric: np.mean(ratios) for metric in METRICS},
        }


class SyntheticStripeTests(unittest.TestCase):
    def test_controls_align_across_reversed_fold_and_preserve_sources(self):
        physical = texture()
        original = {name: face.copy() for name, face in physical.items()}
        rows = list(cases(physical, {**TOP, "reverse": True}, 0))
        controls = {name: faces for name, faces, meta in rows if meta["stripe_kind"] == "none"}
        for name, faces in controls.items():
            cap = _oriented(faces["T"], "bottom")
            side = _oriented(faces["F"], "top")
            difference = np.abs(cap[::-1, 0].astype(int) - side[:, 0].astype(int))
            self.assertTrue(np.all(difference == (0 if name == "A" else 5)))
        a = _oriented(controls["A"]["T"], "bottom")
        np.testing.assert_array_equal(a[:, 32:], np.broadcast_to(a[0, 32], a[:, 32:].shape))
        c = controls["C"]["T"]
        np.testing.assert_array_equal(c, np.broadcast_to(c[0, 0], c.shape))
        self.assertGreater(np.ptp(controls["D0"]["T"].astype(int)), 0)
        for name in physical:
            np.testing.assert_array_equal(physical[name], original[name])

    def test_uniform_artificial_bands_are_detectable_in_rgb_but_blind_to_v1(self):
        rows = {(name, meta["stripe_kind"]): (faces, meta)
                for name, faces, meta in cases(texture(), BOTTOM, 6)}
        original, _ = rows["A", "none"]
        control = _oriented(original["Btm"], "top").astype(int)
        for kind, sign in (("uniform_bright", 1), ("uniform_dark", -1)):
            altered, meta = rows["B", kind]
            actual = _oriented(altered["Btm"], "top").astype(int) - control
            width = meta["stripe_width_px"]
            self.assertEqual(width, 8)
            self.assertTrue(np.all(actual[:, :width] == sign * meta["nominal_contrast_rgb"]))
            self.assertTrue(np.all(actual[:, width:] == 0))
            self.assertAlmostEqual(_stripe_side(altered["Btm"], "top"),
                                   _stripe_side(original["Btm"], "top"), places=6)
        modulated, _ = rows["B", "modulated"]
        self.assertGreater(_stripe_side(modulated["Btm"], "top"), _stripe_side(original["Btm"], "top"))

    def test_numeric_runner_pairs_sources_and_writes_no_images(self):
        physical = texture()
        with tempfile.TemporaryDirectory() as target:
            result = run_synthetic(((f"archived-{i}", physical) for i in range(20)), (TOP, BOTTOM),
                                   mock_measure, 0.0, target, threshold=1.25)
            directory = Path(target)
            self.assertEqual(set(p.name for p in directory.iterdir()),
                             {"synthetic-samples.csv", "synthetic-summary.json"})
            with (directory / "synthetic-samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20 * 9)
            self.assertEqual({row["fold"] for row in rows}, {"F-T", "B-Btm"})
            self.assertEqual(result["design"]["paired_sources"], 20)
            row, = (entry for entry in result["discrimination"]
                    if entry["metric"] == "v1" and entry["scope"] == "fold"
                    and (entry["positive"], entry["negative"]) == ("B", "A")
                    and entry["stripe_kind"] == "uniform_bright" and entry["nominal_contrast_rgb"] is None)
            self.assertEqual(row["n_pairs"], 20)
            self.assertEqual(row["pair_tie_rate"], 1.0)
            self.assertAlmostEqual(row["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
