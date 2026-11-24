"""Validate E01 sample selection and summary arithmetic on temporary fixtures."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from dielinediff.dieline import build_geometry, write_geometry
from dielinediff.summary import SEEDS, build_summary, format_markdown, write_examples


class E01SummaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.briefs = self.root / "briefs.jsonl"
        self.briefs.write_text("".join(
            json.dumps({"id": f"E01-{number:02}", "continuous": number in {1, 11, 12}}) + "\n"
            for number in range(1, 51)
        ))

    def write_results(self, rows):
        path = self.root / "temporary-test-results.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "brief_id", "seed", "split", "continuous", "seam_sdr"))
            for method, brief, values in rows:
                split = "tune" if brief <= 10 else "holdout"
                for seed, value in zip(SEEDS, values):
                    writer.writerow((method, f"E01-{brief:02}", seed, split, brief in {1, 11, 12}, value))
        return path

    def test_complete_seed_pairing_and_subset_arithmetic(self):
        results = self.write_results([
            ("ours", 1, (900, 901, 902)),
            ("ours", 11, (6, 7, 8)),
            ("ours", 12, (10, 11, 12)),
            ("ours", 13, (14, 15, 16)),
            ("b2", 11, (2, 3, 4)),
            ("b2", 12, (20, None, 22)),
            ("b2", 13, (12, 13, 14)),
        ])
        report = build_summary(results, "full", briefs=self.briefs, methods=("ours", "b2", "b4"))
        overall = report["subsets"]["overall"]
        self.assertEqual(overall["planned_briefs"], 40)
        self.assertEqual(overall["observed_briefs"], 3)
        ours, b2, b4 = overall["descriptive"][:3]
        self.assertEqual((ours["n"], ours["mean"], ours["measured_seed_values"]), (3, 11, 9))
        self.assertAlmostEqual(ours["ci95"][0], 1.063449153, places=6)
        self.assertAlmostEqual(ours["ci95"][1], 20.936550847, places=6)
        self.assertEqual((b2["n"], b2["mean"], b2["measured_seed_values"]), (2, 8, 8))
        self.assertEqual((b4["n"], b4["mean"], b4["measured_seed_values"]), (0, None, 0))
        comparison = overall["comparisons"][0]
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(comparison["mean_difference"], 3)
        self.assertEqual(comparison["direction"], "higher")
        self.assertEqual(comparison["p_wilcoxon"], 0.5)
        subset = report["subsets"]["cross_panel"]
        self.assertEqual(subset["planned_briefs"], 2)
        self.assertEqual(subset["comparisons"][0]["n"], 1)
        self.assertIsNone(subset["comparisons"][0]["p_wilcoxon"])
        self.assertIn("Wilcoxon p", format_markdown(report))

    def test_pilot_is_separate_and_descriptive(self):
        results = self.write_results([
            ("ours", 1, (7, 999, 999)),
            ("ours", 11, (91, 91, 91)),
        ])
        report = build_summary(results, "pilot", briefs=self.briefs, methods=("ours",))
        subset = report["subsets"]["overall"]
        self.assertEqual(report["seeds"], [42])
        self.assertEqual(subset["planned_briefs"], 5)
        self.assertEqual(subset["descriptive"][0]["mean"], 7)
        self.assertFalse(subset["comparisons"])

    def test_examples_select_only_common_real_artifacts(self):
        geometry = write_geometry(build_geometry(short_side=128), self.root / "geometry")
        run = self.root / "example-test-run"
        image = Image.new("RGB", (128, 152), (31, 101, 133))
        for method, ids in (("ours", (1, 2, 3, 4, 5)), ("b2", (1, 2, 4, 5))):
            for number in ids:
                sample = run / method / f"E01-{number:02}-42.png"
                sample.parent.mkdir(parents=True, exist_ok=True)
                image.save(sample)
                sample.with_suffix(".run.json").write_text("{}")
        output = self.root / "examples"
        manifest = write_examples(run, geometry, output, "pilot", briefs=self.briefs,
                                  methods=("ours", "b2", "b4"))
        self.assertEqual(manifest["chosen_briefs"], ["E01-01", "E01-02", "E01-04"])
        self.assertEqual(manifest["missing_methods"], ["b4"])
        for name in manifest["figures"]:
            with Image.open(output / name) as figure:
                self.assertEqual(figure.format, "JPEG")
                self.assertGreater(figure.height, 500)


if __name__ == "__main__":
    unittest.main()
