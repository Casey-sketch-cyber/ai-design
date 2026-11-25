import tempfile
import unittest
from pathlib import Path

from dielinediff.e01b import (
    B2_POST, BOTTOM_TUNABLE, Experiment, check_stage, generate, measurement_input, pilot_gate,
    record_review, write_json,
)
from dielinediff.e01b_summary import build_summary


class BottomFrameRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        experiment = Experiment.__new__(Experiment)
        experiment.base = self.directory / "bottom-frame"
        experiment.shared = experiment.base / "legacy-metrics" / "test"
        experiment.legacy_outputs = self.directory / "E01"
        experiment.run = experiment.pilot_run = experiment.base / "runs" / "test"
        experiment.bottom_frame = experiment.physical = True
        experiment.tunable = BOTTOM_TUNABLE
        experiment.config = experiment.pilot_config = {"steps": 30, "transition_mm": 2.0}
        experiment.briefs = [{"id": f"E01-{index:03d}", "continuous": index % 2 == 0}
                             for index in range(1, 51)]
        self.experiment = experiment
        self.coverage = {"covered": 10, "required": 10, "steps_checked": 30, "windows": 1,
                         "minimum_weight": 0.01, "window_sizes_px": [[512, 512]]}

    def save_scores(self, job, raw, preflight):
        job.measurement.parent.mkdir(parents=True, exist_ok=True)
        write_json(job.measurement, {"input": measurement_input(job),
                                     "variants": {"raw": raw, "preflight": preflight}})

    def stage_pilot(self):
        methods = ("b2", "b3", "b4", B2_POST, "v2")
        rows = {
            "b2": (0.5, 2.0, 1.1), "b3": (0.55, 1.6, 1.2), "b4": (0.6, 1.5, 1.25),
            B2_POST: (0.75, 1.3, 1.05), "v2": (0.52, 1.4, 1.2),
        }
        self.experiment.run.mkdir(parents=True)
        (self.experiment.run / "coverage.png").write_bytes(b"coverage evidence")
        jobs = list(self.experiment.jobs("pilot", methods=methods))
        for job in jobs:
            job.image.parent.mkdir(parents=True, exist_ok=True)
            job.image.write_bytes(b"test atlas")
            write_json(job.metadata, {"coverage": self.coverage} if job.needs_generation else {})
            if job.method == B2_POST:
                job.image.with_suffix(".preflight.png").write_bytes(b"test preflight atlas")
            raw = dict(zip(("clip_score", "stripe_index", "back_bottom_stripe_index"), rows[job.method]))
            self.save_scores(job, raw, raw)
        return [job for job in jobs if job.needs_generation]

    def review(self, gate):
        source = self.directory / "synthetic-review.json"
        write_json(source, {"fingerprint": gate["fingerprint"], "reviewer": "Synthetic unit test",
                            "samples": {brief: {"edge_noise": False, "notes": "Fixture"}
                                        for brief in gate["coverage"]}})
        record_review(self.experiment, source)

    def test_both_stripe_metrics_and_both_variants_must_pass(self):
        jobs = self.stage_pilot()
        experiment = self.experiment
        gate = pilot_gate(experiment)
        self.assertTrue(gate["coverage_passed"])
        self.assertTrue(gate["clip_passed"])
        self.assertEqual(gate["clip_means"]["b2"], 0.5)
        for variant in ("raw", "preflight"):
            checks = gate["stripe_checks"][variant]
            self.assertEqual(checks["stripe_index"]["baseline_method"], B2_POST)
            self.assertAlmostEqual(checks["stripe_index"]["threshold"], 1.56)
            self.assertAlmostEqual(checks["back_bottom_stripe_index"]["threshold"], 1.26)
        self.assertTrue(gate["stripe_passed"])
        self.assertFalse(gate["passed"])

        sample = {"clip_score": 0.52, "stripe_index": 1.4, "back_bottom_stripe_index": 1.2}
        for job in jobs:
            self.save_scores(job, sample, {**sample, "back_bottom_stripe_index": 1.35})
        gate = pilot_gate(experiment)
        self.assertTrue(gate["stripe_checks"]["raw"]["back_bottom_stripe_index"]["passed"])
        self.assertFalse(gate["stripe_checks"]["preflight"]["back_bottom_stripe_index"]["passed"])
        self.review(gate)
        self.assertFalse(pilot_gate(experiment)["passed"])
        with self.assertRaisesRegex(ValueError, "Pilot gate"):
            check_stage(experiment, "tune")

        for job in jobs:
            self.save_scores(job, {**sample, "stripe_index": 1.8}, sample)
        gate = pilot_gate(experiment)
        self.assertFalse(gate["stripe_checks"]["raw"]["stripe_index"]["passed"])
        self.assertTrue(gate["stripe_checks"]["raw"]["back_bottom_stripe_index"]["passed"])
        self.assertFalse(gate["visual_review_passed"])

        for job in jobs:
            self.save_scores(job, sample, sample)
        gate = pilot_gate(experiment)
        self.assertTrue(gate["stripe_passed"])
        self.review(gate)
        self.assertTrue(pilot_gate(experiment)["passed"])
        experiment.config = {**experiment.pilot_config, "transition_mm": 2.5}
        check_stage(experiment, "tune")
        experiment.config["top_band"] = 24
        with self.assertRaisesRegex(ValueError, "top_band"):
            check_stage(experiment, "tune")

    def test_archived_protocol_only_reads_completed_inference(self):
        self.stage_pilot()
        experiment = self.experiment
        experiment.bottom_frame = False
        generate(experiment, "pilot", "cpu")
        (experiment.run / "coverage.png").unlink()
        with self.assertRaisesRegex(ValueError, "original protocol's Git revision"):
            generate(experiment, "pilot", "cpu")

    def test_b2_post_stays_cpu_and_participates_in_holdout_pairing(self):
        experiment = self.experiment
        full = list(experiment.jobs("full"))
        self.assertEqual(len(full), 1200)
        self.assertEqual(sum(job.needs_generation for job in full), 450)
        post = [job for job in full if job.method == B2_POST]
        self.assertEqual(len(post), 150)
        self.assertTrue(all(job.uses_legacy_geometry and job.source.parent.name == "b2" for job in post))
        rows = [{"method": job.method, "variant": variant, "brief_id": job.brief["id"],
                 "seed": job.seed, "stripe_index": 1.4 if job.method == "v2" else 1.0}
                for job in full for variant in ("raw", "preflight")]
        summary = build_summary(rows, experiment.briefs, experiment.phase_methods("full"),
                                (42, 1234, 2026), "full", physical=True)
        for variant in ("raw", "preflight"):
            for group, count in (("overall", 40), ("cross_panel", 20)):
                paired = summary["variants"][variant][group]["metrics"]["stripe_index"]["paired"]
                self.assertEqual(set(paired), set(experiment.phase_methods("full")) - {"v2"})
                self.assertEqual(paired[B2_POST]["n"], count)


if __name__ == "__main__":
    unittest.main()
