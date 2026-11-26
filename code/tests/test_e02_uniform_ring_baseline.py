import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dielinediff.e02_uniform_ring_baseline import (
    ORIGINAL, R1, TASK07_RUN, baseline_atlases, regional_conditions_with_negative,
    regional_negative, tokenizer_proof,
)
from dielinediff.methods.sdxl import NEGATIVE


class SpyEncoder:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, *, prompt, negative_prompt, device, do_classifier_free_guidance):
        self.calls.append((list(prompt), list(negative_prompt), device, do_classifier_free_guidance))
        offsets = torch.arange(len(prompt), dtype=torch.float32)
        positive = offsets[:, None, None].expand(-1, 2, 3)
        pooled = offsets[:, None].expand(-1, 3)
        multiplier = 7.0 if negative_prompt[0] == NEGATIVE else 11.0
        return positive, positive * 0 + multiplier, pooled, pooled * 0 + multiplier


class RegionalBaselineTests(unittest.TestCase):
    def test_only_negative_argument_changes_in_original_conditioning(self):
        engine = SimpleNamespace(pipe=SpyEncoder(), device="cpu", config={"regional_weight": 1.0})
        brief = {"continuous": True, "main_color": "#123456", "front_prompt": "front",
                 "side_prompt": "side", "top_prompt": "top"}
        main, original_proof = regional_conditions_with_negative(engine, brief, NEGATIVE)
        suffix = "split panels, stacked horizontal bands"
        changed, r1_proof = regional_conditions_with_negative(engine, brief, regional_negative(suffix))
        self.assertNotIn("encode_prompt", vars(engine.pipe))
        original_call, r1_call = engine.pipe.calls
        self.assertEqual(original_call[0], r1_call[0])
        self.assertEqual(original_call[2:], r1_call[2:])
        self.assertEqual(len(original_call[0]), 6)
        self.assertEqual(original_call[1], [NEGATIVE] * 6)
        self.assertEqual(r1_call[1], [regional_negative(suffix)] * 6)
        self.assertEqual(original_proof["positive_sha256"], r1_proof["positive_sha256"])
        self.assertNotEqual(original_proof["negative_sha256"], r1_proof["negative_sha256"])
        old_regions, old_pool = main
        new_regions, new_pool = changed
        for key in old_regions:
            torch.testing.assert_close(old_regions[key][1], new_regions[key][1])
            self.assertFalse(torch.equal(old_regions[key][0], new_regions[key][0]))
        torch.testing.assert_close(old_pool[1], new_pool[1])

    def test_tokenizer_counts_include_special_tokens_and_reject_truncation(self):
        class Tokens:
            def __init__(self, limit):
                self.model_max_length = limit

            def __call__(self, value, *, add_special_tokens, truncation):
                count = len(value.split())
                assert add_special_tokens and not truncation
                return {"input_ids": [0, *range(1, count + 1), 99]}

        evidence = tokenizer_proof((Tokens(8), Tokens(8)), "split panels")
        self.assertEqual(evidence["tokenizer_1"]["tokens"], 4)
        self.assertEqual(evidence["tokenizer_1"]["input_ids_sha256"], evidence["tokenizer_2"]["input_ids_sha256"])
        with self.assertRaisesRegex(ValueError, "tokenizer 2"):
            tokenizer_proof((Tokens(8), Tokens(3)), "split panels")

    def test_reuses_original_seed42_and_protects_archive_tree(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archive = root / "outputs/e02/step1" / TASK07_RUN
            output = root / "outputs/e02/uniform-ring/runs/check"
            exp = SimpleNamespace(root=root, output=output, geometry=root / "geometry",
                                  briefs=[{"id": f"E01-{index:03d}"} for index in range(1, 51)])
            items = baseline_atlases(exp)
            self.assertEqual(len(items), 40)
            counts = {(method, seed): sum(item.method == method and item.seed == seed for item in items)
                      for method in (ORIGINAL, R1) for seed in (42, 1234)}
            self.assertEqual(set(counts.values()), {10})
            for item in items:
                expected = archive if item.method == ORIGINAL and item.seed == 42 else output
                self.assertTrue(item.path.is_relative_to(expected))
            exp.output = archive / "other_baseline"
            with self.assertRaisesRegex(ValueError, "Task 07 archive"):
                baseline_atlases(exp)


if __name__ == "__main__":
    unittest.main()
