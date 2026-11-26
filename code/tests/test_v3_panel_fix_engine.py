"""Verify that panel wording and diagnostics preserve the registered cap sampler."""

from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

import torch
from diffusers import DDIMScheduler

from dielinediff.e02_panel_x0 import DDIMTrace, PanelTraceMixin
from dielinediff.e02_panel_x0_scan import profile
from dielinediff.methods.v3_capgeom_init import sdedit_schedule
from dielinediff.methods.v3_joint_cap_engine import JointCapEngine
from dielinediff.methods.v3_joint_cap_prompts import joint_cap_negative, joint_cap_prompt
from dielinediff.methods.v3_panel_fix_engine import (
    PanelFixEngine,
    panel_cap_conditions,
    panel_cap_negative,
    panel_cap_prompt,
)


SETTINGS = {
    "style": "flat orthographic print artwork, evenly lit, edge-to-edge illustration",
    "negative_additions": (
        "frame, border, panel, inset picture, card, label, collage, patchwork, split layout, rectangle"
    ),
}
BRIEF = {"side_prompt": "etched woodland and log rings", "main_color": "#aabb99", "continuous": True}


class Tokenizer:
    model_max_length = 77

    def __call__(self, phrases, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return {"input_ids": [list(range(len(phrase.split()) + 2)) for phrase in phrases]}


class PanelFixTests(TestCase):
    def test_prompt_removes_layout_words_without_losing_motifs_or_colour(self):
        open_prompt = panel_cap_prompt(BRIEF, SETTINGS)
        prior = joint_cap_prompt(BRIEF)
        self.assertEqual(open_prompt.split(", log rings")[1], prior.split(", log rings")[1])
        self.assertTrue(open_prompt.startswith(SETTINGS["style"]))
        rendered = panel_cap_prompt({**BRIEF, "side_prompt": (
            "log rings, calm background for an information panel, flower leaves"
        )}, SETTINGS)
        self.assertIn("log rings, flower leaves, #aabb99", rendered)
        self.assertNotIn("information panel", rendered)
        self.assertEqual(panel_cap_negative(SETTINGS), f"{joint_cap_negative()}, {SETTINGS['negative_additions']}")
        with self.assertRaisesRegex(ValueError, "illustration"):
            panel_cap_prompt(BRIEF, {**SETTINGS, "style": "packaging graphic design"})

    def test_conditions_preserve_cfg_order_and_both_tokenizers(self):
        def encode_prompt(**kwargs):
            self.assertEqual(kwargs["prompt"], [panel_cap_prompt(BRIEF, SETTINGS)])
            self.assertEqual(kwargs["negative_prompt"], [panel_cap_negative(SETTINGS)])
            self.assertTrue(kwargs["do_classifier_free_guidance"])
            return tuple(torch.tensor([[n]]) for n in (1, 0, 3, 2))

        engine = SimpleNamespace(
            config={"panel_fix": {"cap_prompt": SETTINGS}}, device="cpu",
            pipe=SimpleNamespace(tokenizer=Tokenizer(), tokenizer_2=Tokenizer(), encode_prompt=encode_prompt),
        )
        positive, negative, text, pooled, proof = panel_cap_conditions(engine, BRIEF)
        self.assertIn("log rings", positive)
        self.assertTrue(negative.endswith("rectangle"))
        self.assertEqual(text[0].tolist(), [[0], [1]])
        self.assertEqual(pooled.tolist(), [[2], [3]])
        self.assertEqual(tuple(proof), ("tokenizer_1", "tokenizer_2"))

    def test_new_gamma_does_not_mutate_task20_settings(self):
        source_gamma = [0.20, 0.30]
        config = {"joint_cap": {"guidance": {"gamma": source_gamma}},
                  "panel_fix": {"guidance_gamma": {"k1": 0.30, "k2": 0.40}}}
        with patch.object(JointCapEngine, "__init__", return_value=None) as parent:
            PanelFixEngine("models", "geometry", config, "cpu")
        self.assertEqual(parent.call_args.args[2]["joint_cap"]["guidance"]["gamma"], (0.20, 0.30, 0.40))
        self.assertEqual(source_gamma, [0.20, 0.30])

    def test_ddim_trace_returns_exact_results_and_uses_full_schedule_indices(self):
        baseline = DDIMScheduler(clip_sample=False)
        reference = DDIMScheduler(clip_sample=False)
        _, baseline_times = sdedit_schedule(baseline, 30, 0.85, "cpu")
        start, trace_times = sdedit_schedule(reference, 30, 0.85, "cpu")
        trace = DDIMTrace(reference, 30, 0.85)
        previous = torch.rand((1, 4, 3, 4), generator=torch.Generator().manual_seed(42))
        eps = previous * 0.5
        raw_steps = []
        actual = previous.clone()
        for raw_time, trace_time in zip(baseline_times, trace_times, strict=True):
            model_output = baseline.step(eps, raw_time, previous, return_dict=False)
            candidate_output = trace.step(eps, trace_time, actual, return_dict=False)
            self.assertTrue(torch.equal(model_output[0], candidate_output[0]))
            self.assertTrue(torch.equal(model_output[1], candidate_output[1]))
            previous, actual = model_output[0], candidate_output[0]
            raw_steps.append(model_output[1].clone())
        self.assertEqual([idx + 1 for idx, _, _ in trace.frames], list(range(6, 31, 3)))
        self.assertTrue(all(torch.equal(sample, raw_steps[idx - start]) for idx, _, sample in trace.frames))
        self.assertTrue(torch.equal(actual, trace.final))

    def test_prior_capture_ignores_late_lut_reencode(self):
        class Codec:
            def encode(self, rgb):
                return rgb + 1

        class TracedCodec(PanelTraceMixin, Codec):
            pass

        engine = TracedCodec()
        engine._cap_trace = SimpleNamespace(steps=[])
        engine.encode(torch.full((1, 3, 1, 1), 0.25))
        engine.encode(torch.full((1, 3, 1, 1), 0.5))
        engine._cap_trace.steps.append(800)
        engine.encode(torch.full((1, 3, 1, 1), 0.75))
        self.assertTrue(torch.all(engine._trace_initial == 0.5))

    def test_native_scan_measures_same_step_in_either_axis(self):
        import numpy as np

        lab = np.zeros((8, 14, 3), dtype=np.float32)
        lab[:4, :, 0] = 55
        lab[4:, :, 0] = 60
        horizontal = profile(lab, "y", 4, 0, 14)
        vertical = profile(lab.swapaxes(0, 1), "x", 4, 0, 14)
        self.assertEqual(horizontal, vertical)
        self.assertEqual(horizontal["delta_l_mean"], 5)
        self.assertEqual(horizontal["same_sign_fraction"], 1)
