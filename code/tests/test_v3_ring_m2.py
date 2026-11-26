import unittest

import numpy as np
import torch
from diffusers.models.attention_processor import Attention

from dielinediff.methods.v3_geometry import RingLayout
from dielinediff.methods.v3_ring_m2 import (
    M2_REGIONAL_SETTINGS, M2Routing, m2_atlas_weights, m2_attention, m2_prompt_plan,
    m2_ring_weights, m2_schedule,
)


class CoarseToFineTests(unittest.TestCase):
    def test_prompt_plan_keeps_global_subject_and_regional_print_requirement(self):
        brief = {
            "continuous": True, "main_color": "#75af6e",
            "front_prompt": "prominent illustration of tea leaves, misty mountains, clear space for product name",
            "side_prompt": "tea leaves and valley",
            "top_prompt": "small etching of tea stems",
        }
        prompts = m2_prompt_plan(brief, M2_REGIONAL_SETTINGS)
        self.assertIn("tea leaves, misty mountains", prompts[0])
        self.assertNotIn("product name", prompts[0])
        self.assertNotIn("prominent illustration", prompts[0])
        self.assertIn("clear space for product name", prompts[1])
        self.assertIn("slightly lower visual density, softer details", prompts[2])
        self.assertNotIn("calm background", prompts[2])
        self.assertIn("simple ornamental treatment", prompts[6])
        self.assertNotIn("clear margins", " ".join(prompts.values()))

    def test_ramp_boundaries_and_equal_increments(self):
        schedule = m2_schedule(30, M2_REGIONAL_SETTINGS)
        self.assertEqual(len(schedule), 30)
        self.assertEqual(schedule[:9], (0.0,) * 9)
        self.assertEqual(schedule[9], 0.0)
        self.assertEqual(schedule[-1], 0.60)
        np.testing.assert_allclose(np.diff(schedule[9:]), 0.03)

    def test_ring_feather_in_physical_coordinates_wraps_and_rolls(self):
        labels = np.tile(np.repeat([3, 1, 4, 2], 10), (8, 1))
        rectangles = {name: (i * 10, 1, (i + 1) * 10, 7)
                      for i, name in enumerate(("L", "F", "R", "B"))}
        layout = RingLayout(320, 64, labels.shape, (1, 7), rectangles, labels, (8, 8))
        topology = {"box_mm": {"width": 40.0, "length": 40.0}}
        weights = m2_ring_weights(topology, layout, 7.0)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0)
        np.testing.assert_allclose(weights[1, 2, 9:11], [5 / 14, 9 / 14])
        np.testing.assert_allclose(weights[3, 0, [39, 0]], [5 / 14, 9 / 14])
        np.testing.assert_allclose(weights[2, 0, [39, 0]], [9 / 14, 5 / 14])
        self.assertEqual(weights[3, 0, 4], 1.0)
        self.assertEqual(weights[1, 0, 4], 0.0)
        routing = M2Routing(weights)
        routing.set_shift(37)
        for stride in (1, 2, 4, 8):
            offset = stride // 2
            expected = np.roll(weights, 37, axis=-1)[:, offset::stride, offset::stride]
            np.testing.assert_array_equal(routing.weights_at_stride(stride), expected)

    def test_flat_feather_uses_only_actual_atlas_folds(self):
        masks = np.zeros((96, 160), dtype=np.uint8)
        faces = {}
        for i, (name, label) in enumerate(zip(("L", "F", "R", "B"), (3, 1, 4, 2), strict=True)):
            x0, x1 = 16 + 32 * i, 48 + 32 * i
            masks[32:80, x0:x1] = label
            faces[name] = {"rect": (x0, 32, x1, 80)}
        masks[8:32, 48:80] = 5
        topology = {"faces": faces, "image": {"pixels_per_mm": 1.0}}
        weights = m2_atlas_weights(topology, masks, 7.0)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0)
        self.assertGreater(weights[1, 5, 5], 0)
        self.assertGreater(weights[3, 5, 5], 0)
        self.assertGreater(weights[1, 5, 9], 0)
        self.assertGreater(weights[3, 5, 9], 0)
        self.assertGreater(weights[2, 5, 13], 0)
        self.assertGreater(weights[3, 5, 13], 0)
        self.assertEqual(weights[5, 3, 7], 1.0)
        self.assertEqual(weights[2, 5, 2], 0.0)
        self.assertEqual(weights[3, 5, 17], 0.0)

    def test_temporal_and_spatial_mixing_agrees_with_direct_attention_and_restores(self):
        generator = torch.Generator().manual_seed(12)
        attn = Attention(query_dim=4, cross_attention_dim=4, heads=1, dim_head=4)
        original = attn.processor
        hidden = torch.randn((2, 2, 4), generator=generator)
        global_context = torch.randn((2, 3, 4), generator=generator)
        regional = torch.randn((2, 3, 4), generator=generator)
        spatial = np.array([[[0.75, 0.0]], [[0.25, 1.0]]], dtype=np.float32)
        routing = M2Routing(spatial)
        raw_context = torch.stack((global_context[0], global_context[1].lerp(regional[1], 0.6)))
        base = attn(hidden, encoder_hidden_states=global_context)
        region = attn(hidden, encoder_hidden_states=raw_context)
        coefficients = torch.tensor([0.25, 1.0])[None, :, None]
        expected = base + coefficients * (region - base)
        with m2_attention((attn,), routing, {0: global_context, 1: regional}, (0, 0.6)) as current:
            self.assertIs(attn.processor, current)
            torch.testing.assert_close(attn(hidden, encoder_hidden_states=global_context), base)
            current.set_phase(1)
            torch.testing.assert_close(attn(hidden, encoder_hidden_states=global_context), expected)
        self.assertIs(attn.processor, original)


if __name__ == "__main__":
    unittest.main()
