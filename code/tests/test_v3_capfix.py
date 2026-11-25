"""Verify Task 08 prompt routing, oriented context, and cap-only resampling on CPU."""

from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from diffusers import DDIMScheduler

from dielinediff.dieline import build_geometry
from dielinediff.methods.v3_capfix import CapFixEngine, cap_prompts, cap_region_ids
from dielinediff.methods.v3_capfix_geometry import build_capfix_layout, cap_at_original_scale
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout


RING_SIZES = ((1280, 1024), (1472, 1152))
CAPS = ("T", "Btm")


class CapFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = build_geometry(pixels_per_mm=6.4).topology

    def test_both_scaled_cores_meet_minimum_and_preserve_known_strips(self):
        for size in RING_SIZES:
            ring = build_ring_layout(self.topology, *size)
            yy, xx = np.indices(ring.latent_shape)
            frozen_ring = torch.as_tensor(np.stack((xx, yy)), dtype=torch.float32)[None]
            initial = frozen_ring.clone()
            for face in CAPS:
                with self.subTest(size=size, face=face):
                    original = build_cap_layout(self.topology, ring, face)
                    unchanged = build_capfix_layout(original, ring, 16, 1)
                    self.assertTrue(torch.equal(unchanged.read_frozen_ring(frozen_ring),
                                                original.read_frozen_ring(frozen_ring)))
                    scaled = build_capfix_layout(original, ring, 16, 2)
                    x0, y0, x1, y1 = scaled.center_rect
                    self.assertGreaterEqual(8 * min(x1 - x0, y1 - y0), 512)
                    self.assertEqual(scaled.width_px % 64, 0)
                    self.assertEqual(scaled.height_px % 64, 0)
                    self.assertEqual(int(scaled.known_mask.sum()), 64 * (x1 - x0 + y1 - y0))
                    self.assertFalse(scaled.known_mask[y0:y1, x0:x1].any())
                    pixels = scaled.read_frozen_ring(frozen_ring)[0].numpy()
                    self.assertFalse(pixels[:, ~scaled.known_mask].any())

                    # The native horizontal seam keeps increasing x across the whole cap.
                    row = y1 if face == "T" else y0 - 1
                    side = "F" if face == "T" else "B"
                    left, top, right, bottom = ring.face_rects[side]
                    cols = pixels[0, row, x0:x1]
                    np.testing.assert_allclose((cols[0], cols[-1]), (left, right - 1))
                    self.assertTrue(np.all(np.diff(cols) >= 0))
                    np.testing.assert_allclose(pixels[1, row, x0:x1], top if face == "T" else bottom - 1)
            self.assertTrue(torch.equal(initial, frozen_ring))

    def test_native_inpaint_pixels_align_to_each_side_in_both_directions(self):
        ring = build_ring_layout(self.topology, *RING_SIZES[0])
        x, y = np.meshgrid(np.arange(ring.width_px), np.arange(ring.height_px))
        rgb = np.stack((x % 251, y % 251, (x // 23) % 251), axis=-1).astype(np.uint8)
        for face in CAPS:
            original = build_cap_layout(self.topology, ring, face)
            for scale in (1, 2):
                with self.subTest(face=face, scale=scale):
                    cap = build_capfix_layout(original, ring, 16, scale)
                    known = np.asarray(cap.read_frozen_ring_rgb(Image.fromarray(rgb)))
                    mask = cap.known_mask.repeat(8, 0).repeat(8, 1)
                    self.assertEqual(known.shape, (cap.height_px, cap.width_px, 3))
                    self.assertFalse(known[~mask].any())
                    x0, y0, x1, y1 = [8 * item for item in cap.center_rect]
                    top_row = y0 - 1 - 8 * cap.bleed_cells["top"]
                    top_side, top_edge, reverse, _ = original.edge_sources["top"]
                    left, top, right, bottom = (8 * value for value in ring.face_rects[top_side])
                    first, last = (right - 1, left) if reverse else (left, right - 1)
                    yy = top if top_edge == "top" else bottom - 1
                    self.assertTrue(np.array_equal(known[top_row, x0], rgb[yy, first]))
                    self.assertTrue(np.array_equal(known[top_row, x1 - 1], rgb[yy, last]))
        np.testing.assert_array_equal(rgb[..., 0], (x % 251).astype(np.uint8))

    def test_downsampling_never_reads_frozen_or_corner_pixels(self):
        ring = build_ring_layout(self.topology, *RING_SIZES[0])
        original = build_cap_layout(self.topology, ring, "Btm")
        cap = build_capfix_layout(original, ring, 16, 2)
        image = np.full((cap.height_px, cap.width_px, 3), (34, 59, 88), dtype=np.uint8)
        labels = cap.known_mask.copy()
        y, x = np.indices(cap.latent_shape)
        x0, y0, x1, y1 = cap.center_rect
        labels |= ((x < x0) | (x >= x1)) & ((y < y0) | (y >= y1))
        image[labels.repeat(8, 0).repeat(8, 1)] = (255, 0, 255)
        ordinary, restored = cap_at_original_scale(cap, Image.fromarray(image))
        self.assertIs(ordinary, original)
        restored = np.asarray(restored)
        region = np.any(restored != 0, axis=2)
        self.assertTrue(np.all(restored[region] == (34, 59, 88)))
        self.assertFalse(restored[original.known_mask.repeat(8, 0).repeat(8, 1)].any())

    def test_emblem_continuation_margin_and_shared_prompt_rules(self):
        ring = build_ring_layout(self.topology, *RING_SIZES[0])
        original = build_cap_layout(self.topology, ring, "T")
        cap = build_capfix_layout(original, ring, 16, 1)
        plain = cap_region_ids(cap, "plain", 0.25)
        emblem = cap_region_ids(cap, "emblem", 0.25)
        self.assertFalse(plain.any())
        self.assertFalse(emblem[cap.known_mask].any())
        yy, xx = np.nonzero(emblem)
        x0, y0, x1, y1 = cap.center_rect
        self.assertGreaterEqual(min(xx.min() - x0, yy.min() - y0,
                                    x1 - 1 - xx.max(), y1 - 1 - yy.max()), 0.25 * min(x1 - x0, y1 - y0))
        brief = {"id": "TEST", "side_prompt": "orchards under pastel skies TEST-PRODUCT",
                 "top_prompt": "small upright apricot emblem, yellow details",
                 "product_name_zh": "TEST-PRODUCT", "continuous": True, "main_color": "#223377"}
        a, b = cap_prompts(brief, {
            "continuation": "{style}, {side_prompt}, {main_color}, matching brushwork",
            "emblem": "{continuation}, {top_prompt}, tiny and centered",
        })
        self.assertIn("orchards under pastel skies", a)
        self.assertIn("matching brushwork", b)
        self.assertNotIn("TEST-PRODUCT", a)
        self.assertNotIn("upright", b)
        self.assertIn("apricot emblem", b)
        self.assertNotIn("apricot emblem", a)

    def test_optional_repaint_revisits_one_cap_step_and_restores_the_scheduler(self):
        engine = object.__new__(CapFixEngine)
        engine.dtype = torch.float32
        engine.device = torch.device("cpu")
        engine.config = {"steps": 5, "repaint_index": 2, "projection_strength": 0}
        engine.pipe = SimpleNamespace(scheduler=DDIMScheduler(num_train_timesteps=100, clip_sample=False))
        before = engine.pipe.scheduler.step
        watched = []

        def predict(sample, time, *_):
            watched.append((int(time), sample.clone()))
            return torch.zeros_like(sample)

        engine._predict = predict
        anchor = torch.ones(1, 4, 64, 64)
        mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        mask[..., :4, :] = True
        result = engine._cap_denoise((512, 512), torch.Generator().manual_seed(42),
                                     None, None, anchor, mask, True)
        self.assertEqual(len(watched), 6)
        self.assertEqual(watched[2][0], watched[3][0])
        self.assertTrue(torch.equal(watched[2][1][..., :4, :], watched[3][1][..., :4, :]))
        self.assertFalse(torch.equal(watched[2][1][..., 4:, :], watched[3][1][..., 4:, :]))
        self.assertTrue(torch.equal(result[..., :4, :], anchor[..., :4, :]))
        self.assertEqual(engine.pipe.scheduler.step, before)


if __name__ == "__main__":
    unittest.main()
