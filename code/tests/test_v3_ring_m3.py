import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from dielinediff.dieline.geometry import build_geometry
from dielinediff.methods.v3 import DieLineDiffV3
from dielinediff.methods.v3_attention import RegionRouting
from dielinediff.methods.v3_capgeom_geometry import (
    build_capgeom_layout, compose_capgeom_atlas, iter_frozen_cap_bleed,
)
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout
from dielinediff.methods.v3_ring_m3 import extended_ring_layout
from dielinediff.methods.v3_ring_roll import M3RingMixin, PhasedRegionRouting, ring_shift


class M3GeometryTest(unittest.TestCase):
    def test_extended_projection_wrap_and_cap_bleed(self):
        import tempfile
        from pathlib import Path
        from dielinediff.dieline.geometry import _write_labels

        geometry = build_geometry(pixels_per_mm=6.4)
        top = geometry.topology
        with tempfile.TemporaryDirectory() as directory:
            _write_labels(geometry, Path(directory))
            labels = np.asarray(Image.open(Path(directory) / "region_map.png"))
            owners = np.asarray(Image.open(Path(directory) / "bleed_owner.png"))
        original = build_ring_layout(top, 1280, 1024)
        expanded = extended_ring_layout(top, (1280, 1152))
        self.assertEqual(original.body_rows, (8, 120))
        self.assertEqual(expanded.body_rows, (16, 128))
        np.testing.assert_allclose(expanded.buffer_mm, (160 / 7, 160 / 7))
        np.testing.assert_array_equal(original.region_ids, expanded.region_ids[8:-8])
        for face, (l, t, r, b) in original.face_rects.items():
            self.assertEqual(expanded.face_rects[face], (l, t + 8, r, b + 8))

        y, x = np.indices((1024, 1280))
        rgb = np.stack((x % 251, y % 251, (x // 17 + y // 11) % 251), -1).astype(np.uint8)
        rgb[0] = (18, 52, 93)
        rgb[-1] = (188, 131, 74)
        padded = np.pad(rgb, ((64, 64), (0, 0), (0, 0)), mode="edge")

        def atlas(ring, canvas):
            caps = {}
            for face in ("T", "Btm"):
                old_cap = build_capgeom_layout(build_cap_layout(top, original, face), original, 16)
                cap = build_capgeom_layout(build_cap_layout(top, ring, face), ring, 16)
                np.testing.assert_array_equal(cap.known_mask, old_cap.known_mask)
                np.testing.assert_array_equal(cap.read_frozen_ring_rgb(canvas),
                                              old_cap.read_frozen_ring_rgb(rgb))
                caps[face] = (cap, np.full((cap.height_px, cap.width_px, 3), 90, dtype=np.uint8))
            image, support = compose_capgeom_atlas(top, labels, owners, ring, canvas, caps,
                                                  return_support=True)
            return np.asarray(image), support

        past, known = atlas(original, rgb)
        current, support = atlas(expanded, padded)
        np.testing.assert_array_equal(past[known > 0], current[known > 0])
        np.testing.assert_array_equal(known[known > 0], support[known > 0])
        rows, columns = np.nonzero((support > 0) & (known == 0))
        _, north, _, south = top["faces"]["F"]["rect"]
        self.assertTrue(np.any(rows < north) and np.any(rows >= south))
        self.assertTrue(np.all((rows < north) | (rows >= south)))
        colors = np.where((rows < north)[:, None], rgb[0, 0], rgb[-1, 0])
        np.testing.assert_array_equal(current[rows, columns], colors)

        for face, column in (("L", 0), ("B", -1)):
            l, t, r, b = top["faces"][face]["rect"]
            x = l if column == 0 else r - 1
            sl, _, sr, _ = original.face_rects[face]
            source_x = 8 * sl + (x - l + 0.5) * 8 * (sr - sl) / (r - l) - 0.5
            if face == "L":
                self.assertLess(source_x, 0)
            else:
                self.assertGreater(source_x, 1279)
            base = int(np.floor(source_x))
            fraction = source_x - base
            expected = np.rint((1 - fraction) * rgb[512, base % 1280, 0]
                               + fraction * rgb[512, (base + 1) % 1280, 0])
            self.assertEqual(current[(b + t) // 2, x, 0], expected)

        patches = list(iter_frozen_cap_bleed(top, labels, owners))
        self.assertEqual(sum(len(p.destination[0]) for p in patches), 20_259)
        for patch in patches:
            np.testing.assert_array_equal(current[patch.destination], current[patch.source])


class M3PhaseTest(unittest.TestCase):
    def test_rolls_before_downsampling_labels(self):
        labels = np.indices((144, 160)).sum(0) % 5
        routing = PhasedRegionRouting(RegionRouting(labels))
        for i in (0, 1, 4, 29):
            shift = ring_shift(i, labels.shape[1])
            routing.set_phase(i, shift)
            expected = np.roll(labels, shift, axis=1)
            for stride in (1, 2, 4):
                center = stride // 2
                query_labels = expected[center::stride, center::stride].ravel()
                actual = np.full(query_labels.shape, -1)
                for label, indices in routing.groups(len(actual), torch.device("cpu")):
                    actual[indices.numpy()] = label
                np.testing.assert_array_equal(actual, query_labels)

    def test_unrolls_before_lut_and_ddim(self):
        initial = torch.randn((1, 4, 144, 160), generator=torch.Generator().manual_seed(42))
        text, pooled = object(), object()
        observed = []
        test = self

        class Scheduler:
            init_noise_sigma = 1

            def set_timesteps(self, steps, device):
                self.timesteps = range(steps)

            def step(self, prediction, timestep, latent, *, return_dict):
                torch.testing.assert_close(prediction, latent * 2)
                torch.testing.assert_close(latent, initial + timestep)
                return (latent + 1,)

        class Native:
            dtype, device = torch.float32, torch.device("cpu")
            config = {"steps": 3, "projection_tail": 1, "projection_interval": 1,
                      "projection_strength": 0.5}
            pipe = SimpleNamespace(scheduler=Scheduler())
            _denoise = DieLineDiffV3._denoise

            def _predict(self, latent, timestep, actual_text, condition, control):
                test.assertIs(actual_text, text)
                test.assertIs(condition["text_embeds"], pooled)
                test.assertEqual(condition["time_ids"][0].tolist(), [1152, 1280, 0, 0, 1152, 1280])
                torch.testing.assert_close(
                    latent, torch.roll(initial + timestep, ring_shift(timestep, 160), -1)
                )
                return latent * 2

            def color_constraint(self, latent, prediction, timestep):
                torch.testing.assert_close(latent, initial + timestep)
                torch.testing.assert_close(prediction, latent * 2)
                return prediction

        class Engine(M3RingMixin, Native):
            pass

        engine = Engine()
        with engine._ring_phase(rolling=True, set_phase=lambda *phase: observed.append(phase)) as phase:
            result = engine._denoise((1280, 1152), torch.Generator().manual_seed(42), text, pooled)
        torch.testing.assert_close(result, initial + 3)
        self.assertEqual(phase.shifts, [0, 37, 74])
        self.assertEqual(observed, [(0, 0), (1, 37), (2, 74)])
        self.assertFalse(hasattr(engine, "_active_ring_phase"))
