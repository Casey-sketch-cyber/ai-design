"""Keep G5 reflection registered to the frozen physical edges and nowhere else."""

import unittest

import numpy as np
from PIL import Image

from dielinediff.dieline import build_geometry
from dielinediff.methods.v3_capgeom_geometry import build_capgeom_layout
from dielinediff.methods.v3_captexture_inject import inject_texture
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout


class TextureInjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = build_geometry(pixels_per_mm=6.4).topology
        cls.ring = build_ring_layout(cls.topology, 1280, 1024)

    def layout(self, face, scale):
        original = build_cap_layout(self.topology, self.ring, face, band_latent=16)
        return build_capgeom_layout(original, self.ring, 16, scale)

    @staticmethod
    def known_pixels(layout):
        return layout.known_mask.repeat(8, axis=0).repeat(8, axis=1)

    def test_preserves_frozen_strips_and_everything_outside_cap_core(self):
        y, x = np.indices((self.ring.height_px, self.ring.width_px))
        coded = np.stack((70 + x % 83, 97 + y % 19, (x // 8 + y // 3) % 255), axis=2)
        art = Image.fromarray(coded.astype(np.uint8))
        for face in ("T", "Btm"):
            with self.subTest(face=face):
                layout = self.layout(face, 2)
                known = self.known_pixels(layout)
                frozen = np.asarray(layout.read_frozen_ring_rgb(art)).astype(np.float32) / 255
                harmonic = np.full(frozen.shape, 0.5, dtype=np.float32)
                harmonic[known] = frozen[known]
                x0, y0, x1, y1 = (8 * v for v in layout.center_rect)
                outside = np.ones(known.shape, dtype=bool)
                outside[y0:y1, x0:x1] = False
                frozen[~known] = np.nan
                result, diagnostic = inject_texture(harmonic, frozen, layout)
                np.testing.assert_array_equal(result[outside], harmonic[outside])
                np.testing.assert_array_equal(result[known], frozen[known])
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(np.all((0 <= result) & (result <= 1)))
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(diagnostic["gaussian_sigma_px"], 8.0)
                self.assertGreater(diagnostic["applied_core_residual_rms"], 0)

    def test_mirrors_each_source_along_its_normal_without_reversing_tangent(self):
        layout = self.layout("T", 1)
        x0, y0, x1, y1 = (8 * v for v in layout.center_rect)
        known = self.known_pixels(layout)
        xx, yy = (x0 + x1) // 2 + 13, (y0 + y1) // 2 - 11
        depth = 7
        cases = {
            "top": ((y0 - 1 - depth, xx), (y0 + depth, xx)),
            "right": ((yy, x1 + depth), (yy, x1 - 1 - depth)),
            "bottom": ((y1 + depth, xx), (y1 - 1 - depth, xx)),
            "left": ((yy, x0 - 1 - depth), (yy, x0 + depth)),
        }
        for edge, (source, destination) in cases.items():
            with self.subTest(edge=edge):
                harmonic = np.full((layout.height_px, layout.width_px, 3), 0.5, dtype=np.float32)
                self.assertTrue(known[source])
                harmonic[source][0] = 0.91
                result, _ = inject_texture(harmonic, harmonic, layout)
                red = result[y0:y1, x0:x1, 0]
                location = np.unravel_index(red.argmax(), red.shape)
                self.assertEqual((location[0] + y0, location[1] + x0), destination)
                self.assertGreater(float(result[destination][0]), 0.8)
                np.testing.assert_array_equal(result[y0:y1, x0:x1, 1:], 0.5)

    def test_constant_color_has_no_injected_signal_and_overlaps_do_not_add_full_strength(self):
        layout = self.layout("Btm", 1)
        x0, y0, x1, y1 = (8 * v for v in layout.center_rect)
        uniform = np.full((layout.height_px, layout.width_px, 3), 0.5, dtype=np.float32)
        base, diagnostics = inject_texture(uniform, uniform, layout)
        np.testing.assert_array_equal(base, uniform)
        self.assertEqual(diagnostics["applied_core_residual_rms"], 0.0)
        for channel in range(3):
            uniform[y0 - 2, x0 + 1, channel] = 0.9
            uniform[y0 + 1, x0 - 2, channel] = 0.9
        overlap, diagnostics = inject_texture(uniform, uniform, layout)
        self.assertTrue(np.all((0.87 < overlap[y0 + 1, x0 + 1]) & (overlap[y0 + 1, x0 + 1] < 0.9)))
        self.assertEqual(diagnostics["clipped_core_pixels"], 0)
        np.testing.assert_array_equal(overlap[y1 - 32:y1, x1 - 32:x1], 0.5)


if __name__ == "__main__":
    unittest.main()
