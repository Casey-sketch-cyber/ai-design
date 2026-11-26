"""Verify the geometry, frequency and phase properties of the U1 prior on CPU."""

import unittest

import numpy as np
from PIL import Image

from dielinediff.dieline import build_geometry
from dielinediff.methods.v3_capgeom_geometry import build_capgeom_layout
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout
from dielinediff.methods.v3_uniform_texture import _extend, inject_uniform_texture


class UniformTextureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = build_geometry(pixels_per_mm=6.4).topology
        cls.ring = build_ring_layout(cls.topology, 1280, 1024)

    def layout(self, face, scale):
        original = build_cap_layout(self.topology, self.ring, face, band_latent=16)
        return build_capgeom_layout(original, self.ring, 16, scale)

    def test_smooth_phase_starts_after_the_fold_and_breaks_axial_mirror_pairs(self):
        depth, along = np.indices((12, 41))
        texture = np.stack((depth + along * 0.5, depth * 2 - along * 0.01, depth - along), axis=2)
        texture = texture.astype(np.float32)
        locked = 5
        extended = _extend(texture, 51, 0.21, locked)
        np.testing.assert_array_equal(extended[:locked], texture[:locked])
        # Depths 7 and 16 have the same reflected source row. The tangential
        # phase makes their interior samples different without an abrupt tile seam.
        self.assertFalse(np.array_equal(extended[7, 8:-8], extended[16, 8:-8]))
        np.testing.assert_array_equal(extended, _extend(texture, 51, 0.21, locked))

    def test_full_core_retains_texture_and_coarse_color_while_holding_frozen_rgb(self):
        y, x = np.indices((self.ring.height_px, self.ring.width_px), dtype=np.float32)
        wave = 125 + 23 * np.sin(x / 2.5 + np.sin(y / 27)) + 16 * np.cos(y / 3.5 + x / 49)
        ring_art = Image.fromarray(np.stack((wave, wave + 14, wave - 20), axis=2).astype(np.uint8))
        for name in ("T", "Btm"):
            layout = self.layout(name, 2)
            frozen = np.asarray(layout.read_frozen_ring_rgb(ring_art)).astype(np.float32) / 255
            known = layout.known_mask.repeat(8, axis=0).repeat(8, axis=1)
            harmonic = np.full_like(frozen, 0.5)
            harmonic[known] = frozen[known]
            frozen[~known] = np.nan
            x0, y0, x1, y1 = (8 * v for v in layout.center_rect)
            outside = np.ones(known.shape, dtype=bool)
            outside[y0:y1, x0:x1] = False
            result, proof = inject_uniform_texture(harmonic, frozen, layout)
            with self.subTest(face=name):
                np.testing.assert_array_equal(result[outside], harmonic[outside])
                np.testing.assert_array_equal(result[known], frozen[known])
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.all((0 <= result) & (result <= 1)))
                self.assertEqual(proof["supported_core_pixels"], (x1 - x0) * (y1 - y0))
                self.assertAlmostEqual(proof["center_envelope_min"], 0.60)
                self.assertEqual(proof["edge_envelope_max"], 1.0)
                self.assertGreater(proof["applied_center_to_edge_rms"], 0.5)
                self.assertEqual(proof["center_rgb_gradient_energy_before"], 0)
                self.assertGreater(proof["center_rgb_gradient_energy_after"], 1e-5)
                self.assertLess(proof["applied_center_coarse_residual_rms"],
                                0.2 * proof["applied_center_residual_rms"])

    def test_constant_source_injects_no_noise(self):
        layout = self.layout("T", 1)
        neutral = np.full((layout.height_px, layout.width_px, 3), 0.625, dtype=np.float32)
        result, proof = inject_uniform_texture(neutral, neutral, layout)
        np.testing.assert_array_equal(result, neutral)
        self.assertEqual(proof["applied_core_residual_rms"], 0.0)
        self.assertIsNone(proof["applied_center_to_edge_rms"])


if __name__ == "__main__":
    unittest.main()
