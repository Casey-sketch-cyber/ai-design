import unittest
from types import SimpleNamespace

import numpy as np

from dielinediff.methods.v3_cap_energy import inject_energy_matched
from dielinediff.methods.v3_capfix_geometry import _pixel_rect, _span, _strip
from dielinediff.methods.v3_geometry import EDGES, LATENT_SCALE
from dielinediff.methods.v3_ring_midscale_cap import inject_texture_at_scale


def cap_fixture(textured=True):
    height, width = 144, 160
    core_rect = (4, 4, 16, 14)
    rect = _pixel_rect(core_rect)
    band_px = 24
    frozen = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    for index, edge in enumerate(EDGES):
        length = _span(rect, edge)
        coords = _strip(rect, edge, length, band_px, 0)
        d, t = np.indices((band_px, length))
        phase = 0.71 * t + 0.22 * d + index * 0.73
        waveform = np.sin(phase) + 0.4 * np.cos(1.33 * t + 0.86 * d + index)
        value = 0.5 + 0.10 * waveform if textured else np.full_like(phase, 0.5)
        frozen[coords] = value[..., None]
        mask[coords] = True
    mask_latent = mask[::LATENT_SCALE, ::LATENT_SCALE]
    layout = SimpleNamespace(width_px=width, height_px=height, scale=1, band=3,
                             center_rect=core_rect, known_mask=mask_latent,
                             bleed_cells=dict.fromkeys(EDGES, 0))
    harmonic = np.full_like(frozen, 0.5)
    harmonic[mask] = frozen[mask]
    return harmonic, frozen, layout, mask


class CapEnergyTest(unittest.TestCase):
    def test_protects_physical_edges_and_adds_real_texture_inside(self):
        prior, frozen, layout, mask = cap_fixture()
        settings = {"core_mm": (24, 20)}
        matched, proof = inject_energy_matched(prior, frozen, layout, settings=settings)
        baseline, _ = inject_texture_at_scale(prior, frozen, layout, sigma_px_at_scale_1=14)
        x0, y0, x1, y1 = _pixel_rect(layout.center_rect)
        yy, xx = np.indices(mask.shape)
        in_core = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        # This fixture has 4 model pixels/mm on both axes.
        collar = np.minimum.reduce((yy - y0, y1 - 1 - yy, xx - x0, x1 - 1 - xx)) < 12
        np.testing.assert_array_equal(matched[~in_core | collar], baseline[~in_core | collar])
        np.testing.assert_array_equal(matched[mask], frozen[mask])
        self.assertEqual(proof["edge_source_rows"], dict.fromkeys(EDGES, 12))
        self.assertEqual(proof["protected_edge_pixels_changed"], 0)
        self.assertGreater(proof["applied_center_to_edge_rms"], proof["e0_center_to_edge_rms"])
        self.assertGreater(proof["center_rgb_gradient_energy_e1"], proof["center_rgb_gradient_energy_e0"])
        self.assertEqual(matched.dtype, np.float32)
        self.assertTrue(np.all((matched >= 0) & (matched <= 1)))

    def test_does_not_invent_texture_from_zero_variance(self):
        prior, frozen, layout, _ = cap_fixture(textured=False)
        matched, proof = inject_energy_matched(prior, frozen, layout, settings={"core_mm": (24, 20)})
        np.testing.assert_array_equal(matched, prior)
        self.assertIsNone(proof["applied_center_to_edge_rms"])
        self.assertEqual(proof["additional_normal_mean_change_max_8bit"], 0)
