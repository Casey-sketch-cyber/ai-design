"""Regression coverage for the physical cap texture diagnostic."""

import numpy as np

from dielinediff.e02_cap_energy_curves import _curve_summary, tangential_energy
from dielinediff.metrics.physical import _stripe_side


def test_tangential_bins_reconstruct_frozen_stripe_and_keep_edge_direction():
    face = np.zeros((100, 120, 3), dtype=np.uint8)
    face[:, ::2] = 15
    face[:12, ::2] = 51
    curve = tangential_energy(face, "top")

    np.testing.assert_allclose(curve[:3], 0.04, rtol=1e-6)
    np.testing.assert_allclose(curve[9:12], (15 / 255) ** 2, rtol=1e-6)
    np.testing.assert_allclose(_curve_summary(curve)["v1_side_reconstruction"],
                               _stripe_side(face, "top"), rtol=1e-6)
    np.testing.assert_array_equal(tangential_energy(face, "bottom")[:3], curve[9:12])
