"""Verify the registered sigma change without disturbing U1's fixed-edge prior."""

import numpy as np
from PIL import Image

from dielinediff.dieline import build_geometry
from dielinediff.methods.v3_capgeom_geometry import build_capgeom_layout
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout
from dielinediff.methods.v3_ring_midscale_cap import inject_texture_at_scale
from dielinediff.methods.v3_uniform_texture import inject_uniform_texture


def test_midscale_texture_changes_only_the_registered_core():
    topology = build_geometry(pixels_per_mm=6.4).topology
    ring = build_ring_layout(topology, 1280, 1024)
    original = build_cap_layout(topology, ring, "Btm", band_latent=16)
    layout = build_capgeom_layout(original, ring, 16, 2)
    y, x = np.indices((ring.height_px, ring.width_px), dtype=np.float32)
    scale_wave = 120 + 30 * np.sin((x + y) / 14) + 12 * np.sin((x - y) / 4)
    image = Image.fromarray(np.repeat(scale_wave.astype(np.uint8)[..., None], 3, axis=-1))
    frozen = np.asarray(layout.read_frozen_ring_rgb(image)).astype(np.float32) / 255
    known = layout.known_mask.repeat(8, 0).repeat(8, 1)
    harmonic = np.full_like(frozen, 0.5)
    harmonic[known] = frozen[known]

    expected, _ = inject_uniform_texture(harmonic, frozen, layout)
    same, _ = inject_texture_at_scale(harmonic, frozen, layout, sigma_px_at_scale_1=4)
    np.testing.assert_array_equal(same, expected)

    wider, proof = inject_texture_at_scale(harmonic, frozen, layout, sigma_px_at_scale_1=14)
    np.testing.assert_array_equal(wider[known], frozen[known])
    assert proof["gaussian_sigma_px"] == 28
    assert proof["center_envelope_min"] >= 0.60
    assert not np.array_equal(wider, same)
