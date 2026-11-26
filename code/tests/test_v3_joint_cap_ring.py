import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from dielinediff.dieline.geometry import BoxSpec, build_geometry, _write_labels
from dielinediff.methods.v3_geometry import CAPS, SIDES, build_ring_layout
from dielinediff.methods.v3_joint_cap_geometry import (
    JointStageOne, build_joint_ring_layout, build_outpaint_layout, coarse_joint_layout,
    compose_joint_stage1, ring_layout_from_metadata, ring_registration,
)
from dielinediff.methods.v3_joint_cap_ring import (
    CONTINUATION_LABEL, SIDE_LABEL, _early_redilation, _outpaint_weights,
    downsample_energy_retention, joint_weights, periodic_bicubic,
)
from dielinediff.methods.v3_joint_cap_stage2 import (
    STAGE2_OPTIONS, _initial_pixels, _layout, _native_y, _preserve_native, stage2_strength_map,
)
from dielinediff.methods.v3_capgeom_geometry import iter_frozen_cap_bleed
from dielinediff.methods.v3_circular import horizontal_circular


class JointCapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geom = build_geometry(pixels_per_mm=6.4)
        cls.topology = geom.topology
        cls.full = build_joint_ring_layout(cls.topology)
        with TemporaryDirectory() as folder:
            _write_labels(geom, Path(folder))
            with Image.open(Path(folder) / "region_map.png") as labels:
                cls.labels = np.array(labels)
            with Image.open(Path(folder) / "bleed_owner.png") as owners:
                cls.owners = np.array(owners)

    def test_geometry_wrap_layout_and_native_rows(self):
        topo, high = self.topology, self.full
        ring = high.ring
        small = coarse_joint_layout(topo, high, 0.70)
        self.assertEqual((ring.width_px, ring.height_px, ring.body_rows), (1280, 1536, (40, 152)))
        self.assertEqual((small.ring.width_px, small.ring.height_px, small.ring.body_rows),
                         (896, 1088, (28, 108)))
        self.assertEqual(high.cap_rects_px["T"], (248, 72, 640, 320))
        self.assertEqual(high.cap_rects_px["Btm"], (888, 1216, 1280, 1464))
        self.assertEqual(ring.face_rects["L"][0], 0)
        self.assertEqual(ring.face_rects["B"][2], 160)
        encoded = json.loads(json.dumps(ring_registration(ring)))
        restored = ring_layout_from_metadata(topo, encoded)
        self.assertEqual(restored.face_rects, ring.face_rects)
        np.testing.assert_array_equal(restored.region_ids, ring.region_ids)
        larger = build_joint_ring_layout(build_geometry(BoxSpec(width=72.0)).topology)
        self.assertGreater(larger.ring.body_rows[0], ring.body_rows[0])

    def test_f_and_b_soft_prompt_routing_crosses_no_native_fold(self):
        for plan in (self.full, coarse_joint_layout(self.topology, self.full, 0.70)):
            ring = plan.ring
            weights = joint_weights(ring)
            self.assertEqual(weights.dtype, np.float32)
            np.testing.assert_allclose(weights.sum(0), 1, rtol=0, atol=1e-7)
            lo, hi = ring.body_rows
            f, b = [ring.face_rects[face][0] + 3 for face in ("F", "B")]
            self.assertEqual(weights[CONTINUATION_LABEL, lo, f], 1)
            self.assertEqual(weights[CONTINUATION_LABEL, lo - 1, f], 1)
            self.assertEqual(weights[CONTINUATION_LABEL, hi - 1, b], 1)
            self.assertEqual(weights[CONTINUATION_LABEL, hi, b], 1)
            mid = (lo + hi) // 2
            self.assertEqual(weights[SIDE_LABEL["F"], mid, f], 1)
            self.assertEqual(weights[SIDE_LABEL["B"], mid, b], 1)
            feather = weights[SIDE_LABEL["F"], :, f]
            self.assertEqual(int(((feather > 0) & (feather < 1)).sum()), 24)

    def test_outpaint_uses_full_circular_width_and_aligned_r0_rows(self):
        ring = build_ring_layout(self.topology, 1280, 1024)
        top = build_outpaint_layout(self.topology, ring, "T")
        bottom = build_outpaint_layout(self.topology, ring, "Btm")
        self.assertEqual(top.size_px, (1280, 640))
        self.assertEqual(bottom.size_px, top.size_px)
        self.assertEqual(top.known_rows_latent, (40, 80))
        self.assertEqual(bottom.known_rows_latent, (0, 40))
        self.assertEqual(top.source_rows_latent, (8, 48))
        self.assertEqual(bottom.source_rows_latent, (80, 120))
        self.assertEqual(top.cap_rect_px, (248, 72, 640, 320))
        self.assertEqual(bottom.cap_rect_px, (888, 320, 1280, 568))
        for face, side in (("T", "F"), ("Btm", "B")):
            weights = _outpaint_weights(ring, face, 80)
            np.testing.assert_array_equal(weights.sum(0), 1)
            for other in SIDES:
                x = ring.face_rects[other][0]
                expected = CONTINUATION_LABEL if other == side else SIDE_LABEL[other]
                np.testing.assert_array_equal(weights.argmax(0)[:, x], expected)

    def test_cyclic_resize_commutes_with_exact_integer_roll(self):
        pixels = np.random.default_rng(20).integers(20, 235, (12, 32, 3), dtype=np.uint8)
        original = Image.fromarray(pixels)
        shifted = Image.fromarray(np.roll(pixels, 3, axis=1))
        doubled = periodic_bicubic(original, (64, 24))
        resized_shift = periodic_bicubic(shifted, (64, 24))
        np.testing.assert_array_equal(np.roll(np.asarray(doubled), 6, axis=1), resized_shift)
        retention = downsample_energy_retention(original, circular=True)
        self.assertGreater(retention["retention_ratio"], 0)
        self.assertLess(retention["retention_ratio"], 1)

    def test_stage2_rgb_and_physical_lock_for_both_caps(self):
        ring = self.full.ring
        y, x = np.indices((ring.height_px, ring.width_px))
        ring_rgb = Image.fromarray(np.stack((x % 251, y % 251, (x + y) % 251), -1).astype(np.uint8))
        caps = {name: ring_rgb.crop(rect) for name, rect in self.full.cap_rects_px.items()}
        proof = {**ring_registration(ring), "retained_side_buffer_px": 64}
        reference = JointStageOne(ring, ring_rgb, caps, proof)
        edited = {}
        fields = {}
        for face in CAPS:
            layout = _layout(self.topology, reference, face, 16, 1)
            original, frozen = _initial_pixels(reference, layout)
            x0, y0, x1, y1 = layout.center_rect
            np.testing.assert_array_equal(
                original.crop(tuple(8 * n for n in layout.center_rect)), caps[face]
            )
            known = layout.known_mask.repeat(8, 0).repeat(8, 1)
            np.testing.assert_array_equal(np.asarray(original)[known], np.asarray(frozen)[known])
            field = stage2_strength_map(layout, self.topology["box_mm"], settings=STAGE2_OPTIONS)
            self.assertTrue(np.all(field[layout.known_mask] == 0))
            field = fields[face] = field[y0:y1, x0:x1]
            h, w = field.shape
            distance = _native_y(face, h, 45)
            self.assertTrue(np.all(field[distance <= 15] == 0))
            self.assertEqual(field.max(), 0.70)
            self.assertEqual(field[h // 2, w // 2], 0.30)
            forged = Image.new("RGB", caps[face].size, (252, 255, 11))
            edited[face], lock = _preserve_native(caps[face], forged, face=face,
                                                box_mm=self.topology["box_mm"], settings=STAGE2_OPTIONS)
            self.assertEqual(lock["locked_native_pixels_changed"], 0)
            self.assertGreater(lock["locked_native_pixels"], 0)
        np.testing.assert_array_equal(fields["T"][::-1], fields["Btm"])
        a, support = compose_joint_stage1(self.topology, self.labels, self.owners, reference,
                                          return_support=True)
        b, after = compose_joint_stage1(self.topology, self.labels, self.owners,
                                        JointStageOne(ring, ring_rgb, edited, proof), return_support=True)
        np.testing.assert_array_equal(support, after)
        a, b = np.asarray(a), np.asarray(b)
        for name in SIDES + CAPS:
            x0, y0, x1, y1 = self.topology["faces"][name]["rect"]
            if name in CAPS:
                mask = _native_y(name, y1 - y0, 45) <= 15
                np.testing.assert_array_equal(a[y0:y1, x0:x1][mask], b[y0:y1, x0:x1][mask])
                self.assertTrue(np.any(a[y0:y1, x0:x1][~mask] != b[y0:y1, x0:x1][~mask]))
            else:
                np.testing.assert_array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
        for patch in iter_frozen_cap_bleed(self.topology, self.labels, self.owners):
            np.testing.assert_array_equal(a[patch.destination], b[patch.destination])

    def test_redilation_scopes_only_registered_early_resnet_convolutions(self):
        model = nn.Module()
        model.mid_block = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1, bias=False))
        model.other = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        layer = model.mid_block[0]
        phase = [False]
        source = torch.arange(49, dtype=layer.weight.dtype).reshape(1, 1, 7, 7)
        with horizontal_circular(model):
            plain = layer(source)
            with _early_redilation(model, phase) as keys:
                self.assertEqual(keys, ("mid_block.0",))
                np.testing.assert_array_equal(layer(source).detach(), plain.detach())
                phase[0] = True
                diff = layer(source)
                self.assertEqual(diff.shape, plain.shape)
                self.assertFalse(np.array_equal(diff.detach(), plain.detach()))
                self.assertEqual(layer.padding, (1, 1))
                self.assertEqual(model.other.dilation, (1, 1))
            np.testing.assert_array_equal(layer(source).detach(), plain.detach())
        self.assertNotIn("forward", layer.__dict__)
