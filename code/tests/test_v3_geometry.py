"""Verify the E02 cyclic ring, frozen cap strips, and print registration on CPU."""

import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dielinediff.dieline import build_geometry, write_geometry
from dielinediff.dieline.geometry import EDGES, REGION_LABELS
from dielinediff.methods.v3_geometry import (
    CAPS,
    SIDES,
    atlas_canvas_size,
    atlas_region_ids,
    build_cap_layout,
    build_ring_layout,
    compose_atlas,
    compose_support_counts,
)


SCALES = ((1280, 1024), (1472, 1152))


class V3GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geo = build_geometry(pixels_per_mm=6.4)
        cls.topology = geo.topology
        storage = tempfile.TemporaryDirectory()
        cls.addClassCleanup(storage.cleanup)
        folder = write_geometry(geo, storage.name)
        with Image.open(folder / "region_map.png") as image:
            cls.labels = np.asarray(image).copy()
        with Image.open(folder / "bleed_owner.png") as image:
            cls.bleed_owner = np.asarray(image).copy()

    def test_flat_region_ids_maintain_the_atlas_origin(self):
        self.assertEqual(atlas_canvas_size(self.labels), (1600, 1792))
        ids = atlas_region_ids(self.labels)
        expected = self.labels[4::8, 4::8]
        expected = np.where(expected <= 6, expected, 0)
        np.testing.assert_array_equal(ids[:, :199], expected)
        self.assertTrue((ids[:, 199:] == 0).all())

    def test_ring_uses_true_open_buffers_and_cyclic_face_order(self):
        for w, h in SCALES:
            with self.subTest(pixels=(w, h)):
                ring = build_ring_layout(self.topology, w, h)
                self.assertEqual(ring.latent_shape, (h // 8, w // 8))
                self.assertEqual(tuple(ring.face_rects), SIDES)
                self.assertEqual(ring.body_rows, (8, h // 8 - 8))
                expected_mm = 80 / 7 if w == 1280 else 10
                np.testing.assert_allclose(ring.buffer_mm, [expected_mm] * 2)
                prev = 0
                for name in SIDES:
                    x0, y0, x1, y1 = ring.face_rects[name]
                    self.assertEqual(x0, prev)
                    self.assertEqual((y0, y1), ring.body_rows)
                    self.assertTrue((ring.region_ids[:, x0:x1] == REGION_LABELS[name]).all())
                    prev = x1
                self.assertEqual(prev, w // 8)
                self.assertEqual(ring.region_ids[0, 0], REGION_LABELS["L"])
                self.assertEqual(ring.region_ids[0, -1], REGION_LABELS["B"])

    def test_all_cap_strips_follow_obj_endpoints_and_cannot_modify_the_ring(self):
        neighbors = {}
        for seam in self.topology["adjacencies"]:
            a, b = seam["a"], seam["b"]
            neighbors[a["face"], a["edge"]] = (b["face"], b["edge"])
            neighbors[b["face"], b["edge"]] = (a["face"], a["edge"])
        faces = self.topology["faces"]

        def edge_points(face, edge):
            return [faces[face]["corners_mm"][i] for i in EDGES[edge]]

        for w, h in SCALES:
            ring = build_ring_layout(self.topology, w, h)
            pixels = torch.arange(np.prod(ring.latent_shape), dtype=torch.float32).reshape(1, 1, *ring.latent_shape)
            pristine = pixels.clone()
            for face in CAPS:
                with self.subTest(pixels=(w, h), face=face):
                    cap = build_cap_layout(self.topology, ring, face)
                    gathered = cap.read_frozen_ring(pixels)
                    self.assertGreaterEqual(min(cap.width_px, cap.height_px), 512)
                    x0, y0, x1, y1 = cap.center_rect
                    self.assertEqual(int(cap.known_mask.sum()), 32 * ((x1 - x0) + (y1 - y0)))
                    self.assertFalse(cap.known_mask[y0:y1, x0:x1].any())
                    yy, xx = np.indices(cap.latent_shape)
                    corners = ((xx < x0) | (xx >= x1)) & ((yy < y0) | (yy >= y1))
                    self.assertFalse(cap.known_mask[corners].any())
                    self.assertFalse(gathered[0, 0].numpy()[~cap.known_mask].any())
                    self.assertEqual(len(np.unique(cap.ring_ids[cap.known_mask])), int(cap.known_mask.sum()))
                    for edge in EDGES:
                        side, side_edge = neighbors[face, edge]
                        a, b = edge_points(face, edge), edge_points(side, side_edge)
                        reverse = a != b
                        self.assertEqual(a, b[::-1] if reverse else b)
                        halo = cap.bleed_cells[edge]
                        native = (face, edge) in (("T", "bottom"), ("Btm", "top"))
                        self.assertEqual(halo, 0 if native else 3)
                        span = x1 - x0 if edge in ("top", "bottom") else y1 - y0
                        along = np.arange(span)
                        near = 15 - np.arange(16)[:, None]
                        outward = near + halo
                        if edge == "top":
                            cx, cy = x0 + along, y0 - 1 - outward
                        elif edge == "right":
                            cx, cy = x1 + outward, y0 + along
                        elif edge == "bottom":
                            cx, cy = x0 + along, y1 + outward
                        else:
                            cx, cy = x0 - 1 - outward, y0 + along
                        self.assertTrue(cap.known_mask[cy, cx].all())
                        indices = gathered[0, 0].numpy()[cy, cx].astype(int)
                        rsx, rsy = indices % ring.latent_shape[1], indices // ring.latent_shape[1]
                        rx0, ry0, _, ry1 = ring.face_rects[side]
                        expected_x = rx0 + (span - 1 - along if reverse else along)
                        expected_y = ry0 + near if side_edge == "top" else ry1 - 1 - near
                        np.testing.assert_array_equal(rsx, np.broadcast_to(expected_x, rsx.shape))
                        np.testing.assert_array_equal(rsy, np.broadcast_to(expected_y, rsy.shape))
                    gathered.fill_(-1)
                    self.assertTrue(torch.equal(pixels, pristine))

    def test_compositing_preserves_ring_and_exports_only_free_cap_pixels(self):
        labels = self.labels
        for w, h in SCALES:
            with self.subTest(pixels=(w, h)):
                ring = build_ring_layout(self.topology, w, h)
                r = np.full((h, w, 3), (22, 80, 106), dtype=np.uint8)
                caps = {}
                for name in CAPS:
                    cap = build_cap_layout(self.topology, ring, name)
                    pixels = np.full((cap.height_px, cap.width_px, 3), (30, 122, 48), dtype=np.uint8)
                    yy, xx = np.indices(cap.latent_shape)
                    x0, y0, x1, y1 = cap.center_rect
                    corners = ((xx < x0) | (xx >= x1)) & ((yy < y0) | (yy >= y1))
                    never_export = corners | cap.known_mask
                    pixels[never_export.repeat(8, 0).repeat(8, 1)] = (255, 0, 255)
                    caps[name] = cap, pixels
                ring_only = np.asarray(compose_atlas(self.topology, labels, self.bleed_owner, ring, r, {}))
                atlas, support = compose_atlas(
                    self.topology, labels, self.bleed_owner, ring, r, caps, return_support=True,
                )
                atlas = np.asarray(atlas)
                sides = np.isin(support, [REGION_LABELS[name] for name in SIDES])
                np.testing.assert_array_equal(atlas[sides], ring_only[sides])
                cap_pixels = np.isin(support, [REGION_LABELS[name] for name in CAPS])
                self.assertTrue((atlas[cap_pixels] == (30, 122, 48)).all())
                self.assertTrue((atlas[support == 0] == 255).all())
                counts = compose_support_counts(labels, self.bleed_owner, support)
                for name in SIDES + CAPS:
                    region = counts["regions"][name]
                    self.assertEqual(region["supported"], region["total"])
                self.assertEqual(counts["regions"]["Glue"]["supported"], counts["regions"]["Glue"]["total"])
                bleed = counts["regions"]["Bleed"]
                self.assertGreater(bleed["supported"], 80_000)
                self.assertLess(bleed["supported"], bleed["total"])
                for cap in CAPS:
                    self.assertGreater(counts["bleed_owners"][cap]["supported"], 0)


if __name__ == "__main__":
    unittest.main()
