"""Check G1 edge attachment and direct physical bleed sampling on CPU."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from dielinediff.dieline import build_geometry
from dielinediff.dieline.geometry import REGION_LABELS, _write_labels
from dielinediff.methods.v3_capfix_geometry import build_capfix_layout, cap_at_original_scale
from dielinediff.methods.v3_capgeom_geometry import build_capgeom_layout, compose_capgeom_atlas
from dielinediff.methods.v3_geometry import build_cap_layout, build_ring_layout, compose_atlas


EDGES = ("top", "right", "bottom", "left")
NEIGHBORS = {
    "T": {"top": ("B", "top", True), "right": ("R", "top", True),
          "bottom": ("F", "top", False), "left": ("L", "top", False)},
    "Btm": {"top": ("B", "bottom", False), "right": ("L", "bottom", False),
            "bottom": ("F", "bottom", True), "left": ("R", "bottom", True)},
}
NATIVE = {"T": "bottom", "Btm": "top"}


def outer_strip(array, rect, edge, depth, shift=0):
    """View a strip with outside distance on the first axis and edge order on the second."""
    x0, y0, x1, y1 = rect
    if edge == "top":
        return array[y0 - shift - depth:y0 - shift, x0:x1][::-1]
    if edge == "right":
        return array[y0:y1, x1 + shift:x1 + shift + depth].swapaxes(0, 1)
    if edge == "bottom":
        return array[y1 + shift:y1 + shift + depth, x0:x1]
    return array[y0:y1, x0 - shift - depth:x0 - shift][:, ::-1].swapaxes(0, 1)


def coordinate_rgb(height, width):
    y, x = np.indices((height, width))
    return np.stack((x % 251, y % 251, (x // 31 + y // 43) % 251), axis=-1).astype(np.uint8)


class TightGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_geometry(pixels_per_mm=6.4)
        cls.topology = cls.geometry.topology
        cls.ring = build_ring_layout(cls.topology, 1280, 1024)
        cls.rgb = coordinate_rgb(cls.ring.height_px, cls.ring.width_px)

    def test_g1_moves_each_coordinate_coded_strip_directly_to_the_same_core(self):
        y, x = np.indices(self.ring.latent_shape)
        coordinates = torch.from_numpy(np.stack((x, y)).astype(np.float32))[None]
        for face in NEIGHBORS:
            original = build_cap_layout(self.topology, self.ring, face)
            for scale in (1, 2):
                with self.subTest(face=face, scale=scale):
                    g0 = build_capfix_layout(original, self.ring, 16, scale)
                    g1 = build_capgeom_layout(original, self.ring, 16, scale)
                    self.assertEqual(g0.center_rect, g1.center_rect)
                    self.assertEqual((g0.width_px, g0.height_px), (g1.width_px, g1.height_px))
                    self.assertTrue(all(halo == 0 for halo in g1.bleed_cells.values()))
                    old_values = g0.read_frozen_ring(coordinates)[0].permute(1, 2, 0).numpy()
                    new_values = g1.read_frozen_ring(coordinates)[0].permute(1, 2, 0).numpy()
                    coverage = np.zeros(g1.latent_shape, dtype=bool)
                    for edge in EDGES:
                        depth = 16 * scale
                        view = outer_strip(coverage, g1.center_rect, edge, depth)
                        self.assertEqual(view.shape[0], depth)
                        self.assertFalse(view.any())
                        view[:] = True
                        old_strip = outer_strip(old_values, g0.center_rect, edge, depth, g0.bleed_cells[edge])
                        new_strip = outer_strip(new_values, g1.center_rect, edge, depth)
                        np.testing.assert_array_equal(new_strip, old_strip)
                    np.testing.assert_array_equal(g1.known_mask, coverage)
                    self.assertFalse(new_values[~coverage].any())
                    base_ids = g1.original.ring_ids[g1.original.known_mask]
                    self.assertEqual(np.unique(base_ids).size, base_ids.size)
                    self.assertEqual(set(base_ids), set(original.ring_ids[original.known_mask]))

    def test_g1_reads_first_rgb_row_of_each_true_neighbor_in_folded_order(self):
        for face, known_neighbors in NEIGHBORS.items():
            old = build_cap_layout(self.topology, self.ring, face)
            for scale in (1, 2):
                layout = build_capgeom_layout(old, self.ring, 16, scale)
                known = np.asarray(layout.read_frozen_ring_rgb(Image.fromarray(self.rgb)))
                core = tuple(8 * value for value in layout.center_rect)
                for edge, (side, side_edge, reverse) in known_neighbors.items():
                    with self.subTest(face=face, scale=scale, edge=edge):
                        x0, y0, x1, y1 = (8 * value for value in self.ring.face_rects[side])
                        expected = self.rgb[y0 if side_edge == "top" else y1 - 1, x0:x1]
                        if reverse:
                            expected = expected[::-1]
                        actual = outer_strip(known, core, edge, 1)[0]
                        np.testing.assert_array_equal(actual[[0, -1]], expected[[0, -1]])
                        if scale == 1:
                            np.testing.assert_array_equal(actual, expected)
                mask = layout.known_mask.repeat(8, 0).repeat(8, 1)
                self.assertFalse(known[~mask].any())

    def test_cap_bleed_bytes_match_physical_neighbors_and_only_fill_three_owned_strips_per_cap(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            _write_labels(self.geometry, destination)
            with Image.open(destination / "region_map.png") as image:
                labels = np.array(image)
            with Image.open(destination / "bleed_owner.png") as image:
                owners = np.array(image)
        ring_rgb = Image.fromarray(self.rgb)
        cap_core = (37, 76, 102)
        caps = {}
        for face in NEIGHBORS:
            layout = build_capgeom_layout(build_cap_layout(self.topology, self.ring, face), self.ring, 16)
            # Fill generated context and corners with a sentinel to detect any accidental export.
            artwork = Image.new("RGB", (layout.width_px, layout.height_px), (255, 0, 255))
            artwork.paste(cap_core, tuple(8 * value for value in layout.center_rect))
            caps[face] = cap_at_original_scale(layout, artwork)
        base, base_support = compose_atlas(self.topology, labels, owners, self.ring, ring_rgb, caps,
                                           return_support=True)
        actual, actual_support = compose_capgeom_atlas(self.topology, labels, owners, self.ring,
                                                       ring_rgb, caps, return_support=True)
        frozen_pixels = np.asarray(base)
        result = np.asarray(actual)
        new_support = np.zeros_like(base_support)
        cap_counts = {}
        for cap, neighbors in NEIGHBORS.items():
            rect = self.topology["faces"][cap]["rect"]
            cap_tag = REGION_LABELS[cap]
            np.testing.assert_array_equal(result[labels == cap_tag],
                                          np.broadcast_to(cap_core, result[labels == cap_tag].shape))
            for edge, (side, side_edge, reverse) in neighbors.items():
                marker = outer_strip(new_support, rect, edge, 19)
                permitted = (outer_strip(labels, rect, edge, 19) == REGION_LABELS["Bleed"]) & (
                    outer_strip(owners, rect, edge, 19) == cap_tag)
                if edge == NATIVE[cap]:
                    self.assertFalse(permitted.any())
                    continue
                self.assertFalse(marker.any())
                marker[permitted] = cap_tag
                self.assertTrue(permitted.any())
                cap_counts[cap, edge] = int(permitted.sum())
                sx0, sy0, sx1, sy1 = self.topology["faces"][side]["rect"]
                side_bytes = frozen_pixels[sy0:sy0 + 19, sx0:sx1] if side_edge == "top" else (
                    frozen_pixels[sy1 - 19:sy1, sx0:sx1][::-1])
                if reverse:
                    side_bytes = side_bytes[:, ::-1]
                dst = outer_strip(result, rect, edge, 19)
                np.testing.assert_array_equal(dst[permitted], side_bytes[permitted])
        self.assertEqual(cap_counts, {
            ("T", "top"): 134, ("T", "right"): 4803, ("T", "left"): 4858,
            ("Btm", "right"): 5472, ("Btm", "bottom"): 134, ("Btm", "left"): 4858,
        })
        untouched = new_support == 0
        np.testing.assert_array_equal(result[untouched], frozen_pixels[untouched])
        np.testing.assert_array_equal(actual_support, np.where(untouched, base_support, new_support))
        self.assertFalse(np.equal(result, (255, 0, 255)).all(2).any())


if __name__ == "__main__":
    unittest.main()
