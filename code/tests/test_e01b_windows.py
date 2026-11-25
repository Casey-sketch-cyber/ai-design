"""Verify E01b geometry, cyclic orientation, and physical latent coverage."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dielinediff.dieline import build_geometry, write_geometry
from dielinediff.methods.windows_v2 import build_windows, save_coverage_heatmap


class E01bWindowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_geometry(pixels_per_mm=6.4)
        temp = tempfile.TemporaryDirectory(prefix="e01b-window-tests-")
        cls.addClassCleanup(temp.cleanup)
        cls.directory = write_geometry(cls.geometry, temp.name)
        with Image.open(cls.directory / "region_map.png") as image:
            cls.labels = np.asarray(image).copy()

    def plan(self, synchronized):
        return build_windows(self.geometry.topology, self.labels, synchronized=synchronized, device="cpu")

    def test_geometry_keeps_the_original_entry_point_and_aligns_e01b(self):
        legacy = build_geometry()
        self.assertEqual((legacy.topology["image"]["width"], legacy.topology["image"]["height"]), (1024, 1160))
        image = self.geometry.topology["image"]
        self.assertEqual((image["width"], image["height"], image["pixels_per_mm"]), (1592, 1792, 6.4))
        faces = self.geometry.topology["faces"]
        self.assertEqual(sum(faces[name]["rect"][2] - faces[name]["rect"][0] for name in "FRBL"), 1472)
        self.assertTrue(all(faces[name]["rect"][3] - faces[name]["rect"][1] == 1024 for name in "FRBL"))

    def test_cyclic_ring_includes_the_physical_glue_seam_and_lids(self):
        plan = self.plan(True)
        ring = [window for window in plan.windows if window.name.startswith("ring-")]
        self.assertEqual([window.primary_face for window in ring[::2]], ["R", "B", "F"])
        self.assertEqual(len(ring), 6)
        self.assertTrue(all(window.shape == (128, 128) for window in ring))
        width = plan.shape[1]
        face_rects = self.geometry.topology["faces"]
        x_blast = (face_rects["B"]["rect"][2] + 3) // 8 - 1
        x_lfirst = (face_rects["L"]["rect"][0] + 3) // 8
        self.assertTrue(any(torch.any((window.read_ids.reshape(window.shape)[:, :-1] % width == x_blast)
                                      & (window.read_ids.reshape(window.shape)[:, 1:] % width == x_lfirst))
                            for window in ring))
        for name in ("T", "Btm"):
            lid = next(window for window in plan.windows if window.name == f"lid-{name}")
            self.assertEqual(set(lid.faces), {name, "F", "R", "B", "L"})
            self.assertGreaterEqual(min(lid.shape), 64)

    def test_sync_and_independent_merge_cover_every_real_required_slot(self):
        for synchronized in (True, False):
            with self.subTest(synchronized=synchronized):
                plan = self.plan(synchronized)
                height, width = plan.shape
                pixels = torch.linspace(0, 1, height * width)[None, None]
                total = torch.zeros_like(pixels)
                count = torch.zeros(height * width)
                for window in plan.windows:
                    self.assertGreaterEqual(min(window.shape), 64)
                    self.assertTrue(torch.equal(window.read_ids[window.write_sources], window.write_ids))
                    if not synchronized:
                        mask = (plan.labels.ravel()[window.read_ids.numpy()] <= 6)
                        distinct = {int(i) for i in plan.labels.ravel()[window.read_ids.numpy()][mask] if i}
                        self.assertEqual(len(distinct), 1)
                    window.merge(window.gather(pixels), total, count)
                plan.assert_covered(count)
                self.assertEqual(int(plan.required.sum()), 29689)
                self.assertEqual(int(np.sum(plan.labels == 10)), 2105)
                self.assertTrue(torch.allclose((total[0, 0] / count)[plan.required], pixels[0, 0][plan.required],
                                               atol=2e-7))
                self.assertEqual(int(torch.count_nonzero(count[~plan.required])), 0)
                first_bleed = int(np.flatnonzero(plan.labels.ravel() == 10)[0])
                count[first_bleed] = 0
                with self.assertRaisesRegex(RuntimeError, "1 uncovered latent cells"):
                    plan.assert_covered(count)

    def test_heatmap_is_written(self):
        output = Path(self.directory) / "coverage-test.png"
        save_coverage_heatmap(self.plan(True), output)
        with Image.open(output) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, self.labels.shape[1] // 8)


if __name__ == "__main__":
    unittest.main()
