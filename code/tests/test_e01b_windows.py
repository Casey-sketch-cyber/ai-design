"""Verify E01b geometry, cyclic orientation, and physical latent coverage."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dielinediff.dieline import build_geometry, write_geometry
from dielinediff.dieline.geometry import EDGES, REGION_LABELS
from dielinediff.methods.ours_tf_v2 import window_prompt
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
            lid = next(window for window in plan.windows if window.name in (f"lid-{name}", f"frame-{name}"))
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
                    self.assertEqual(window.write_ids.unique().numel(), window.write_ids.numel())
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

    def test_every_lid_edge_matches_obj_direction_without_repeated_neighbor_rows(self):
        plan = self.plan(True)
        topology = self.geometry.topology
        faces = topology["faces"]
        rects = {name: np.asarray([(value + 3) // 8 for value in face["rect"]])
                 for name, face in faces.items()}
        neighbors = {}
        for common in topology["adjacencies"]:
            first, second = common["a"], common["b"]
            for current, other in ((first, second), (second, first)):
                neighbors[current["face"], current["edge"]] = other["face"], other["edge"]
        atlas_w = plan.shape[1]
        physical_bleed_slots = 0
        for name in ("T", "Btm"):
            window = next(window for window in plan.windows if window.name in (f"lid-{name}", f"frame-{name}"))
            ids = window.read_ids.numpy().reshape(window.shape)
            write = np.zeros(ids.size, dtype=bool)
            write[window.write_sources.numpy()] = True
            write = write.reshape(window.shape)
            region = plan.labels.ravel()[ids]
            read = ids
            x0, y0, x1, y1 = rects[name]
            crop_w, crop_h = x1 - x0, y1 - y0
            margin = (read.shape[0] - crop_h) // 2

            def local_line(edge, along):
                depths = np.arange(margin)
                if edge == "top":
                    return along, -depths - 1
                if edge == "right":
                    return crop_w + depths, along
                if edge == "bottom":
                    return along, crop_h + depths
                return -depths - 1, along

            for edge, (start, end) in EDGES.items():
                mate, mate_edge = neighbors[name, edge]
                q0, q1 = EDGES[mate_edge]
                p_start, p_end = (faces[name]["corners_mm"][i] for i in (start, end))
                q_start, q_end = (faces[mate]["corners_mm"][i] for i in (q0, q1))
                if p_start == q_start:
                    self.assertEqual(p_end, q_end)
                    reverse = False
                else:
                    self.assertEqual((p_start, p_end), (q_end, q_start))
                    reverse = True
                length = crop_w if edge in ("top", "bottom") else crop_h
                for i in range(length):
                    dx, dy = local_line(edge, i)
                    segment = read[dy + margin, dx + margin]
                    write_segment = write[dy + margin, dx + margin]
                    mate_mask = plan.labels.ravel()[segment] == REGION_LABELS[mate]
                    self.assertGreaterEqual(int(mate_mask.sum()), 16, (name, edge, i))
                    ax, ay = segment[mate_mask] % atlas_w, segment[mate_mask] // atlas_w
                    rx0, ry0, rx1, ry1 = rects[mate]
                    if mate_edge == "top":
                        along, depth, count = ax - rx0, ay - ry0, rx1 - rx0
                    elif mate_edge == "right":
                        along, depth, count = ay - ry0, rx1 - 1 - ax, ry1 - ry0
                    elif mate_edge == "bottom":
                        along, depth, count = ax - rx0, ry1 - 1 - ay, rx1 - rx0
                    else:
                        along, depth, count = ay - ry0, ax - rx0, ry1 - ry0
                    u = (i + 0.5) / length
                    expected = min(int((1 - u if reverse else u) * count), count - 1)
                    np.testing.assert_array_equal(along, expected)
                    np.testing.assert_array_equal(depth, np.arange(len(depth)))
                    native_pair = (name, mate) in (("T", "F"), ("Btm", "B"))
                    if native_pair:
                        np.testing.assert_array_equal(write_segment[mate_mask], depth < 2)
                    else:
                        self.assertFalse(write_segment[mate_mask].any())
                    native_x, native_y = np.broadcast_arrays(x0 + dx, y0 + dy)
                    for d, (nx, ny) in enumerate(zip(native_x[:3], native_y[:3], strict=True)):
                        if 0 <= nx < atlas_w and 0 <= ny < plan.shape[0] and plan.labels[ny, nx] == 10:
                            self.assertEqual(segment[d], ny * atlas_w + nx)
                            self.assertTrue(write_segment[d])
                            physical_bleed_slots += 1

            # Both windows use atlas direction. The bottom's viewing rotation remains in the OBJ metadata.
            core = region == REGION_LABELS[name]
            cy, cx = np.nonzero(core)
            y, x = (cy.min() + cy.max()) // 2, (cx.min() + cx.max()) // 2
            tl, tr, _, bl = np.asarray(faces[name]["corners_mm"])

            def position(source_id):
                xx = (source_id % atlas_w + 0.5 - x0) / crop_w
                yy = (source_id // atlas_w + 0.5 - y0) / crop_h
                return tl + xx * (tr - tl) + yy * (bl - tl)

            base = position(ids[y, x])
            right = position(ids[y, x + 1]) - base
            down = position(ids[y + 1, x]) - base
            self.assertGreater(right[0] * (1 if name == "T" else -1), 0)
            self.assertGreater(np.cross(right, -down)[1] * (1 if name == "T" else -1), 0)
            self.assertEqual(faces[name]["upright_rotation"], 0 if name == "T" else 180)
        self.assertGreater(physical_bleed_slots, 0)

    def test_lid_and_ring_share_only_the_physical_transition_width(self):
        atlas = self.geometry.topology
        for transition_mm, rows in ((1.0, 1), (2.0, 2), (5.0, 4)):
            with self.subTest(transition_mm=transition_mm):
                plan = build_windows(atlas, self.labels, device="cpu", transition_mm=transition_mm)
                _, atlas_w = plan.shape
                rings = [window for window in plan.windows if window.name.startswith("ring-")]
                ring_writes = np.unique(np.concatenate([window.write_ids.numpy() for window in rings]))
                for face, other, edge, name in (("T", "F", 3, "lid-T"), ("Btm", "B", 1, "frame-Btm")):
                    lid = next(window for window in plan.windows if window.name == name)
                    rect = atlas["faces"][face]["rect"]
                    x0, _, x1, _ = ((value + 3) // 8 for value in rect)
                    fold = (rect[edge] + 3) // 8
                    ys, xs = np.mgrid[fold - rows:fold + rows, x0:x1]
                    expected = ys.ravel() * atlas_w + xs.ravel()
                    shared = np.intersect1d(lid.write_ids.numpy(), ring_writes)
                    print_labels = plan.labels.ravel()[shared]
                    shared = shared[np.isin(print_labels, [REGION_LABELS[face], REGION_LABELS[other]])]
                    np.testing.assert_array_equal(shared, expected)
                    for window in [*rings, lid]:
                        ids = window.read_ids.numpy().reshape(window.shape)
                        raw_ids = window.write_ids.numpy()
                        selected = np.isin(raw_ids, expected)
                        write_sources = window.write_sources.numpy()[selected]
                        actual = raw_ids[selected]
                        wy, wx = np.divmod(write_sources, ids.shape[1])
                        ax = actual % atlas_w
                        forward = (ax < x1 - 1) & (wx < ids.shape[1] - 1)
                        backward = (ax > x0) & (wx > 0)
                        down = wy < ids.shape[0] - 1
                        up = wy > 0
                        np.testing.assert_array_equal(ids[wy[forward], wx[forward] + 1], actual[forward] + 1)
                        np.testing.assert_array_equal(ids[wy[backward], wx[backward] - 1], actual[backward] - 1)
                        np.testing.assert_array_equal(ids[wy[down] + 1, wx[down]], actual[down] + atlas_w)
                        np.testing.assert_array_equal(ids[wy[up] - 1, wx[up]], actual[up] - atlas_w)
                    # Moving deeper into the body decreases the contribution from its neighboring lid.
                    if rows > 1:
                        x_mid = (x0 + x1) // 2
                        depth_one = ((fold if face == "T" else fold - 1) * atlas_w) + x_mid
                        depth_two = depth_one + atlas_w * (1 if face == "T" else -1)
                        weights = dict(zip(lid.write_ids.tolist(), lid.write_weights.tolist(), strict=True))
                        self.assertGreater(weights[depth_one], weights[depth_two])

    def test_physical_side_folds_share_native_coordinates_and_ring_wrap_survives(self):
        plan = self.plan(True)
        atlas = self.geometry.topology
        rings = [window for window in plan.windows if window.name.startswith("ring-")]
        atlas_w = plan.shape[1]
        centers = (np.arange(atlas_w) + 0.5) * 8
        for left, right in (("L", "F"), ("F", "R"), ("R", "B")):
            edge = atlas["faces"][left]["rect"][2]
            self.assertEqual(edge, atlas["faces"][right]["rect"][0])
            xs = np.flatnonzero(np.abs(centers - edge) < 2 * atlas["image"]["pixels_per_mm"])
            top, bottom = ((value + 3) // 8 for value in atlas["faces"][left]["rect"][1::2])
            self.assertGreaterEqual(len(xs), 2)
            for row in (top, (top + bottom) // 2, bottom - 1):
                expected = row * atlas_w + xs
                covered = [window for window in rings if np.isin(expected, window.write_ids.numpy()).all()]
                self.assertTrue(covered)
                for window in covered:
                    gather = window.read_ids.numpy().reshape(window.shape)
                    start = np.argwhere(gather == expected[0])[0]
                    r, column = map(int, start)
                    np.testing.assert_array_equal(gather[r, column:column + len(xs)], expected)

    def test_independent_bottom_reflects_owned_art_with_no_cross_face_context(self):
        plan = self.plan(False)
        bottom = next(window for window in plan.windows if window.name == "face-Btm")
        atlas_w = plan.shape[1]
        x0, y0, x1, _ = ((value + 3) // 8 for value in self.geometry.topology["faces"]["Btm"]["rect"])
        x = (x0 + x1) // 2
        self.assertEqual(plan.labels[y0 - 1, x], REGION_LABELS["B"])
        ids = bottom.read_ids.numpy().reshape(bottom.shape)
        owned = y0 * atlas_w + x
        source = bottom.write_sources[bottom.write_ids == owned].item()
        row, col = divmod(source, ids.shape[1])
        self.assertGreater(row, 0)
        self.assertEqual(ids[row, col + 1], owned + 1)
        self.assertEqual(ids[row - 1, col], (y0 + 1) * atlas_w + x)

    def test_bottom_prompt_is_auxiliary_and_has_no_product_name(self):
        product = "TEST-BOX-PRODUCT-NAME"
        brief = {
            "category": "green_tea",
            "front_prompt": f"front portrait and {product}",
            "side_prompt": f"winding river foliage, {product}",
            "top_prompt": "a centered upright crest",
            "continuous": True,
            "main_color": "#527555",
            "product_name_zh": product,
        }
        prompt = window_prompt(brief, "Btm")
        self.assertIn("winding river foliage", prompt)
        self.assertIn("green tea packaging motifs", prompt)
        self.assertIn("#527555", prompt)
        self.assertNotIn(product, prompt)
        self.assertNotIn("a centered upright crest", prompt)
        self.assertNotIn("front portrait", prompt)

    def test_heatmap_is_written(self):
        output = Path(self.directory) / "coverage-test.png"
        save_coverage_heatmap(self.plan(True), output)
        with Image.open(output) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, self.labels.shape[1] // 8)


if __name__ == "__main__":
    unittest.main()
