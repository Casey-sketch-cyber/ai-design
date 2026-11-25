"""Geometry, UV registration and actual folded-view regression tests."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image
from shapely.geometry import Point

from dielinediff.dieline import BoxSpec, build_geometry, write_geometry
from dielinediff.dieline.geometry import EDGES, REGION_LABELS
from dielinediff.render.orthographic import VIEWS, load_mesh
from dielinediff.render.selfcheck import write_selfcheck


class DielineTest(unittest.TestCase):
    def test_rte_parameters_and_vector_layers(self):
        geometry = build_geometry(BoxSpec(length=84, width=54, height=180, bleed=4), short_side=320)
        a = build_geometry(BoxSpec(paper_thickness=0.35))
        b = build_geometry(BoxSpec(paper_thickness=0.75))
        self.assertGreater(a.outline.area, b.outline.area)
        self.assertEqual(a.faces_mm, b.faces_mm)
        left, top, _, bottom = geometry.faces_mm["L"]
        bleed_point = Point(left - 4, (top + bottom) / 2)
        self.assertAlmostEqual(geometry.bleed_outline.boundary.distance(bleed_point), 0)
        self.assertGreater(geometry.faces_mm["T"][1], 0)
        self.assertGreater(geometry.faces_mm["Btm"][1], geometry.faces_mm["B"][1])
        with self.assertRaises(ValueError):
            build_geometry(BoxSpec(safety=25))
        with self.assertRaises(ValueError):
            build_geometry(short_side=1003)
        with tempfile.TemporaryDirectory() as directory:
            directory = write_geometry(geometry, directory)
            svg = ElementTree.parse(directory / "dieline.svg")
            layers = {element.attrib["id"] for element in svg.getroot()}
            self.assertEqual(layers, {"cut", "crease", "bleed", "safe"})

    def test_atlas_and_mesh_use_the_same_faces_and_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = write_geometry(build_geometry(short_side=384), directory)
            topology = json.loads((directory / "topology.json").read_text())
            width, height = topology["image"]["width"], topology["image"]["height"]
            with Image.open(directory / "region_map.png") as image:
                labels = np.array(image)
            with Image.open(directory / "bleed_owner.png") as image:
                bleed_owner = np.array(image)
            mesh = load_mesh(directory / "box.obj")
            self.assertEqual(labels.shape, (height, width))
            self.assertEqual(set(np.unique(labels)), set(REGION_LABELS.values()))
            np.testing.assert_array_equal(bleed_owner > 0, labels == REGION_LABELS["Bleed"])
            self.assertEqual(len(mesh.faces), 12)
            self.assertEqual(len(np.unique(mesh.vertices, axis=0)), 8)
            for name, face in topology["faces"].items():
                x0, y0, x1, y1 = face["rect"]
                self.assertTrue((labels[y0:y1, x0:x1] == REGION_LABELS[name]).all())
                self.assertAlmostEqual(face["uv"][0], x0 / width)
                self.assertAlmostEqual(face["uv"][1], 1 - y1 / height)
                self.assertAlmostEqual(face["uv"][2], x1 / width)
                self.assertAlmostEqual(face["uv"][3], 1 - y0 / height)
                outward, _ = VIEWS[name]
                outward_triangles = mesh.faces[np.all(np.isclose(mesh.face_normals, outward), axis=1)]
                self.assertEqual(len(outward_triangles), 2)
                uv = mesh.visual.uv[outward_triangles].reshape(-1, 2)
                np.testing.assert_allclose(np.concatenate((uv.min(0), uv.max(0))), face["uv"])
            endpoints = set()
            for edge in topology["adjacencies"]:
                first, second = edge["a"], edge["b"]
                a = topology["faces"][first["face"]]["corners_mm"]
                b = topology["faces"][second["face"]]["corners_mm"]
                common = tuple(tuple(a[index]) for index in EDGES[first["edge"]])
                opposite = tuple(tuple(b[index]) for index in EDGES[second["edge"]])
                self.assertEqual(common, opposite[::-1] if edge["reverse"] else opposite)
                endpoints.add(frozenset(common))
            self.assertEqual(len(endpoints), 12)

    def test_arrow_and_checker_pass_in_the_rendered_views(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            geometry = write_geometry(build_geometry(short_side=384), directory / "geometry")
            results = write_selfcheck(geometry, directory / "renders")
            self.assertEqual(set(results["upright"]), set(VIEWS) - {"iso"})
            self.assertTrue(all(results["upright"].values()))
            self.assertEqual(len(results["seams"]), 12)
            self.assertGreater(min(edge["checker_pixel_agreement"] for edge in results["seams"]), 0.97)


if __name__ == "__main__":
    unittest.main()
