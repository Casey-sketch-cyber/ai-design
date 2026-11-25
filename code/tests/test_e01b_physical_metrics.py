import numpy as np
from PIL import Image

from dielinediff.dieline.geometry import build_geometry, write_geometry
from dielinediff.metrics.e01b import preflight_atlas
from dielinediff.metrics.physical import PhysicalEvaluator, stripe_energy


def _physical_geometries(tmp_path):
    for name, pixels_per_mm in (("legacy", None), ("new", 6.4)):
        directory = tmp_path / name
        write_geometry(build_geometry(pixels_per_mm=pixels_per_mm), directory)
        yield directory


def test_common_scale_and_fixed_region_independence(tmp_path):
    common = []
    for directory in _physical_geometries(tmp_path):
        with Image.open(directory / "region_map.png") as label_image:
            labels = np.array(label_image)
        source = np.full((*labels.shape, 3), [36, 37, 38], dtype=np.uint8)
        source[np.isin(labels, [1, 2, 3, 4, 5, 6])] = [120, 145, 128]
        source[labels == 10] = [131, 154, 137]
        evaluator = PhysicalEvaluator(directory)
        assert len(evaluator.creases) == 5
        assert all(cap.sum() == 4 for cap in evaluator.caps.values())
        prepared = preflight_atlas(source, labels, "#315728")
        assert not np.array_equal(source, prepared)

        raw = evaluator.measure(source, lpips=False)
        assert raw == evaluator.measure(prepared, lpips=False)
        assert raw["stripe_index"] == raw["back_bottom_stripe_index"] == 1
        assert raw["seam_sdr"] == raw["top_seam_sdr"] == raw["bottom_seam_sdr"] == 0
        assert raw["bleed_fill_pct"] > 99.5
        common.append((evaluator.topology, len(evaluator.bleed_support)))
    assert common[0] == common[1]


def test_stripe_energy_is_sensitive_to_a_tangential_defect(tmp_path):
    directory = next(_physical_geometries(tmp_path))
    evaluator = PhysicalEvaluator(directory)
    faces = {}
    for name, face in evaluator.topology["faces"].items():
        x0, y0, x1, y1 = face["rect"]
        texture = np.full((y1 - y0, x1 - x0, 3), [141, 103, 59], dtype=np.uint8)
        texture[:, texture.shape[1] // 2:] = (70, 145, 135)
        faces[name] = texture
    assert stripe_energy(faces, evaluator.creases) == (1.0, 1.0)

    faces["Btm"][:12, ::2] = (10, 220, 245)
    faces["Btm"][:12, 1::2] = (232, 50, 65)
    whole, back_bottom = stripe_energy(faces, evaluator.creases)
    assert whole == back_bottom
    assert back_bottom > 20
