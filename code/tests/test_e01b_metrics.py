import numpy as np
import pytest
from PIL import Image

from dielinediff.metrics.e01b import b2_post_atlas, preflight_atlas


def test_preflight_changes_only_fixed_labels_and_preserves_raw_pixels(tmp_path):
    labels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    raw = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    source = tmp_path / "native.png"
    Image.fromarray(raw).save(source)
    original_file = source.read_bytes()

    result = preflight_atlas(source, labels, "#42765a")

    np.testing.assert_array_equal(result[~np.isin(labels, [0, 7, 8, 9])],
                                  raw[~np.isin(labels, [0, 7, 8, 9])])
    np.testing.assert_array_equal(result[np.isin(labels, [0, 7])], [[255, 255, 255]] * 2)
    np.testing.assert_array_equal(result[np.isin(labels, [8, 9])], [[66, 118, 90]] * 2)
    assert source.read_bytes() == original_file
    assert not np.shares_memory(preflight_atlas(raw, labels, "#42765a"), raw)


def test_preflight_uses_white_paper_alpha_compositing():
    image = Image.new("RGBA", (2, 1), (0, 0, 0, 128))
    result = preflight_atlas(image, np.array([[1, 7]], dtype=np.uint8), "#000000")
    np.testing.assert_array_equal(result[0], [[127, 127, 127], [255, 255, 255]])


def test_preflight_rejects_resizing_and_transparent_brief_colors():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="native"):
        preflight_atlas(rgb, np.zeros((2, 2)), "#447722")
    with pytest.raises(ValueError, match="opaque RGB"):
        preflight_atlas(rgb, np.zeros((4, 4)), "#44772280")

def test_b2_post_applies_the_rgb_lut_once_before_shared_preflight(tmp_path):
    labels = np.array([[1, 0, 7, 8, 9, 10]], dtype=np.uint8)
    rgb = np.array([[[41, 66, 138], [90, 18, 133], [18, 37, 81],
                     [100, 250, 20], [130, 77, 51], [134, 196, 52]]], dtype=np.uint8)
    source = tmp_path / "b2.png"
    Image.fromarray(rgb).save(source)
    original_file = source.read_bytes()
    lut = tmp_path / "color.npy"
    axis = np.linspace(0, 1, 33, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    np.save(lut, np.stack((g, b, r), axis=-1))

    raw = b2_post_atlas(source, lut)
    np.testing.assert_array_equal(raw, rgb[..., [1, 2, 0]])

    prepared = preflight_atlas(raw, labels, "#123456")
    np.testing.assert_array_equal(prepared[0, [0, 5]], raw[0, [0, 5]])
    np.testing.assert_array_equal(prepared[0, 1:3], [[255, 255, 255]] * 2)
    np.testing.assert_array_equal(prepared[0, 3:5], [[18, 52, 86]] * 2)
    assert source.read_bytes() == original_file
