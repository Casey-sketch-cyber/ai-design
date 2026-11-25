import numpy as np
import pytest
from PIL import Image

from dielinediff.metrics.e01b import preflight_atlas


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
