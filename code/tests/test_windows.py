import numpy as np

from dielinediff.methods.windows import latent_rect


def test_latent_rect_matches_raster_sampling():
    rect = (4, 12, 28, 44)
    labels = np.zeros((64, 64), dtype=bool)
    labels[rect[1]:rect[3], rect[0]:rect[2]] = True
    sampled = labels[4::8, 4::8]
    x0, y0, x1, y1 = latent_rect(rect)
    expected = np.zeros_like(sampled)
    expected[y0:y1, x0:x1] = True
    np.testing.assert_array_equal(sampled, expected)
