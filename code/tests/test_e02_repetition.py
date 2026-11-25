"""Registration and cosine invariants for E02's DINOv2 repetition score."""

import numpy as np
from PIL import Image
import torch

from dielinediff.metrics.e02_repetition import crop_regions, summarize_embeddings


def test_crop_regions_follows_physical_panels_and_odd_front_middle():
    rgb = np.zeros((3, 12, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(12)[None]
    topology = {"faces": {
        "B": {"rect": [9, 0, 12, 3]}, "F": {"rect": [2, 0, 7, 3]},
        "R": {"rect": [7, 0, 9, 3]}, "L": {"rect": [0, 0, 2, 3]},
    }}
    crops = crop_regions(Image.fromarray(rgb), topology)
    assert [image.size for image in crops] == [(2, 3), (5, 3), (2, 3), (3, 3), (2, 3), (3, 3)]
    assert [np.asarray(image)[0, 0, 0].item() for image in crops] == [0, 2, 7, 9, 2, 4]


def test_repetitions_score_higher_than_different_subjects():
    same = torch.ones((6, 6))
    different = torch.eye(6)
    actual = summarize_embeddings(torch.stack((same, different)), 0.5)
    torch.testing.assert_close(actual[0], torch.ones(4))
    torch.testing.assert_close(actual[1], torch.zeros(4))
