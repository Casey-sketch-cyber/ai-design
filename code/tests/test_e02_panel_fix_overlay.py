"""Exact coordinate and lattice tests for the Task 22 panel diagnostic."""

import numpy as np

from dielinediff.e02_panel_fix_overlay import _grid_evidence, _points_to_segments, mask_segments, native_to_model


def test_native_cap_pixel_edges_reproject_to_archived_tight_core():
    core = (304, 296, 1088, 808)
    actual = native_to_model([(0, 0), (448, 288), (412, 34)], (448, 288), core)
    np.testing.assert_allclose(actual, [(304,296), (1088,808), (1025,356 + 4/9)])


def test_mask_boundaries_are_pixel_edges_and_distances_ignore_tangent_outside_segment():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    lines = mask_segments(mask)
    assert len(lines) == 4
    np.testing.assert_allclose(_points_to_segments(np.asarray([(3,2), (5,3), (10,7)]), lines),
                               [0,1,5])


def test_lattice_null_uses_quantized_atlas_destinations_and_one_parallel_line():
    core = (304, 296, 1088, 808)
    line = np.linspace([1025,380], [1025,590], 97)
    grid = _grid_evidence(line, 0, (448,288), core, 1.75)
    assert grid["whole_edge_within_1px"] and grid["nearest_parallel_grid_coordinate_px"] == 1024
    assert 0.45 < grid["native_integer_null_2px"] < 0.59
    sloped = np.linspace([1022,380], [1026.5,590], 97)
    grid = _grid_evidence(sloped, 0, (448,288), core, 1.75)
    assert not grid["whole_edge_within_2px"]
