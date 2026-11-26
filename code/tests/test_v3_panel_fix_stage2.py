"""Check that the Stage 2 strength and source-blend masks protect the native fold."""

from types import SimpleNamespace

import numpy as np

from dielinediff.methods.v3_panel_fix_stage2 import _area_axis, _fields, panel_fields


BOX = {"length": 70.0, "width": 45.0}
SETTINGS = {"native_lock_mm": 15.0, "native_unlock_mm": 3.0, "edge_transition_mm": 20.0,
            "edge_strength": 0.85, "interior_strength": 0.30, "mix_edge_transition_mm": 20.0}


def test_native_fold_orientation_and_aligned_seam_limits():
    td, top, tm = _fields("T", (90, 140), BOX, SETTINGS)
    bd, bottom, bm = _fields("Btm", (90, 140), BOX, SETTINGS)
    np.testing.assert_array_equal(top, bottom[::-1])
    np.testing.assert_array_equal(tm, bm[::-1])
    assert np.count_nonzero(top[td <= 15]) == 0
    assert np.count_nonzero(bm[bd <= 15]) == 0
    assert np.isclose(top.max(), 0.85)
    assert np.isclose(tm.max(), 1.0)
    inside = np.flatnonzero((td > 18) & (td < 24))
    np.testing.assert_allclose(top[inside, 70], 0.30)
    np.testing.assert_allclose(tm[inside, 70], 0.0)
    assert top[0, -1] == top[0, 0]


def test_no_ring_context_is_editable():
    known = np.ones((45, 58), dtype=bool)
    known[4:41, 3:55] = False
    layout = SimpleNamespace(face="Btm", latent_shape=known.shape,
                             center_rect=(3, 4, 55, 41), known_mask=known)
    field = panel_fields(layout, BOX, SETTINGS)
    assert not field[known].any() and field[~known].any()
    assert np.all(field[4:16] == 0)



def test_fractional_footprints_keep_the_whole_u1_core():
    source = np.arange(5, dtype=np.float32)[:, None, None]
    np.testing.assert_allclose(_area_axis(source, 0, 2).ravel(), (0.8, 3.2))
    rgb = np.full((512, 784, 3), 0.25, dtype=np.float32)
    actual = _area_axis(_area_axis(rgb, 0, 248), 1, 392)
    np.testing.assert_array_equal(actual.astype(np.float32), np.full((248, 392, 3), 0.25, dtype=np.float32))
