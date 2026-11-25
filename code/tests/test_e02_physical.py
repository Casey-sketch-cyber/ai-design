import numpy as np
from PIL import Image, ImageDraw
import pytest

from dielinediff.dieline.geometry import build_geometry, write_geometry
from dielinediff.metrics.e01b import preflight_atlas
from dielinediff.metrics.e02_physical import E02PhysicalEvaluator, edge_continuity
from dielinediff.metrics.physical import PhysicalEvaluator


@pytest.fixture
def geometry(tmp_path):
    def make(scale):
        output = tmp_path / str(scale)
        write_geometry(build_geometry(pixels_per_mm=scale), output)
        return output
    return make


def _labels(directory, name="region_map"):
    with Image.open(directory / f"{name}.png") as image:
        return np.asarray(image)


def _rgb(gray):
    return np.broadcast_to(gray[..., None], (*gray.shape, 3)).copy()


def _close_legacy_metrics(e02, legacy, source):
    expected = legacy.measure(source, lpips=False)
    measured = e02.measure(source, lpips=False)
    assert {key: measured[key] for key in expected} == expected
    return measured


def test_m1_partition_absolute_and_reference_denominator(geometry):
    directory = geometry(4)
    e02 = E02PhysicalEvaluator(directory)
    legacy = PhysicalEvaluator(directory)
    labels = _labels(directory)
    source = _rgb(np.full(labels.shape, 130, dtype=np.uint8))
    # Put a continuous ribbon on the two sides of L–B, away from other folds.
    for name, side in (("L", "left"), ("B", "right")):
        x0, y0, x1, y1 = e02.native_topology["faces"][name]["rect"]
        x_start, x_end = (x0, x0 + 20) if side == "left" else (x1 - 20, x1)
        source[y0 + 80:y0 + 100, x_start:x_end] = 220
    continuous = _close_legacy_metrics(e02, legacy, source)
    assert continuous["native_5_delta_e00"] == continuous["non_native_7_delta_e00"] == 0
    assert continuous["wrap_lb_sdr"] == 0
    assert 0 < continuous["inside_dE00"] < continuous["sdr_denominator"] == 1
    assert continuous["m2_salient_seam_count"] == 1

    x0, y0, _, _ = e02.native_topology["faces"]["L"]["rect"]
    shifted = source.copy()
    area = source[y0:y0 + 160, x0:x0 + 20]
    shifted[y0:y0 + 160, x0:x0 + 20] = np.roll(area, 16, axis=0)
    broken = e02.measure(shifted, lpips=False)
    assert broken["native_5_delta_e00"] == 0
    assert broken["non_native_7_delta_e00"] > 0
    assert broken["wrap_lb_delta_e00"] == pytest.approx(broken["non_native_7_delta_e00"] * 7)
    assert broken["cross_seam_edge_cont"] < continuous["cross_seam_edge_cont"]


def _leaf_strip(seed):
    rng = np.random.default_rng(seed)
    strip = Image.new("RGB", (256, 16), "#667926")
    draw = ImageDraw.Draw(strip)
    palette = ("#7e9641", "#606e13", "#aead60")
    for _ in range(140):
        center = rng.uniform((-2, -2), (258, 18))
        angle = rng.uniform(0, np.pi)
        major = np.array([np.cos(angle), np.sin(angle)]) * rng.uniform(2, 5)
        minor = np.array([-major[1], major[0]]) * 0.38
        xy = tuple(map(tuple, (center + major, center + minor, center - major, center - minor)))
        draw.polygon(xy, fill=palette[rng.integers(len(palette))])
    return np.asarray(strip).transpose(1, 0, 2)


def test_m2_shift_and_independent_same_palette_negative():
    along, depth = np.indices((256, 16))
    first = _rgb(115 + 120 * np.exp(-0.5 * ((along + 0.72 * (depth + 0.5) - 115) / 5) ** 2))
    second = _rgb(115 + 120 * np.exp(-0.5 * ((along - 0.72 * (depth + 0.5) - 115) / 5) ** 2))
    scores = [edge_continuity(first, np.roll(second, p, axis=0)) for p in (0, 4, 8, 16)]
    negative = edge_continuity(_leaf_strip(12), _leaf_strip(93))
    print(f"M2 ridge shifts 0/4/8/16 px: {scores}; independent leaf palette: {negative}")
    assert 1 >= scores[0] > 0.8
    assert scores[0] > scores[1] > scores[2] > scores[3] >= 0
    assert negative < scores[0] - 0.4


def test_m2_returns_no_score_for_solid_atlas(geometry):
    evaluator = E02PhysicalEvaluator(geometry(4))
    solid = np.full((*evaluator.native_shape, 3), 120, dtype=np.uint8)
    measured = evaluator.measure(solid, lpips=False)
    assert measured["cross_seam_edge_cont"] is None
    assert measured["m2_salient_seam_count"] == 0


def _m3_atlases(directory, evaluator):
    labels = _labels(directory)
    owner = _labels(directory, "bleed_owner")
    face = np.isin(labels, [1, 2, 3, 4, 5, 6])
    bleed = labels == 10
    scale = evaluator.native_topology["image"]["pixels_per_mm"]
    y, x = np.indices(labels.shape)
    period = (x + y) / (scale * 25)
    wave = np.round(145 + 125 * (np.abs((period + 0.5) % 1 - 0.5) - 0.25)).astype(np.uint8)
    texture = np.full_like(wave, 255)
    texture[face | bleed] = wave[face | bleed]
    copy = texture.copy()
    for index, name in enumerate(("F", "B", "L", "R", "T", "Btm"), start=1):
        mask = bleed & (owner == index)
        x0, y0, x1, y1 = evaluator.native_topology["faces"][name]["rect"]
        copy[mask] = wave[np.clip(y[mask], y0, y1 - 1), np.clip(x[mask], x0, x1 - 1)]
    blank = texture.copy()
    blank[bleed] = 255
    solid = np.full_like(wave, 255)
    solid[face | bleed] = 145
    return [_rgb(atlas) for atlas in (copy, texture, solid, blank)]


@pytest.mark.parametrize("scale", [4, 6.4])
def test_m3_anti_copy_flat_and_white_controls(geometry, scale):
    directory = geometry(scale)
    e02 = E02PhysicalEvaluator(directory)
    legacy = PhysicalEvaluator(directory)
    copied, genuine, solid, blank = _m3_atlases(directory, e02)
    results = [_close_legacy_metrics(e02, legacy, atlas) for atlas in (copied, genuine, solid, blank)]
    m3 = [result["bleed_fill_v2_pct"] for result in results]
    print(f"M3 {scale} px/mm copy/genuine/solid/white: {m3}")
    if scale == 4:
        assert m3[0] == 0
    else:
        assert m3[0] < 5
    assert m3[1] > 60
    assert m3[2] > 99.5
    assert m3[3] == 0
    assert results[2]["d2_flat_bg_trap_pct"] > 99.5
    assert results[3]["d2_white_halo_pct"] > 99.5
    for result in results:
        assert sum(val for key, val in result.items() if key.startswith("d2_")) == pytest.approx(100)
    preflight = preflight_atlas(genuine, _labels(directory), "#315728")
    assert e02.measure(preflight, lpips=False) == results[1]


def test_m3_rejects_copy_after_full_atlas_bilinear_resize(geometry):
    fine, standard = geometry(6.4), geometry(4)
    source = E02PhysicalEvaluator(fine)
    target = E02PhysicalEvaluator(standard)
    copied = _m3_atlases(fine, source)[0]
    resampled = Image.fromarray(copied).resize(target.native_shape[::-1], Image.Resampling.BILINEAR)
    rate = target.measure(resampled, lpips=False)["bleed_fill_v2_pct"]
    print(f"M3 full-atlas 6.4→4 px/mm bilinear copy: {rate:.4f}%")
    assert rate < 5
