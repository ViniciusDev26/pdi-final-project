import math

import cv2
import numpy as np
import pytest

from pdi import features, run
from pdi.features import FEATURE_GROUPS, FEATURE_NAMES, extract

SIZE = 300
SKIN = (200, 160, 140)
DARK_BROWN = (60, 10, 5)


def _canvas(color=SKIN) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), color, dtype=np.uint8)


def _circle_mask(radius: int = 60, center=(150, 150)) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(m, center, radius, 1, thickness=-1)
    return m.astype(bool)


def _star_mask() -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    radii = np.where(np.arange(16) % 2 == 0, 90, 40)
    pts = np.stack([150 + radii * np.cos(angles), 150 + radii * np.sin(angles)], axis=1)
    m = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m.astype(bool)


def _paint(mask: np.ndarray, color=DARK_BROWN) -> np.ndarray:
    img = _canvas()
    img[mask] = color
    return img


def test_names_are_unique_and_grouped():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert set(FEATURE_GROUPS) == {"forma", "cor", "textura"}
    assert len(FEATURE_GROUPS["cor"]) == 14 and len(FEATURE_GROUPS["textura"]) == 14


def test_returns_every_feature_in_order():
    mask = _circle_mask()
    feats = extract(_paint(mask), mask)
    assert tuple(feats) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in feats.values())


@pytest.mark.parametrize("mask", [None, np.zeros((SIZE, SIZE), bool)])
def test_empty_mask_gives_nan(mask):
    feats = extract(_canvas(), mask)
    assert tuple(feats) == FEATURE_NAMES
    assert all(math.isnan(v) for v in feats.values())


def test_circle_is_symmetric_compact_and_regular():
    mask = _circle_mask(60)
    f = extract(_paint(mask), mask)
    assert f["assimetria_eixo_maior"] < 0.03 and f["assimetria_eixo_menor"] < 0.03
    assert 0.9 < f["compacidade"] < 1.15
    assert f["solidez"] > 0.97
    assert f["variacao_raio"] < 0.03
    assert 0.9 < f["dimensao_fractal"] < 1.2
    assert f["area_relativa"] == pytest.approx(np.pi * 60**2 / SIZE**2, rel=0.02)
    assert f["diametro_max_px"] == pytest.approx(120, abs=3)


def test_irregular_border_scores_worse_than_circle():
    circle, star = _circle_mask(60), _star_mask()
    fc, fs = extract(_paint(circle), circle), extract(_paint(star), star)
    assert fs["compacidade"] > fc["compacidade"] + 0.5
    assert fs["solidez"] < fc["solidez"] - 0.2
    assert fs["variacao_raio"] > fc["variacao_raio"] + 0.1


def test_asymmetric_shape_has_higher_asymmetry():
    mask = _circle_mask(60) | _circle_mask(30, center=(150, 215))  # protuberância de um lado
    f = extract(_paint(mask), mask)
    assert max(f["assimetria_eixo_maior"], f["assimetria_eixo_menor"]) > 0.1


def test_sharp_border_has_stronger_gradient_than_blurred():
    mask = _circle_mask(60)
    sharp = _paint(mask)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 10)
    assert extract(sharp, mask)["sobel_borda"] > 2 * extract(blurred, mask)["sobel_borda"]


def test_color_count_and_contrast():
    mask = _circle_mask(60)
    uniform = extract(_paint(mask), mask)
    assert uniform["n_cores"] == 1
    assert uniform["delta_e_pele"] > 20

    two = _paint(mask)
    two[mask & (np.arange(SIZE)[None, :] < 150)] = features.REFERENCE_COLORS["azul_acinzentado"]
    assert extract(two, mask)["n_cores"] == 2

    same_as_skin = extract(_canvas(), mask)
    assert same_as_skin["delta_e_pele"] < 1


def test_texture_histogram_and_glcm():
    mask = _circle_mask(60)
    rng = np.random.default_rng(0)
    noisy = np.clip(_paint(mask).astype(int) + rng.integers(-40, 41, (SIZE, SIZE, 1)), 0, 255)
    f_flat = extract(_paint(mask), mask)
    f_noisy = extract(noisy.astype(np.uint8), mask)
    lbp = [f_noisy[f"lbp_{i}"] for i in range(features.LBP_BINS)]
    assert sum(lbp) == pytest.approx(1.0)
    assert f_noisy["glcm_contrast"] > f_flat["glcm_contrast"]
    assert f_noisy["glcm_homogeneity"] < f_flat["glcm_homogeneity"]


def test_works_on_pipeline_output():
    mask = _circle_mask(60)
    result = run(_paint(mask))
    f = extract(result.image, result.lesion_mask, result.roi)
    assert not any(math.isnan(v) for k, v in f.items() if k != "glcm_correlation")
