import cv2
import numpy as np
import pytest

from pdi import STAGE_IDS, PipelineConfig, preprocess, run
from pdi.metrics import dice, iou
from pdi.segment import segment

SKIN = (200, 160, 140)
LESION = (90, 55, 40)
SIZE = 300


def _skin(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), SKIN, dtype=np.int16)
    img += rng.integers(-4, 5, size=img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _lesion_image() -> tuple[np.ndarray, np.ndarray]:
    img = _skin()
    ref = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(ref, (150, 150), 60, 1, thickness=-1)
    img[ref.astype(bool)] = LESION
    return img, ref.astype(bool)


def _add_vignette(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    out[(yy - 150) ** 2 + (xx - 150) ** 2 > 145**2] = (5, 5, 5)
    return out


def _add_hair(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for x in (40, 110, 250):
        cv2.line(out, (x, 0), (x + 30, SIZE - 1), (20, 15, 10), thickness=2)
    return out


# ---------------------------------------------------------------- config


def test_raw_config_disables_every_stage_and_keeps_image():
    img, _ = _lesion_image()
    result = run(img, PipelineConfig.raw())
    assert result.config.enabled_stages == ()
    assert [s.enabled for s in result.stages] == [False] * 5
    np.testing.assert_array_equal(result.image, img)


def test_only_and_without_build_ablation_scenarios():
    assert PipelineConfig.only("P1", "P2").enabled_stages == ("P1", "P2")  # A2
    assert PipelineConfig().without("P3").enabled_stages == ("P1", "P2", "P4", "P5")  # L3
    assert PipelineConfig().enabled_stages == STAGE_IDS  # A5


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig().without("P9")


def test_full_pipeline_preserves_shape_and_records_every_stage():
    img, _ = _lesion_image()
    result = run(img)
    assert result.image.shape == img.shape and result.image.dtype == np.uint8
    assert [s.id for s in result.stages] == list(STAGE_IDS)
    assert set(result.timings) == {"P0", *STAGE_IDS, "S"}
    assert result.lesion_mask is not None and result.lesion_mask.shape == img.shape[:2]


def test_run_does_not_modify_input():
    img = _add_hair(_add_vignette(_lesion_image()[0]))
    before = img.copy()
    run(img)
    np.testing.assert_array_equal(img, before)


def test_run_rejects_non_rgb_uint8():
    with pytest.raises(ValueError):
        run(np.zeros((10, 10), dtype=np.uint8))


# ---------------------------------------------------------------- etapas


def test_vignette_is_detected_and_filled():
    img = _add_vignette(_lesion_image()[0])
    res = preprocess.remove_vignette(img)
    assert res.detected
    assert res.mask[0, 0] and not res.mask[150, 150]
    assert res.image[0, 0].mean() > 100  # canto agora tem cor de pele


def test_no_vignette_on_clean_image():
    res = preprocess.remove_vignette(_lesion_image()[0])
    assert not res.detected and not res.mask.any()


def test_slightly_dark_corners_are_not_a_vignette():
    img = _lesion_image()[0]
    for y, x in ((0, 0), (0, SIZE - 12), (SIZE - 12, 0), (SIZE - 12, SIZE - 12)):
        img[y : y + 12, x : x + 12] = (10, 10, 10)  # < 1% da área
    res = preprocess.remove_vignette(img)
    assert not res.detected


def test_vignette_transition_ring_is_removed():
    img = _lesion_image()[0]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.sqrt((yy - 150) ** 2 + (xx - 150) ** 2)
    # degradê de 8 px entre a pele e o preto, como nas imagens reais
    fade = np.clip((r - 137) / 8, 0, 1)[..., None]
    img = (img * (1 - fade) + 5 * fade).astype(np.uint8)
    res = preprocess.remove_vignette(img)
    assert res.detected
    ring = (r > 139) & (r < 145)
    assert res.mask[ring].mean() > 0.95
    assert not res.mask[r < 125].any()


def test_central_dark_lesion_is_not_mistaken_for_vignette():
    img = _add_vignette(_lesion_image()[0])
    img[img.sum(axis=2) < 300] = (5, 5, 5)  # lesão tão escura quanto a vinheta
    res = preprocess.remove_vignette(img)
    assert not res.mask[150, 150]


def test_hair_is_detected_and_removed():
    clean = _skin()
    hairy = _add_hair(clean)
    out, mask = preprocess.remove_hair(hairy, preprocess.full_roi(hairy))
    hair_px = np.abs(hairy.astype(int) - clean.astype(int)).sum(axis=2) > 50
    assert mask[hair_px].mean() > 0.9  # quase todo pelo foi detectado
    assert np.abs(out[hair_px].astype(int) - clean[hair_px].astype(int)).mean() < 20


def test_shades_of_gray_removes_color_cast():
    img = _skin().astype(np.float64)
    cast = np.clip(img * (1.3, 1.0, 0.7), 0, 255).astype(np.uint8)
    out = preprocess.shades_of_gray(cast, preprocess.full_roi(cast))
    means = out.reshape(-1, 3).mean(axis=0)
    cast_means = cast.reshape(-1, 3).mean(axis=0)
    assert np.ptp(means) < np.ptp(cast_means) / 5


def test_disabled_stage_passes_image_through():
    img = _add_hair(_lesion_image()[0])
    result = run(img, PipelineConfig.only("P2"))
    np.testing.assert_array_equal(result.stage("P3").image, result.stage("P2").image)
    assert result.stage("P2").mask is not None and result.stage("P2").mask.any()


# ---------------------------------------------------------------- segmentação


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_segmentation_with_postprocessing_finds_lesion(method):
    img, ref = _lesion_image()
    mask = segment(img, preprocess.full_roi(img), method=method, postprocessing=True)
    assert dice(mask, ref) > 0.9


def test_segmentation_ignores_vignette():
    img, ref = _lesion_image()
    result = run(_add_vignette(img), PipelineConfig.only("P1"))
    assert result.vignette_detected
    assert dice(result.lesion_mask, ref) > 0.9


def test_segmentation_prefers_lesion_over_dark_border_strip():
    img = _skin()
    ref = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(ref, (170, 140), 25, 1, thickness=-1)  # lesão pequena, fora do centro
    ref = ref.astype(bool)
    img[ref] = LESION
    img[:, :40] = (120, 90, 80)  # sombra larga na borda, maior que a lesão
    mask = segment(img, preprocess.full_roi(img))
    assert dice(mask, ref) > 0.9


def test_segmentation_keeps_large_central_lesion_touching_border():
    img = _skin()
    ref = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.ellipse(ref, (150, 150), (170, 90), 0, 0, 360, 1, thickness=-1)  # sai pelas laterais
    ref = ref.astype(bool)
    img[ref] = LESION
    mask = segment(img, preprocess.full_roi(img))
    assert dice(mask, ref) > 0.9


def test_segmentation_can_be_disabled():
    img, _ = _lesion_image()
    result = run(img, PipelineConfig(segmentation=None))
    assert result.lesion_mask is None and "S" not in result.timings


def test_dice_and_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[1:3] = True
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a & False, b & False) == 1.0


# ---------------------------------------------------------------- técnicas alternativas


def test_using_builds_technique_swap_scenarios():
    t4 = PipelineConfig().using(P4="equalize")  # cenário T4
    assert t4.technique("P4") == "equalize"
    assert t4.technique("P3") == "shades_of_gray"
    assert PipelineConfig().using(P3=None) == PipelineConfig().without("P3")


def test_invalid_technique_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(contrast="sharpen")
    with pytest.raises(ValueError):
        PipelineConfig().using(P2="laser")
    with pytest.raises(ValueError):
        PipelineConfig(segmentation="kmeans")


def test_config_roundtrips_through_dict():
    cfg = PipelineConfig().using(P3="white_patch", P5="gaussian").without("P2")
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"P9": "x"})


@pytest.mark.parametrize(
    "stage_id",
    [sid for sid in STAGE_IDS],
)
def test_every_registered_technique_runs(stage_id):
    from pdi import STAGES

    img = _add_hair(_add_vignette(_lesion_image()[0]))
    stage = next(s for s in STAGES if s.id == stage_id)
    for technique in stage.techniques:
        result = run(img, PipelineConfig().using(**{stage_id: technique.key}))
        out = result.stage(stage_id)
        assert out.technique == technique.key and out.technique_label == technique.label
        assert result.image.shape == img.shape and result.image.dtype == np.uint8


def test_stage_output_reports_disabled_stage():
    result = run(_lesion_image()[0], PipelineConfig().without("P4"))
    assert not result.stage("P4").enabled and result.stage("P4").technique is None


def test_hair_closing_removes_thin_dark_lines():
    clean = _skin()
    hairy = _add_hair(clean)
    out, mask = preprocess.remove_hair_closing(hairy, preprocess.full_roi(hairy))
    hair_px = np.abs(hairy.astype(int) - clean.astype(int)).sum(axis=2) > 50
    assert mask[hair_px].mean() > 0.9
    assert np.abs(out[hair_px].astype(int) - clean[hair_px].astype(int)).mean() < 20


@pytest.mark.parametrize("fn", [preprocess.gray_world, preprocess.white_patch])
def test_color_constancy_alternatives_remove_cast(fn):
    img = _skin().astype(np.float64)
    cast = np.clip(img * (1.3, 1.0, 0.7), 0, 255).astype(np.uint8)
    out = fn(cast, preprocess.full_roi(cast))
    assert np.ptp(out.reshape(-1, 3).mean(axis=0)) < np.ptp(cast.reshape(-1, 3).mean(axis=0)) / 3


def test_equalize_stretches_low_contrast_image():
    img, _ = _lesion_image()
    low = (img.astype(np.float64) * 0.3 + 100).astype(np.uint8)  # histograma comprimido
    out = preprocess.equalize(low, preprocess.full_roi(low))
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # noqa: E731
    assert gray(out).std() > 2 * gray(low).std()


def test_gaussian_smooths_noise():
    rng = np.random.default_rng(1)
    noisy = np.clip(_skin().astype(int) + rng.normal(0, 20, (SIZE, SIZE, 3)), 0, 255)
    noisy = noisy.astype(np.uint8)
    out = preprocess.gaussian(noisy, preprocess.full_roi(noisy))
    assert out.std(axis=(0, 1)).mean() < noisy.std(axis=(0, 1)).mean() / 2


# ---------------------------------------------------------------- P0


def test_p0_is_a_copy_for_dataset_sized_images():
    img, _ = _lesion_image()
    out = preprocess.normalize_input(img)
    np.testing.assert_array_equal(out, img)
    assert out is not img


def test_p0_crops_central_square_and_resizes():
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, 150:750] = (200, 160, 140)  # quadrado central 600 × 600; laterais pretas
    out = preprocess.normalize_input(img)
    assert out.shape == (300, 300, 3)
    assert (out == (200, 160, 140)).all()


def test_run_normalizes_input_of_any_size():
    img = cv2.resize(_lesion_image()[0], (800, 600))
    result = run(img)
    assert result.original_shape == (600, 800, 3)
    assert result.input.shape == result.image.shape == (SIZE, SIZE, 3)
    assert result.lesion_mask.shape == (SIZE, SIZE)


# ---------------------------------------------------------------- P2 (correção)


def test_hair_mask_ignores_round_dark_globules_inside_lesion():
    img, lesion = _lesion_image()
    for cx, cy in ((130, 130), (170, 140), (150, 175)):  # glóbulos escuros na lesão
        cv2.circle(img, (cx, cy), 3, (10, 8, 6), thickness=-1)
    mask = preprocess.hair_mask(img, preprocess.full_roi(img))
    assert mask[lesion].mean() < 0.01


def test_keep_elongated_separates_lines_from_blobs():
    m = np.zeros((100, 100), np.uint8)
    cv2.line(m, (5, 10), (95, 30), 1, 2)
    cv2.circle(m, (50, 70), 6, 1, -1)
    kept = preprocess.keep_elongated(m.astype(bool), 4.0)
    assert kept[20, 50] and not kept[70, 50]
