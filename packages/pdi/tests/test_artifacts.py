import numpy as np

from pdi import artifacts
from pdi.metrics import delta_e, psnr, ssim


def _img():
    return np.full((300, 300, 3), (200, 160, 140), dtype=np.uint8)


def test_add_hair_is_deterministic_and_darkens_only_the_mask_region():
    a, ma = artifacts.add_hair(_img(), np.random.default_rng(1))
    b, mb = artifacts.add_hair(_img(), np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert ma.any() and (ma == mb).all()
    assert a[ma].mean() < _img()[ma].mean() - 20
    assert psnr(a, _img()) < 40 and ssim(a, _img()) < 1


def test_vignette_and_color_cast():
    out, mask = artifacts.add_vignette(_img())
    assert mask[0, 0] and not mask[150, 150] and out[0, 0].max() < 20
    cast = artifacts.add_color_cast(_img(), (1.1, 1.0, 0.8))
    assert delta_e(cast, _img()) > 5
