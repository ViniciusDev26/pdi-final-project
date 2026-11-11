import numpy as np
import torch

from training import cnn


def _lesion():
    img = np.full((300, 300, 3), (200, 160, 140), np.uint8)
    yy, xx = np.mgrid[:300, :300]
    img[(yy - 150) ** 2 + (xx - 150) ** 2 < 60**2] = (90, 55, 40)
    return img


def test_transforms_give_224_normalized_tensors():
    x = torch.from_numpy(_lesion()).permute(2, 0, 1)
    for t in (cnn.train_transform(), cnn.eval_transform()):
        out = t(x)
        assert out.shape == (3, 224, 224) and out.dtype == torch.float32


def test_lesion_input_zeroes_skin_and_keeps_image_without_mask():
    img = _lesion()
    mask = np.zeros((300, 300), bool)
    mask[100:200, 100:200] = True
    out = cnn.apply_input(img, mask, "lesion")
    assert out[0, 0].sum() == 0 and (out[150, 150] == img[150, 150]).all()
    assert (cnn.apply_input(img, np.zeros_like(mask), "lesion") == img).all()
    assert (cnn.apply_input(img, mask, "preprocessed") == img).all()


def test_model_head_and_gradcam(monkeypatch):
    torch.manual_seed(0)
    model = cnn.build_model(pretrained=False).eval()
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 1)
    monkeypatch.setitem(cnn._loaded, "C7", (model, {}))
    prob, cam, seen = cnn.gradcam("C7", _lesion())
    assert 0 <= prob <= 1 and cam.shape == (300, 300) and seen.shape == (300, 300, 3)
    assert cam.min() >= 0 and cam.max() <= 1
    assert cnn.predict_images("C7", [_lesion()]).shape == (1,)
