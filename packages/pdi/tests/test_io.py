from io import BytesIO

import numpy as np
from PIL import Image

from pdi import load_rgb


def _png_bytes(mode: str) -> bytes:
    buf = BytesIO()
    Image.new(mode, (4, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_rgb_from_bytes_returns_hwc_uint8():
    img = load_rgb(_png_bytes("RGB"))
    assert img.shape == (3, 4, 3)
    assert img.dtype == np.uint8


def test_load_rgb_converts_grayscale_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    path.write_bytes(_png_bytes("L"))
    assert load_rgb(path).shape == (3, 4, 3)
