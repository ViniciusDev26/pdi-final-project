from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(source: str | Path | bytes) -> np.ndarray:
    """Carrega uma imagem como array RGB uint8 de shape (H, W, 3)."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    with Image.open(source) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
