"""Conversão de imagens e máscaras para o JSON da API (PNG em base64)."""

import base64

import cv2
import numpy as np


def png_b64(image: np.ndarray) -> str:
    """Codifica uma imagem RGB uint8 ou uma máscara booleana em PNG base64."""
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("falha ao codificar PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
