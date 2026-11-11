"""Figuras para o simulador e para os slides (contorno da máscara, grade comparativa)."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTOUR_COLOR = (0, 255, 0)
CAPTION_HEIGHT = 44
CAPTION_FONT_SIZE = 14


def overlay_contour(image: np.ndarray, mask: np.ndarray | None, thickness: int = 2) -> np.ndarray:
    """Desenha o contorno da máscara sobre uma cópia da imagem RGB."""
    out = image.copy()
    if mask is None or not mask.any():
        return out
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, CONTOUR_COLOR, thickness)
    return out


@lru_cache
def _font(size: int) -> ImageFont.ImageFont:
    """DejaVu Sans (vem com o matplotlib e tem acentos); senão, a fonte padrão do Pillow."""
    try:
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(path), size)
    except (ImportError, OSError):
        return ImageFont.load_default(size=size)


def _fit(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    """Encurta o texto com reticências até caber na largura."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def captioned(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Acrescenta uma faixa branca com até duas linhas de legenda acima da imagem."""
    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + CAPTION_HEIGHT), "white")
    canvas.paste(Image.fromarray(image), (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = _font(CAPTION_FONT_SIZE)
    for i, line in enumerate(lines[:2]):
        text = _fit(draw, line, font, w - 12)
        draw.text((6, 4 + i * (CAPTION_FONT_SIZE + 4)), text, fill="black", font=font)
    return np.asarray(canvas)


def grid(rows: list[list[np.ndarray]], gap: int = 6) -> np.ndarray:
    """Monta uma grade de imagens RGB do mesmo tamanho, com espaçamento branco."""
    th, tw = rows[0][0].shape[:2]
    ncols = max(len(r) for r in rows)
    out = np.full((len(rows) * (th + gap) - gap, ncols * (tw + gap) - gap, 3), 255, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            y, x = i * (th + gap), j * (tw + gap)
            out[y : y + th, x : x + tw] = tile
    return out
