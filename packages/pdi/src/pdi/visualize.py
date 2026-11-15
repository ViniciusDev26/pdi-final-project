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


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Sobrepõe um mapa em [0, 1] (ex.: Grad-CAM) à imagem RGB, em escala de cor JET."""
    h, w = image.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def overlay_two_contours(
    image: np.ndarray, auto: np.ndarray | None, manual: np.ndarray | None
) -> np.ndarray:
    """Contorno automático em vermelho e manual (referência) em verde."""
    out = image.copy()
    for mask, color in ((manual, (0, 255, 0)), (auto, (255, 0, 0))):
        if mask is not None and mask.any():
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(out, contours, -1, color, 2)
    return out
