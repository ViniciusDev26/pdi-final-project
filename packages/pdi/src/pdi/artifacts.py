"""Artefatos sintéticos (plano experimental, seção 6.4).

Servem às métricas locais das etapas (pelos para o PSNR/SSIM de P2, dominante de cor para
o ΔE de P3) e ao cenário V5 (vinheta artificial). Não fazem parte do simulador.
Todas as funções são determinísticas dado o `rng`.
"""

import cv2
import numpy as np

# o pelo multiplica o fundo por um fator neste intervalo: de pelos pretos (0,15) a pelos
# finos ou claros, pouco contrastados (0,7), como os que aparecem na base
HAIR_DARKENING = (0.15, 0.7)
HAIR_THICKNESS = (1, 3)
VIGNETTE_RADIUS = 0.47  # raio do círculo útil, em fração do lado da imagem
VIGNETTE_SOFTNESS = 6  # px de degradê na transição
VIGNETTE_LEVEL = 8


def _bezier(p0, p1, p2, n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    return np.rint(pts).astype(np.int32)


def add_hair(
    image: np.ndarray, rng: np.random.Generator, n: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Desenha `n` pelos curvos e finos atravessando a imagem. Devolve (imagem, máscara)."""
    h, w = image.shape[:2]
    out = image.astype(np.float64)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        p0 = rng.uniform((0, 0), (w, h))
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(0.4, 1.0) * max(h, w)
        direction = np.array([np.cos(angle), np.sin(angle)])
        p2 = p0 + direction * length
        bend = np.array([-direction[1], direction[0]]) * rng.uniform(-0.15, 0.15) * length
        p1 = (p0 + p2) / 2 + bend
        pts = _bezier(p0, p1, p2)
        thickness = int(rng.integers(HAIR_THICKNESS[0], HAIR_THICKNESS[1] + 1))
        factor = rng.uniform(*HAIR_DARKENING)
        alpha = np.zeros((h, w), dtype=np.float32)
        cv2.polylines(alpha, [pts], False, 1.0, thickness, lineType=cv2.LINE_AA)
        darken = 1 - (1 - factor) * alpha[..., None]
        out *= darken * np.array([1.0, 0.97, 0.94])[None, None] ** alpha[..., None]
        cv2.polylines(mask, [pts], False, 1, thickness, lineType=cv2.LINE_8)
    return np.clip(out, 0, 255).astype(np.uint8), mask.astype(bool)


def add_color_cast(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Dominante de cor: multiplica cada canal RGB por um ganho (simula outra iluminação)."""
    out = image.astype(np.float64) * np.asarray(gains)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_vignette(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vinheta circular de dermatoscópio (cenário V5). Devolve (imagem, máscara da vinheta)."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    radius = VIGNETTE_RADIUS * min(h, w)
    alpha = np.clip((dist - radius) / VIGNETTE_SOFTNESS, 0, 1)[..., None]  # 1 = vinheta
    out = image.astype(np.float64) * (1 - alpha) + VIGNETTE_LEVEL * alpha
    return np.clip(out, 0, 255).astype(np.uint8), alpha[..., 0] > 0.5
