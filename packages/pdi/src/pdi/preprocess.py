"""Etapas de pré-processamento P1–P5 (ver docs/PDI_Etapa2_Plano_Experimental.md, seção 3).

Todas as funções recebem e devolvem imagens RGB uint8 (H, W, 3), não alteram a entrada e
recebem `roi`: máscara booleana (H, W) da região útil da imagem (fora da vinheta).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# P1 — vinheta
VIGNETTE_DARK_THRESHOLD = 40  # nível de cinza abaixo do qual a borda é considerada vinheta
VIGNETTE_CORNER_SIZE = 10  # lado (px) do quadrado de cada canto inspecionado
VIGNETTE_MIN_DARK_CORNERS = 2  # nº mínimo de cantos escuros para declarar vinheta
# área mínima (fração da imagem) para ser vinheta de dermatoscópio e não só cantos
# levemente escurecidos; com 3% a detecção reproduz a análise exploratória da Etapa 1
# (≈30% das malignas de treino, ≈43% das de teste, <1,5% das benignas)
VIGNETTE_MIN_AREA = 0.03
# a transição vinheta → pele é um degradê: a máscara cresce sobre pixels mais escuros que
# esta fração do brilho da pele, até esta distância (px) do núcleo escuro
VIGNETTE_RING_RELATIVE = 0.85
VIGNETTE_RING_MAX_DIST = 20
VIGNETTE_MARGIN = 7  # dilatação final (px, elipse) da máscara, margem de segurança

# P2 — pelos (DullRazor)
HAIR_KERNEL_SIZE = 9  # elemento estruturante do black-hat (imagens de 300 px)
HAIR_THRESHOLD = 12  # resposta mínima do black-hat para um pixel ser pelo
HAIR_INPAINT_RADIUS = 3

# P3 — Shades of Gray
SHADES_OF_GRAY_P = 6

# P4 — CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# P5 — filtragem
MEDIAN_KERNEL_SIZE = 5


@dataclass(frozen=True)
class VignetteResult:
    image: np.ndarray
    mask: np.ndarray  # True onde há vinheta
    detected: bool


def full_roi(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=bool)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _border_components(mask: np.ndarray) -> np.ndarray:
    """Mantém só os componentes conexos de `mask` que tocam a borda da imagem."""
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return np.isin(labels, border[border != 0])


def remove_vignette(image: np.ndarray) -> VignetteResult:
    """P1: detecta a borda escura do dermatoscópio e a preenche com a cor da pele.

    A vinheta é declarada quando ao menos `VIGNETTE_MIN_DARK_CORNERS` cantos são escuros e
    a região escura ocupa ao menos `VIGNETTE_MIN_AREA` da imagem. O núcleo da máscara são
    os componentes escuros que tocam a borda da imagem (uma lesão escura no centro não é
    apagada); depois ele cresce sobre o degradê de transição, para não deixar um anel.
    """
    gray = _gray(image)
    s = VIGNETTE_CORNER_SIZE
    corners = (gray[:s, :s], gray[:s, -s:], gray[-s:, :s], gray[-s:, -s:])
    dark_corners = sum(float(c.mean()) < VIGNETTE_DARK_THRESHOLD for c in corners)
    empty = np.zeros(gray.shape, dtype=bool)
    if dark_corners < VIGNETTE_MIN_DARK_CORNERS:
        return VignetteResult(image.copy(), empty, detected=False)

    core = _border_components(gray < VIGNETTE_DARK_THRESHOLD)
    if core.mean() < VIGNETTE_MIN_AREA or core.all():
        return VignetteResult(image.copy(), empty, detected=False)

    # cresce o núcleo sobre a transição (pixels ainda escuros perto da vinheta)
    skin_level = float(np.median(gray[~core]))
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    ring = (gray < VIGNETTE_RING_RELATIVE * skin_level) & (dist <= VIGNETTE_RING_MAX_DIST)
    mask = _border_components(core | ring)
    margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (VIGNETTE_MARGIN, VIGNETTE_MARGIN))
    mask = cv2.dilate(mask.astype(np.uint8), margin).astype(bool)

    out = image.copy()
    out[mask] = np.median(image[~mask], axis=0).astype(np.uint8)
    return VignetteResult(out, mask, detected=True)


def hair_mask(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Máscara de pelos pelo black-hat morfológico (DullRazor; LEE et al., 1997)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAIR_KERNEL_SIZE, HAIR_KERNEL_SIZE))
    blackhat = cv2.morphologyEx(_gray(image), cv2.MORPH_BLACKHAT, kernel)
    mask = (blackhat > HAIR_THRESHOLD) & roi
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def remove_hair(image: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P2: detecta pelos e os substitui por inpainting. Devolve (imagem, máscara de pelos)."""
    mask = hair_mask(image, roi)
    if not mask.any():
        return image.copy(), mask
    out = cv2.inpaint(image, mask.astype(np.uint8), HAIR_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out, mask


def shades_of_gray(image: np.ndarray, roi: np.ndarray, p: int = SHADES_OF_GRAY_P) -> np.ndarray:
    """P3: constância de cor Shades of Gray (FINLAYSON; TREZZI, 2004), estimada só na ROI."""
    pixels = image[roi].astype(np.float64)
    norms = np.power(np.mean(np.power(pixels, p), axis=0), 1.0 / p)
    gains = norms.mean() / np.maximum(norms, 1e-6)
    out = image.astype(np.float64) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P4: CLAHE no canal L do CIELab; a cromaticidade (a, b) não é alterada."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[..., 0] = op.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out[~roi] = image[~roi]
    return out


def denoise(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P5: filtro da mediana, que suaviza ruído preservando a borda da lesão."""
    out = cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)
    out[~roi] = image[~roi]
    return out
