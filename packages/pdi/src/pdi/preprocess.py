"""Etapas de pré-processamento P1–P5 (ver docs/PDI_Etapa2_Plano_Experimental.md, seção 3).

Todas as funções recebem e devolvem imagens RGB uint8 (H, W, 3), não alteram a entrada e
recebem `roi`: máscara booleana (H, W) da região útil da imagem (fora da vinheta).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

# P0 — normalização da entrada
INPUT_SIZE = 300  # lado (px) das imagens da base; os kernels abaixo foram dimensionados para ele

# P1 — vinheta
VIGNETTE_DARK_THRESHOLD = 40  # nível de cinza abaixo do qual a borda é considerada vinheta
VIGNETTE_CORNER_SIZE = 10  # lado (px) do quadrado de cada canto inspecionado
VIGNETTE_MIN_DARK_CORNERS = 2  # nº mínimo de cantos escuros para declarar vinheta
# área mínima (fração da imagem) para ser vinheta de dermatoscópio e não só cantos
# levemente escurecidos. Calibrado em treino/validação (regra do teste intocado): com 3%
# a detecção dá 29,2% das malignas e 1,1% das benignas no treino (validação: 28,3% e
# 1,1%), coerente com a análise exploratória da Etapa 1 (≈31% e 0,6%)
VIGNETTE_MIN_AREA = 0.03
# a transição vinheta → pele é um degradê: a máscara cresce sobre pixels mais escuros que
# esta fração do brilho da pele, até esta distância (px) do núcleo escuro
VIGNETTE_RING_RELATIVE = 0.85
VIGNETTE_RING_MAX_DIST = 20
VIGNETTE_MARGIN = 7  # dilatação final (px, elipse) da máscara, margem de segurança

# P2 — pelos (DullRazor)
HAIR_KERNEL_SIZE = 9  # elemento estruturante do black-hat (imagens de 300 px)
HAIR_INPAINT_RADIUS = 3
# Correção do P2: com limiar 12 e sem filtro de forma, o DullRazor marcava como pelo 41% dos
# pixels da lesão (texturas e glóbulos escuros). Duas constantes, calibradas em
# training.calibrate_p2 (150 imagens de treino + 150 de validação, pelos sintéticos de
# pdi.artifacts; regra do teste intocado):
# - HAIR_THRESHOLD: resposta mínima do black-hat para um pixel ser pelo;
# - HAIR_MIN_ELONGATION: só componentes finos e alongados são pelo, com alongamento =
#   comprimento² / área (comprimento = pixels do esqueleto) = comprimento ÷ largura média.
# Resultado (treino; validação entre parênteses), PSNR da restauração / cobertura dos pelos
# sintéticos / fração da lesão apagada sem pelo sintético:
#   12, 0 (original) → 32,1 dB / 99,8% / 41,4%  (33,2 / 99,8% / 38,6%)
#   40, 0 (maior PSNR) → 35,6 dB / 96,7% / 5,0%  (37,1 / 95,7% / 3,1%)
#   40, 4 (escolhido)  → 35,0 dB / 94,8% / 2,6%  (36,6 / 93,8% / 1,6%)
# O filtro de alongamento custa 0,5 dB e reduz à metade o que é apagado dentro da lesão,
# que é o defeito a corrigir; limiares maiores perdem pelos claros (50, 0: 91% na validação).
HAIR_THRESHOLD = 40
HAIR_MIN_ELONGATION = 4.0

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


def normalize_input(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """P0: recorta o quadrado central e redimensiona para `size` × `size` (interpolação por área).

    Sempre ligada e fora da ablação. Nas imagens da base (todas 300 × 300) é uma cópia; existe
    para que imagens de outros dermatoscópios cheguem na escala dos kernels de P1–P5 e S.
    """
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    square = image[top : top + side, left : left + side]
    if side == size:
        return square.copy()
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


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
    mask = keep_elongated(mask, HAIR_MIN_ELONGATION)
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def keep_elongated(mask: np.ndarray, min_elongation: float) -> np.ndarray:
    """Mantém só os componentes conexos com comprimento² / área ≥ `min_elongation`."""
    if min_elongation <= 0 or not mask.any():
        return mask
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    area = np.bincount(labels.ravel(), minlength=n)
    length = np.bincount(labels[skeletonize(mask)], minlength=n)
    elongation = length.astype(np.float64) ** 2 / np.maximum(area, 1)
    keep = elongation >= min_elongation
    keep[0] = False
    return keep[labels]


def remove_hair(image: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P2: detecta pelos e os substitui por inpainting. Devolve (imagem, máscara de pelos)."""
    mask = hair_mask(image, roi)
    if not mask.any():
        return image.copy(), mask
    out = cv2.inpaint(image, mask.astype(np.uint8), HAIR_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out, mask


def remove_hair_closing(image: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P2 (alternativa): fechamento morfológico, que apaga estruturas escuras e finas.

    Mais simples que o DullRazor: não há inpainting, os pixels de pelo recebem o valor do
    fechamento. Devolve (imagem, máscara dos pixels alterados).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAIR_KERNEL_SIZE, HAIR_KERNEL_SIZE))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    diff = closed.astype(np.int16) - image.astype(np.int16)
    mask = (diff.max(axis=2) > HAIR_THRESHOLD) & roi
    out = image.copy()
    out[mask] = closed[mask]
    return out, mask


def shades_of_gray(image: np.ndarray, roi: np.ndarray, p: float = SHADES_OF_GRAY_P) -> np.ndarray:
    """P3: constância de cor Shades of Gray (FINLAYSON; TREZZI, 2004), estimada só na ROI.

    A norma de Minkowski de ordem `p` estima a cor da iluminação em cada canal:
    p = 1 é o Gray World (média), p = ∞ é o White Patch (máximo).
    """
    pixels = image[roi].astype(np.float64)
    if np.isinf(p):
        norms = pixels.max(axis=0)
    else:
        norms = np.power(np.mean(np.power(pixels, p), axis=0), 1.0 / p)
    gains = norms.mean() / np.maximum(norms, 1e-6)
    out = image.astype(np.float64) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def gray_world(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P3 (alternativa): Gray World, caso p = 1 do Shades of Gray."""
    return shades_of_gray(image, roi, p=1)


def white_patch(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P3 (alternativa): White Patch (max-RGB), caso p = ∞ do Shades of Gray."""
    return shades_of_gray(image, roi, p=np.inf)


def _apply_to_l_channel(image: np.ndarray, roi: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = fn(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out[~roi] = image[~roi]
    return out


def clahe(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P4: CLAHE no canal L do CIELab; a cromaticidade (a, b) não é alterada."""
    op = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return _apply_to_l_channel(image, roi, op.apply)


def equalize(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P4 (alternativa): equalização global de histograma no canal L do CIELab."""
    return _apply_to_l_channel(image, roi, cv2.equalizeHist)


def denoise(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P5: filtro da mediana, que suaviza ruído preservando a borda da lesão."""
    out = cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)
    out[~roi] = image[~roi]
    return out


def gaussian(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """P5 (alternativa): filtro gaussiano, que suaviza ruído mas também a borda."""
    out = cv2.GaussianBlur(image, (MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE), 0)
    out[~roi] = image[~roi]
    return out
