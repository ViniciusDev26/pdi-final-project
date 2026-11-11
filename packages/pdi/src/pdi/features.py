"""Etapa F: características da lesão (lista fechada da seção 3.1 do plano experimental).

Todas são calculadas dentro da máscara da lesão (saída de S) sobre a imagem pré-processada.
`extract` devolve sempre as mesmas chaves, na ordem de `FEATURE_NAMES`; com máscara vazia
ou pequena demais (< `MIN_LESION_PIXELS`) todos os valores são NaN e o classificador decide
como tratar (imputação no treino).

Incluir ou remover uma característica é uma alteração registrada no histórico do plano.
"""

import math

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

MIN_LESION_PIXELS = 50

# B — nitidez da borda
CANNY_LOW, CANNY_HIGH = 50, 150  # limiares de histerese sobre o canal L (0–255)
BORDER_BAND = 5  # largura (px) da faixa em torno do contorno para a densidade de Canny

# C — cores dermatoscópicas de referência (RGB), como usadas na literatura de ABCD
# automatizado (p. ex. Seidenari et al.; Celebi et al.). Cada pixel da lesão é atribuído à
# referência mais próxima em CIELab; a cor conta quando ocupa ≥ `COLOR_MIN_FRACTION`.
REFERENCE_COLORS: dict[str, tuple[int, int, int]] = {
    "branco": (255, 255, 255),
    "vermelho": (204, 51, 51),
    "marrom_claro": (153, 102, 51),
    "marrom_escuro": (51, 0, 0),
    "azul_acinzentado": (51, 153, 153),
    "preto": (0, 0, 0),
}
COLOR_MIN_FRACTION = 0.05
# faixa de pele ao redor da lesão para o contraste ΔE: entre estas distâncias (px) do contorno
SKIN_RING_INNER, SKIN_RING_OUTER = 5, 20

# textura
GLCM_LEVELS = 32  # quantização do canal L para a GLCM
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # LBP uniforme: P + 1 padrões uniformes + 1 bin para os não uniformes

_LAB_NAMES = ("L", "a", "b")
_HSV_NAMES = ("H", "S", "V")

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "forma": (
        # A — assimetria
        "assimetria_eixo_maior",
        "assimetria_eixo_menor",
        # B — borda (forma)
        "compacidade",
        "solidez",
        "dimensao_fractal",
        "variacao_raio",
        # B — nitidez
        "sobel_borda",
        "canny_borda",
        # D — dimensão
        "area_relativa",
        "diametro_max_px",
    ),
    "cor": (
        *(f"{c}_{stat}" for c in _LAB_NAMES + _HSV_NAMES for stat in ("media", "dp")),
        "n_cores",
        "delta_e_pele",
    ),
    "textura": (
        *(f"glcm_{p}" for p in GLCM_PROPS),
        *(f"lbp_{i}" for i in range(LBP_BINS)),
    ),
}
FEATURE_NAMES: tuple[str, ...] = tuple(n for names in FEATURE_GROUPS.values() for n in names)


def extract(image: np.ndarray, mask: np.ndarray | None, roi: np.ndarray | None = None) -> dict:
    """Extrai as características de uma imagem RGB uint8 e da máscara booleana da lesão.

    `roi` (região útil, fora da vinheta) limita a faixa de pele usada no contraste ΔE.
    """
    if mask is None or int(mask.sum()) < MIN_LESION_PIXELS:
        return dict.fromkeys(FEATURE_NAMES, math.nan)
    mask = mask.astype(bool)
    roi = np.ones(mask.shape, dtype=bool) if roi is None else roi.astype(bool)

    contour = _largest_contour(mask)
    lab = _lab(image)
    l_u8 = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]

    feats: dict[str, float] = {}
    feats |= _asymmetry(mask)
    feats |= _border_shape(mask, contour)
    feats |= _border_sharpness(l_u8, mask, contour)
    feats |= _dimension(mask, contour)
    feats |= _color(image, lab, mask, roi)
    feats |= _texture(l_u8, mask)
    return {name: float(feats[name]) for name in FEATURE_NAMES}


# ---------------------------------------------------------------- utilidades


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def _lab(image: np.ndarray) -> np.ndarray:
    """CIELab em unidades reais: L ∈ [0, 100], a e b aproximadamente em [−128, 127]."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _contour_band(contour: np.ndarray, shape: tuple[int, int], width: int) -> np.ndarray:
    band = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(band, [contour], -1, 1, thickness=width)
    return band.astype(bool)


# ---------------------------------------------------------------- forma


def _asymmetry(mask: np.ndarray) -> dict[str, float]:
    """A: fração da lesão sem correspondente na reflexão em torno de cada eixo principal.

    0 = simétrica. Os eixos são os autovetores da covariância das coordenadas da lesão; a
    reflexão é feita nas coordenadas, sem rotacionar a imagem (não há corte na borda).
    """
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, vecs = np.linalg.eigh(np.cov(centered.T))
    major, minor = vecs[:, 1], vecs[:, 0]
    u, v = centered @ major, centered @ minor
    h, w = mask.shape
    out = {}
    for name, (ru, rv) in (
        ("assimetria_eixo_maior", (u, -v)),  # reflexão em torno do eixo maior
        ("assimetria_eixo_menor", (-u, v)),  # reflexão em torno do eixo menor
    ):
        pts = np.outer(ru, major) + np.outer(rv, minor) + centroid
        px = np.rint(pts).astype(int)
        inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        hit = np.zeros(len(px), dtype=bool)
        hit[inside] = mask[px[inside, 1], px[inside, 0]]
        out[name] = 1.0 - hit.mean()
    return out


def _fractal_dimension(contour: np.ndarray, shape: tuple[int, int]) -> float:
    """Dimensão de box-counting do contorno (1 = curva suave; perto de 2 = muito irregular)."""
    edge = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(edge, [contour], -1, 1, thickness=1)
    ys, xs = np.nonzero(edge)
    span = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 2)
    sizes = 2 ** np.arange(1, int(np.log2(span)))
    if len(sizes) < 2:
        return math.nan
    counts = [len({(x // s, y // s) for x, y in zip(xs, ys, strict=True)}) for s in sizes]
    slope, _ = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return float(slope)


def _border_shape(mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    area = float(mask.sum())
    perimeter = cv2.arcLength(contour, closed=True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    pts = contour[:, 0, :].astype(np.float64)
    ys, xs = np.nonzero(mask)
    radii = np.linalg.norm(pts - (xs.mean(), ys.mean()), axis=1)
    return {
        "compacidade": perimeter**2 / (4 * np.pi * area),
        "solidez": cv2.contourArea(contour) / hull_area if hull_area > 0 else math.nan,
        "dimensao_fractal": _fractal_dimension(contour, mask.shape),
        "variacao_raio": radii.std() / radii.mean() if radii.mean() > 0 else math.nan,
    }


def _border_sharpness(l_u8: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    gx = cv2.Sobel(l_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_u8, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    pts = contour[:, 0, :]
    edges = cv2.Canny(l_u8, CANNY_LOW, CANNY_HIGH) > 0
    band = _contour_band(contour, mask.shape, BORDER_BAND)
    return {
        "sobel_borda": magnitude[pts[:, 1], pts[:, 0]].mean(),
        "canny_borda": edges[band].mean(),
    }


def _dimension(mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    hull = cv2.convexHull(contour)[:, 0, :].astype(np.float64)
    diffs = hull[:, None, :] - hull[None, :, :]
    return {
        "area_relativa": mask.mean(),
        "diametro_max_px": np.sqrt((diffs**2).sum(axis=2)).max(),
    }


# ---------------------------------------------------------------- cor


def _reference_lab() -> np.ndarray:
    rgb = np.array([list(REFERENCE_COLORS.values())], dtype=np.uint8)
    return _lab(rgb)[0]


def _color(image: np.ndarray, lab: np.ndarray, mask: np.ndarray, roi: np.ndarray) -> dict:
    out: dict[str, float] = {}
    lesion_lab = lab[mask]
    for i, name in enumerate(_LAB_NAMES):
        out[f"{name}_media"] = lesion_lab[:, i].mean()
        out[f"{name}_dp"] = lesion_lab[:, i].std()

    # HSV com H em graus [0, 360) e S, V em [0, 1]; H é circular (média e dp circulares)
    hsv = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)[mask]
    hue = np.deg2rad(hsv[:, 0])
    resultant = min(float(np.hypot(np.sin(hue).mean(), np.cos(hue).mean())), 1.0)
    out["H_media"] = np.rad2deg(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) % 360
    out["H_dp"] = np.rad2deg(np.sqrt(-2 * np.log(max(resultant, 1e-12))))
    for i, name in ((1, "S"), (2, "V")):
        out[f"{name}_media"] = hsv[:, i].mean()
        out[f"{name}_dp"] = hsv[:, i].std()

    refs = _reference_lab()
    nearest = np.linalg.norm(lesion_lab[:, None, :] - refs[None], axis=2).argmin(axis=1)
    fractions = np.bincount(nearest, minlength=len(refs)) / len(nearest)
    out["n_cores"] = float((fractions >= COLOR_MIN_FRACTION).sum())

    # ΔE (CIE76) entre as médias da lesão e de uma faixa de pele ao seu redor
    m = mask.astype(np.uint8)
    dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    ring = (dist > SKIN_RING_INNER) & (dist <= SKIN_RING_OUTER) & roi
    if ring.sum() < MIN_LESION_PIXELS:
        out["delta_e_pele"] = math.nan
    else:
        out["delta_e_pele"] = np.linalg.norm(lesion_lab.mean(axis=0) - lab[ring].mean(axis=0))
    return out


# ---------------------------------------------------------------- textura


def _texture(l_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}

    # GLCM só com pares de pixels da lesão: fora dela o nível é 0 e a linha/coluna 0 é
    # descartada antes de normalizar
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = l_u8[y0:y1, x0:x1].astype(np.int32) * GLCM_LEVELS // 256 + 1
    crop[~mask[y0:y1, x0:x1]] = 0
    glcm = graycomatrix(
        crop.astype(np.uint8), [1], GLCM_ANGLES, levels=GLCM_LEVELS + 1, symmetric=True
    )[1:, 1:].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)
    for prop in GLCM_PROPS:
        out[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()

    lbp = local_binary_pattern(l_u8, LBP_P, LBP_R, method="uniform")
    hist = np.bincount(lbp[mask].astype(int), minlength=LBP_BINS)[:LBP_BINS]
    hist = hist / hist.sum()
    for i, v in enumerate(hist):
        out[f"lbp_{i}"] = v
    return out
