"""Classificador 100% PDI: regra ABCD da dermatoscopia (Total Dermoscopy Score, TDS).

Stolz et al. (1994): TDS = 1,3·A + 0,1·B + 0,5·C + 0,5·D, com
- A, assimetria: número de eixos principais em que a lesão é assimétrica (0–2);
- B, borda: número de segmentos, de 8, com corte abrupto do pigmento (0–8);
- C, cores: número de cores dermatoscópicas presentes, de 6 (1–6);
- D, estruturas diferenciais: quantas de 5 estruturas aparecem (rede pigmentar, áreas sem
  estrutura, pontos, glóbulos, estrias), no mínimo 1 (1–5).
TDS < 4,75 = benigna; 4,75–5,45 = suspeita; > 5,45 = altamente suspeita de melanoma. Na
triagem, a partir de 4,75 a lesão é encaminhada.

Nada aqui é aprendido com os rótulos (benigna/maligna):
- pesos e cortes do TDS vêm da literatura;
- as medidas são PDI (morfologia, gradiente de Sobel, estatística local, componentes conexos);
- os critérios do TDS são julgamentos visuais do dermatologista; para convertê-los em sim/não
  automaticamente, cada medida é comparada com um corte `Cutoffs`, que é a **mediana da medida
  nas lesões do treino, calculada sem olhar os rótulos** ("mais assimétrica / mais abrupta /
  com mais pontos que a lesão típica da base"). Ver `training.rule_tds`.

As medidas de A e C reutilizam as características F (`pdi.features`); B e D são calculadas
aqui sobre a imagem pré-processada (A5) e a máscara da lesão.
"""

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Stolz et al. (1994)
WEIGHTS = {"A": 1.3, "B": 0.1, "C": 0.5, "D": 0.5}
CUTOFF_SUSPICIOUS = 4.75
CUTOFF_MELANOMA = 5.45
TDS_MIN, TDS_MAX = 1.0, 8.9  # A=0, B=0, C=1, D=1  …  A=2, B=8, C=6, D=5

N_SECTORS = 8
# Estruturas: diferença mínima de luminosidade (canal L, 0–255) para uma estrutura escura se
# destacar do entorno no black-hat. 10 níveis ≈ 4% da escala; fixado a priori, sem calibração.
STRUCTURE_CONTRAST = 10
DOT_MAX_AREA = 12  # px: pontos ≈ 1–4 px de diâmetro em 300 × 300
GLOBULE_AREA = (13, 120)
MIN_CIRCULARITY = 0.6
STREAK_MIN_ELONGATION = 3.0
STRUCTURELESS_STD = 4.0  # desvio padrão local do canal L abaixo do qual a área é homogênea
PERIPHERY = 0.7  # estrias: fora de 70% do raio médio (anel da borda)

MEASURE_NAMES = (
    "borda_setor",  # gradiente médio em cada setor (comparado setor a setor com o corte)
    "pontos",
    "globulos",
    "rede",
    "sem_estrutura",
    "estrias",
)
STRUCTURES = ("rede", "sem_estrutura", "pontos", "globulos", "estrias")


@dataclass(frozen=True)
class Cutoffs:
    """Cortes de cada medida (medianas sem rótulo do treino)."""

    assimetria: float
    borda_setor: float
    rede: float
    sem_estrutura: float
    pontos: float
    globulos: float
    estrias: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Cutoffs":
        return cls(**data)


def _l_channel(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def _border_sectors(l_u8: np.ndarray, mask: np.ndarray) -> list[float]:
    """Gradiente de Sobel médio sobre o contorno em cada um dos 8 setores angulares."""
    contour = _largest_contour(mask)[:, 0, :]
    gx = cv2.Sobel(l_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_u8, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    ys, xs = np.nonzero(mask)
    angles = np.arctan2(contour[:, 1] - ys.mean(), contour[:, 0] - xs.mean())
    sector = ((angles + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    values = magnitude[contour[:, 1], contour[:, 0]]
    return [float(values[sector == k].mean()) if (sector == k).any() else 0.0 for k in range(8)]


def _components(binary: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, 1)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = 4 * math.pi * area / perimeter**2 if perimeter > 0 else 1.0
        ys, xs = np.nonzero(comp)
        extent = max(np.ptp(xs), np.ptp(ys)) + 1
        elongation = extent**2 / area
        yield comp, area, circularity, elongation


def _structures(l_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Medidas das 5 estruturas diferenciais dentro da lesão."""
    mask = mask.astype(bool)
    area = mask.sum()
    small = cv2.morphologyEx(
        l_u8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    large = cv2.morphologyEx(
        l_u8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )
    dark_small = (small > STRUCTURE_CONTRAST) & mask
    dark_large = (large > STRUCTURE_CONTRAST) & mask & ~dark_small

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    radius = np.hypot(yy - cy, xx - cx)
    mean_radius = math.sqrt(area / math.pi)
    periphery = radius > PERIPHERY * mean_radius

    dots = globules = streaks = 0
    network = np.zeros_like(mask)
    for comp, a, circ, elong in _components(dark_small):
        if a <= DOT_MAX_AREA and circ >= MIN_CIRCULARITY:
            dots += 1
        elif elong >= STREAK_MIN_ELONGATION and (comp & periphery).sum() > 0.5 * a:
            streaks += 1
        elif a > DOT_MAX_AREA:
            network |= comp  # linhas finas escuras conectadas: malha da rede pigmentar
    for _, a, circ, _ in _components(dark_large):
        if GLOBULE_AREA[0] <= a <= GLOBULE_AREA[1] and circ >= MIN_CIRCULARITY:
            globules += 1

    l_f = l_u8.astype(np.float32)
    mean = cv2.blur(l_f, (7, 7))
    local_std = np.sqrt(np.maximum(cv2.blur(l_f**2, (7, 7)) - mean**2, 0))
    return {
        "pontos": float(dots),
        "globulos": float(globules),
        "rede": float(network.sum() / area),
        "sem_estrutura": float(((local_std < STRUCTURELESS_STD) & mask).sum() / area),
        "estrias": float(streaks),
    }


def measures(image: np.ndarray, mask: np.ndarray | None) -> dict | None:
    """Medidas de B e D sobre a imagem pré-processada; None se não houver lesão."""
    if mask is None or mask.sum() < 50:
        return None
    l_u8 = _l_channel(image)
    return {"borda_setor": _border_sectors(l_u8, mask), **_structures(l_u8, mask)}


def criteria(feats: dict, meas: dict, cut: Cutoffs) -> dict[str, float]:
    """A, B, C e D do TDS a partir das características F (A, C) e das medidas (B, D)."""
    asym = [feats["assimetria_eixo_maior"], feats["assimetria_eixo_menor"]]
    a = sum(v > cut.assimetria for v in asym if v is not None and not math.isnan(v))
    b = sum(v > cut.borda_setor for v in meas["borda_setor"])
    colors = feats.get("n_cores")
    c = 1 if colors is None or math.isnan(colors) else int(min(max(colors, 1), 6))
    present = [s for s in STRUCTURES if meas[s] > getattr(cut, s)]
    d = max(1, len(present))
    return {"A": a, "B": b, "C": c, "D": d, "estruturas": present}


def score(crit: dict) -> float:
    return sum(WEIGHTS[k] * crit[k] for k in WEIGHTS)


def normalized(tds: float) -> float:
    """TDS levado a [0, 1] (só para exibir junto às probabilidades; não é probabilidade)."""
    return (tds - TDS_MIN) / (TDS_MAX - TDS_MIN)
