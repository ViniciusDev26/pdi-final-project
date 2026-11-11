"""Etapa S: segmentação da lesão (cenários S1–S4 do plano experimental)."""

from typing import Literal

import cv2
import numpy as np

SegmentationMethod = Literal["otsu", "adaptive"]

ADAPTIVE_BLOCK_SIZE = 51  # vizinhança (px, ímpar) da limiarização adaptativa
ADAPTIVE_C = 2
MORPH_KERNEL_SIZE = 5


def threshold(gray: np.ndarray, roi: np.ndarray, method: SegmentationMethod) -> np.ndarray:
    """Máscara bruta da lesão (pixels mais escuros que a pele), restrita à ROI."""
    if method == "otsu":
        # limiar de Otsu calculado só com os pixels da ROI (ignora a vinheta)
        values = gray[roi].reshape(-1, 1)
        t, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = gray <= t
    elif method == "adaptive":
        mask = (
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                ADAPTIVE_BLOCK_SIZE,
                ADAPTIVE_C,
            )
            > 0
        )
    else:
        raise ValueError(f"método de segmentação desconhecido: {method!r}")
    return mask & roi


def _roi_border(roi: np.ndarray) -> np.ndarray:
    """Faixa de 2 px na fronteira da região útil (borda da imagem ou limite da vinheta)."""
    padded = np.pad(roi, 1, constant_values=False).astype(np.uint8)
    eroded = cv2.erode(padded, np.ones((5, 5), np.uint8))[1:-1, 1:-1].astype(bool)
    return roi & ~eroded


def postprocess(mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Abertura + fechamento, depois escolhe um único componente e preenche seus buracos.

    Ordem de preferência (a lesão costuma estar centralizada na dermatoscopia):
    1. o componente que contém o centro da imagem (inclusive lesões grandes que encostam
       na borda);
    2. o maior componente que não toca a fronteira da ROI (componentes na fronteira costumam
       ser sombras de iluminação ou restos da vinheta);
    3. o maior componente.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    if not contours:
        return out.astype(bool)

    border = _roi_border(roi)

    def touches_border(c: np.ndarray) -> bool:
        xs, ys = c[:, 0, 0], c[:, 0, 1]
        return bool(border[ys, xs].any())

    h, w = m.shape
    center = (w / 2, h / 2)
    central = [c for c in contours if cv2.pointPolygonTest(c, center, False) >= 0]
    inner = [c for c in contours if not touches_border(c)]
    best = max(central or inner or contours, key=cv2.contourArea)
    cv2.drawContours(out, [best], -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def segment(
    image: np.ndarray,
    roi: np.ndarray,
    method: SegmentationMethod = "otsu",
    postprocessing: bool = True,
) -> np.ndarray:
    """Segmenta a lesão de uma imagem RGB uint8. Devolve máscara booleana (H, W)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = threshold(gray, roi, method)
    return postprocess(mask, roi) if postprocessing else mask
