"""Métricas de imagem e de segmentação."""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    total = pred.sum() + ref.sum()
    return 1.0 if total == 0 else float(2 * (pred & ref).sum() / total)


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    union = (pred | ref).sum()
    return 1.0 if union == 0 else float((pred & ref).sum() / union)


def psnr(restored: np.ndarray, reference: np.ndarray) -> float:
    """PSNR (dB) entre duas imagens RGB uint8; infinito se forem idênticas."""
    if np.array_equal(restored, reference):
        return float("inf")
    return float(peak_signal_noise_ratio(reference, restored, data_range=255))


def ssim(restored: np.ndarray, reference: np.ndarray) -> float:
    return float(structural_similarity(reference, restored, channel_axis=2, data_range=255))


def lab(image: np.ndarray) -> np.ndarray:
    """CIELab em unidades reais (L ∈ [0, 100])."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def delta_e(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """ΔE (CIE76) entre as cores médias de duas imagens RGB, opcionalmente dentro da máscara."""
    la, lb = lab(a), lab(b)
    if mask is not None:
        la, lb = la[mask], lb[mask]
    return float(np.linalg.norm(la.reshape(-1, 3).mean(axis=0) - lb.reshape(-1, 3).mean(axis=0)))


def rms_contrast(image: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Contraste RMS: desvio-padrão do canal L (0–100) na região útil."""
    luminance = lab(image)[..., 0]
    return float(luminance[roi].std() if roi is not None else luminance.std())
