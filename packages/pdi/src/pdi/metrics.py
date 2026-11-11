"""Métricas de imagem e de segmentação."""

import numpy as np


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    total = pred.sum() + ref.sum()
    return 1.0 if total == 0 else float(2 * (pred & ref).sum() / total)


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    union = (pred | ref).sum()
    return 1.0 if union == 0 else float((pred & ref).sum() / union)
