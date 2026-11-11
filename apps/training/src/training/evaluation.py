"""Métricas de classificação, limiar de decisão e intervalos de confiança (plano, seção 2).

- Limiar: escolhido **na validação**, o maior limiar com sensibilidade ≥ 0,90.
- Métrica principal: AUC-ROC no teste; secundárias no limiar fixado.
- Incerteza: IC 95% por bootstrap (1.000 reamostragens do teste, semente 42); diferenças
  entre cenários por bootstrap **pareado** (mesmas reamostragens nos dois cenários).
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from training.config import SEED

TARGET_SENSITIVITY = 0.90
N_BOOTSTRAP = 1000
METRICS = (
    "auc",
    "sensibilidade",
    "especificidade",
    "precisao",
    "f1",
    "acuracia_balanceada",
    "acuracia",
)


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray, target: float = TARGET_SENSITIVITY):
    """Maior limiar cuja sensibilidade na validação é ≥ `target`."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    positives = np.sort(p_val[y_val == 1])[::-1]
    if len(positives) == 0:
        return 0.5
    k = int(np.ceil(target * len(positives)))  # quantos positivos precisam ficar acima
    return float(positives[k - 1])


def _at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens and not np.isnan(prec) else np.nan
    return {
        "sensibilidade": sens,
        "especificidade": spec,
        "precisao": prec,
        "f1": f1,
        "acuracia_balanceada": (sens + spec) / 2,
        "acuracia": (tp + tn) / len(y),
    }


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return {"auc": auc, **_at_threshold(y, p, threshold)}


def bootstrap_indices(n: int, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_weights(n: int, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    """Reamostragens como pesos: W[b, i] = quantas vezes o item i aparece na reamostragem b.

    Equivale a `bootstrap_indices` (mesmas reamostragens), mas permite calcular as métricas
    de todas as reamostragens de uma vez.
    """
    idx = bootstrap_indices(n, n_boot, seed)
    w = np.zeros((n_boot, n))
    np.add.at(w, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
    return w


def _weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUC (Mann-Whitney, empates valem 1/2) para cada linha de pesos `w` (B × n)."""
    order = np.argsort(p, kind="mergesort")
    ps, ys, ws = p[order], y[order], w[:, order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])  # grupos de escores empatados
    neg = np.add.reduceat(ws * (ys == 0), starts, axis=1)
    pos = np.add.reduceat(ws * (ys == 1), starts, axis=1)
    neg_before = np.cumsum(neg, axis=1) - neg
    total = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (neg_before + 0.5 * neg)).sum(axis=1) / total


def _weighted_threshold_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, w: np.ndarray
) -> dict[str, np.ndarray]:
    pred = p >= threshold
    tp, fn = w @ ((y == 1) & pred), w @ ((y == 1) & ~pred)
    tn, fp = w @ ((y == 0) & ~pred), w @ ((y == 0) & pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
        return {
            "sensibilidade": sens,
            "especificidade": spec,
            "precisao": prec,
            "f1": 2 * prec * sens / (prec + sens),
            "acuracia_balanceada": (sens + spec) / 2,
            "acuracia": (tp + tn) / w.sum(axis=1),
        }


def bootstrap_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, w: np.ndarray
) -> dict[str, np.ndarray]:
    """Todas as métricas para cada reamostragem (linhas de `w`)."""
    return {"auc": _weighted_auc(y, p, w), **_weighted_threshold_metrics(y, p, threshold, w)}


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """Métricas no limiar com IC 95%: {métrica: {"valor", "ic95": [inf, sup]}} + matriz."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    point = point_metrics(y, p, threshold)
    samples = bootstrap_metrics(y, p, threshold, bootstrap_weights(len(y), n_boot))
    out = {
        m: {
            "valor": float(point[m]),
            "ic95": [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))],
        }
        for m, s in samples.items()
    }
    pred = (p >= threshold).astype(int)
    # [[VN, FP], [FN, VP]]
    out["matriz_confusao"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    out["n"] = int(len(y))
    out["positivos"] = int(y.sum())
    return out


def paired_difference(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str = "auc", **kw
) -> dict[str, float]:
    """Diferença métrica(a) − métrica(b) com IC 95% por bootstrap pareado."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    t_a, t_b = kw.get("threshold_a", 0.5), kw.get("threshold_b", 0.5)
    full = point_metrics(y, p_a, t_a)[metric] - point_metrics(y, p_b, t_b)[metric]
    w = bootstrap_weights(len(y), kw.get("n_boot", N_BOOTSTRAP))
    diffs = bootstrap_metrics(y, p_a, t_a, w)[metric] - bootstrap_metrics(y, p_b, t_b, w)[metric]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diferenca": float(full), "ic95": [float(lo), float(hi)]}


def roc_points(y: np.ndarray, p: np.ndarray, max_points: int = 200) -> dict[str, list[float]]:
    """Curva ROC reduzida para gráficos (no máximo `max_points` pontos)."""
    fpr, tpr, _ = roc_curve(y, p)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}


def to_jsonable(obj):
    """Converte tipos do numpy e troca NaN/infinito por None (JSON válido)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
