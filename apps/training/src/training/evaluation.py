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


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """Métricas no limiar com IC 95%: {métrica: {"valor", "ic95": [inf, sup]}} + matriz."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    point = point_metrics(y, p, threshold)
    samples = {m: [] for m in METRICS}
    for idx in bootstrap_indices(len(y), n_boot):
        for m, v in point_metrics(y[idx], p[idx], threshold).items():
            samples[m].append(v)
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
    y, p_a, p_b = np.asarray(y).astype(int), np.asarray(p_a), np.asarray(p_b)
    t_a, t_b = kw.get("threshold_a", 0.5), kw.get("threshold_b", 0.5)

    def value(idx):
        return (
            point_metrics(y[idx], p_a[idx], t_a)[metric]
            - point_metrics(y[idx], p_b[idx], t_b)[metric]
        )

    full = value(np.arange(len(y)))
    diffs = [value(idx) for idx in bootstrap_indices(len(y), kw.get("n_boot", N_BOOTSTRAP))]
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
