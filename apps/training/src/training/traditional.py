"""Modelos tradicionais sobre as características F (protocolo congelado, seção 5.7).

- Imputação pela mediana (NaN = lesão não segmentada) e padronização ajustadas só no treino
  (dentro do Pipeline, portanto também dentro de cada fold).
- Busca em grade, validação cruzada estratificada 5-fold **no treino**, métrica AUC.
- Modelo final reajustado em todo o treino; SVM com probabilidades por Platt.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from training.config import SEED

ModelKind = Literal["svm", "rf"]

GRIDS: dict[str, dict[str, list]] = {
    "svm": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    "rf": {
        "clf__max_features": ["sqrt", 0.3],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_depth": [None, 20],
    },
}
RF_TREES = 500


def _estimator(kind: ModelKind, probability: bool, n_jobs: int) -> Pipeline:
    if kind == "svm":
        clf = SVC(kernel="rbf", probability=probability, random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=RF_TREES, random_state=SEED, n_jobs=n_jobs)
    else:
        raise ValueError(f"modelo desconhecido: {kind!r}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


@dataclass
class FitResult:
    model: Pipeline
    best_params: dict
    cv_auc: float
    cv_auc_std: float


def fit(kind: ModelKind, X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> FitResult:
    """Busca em grade (5-fold no treino) e reajuste final com os melhores parâmetros."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # SVM: a busca usa decision_function (mesma AUC, sem o custo do Platt); RF paraleliza
    # os folds, cada floresta com 1 núcleo
    search = GridSearchCV(
        _estimator(kind, probability=False, n_jobs=1),
        GRIDS[kind],
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(X, y)
    best = search.best_params_
    i = search.best_index_
    model = _estimator(kind, probability=True, n_jobs=n_jobs).set_params(**best)
    model.fit(X, y)
    return FitResult(
        model=model,
        best_params={k.removeprefix("clf__"): v for k, v in best.items()},
        cv_auc=float(search.cv_results_["mean_test_score"][i]),
        cv_auc_std=float(search.cv_results_["std_test_score"][i]),
    )


def predict_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
