"""Inferência com os modelos treinados dos cenários (Triagem e Simulador).

Cada cenário com modelo salvo em `results/<ID>/` (`model.joblib` ou `model.pt`) e
`metrics.json` vira um `TrainedModel`. Um modelo só vale para a entrada com que foi
treinado (plano, seção 6.2): o pipeline do cenário e, nos tradicionais, o subconjunto de
características. Por isso o Simulador só mostra probabilidade quando a combinação montada
é exatamente o pipeline de um cenário (`matching`).

V1 fica de fora: o preditor é só o indicador de vinheta, não uma imagem.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pdi import PipelineConfig, features
from pdi import run as run_pipeline
from training import status
from training.scenarios import SCENARIOS, Scenario

EXCLUDED = frozenset({"V1"})


@dataclass(frozen=True)
class TrainedModel:
    sid: str
    description: str
    kind: str  # "tradicional" ou "cnn"
    model: str  # svm, rf, cnn
    config: PipelineConfig  # pipeline da entrada do modelo
    feature_groups: tuple[str, ...] | None
    threshold: float
    auc_val: float | None
    auc_test: float | None

    def label(self, p: float) -> str:
        return "malignant" if p >= self.threshold else "benign"


def input_config(sc: Scenario) -> PipelineConfig:
    """Pipeline aplicado antes do modelo (C6 usa a imagem bruta, após P0)."""
    if sc.model == "cnn" and sc.cnn_input == "raw":
        return PipelineConfig.raw()
    return sc.pipeline


def _model_file(sid: str):
    d = status.scenario_dir(sid)
    for name in ("model.pt", "model.joblib"):
        if (d / name).exists():
            return d / name
    return None


def available() -> list[TrainedModel]:
    out = []
    for sid, sc in SCENARIOS.items():
        if sid in EXCLUDED or sc.model is None or sc.pipeline is None:
            continue
        metrics_path = status.scenario_dir(sid) / "metrics.json"
        if _model_file(sid) is None or not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        if "limiar" not in m:
            continue
        out.append(
            TrainedModel(
                sid=sid,
                description=sc.description,
                kind="cnn" if sc.model == "cnn" else "tradicional",
                model=sc.model,
                config=input_config(sc),
                feature_groups=sc.feature_groups,
                threshold=float(m["limiar"]),
                auc_val=m.get("validacao", {}).get("auc"),
                auc_test=m.get("teste", {}).get("auc", {}).get("valor"),
            )
        )
    return out


def get(sid: str) -> TrainedModel:
    for m in available():
        if m.sid == sid:
            return m
    raise KeyError(f"{sid} não tem modelo treinado")


def matching(config: PipelineConfig) -> list[TrainedModel]:
    """Modelos treinados exatamente sobre este pipeline."""
    return [m for m in available() if m.config == config]


@lru_cache(maxsize=32)
def _joblib(sid: str):
    import joblib

    return joblib.load(status.scenario_dir(sid) / "model.joblib")


def feature_vector(feats: dict, groups: tuple[str, ...] | None) -> np.ndarray:
    names = (
        features.FEATURE_NAMES
        if groups is None
        else tuple(n for g in groups for n in features.FEATURE_GROUPS[g])
    )
    row = [feats.get(n) for n in names]
    return np.array([[math.nan if v is None else v for v in row]], dtype=float)


def predict_traditional(m: TrainedModel, feats: dict) -> float:
    """Probabilidade a partir das características (F) já extraídas no pipeline do modelo."""
    model = _joblib(m.sid)
    return float(model.predict_proba(feature_vector(feats, m.feature_groups))[0, 1])


def predict(m: TrainedModel, image: np.ndarray, feats: dict | None = None) -> float:
    """Probabilidade de melanoma para a imagem (após P0). `feats` evita refazer o pipeline."""
    if m.kind == "cnn":
        from training import cnn

        return float(cnn.predict_images(m.sid, [image])[0])
    if feats is None:
        res = run_pipeline(image, m.config)
        feats = features.extract(res.image, res.lesion_mask, res.roi)
    return predict_traditional(m, feats)
