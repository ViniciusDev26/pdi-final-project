"""Contrato entre o modelo treinado e a API.

A API só depende de um objeto que implemente `Predictor`. O modelo da Triagem é o
escolhido pela regra da seção 2 do plano (cenário C com maior AUC na validação; hoje o
C6) e é carregado de `results/<ID>/`. Sem nenhum modelo treinado (por exemplo, num clone
sem os arquivos do Git LFS), a API usa o `DummyPredictor`, sem valor diagnóstico, para que
o sistema continue funcionando.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from training import inference, report, status


@dataclass(frozen=True)
class Prediction:
    probability_malignant: float
    threshold: float
    gradcam: np.ndarray | None = None  # mapa 300 × 300 em [0, 1] (só CNN)

    @property
    def label(self) -> str:
        return "malignant" if self.probability_malignant >= self.threshold else "benign"


class Predictor(Protocol):
    name: str
    version: str
    description: str

    def predict(self, image: np.ndarray) -> Prediction: ...


class DummyPredictor:
    """Sem valor diagnóstico: devolve uma probabilidade derivada do brilho médio."""

    name = "dummy"
    version = "0.0.0"
    description = "Modelo de demonstração, sem valor diagnóstico"

    def predict(self, image: np.ndarray) -> Prediction:
        darkness = 1.0 - float(image.mean()) / 255.0
        return Prediction(probability_malignant=round(darkness, 4), threshold=0.5)


class ScenarioPredictor:
    """Modelo treinado de um cenário; para a CNN, devolve também o Grad-CAM."""

    def __init__(self, model: inference.TrainedModel):
        self.model = model
        self.name = model.sid
        self.description = model.description
        metrics = json.loads((status.scenario_dir(model.sid) / "metrics.json").read_text())
        self.version = str(metrics.get("commit", "desconhecido"))

    def predict(self, image: np.ndarray) -> Prediction:
        if self.model.kind == "cnn":
            from training import cnn

            prob, cam, _ = cnn.gradcam(self.model.sid, image)
            return Prediction(prob, self.model.threshold, gradcam=cam)
        return Prediction(inference.predict(self.model, image), self.model.threshold)


@lru_cache(maxsize=1)
def triage_predictor() -> Predictor:
    """Modelo da Triagem pela regra da seção 2; o dummy se não houver modelo treinado."""
    choice = report.triage_choice(report.load_metrics())
    if choice is not None:
        try:
            return ScenarioPredictor(inference.get(choice["id"]))
        except KeyError:
            pass
    return DummyPredictor()
