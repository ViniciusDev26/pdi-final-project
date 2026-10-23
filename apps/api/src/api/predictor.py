"""Contrato entre o modelo treinado e a API.

O treinamento exporta um modelo; a API só depende de um objeto que
implemente `Predictor`. Enquanto não há modelo treinado, usa-se o
`DummyPredictor`, que permite desenvolver e testar o front e a API.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Prediction:
    probability_malignant: float
    threshold: float

    @property
    def label(self) -> str:
        return "malignant" if self.probability_malignant >= self.threshold else "benign"


class Predictor(Protocol):
    name: str
    version: str

    def predict(self, image: np.ndarray) -> Prediction: ...


class DummyPredictor:
    """Sem valor diagnóstico: devolve uma probabilidade derivada do brilho médio."""

    name = "dummy"
    version = "0.0.0"

    def predict(self, image: np.ndarray) -> Prediction:
        darkness = 1.0 - float(image.mean()) / 255.0
        return Prediction(probability_malignant=round(darkness, 4), threshold=0.5)
