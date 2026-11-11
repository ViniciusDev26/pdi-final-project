"""Biblioteca de PDI compartilhada entre o treinamento e a API.

Todo pré-processamento usado no treino deve morar aqui, para que a API
aplique exatamente as mesmas transformações na inferência.
"""

from pdi.io import load_rgb
from pdi.pipeline import (
    SEGMENTATION_METHODS,
    STAGE_IDS,
    STAGES,
    PipelineConfig,
    PipelineResult,
    run,
)

__all__ = [
    "SEGMENTATION_METHODS",
    "STAGES",
    "STAGE_IDS",
    "PipelineConfig",
    "PipelineResult",
    "load_rgb",
    "run",
]
