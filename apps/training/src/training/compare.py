"""Simulador, modo "uma imagem" (plano experimental, seção 6.2).

Executa várias combinações de técnicas sobre a mesma imagem e calcula as diferenças entre
elas. Por enquanto compara as imagens intermediárias e as máscaras; as diferenças de
indicadores ABCD e de probabilidade entram quando houver características e modelos.
"""

from dataclasses import dataclass

import numpy as np

from pdi import SEGMENTATION_METHODS, STAGES, PipelineConfig, PipelineResult, run
from pdi.metrics import dice
from pdi.visualize import captioned, grid, overlay_contour

MAX_COMBINATIONS = 4


@dataclass(frozen=True)
class Combination:
    label: str
    config: PipelineConfig


@dataclass(frozen=True)
class Comparison:
    combinations: tuple[Combination, ...]
    results: tuple[PipelineResult, ...]
    mask_dice: np.ndarray  # (n, n): Dice entre as máscaras; NaN se alguma não existir


def compare(image: np.ndarray, combinations: list[Combination]) -> Comparison:
    if not 1 <= len(combinations) <= MAX_COMBINATIONS:
        raise ValueError(f"informe de 1 a {MAX_COMBINATIONS} combinações")
    results = tuple(run(image, c.config) for c in combinations)
    n = len(results)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = results[i].lesion_mask, results[j].lesion_mask
            if a is not None and b is not None:
                matrix[i, j] = dice(a, b)
    return Comparison(tuple(combinations), results, matrix)


def segmentation_label(config: PipelineConfig) -> str:
    if config.segmentation is None:
        return "desligada"
    post = "com pós-processamento" if config.seg_postprocess else "sem pós-processamento"
    return f"{SEGMENTATION_METHODS[config.segmentation]}, {post}"


def figure(comparison: Comparison) -> np.ndarray:
    """Grade: uma linha por combinação; colunas = entrada, P1–P5 e máscara final."""
    rows = []
    for combo, result in zip(comparison.combinations, comparison.results, strict=True):
        tiles = [captioned(result.input, [combo.label, "Entrada"])]
        for stage, out in zip(STAGES, result.stages, strict=True):
            caption = out.technique_label if out.enabled else "desligada"
            tiles.append(captioned(out.image, [f"{stage.id} {stage.name}", caption]))
        mask_tile = overlay_contour(result.image, result.lesion_mask)
        tiles.append(captioned(mask_tile, ["S Segmentação", segmentation_label(result.config)]))
        rows.append(tiles)
    return grid(rows)
