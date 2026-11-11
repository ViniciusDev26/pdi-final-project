"""Cenários congelados do plano experimental (docs/PDI_Etapa2_Plano_Experimental.md).

IDs são permanentes: um cenário nunca é renomeado nem reaproveitado. Alterações aqui
precisam ser registradas no histórico do plano.

Este módulo não importa torch: a API o usa para listar os cenários e oferecer as
configurações de pipeline como presets do simulador.
"""

from dataclasses import dataclass

from pdi import PipelineConfig


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str  # A, L, T, S (C e V entram quando houver modelos)
    description: str
    pipeline: PipelineConfig


_FULL = PipelineConfig()  # A5

_PIPELINE_SCENARIOS: tuple[Scenario, ...] = (
    # 5.1 acumulativos
    Scenario("A0", "A", "Imagem bruta, sem pré-processamento", PipelineConfig.raw()),
    Scenario("A1", "A", "Só P1 (vinheta)", PipelineConfig.only("P1")),
    Scenario("A2", "A", "P1 + P2 (pelos)", PipelineConfig.only("P1", "P2")),
    Scenario("A3", "A", "P1 a P3 (+ cor)", PipelineConfig.only("P1", "P2", "P3")),
    Scenario("A4", "A", "P1 a P4 (+ contraste)", PipelineConfig.only("P1", "P2", "P3", "P4")),
    Scenario("A5", "A", "Pipeline completo (padrão)", _FULL),
    # 5.2 remoção
    Scenario("L1", "L", "Completo sem P1 (vinheta)", _FULL.without("P1")),
    Scenario("L2", "L", "Completo sem P2 (pelos)", _FULL.without("P2")),
    Scenario("L3", "L", "Completo sem P3 (cor)", _FULL.without("P3")),
    Scenario("L4", "L", "Completo sem P4 (contraste)", _FULL.without("P4")),
    Scenario("L5", "L", "Completo sem P5 (filtro)", _FULL.without("P5")),
    # 6.3 técnica alternativa
    Scenario("T1", "T", "P2 = fechamento morfológico", _FULL.using(P2="closing")),
    Scenario("T2", "T", "P3 = Gray World", _FULL.using(P3="gray_world")),
    Scenario("T3", "T", "P3 = White Patch", _FULL.using(P3="white_patch")),
    Scenario("T4", "T", "P4 = equalização global de histograma", _FULL.using(P4="equalize")),
    Scenario("T5", "T", "P5 = gaussiano", _FULL.using(P5="gaussian")),
    # 5.5 segmentação
    Scenario(
        "S1",
        "S",
        "Otsu global, sem pós-processamento",
        PipelineConfig(segmentation="otsu", seg_postprocess=False),
    ),
    Scenario(
        "S2",
        "S",
        "Adaptativa, sem pós-processamento",
        PipelineConfig(segmentation="adaptive", seg_postprocess=False),
    ),
    Scenario("S3", "S", "Otsu + morfologia + componente central (padrão)", _FULL),
    Scenario(
        "S4",
        "S",
        "Adaptativa + morfologia + componente central",
        PipelineConfig(segmentation="adaptive"),
    ),
)

SCENARIOS: dict[str, Scenario] = {s.id: s for s in _PIPELINE_SCENARIOS}
