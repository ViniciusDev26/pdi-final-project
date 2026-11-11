"""Cenários congelados do plano experimental (docs/PDI_Etapa2_Plano_Experimental.md).

IDs são permanentes: um cenário nunca é renomeado nem reaproveitado. Alterações aqui
precisam ser registradas no histórico do plano.

Este módulo não importa torch: a API o usa para listar os cenários e oferecer as
configurações de pipeline como presets do simulador.
"""

from dataclasses import dataclass
from typing import Literal

from pdi import PipelineConfig

Kind = Literal[
    "ablation",  # A, L, T: características do pipeline + Random Forest de referência
    "segmentation",  # S: Dice/IoU nas máscaras anotadas do teste
    "traditional",  # C1–C5: características F + SVM/RF
    "cnn",  # C6, C7, V3: EfficientNet-B0
    "vignette_shortcut",  # V1
    "vignette_split",  # V2
    "gradcam",  # V4
    "synthetic_vignette",  # V5
]
CnnInput = Literal["raw", "preprocessed", "lesion"]


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str  # A, L, T, S, C, V
    description: str
    pipeline: PipelineConfig | None  # pipeline de PDI usado (None em V1, V2, V4, V5)
    kind: Kind = "ablation"
    model: str | None = "rf"  # rf, svm, cnn
    feature_groups: tuple[str, ...] | None = None  # None = todas as características
    cnn_input: CnnInput | None = None
    depends: tuple[str, ...] = ()  # cenários cujos modelos/predições são usados
    estimate: str = "≈ 1 min"  # tempo estimado no desktop (mostrado antes de executar)

    @property
    def uses_gpu(self) -> bool:
        return self.model == "cnn" or self.kind in ("gradcam",)


_FULL = PipelineConfig()  # A5

_PIPELINE_SCENARIOS: tuple[Scenario, ...] = (
    # 5.1 acumulativos (todos com o Random Forest de referência da ablação)
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
        kind="segmentation",
        model=None,
        estimate="< 1 min",
    ),
    Scenario(
        "S2",
        "S",
        "Adaptativa, sem pós-processamento",
        PipelineConfig(segmentation="adaptive", seg_postprocess=False),
        kind="segmentation",
        model=None,
        estimate="< 1 min",
    ),
    Scenario(
        "S3",
        "S",
        "Otsu + morfologia + componente central (padrão)",
        _FULL,
        kind="segmentation",
        model=None,
        estimate="< 1 min",
    ),
    Scenario(
        "S4",
        "S",
        "Adaptativa + morfologia + componente central",
        PipelineConfig(segmentation="adaptive"),
        kind="segmentation",
        model=None,
        estimate="< 1 min",
    ),
)

# 5.3 comparação de modelos (Q1), todos com o pipeline vencedor da ablação (A5)
_MODEL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "C1",
        "C",
        "Características F (todas) + SVM (RBF)",
        _FULL,
        "traditional",
        "svm",
        estimate="≈ 3 min",
    ),
    Scenario("C2", "C", "Características F (todas) + Random Forest", _FULL, "traditional"),
    Scenario(
        "C3",
        "C",
        "Só forma (A, B, D) + Random Forest",
        _FULL,
        "traditional",
        feature_groups=("forma",),
    ),
    Scenario(
        "C4", "C", "Só cor (C) + Random Forest", _FULL, "traditional", feature_groups=("cor",)
    ),
    Scenario(
        "C5",
        "C",
        "Só textura (GLCM, LBP) + Random Forest",
        _FULL,
        "traditional",
        feature_groups=("textura",),
    ),
    Scenario(
        "C6",
        "C",
        "CNN (EfficientNet-B0) na imagem bruta",
        PipelineConfig.raw(),
        "cnn",
        "cnn",
        cnn_input="raw",
        estimate="≈ 12 min (GPU)",
    ),
    Scenario(
        "C7",
        "C",
        "CNN (EfficientNet-B0) na imagem pré-processada",
        _FULL,
        "cnn",
        "cnn",
        cnn_input="preprocessed",
        estimate="≈ 12 min (GPU)",
    ),
)

# 5.4 viés e artefatos (Q3)
_BIAS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "V1",
        "V",
        'Classificador só com o indicador "tem vinheta"',
        None,
        "vignette_shortcut",
        "logistic",
        estimate="< 1 min",
    ),
    Scenario(
        "V2",
        "V",
        "C2, C6 e C7 avaliados com e sem vinheta",
        None,
        "vignette_split",
        None,
        depends=("C2", "C6", "C7"),
        estimate="< 1 min",
    ),
    Scenario(
        "V3",
        "V",
        "CNN com a pele fora da máscara zerada (só a lesão)",
        _FULL,
        "cnn",
        "cnn",
        cnn_input="lesion",
        estimate="≈ 12 min (GPU)",
    ),
    Scenario(
        "V4",
        "V",
        "Grad-CAM nas imagens anotadas do teste: ativação dentro da máscara",
        None,
        "gradcam",
        None,
        depends=("C6", "C7"),
        estimate="≈ 1 min (GPU)",
    ),
    Scenario(
        "V5",
        "V",
        "Vinheta artificial em 50 benignas: variação da probabilidade",
        None,
        "synthetic_vignette",
        None,
        depends=("C2", "L1", "C6", "C7"),
        estimate="≈ 1 min",
    ),
)

SCENARIOS: dict[str, Scenario] = {
    s.id: s for s in (*_PIPELINE_SCENARIOS, *_MODEL_SCENARIOS, *_BIAS_SCENARIOS)
}
GROUPS = ("A", "L", "T", "S", "C", "V")


def presets() -> list[Scenario]:
    """Cenários que definem só um pipeline de PDI: presets do simulador."""
    return [s for s in SCENARIOS.values() if s.group in ("A", "L", "T", "S")]
