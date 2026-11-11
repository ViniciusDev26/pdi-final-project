"""Pipeline configurável: pré-processamento (P1–P5) seguido da segmentação (S).

Cada etapa de pré-processamento pode ser desligada ou executada com uma das técnicas do
registro `STAGES` (plano experimental, seção 6.1). Isso permite montar os cenários A, L e
T de docs/PDI_Etapa2_Plano_Experimental.md e qualquer combinação do simulador.
`run` devolve todas as imagens intermediárias e o tempo de cada etapa.

Treino e API devem chamar sempre `run`, para que a inferência aplique exatamente o mesmo
processamento usado no treino.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from pdi import preprocess
from pdi.segment import SegmentationMethod, segment

# uma técnica recebe (imagem, roi) e devolve (imagem, máscara opcional produzida pela etapa)
StageFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray | None]]


@dataclass(frozen=True)
class Technique:
    key: str
    label: str
    fn: StageFn


@dataclass(frozen=True)
class Stage:
    id: str  # ID do plano experimental
    field: str  # campo correspondente em PipelineConfig
    name: str
    techniques: tuple[Technique, ...]  # a primeira é a padrão

    @property
    def default(self) -> str:
        return self.techniques[0].key

    def technique(self, key: str) -> Technique:
        for t in self.techniques:
            if t.key == key:
                return t
        valid = [t.key for t in self.techniques]
        raise ValueError(f"técnica {key!r} inválida para {self.id}; válidas: {valid}")


def _no_mask(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> StageFn:
    return lambda img, roi: (fn(img, roi), None)


def _vignette(img: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = preprocess.remove_vignette(img)
    return res.image, res.mask


# ordem fixa das etapas
STAGES: tuple[Stage, ...] = (
    Stage(
        "P1",
        "vignette",
        "Recorte da vinheta",
        (Technique("fill", "Preenchimento com a cor da pele", _vignette),),
    ),
    Stage(
        "P2",
        "hair",
        "Remoção de pelos",
        (
            Technique("dullrazor", "DullRazor (black-hat + inpainting)", preprocess.remove_hair),
            Technique("closing", "Fechamento morfológico", preprocess.remove_hair_closing),
        ),
    ),
    Stage(
        "P3",
        "color",
        "Normalização de cor",
        (
            Technique(
                "shades_of_gray", "Shades of Gray (p = 6)", _no_mask(preprocess.shades_of_gray)
            ),
            Technique("gray_world", "Gray World (p = 1)", _no_mask(preprocess.gray_world)),
            Technique("white_patch", "White Patch (p = ∞)", _no_mask(preprocess.white_patch)),
        ),
    ),
    Stage(
        "P4",
        "contrast",
        "Realce de contraste",
        (
            Technique("clahe", "CLAHE (canal L)", _no_mask(preprocess.clahe)),
            Technique(
                "equalize", "Equalização global de histograma", _no_mask(preprocess.equalize)
            ),
        ),
    ),
    Stage(
        "P5",
        "denoise",
        "Filtragem",
        (
            Technique("median", "Mediana", _no_mask(preprocess.denoise)),
            Technique("gaussian", "Gaussiano", _no_mask(preprocess.gaussian)),
        ),
    ),
)
STAGE_IDS = tuple(s.id for s in STAGES)
_BY_ID = {s.id: s for s in STAGES}

SEGMENTATION_METHODS: dict[str, str] = {
    "otsu": "Otsu (global)",
    "adaptive": "Limiarização adaptativa (gaussiana)",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Técnica de cada etapa; `None` desliga a etapa. Os padrões formam o cenário A5."""

    vignette: str | None = "fill"
    hair: str | None = "dullrazor"
    color: str | None = "shades_of_gray"
    contrast: str | None = "clahe"
    denoise: str | None = "median"
    segmentation: SegmentationMethod | None = "otsu"  # None desliga a etapa S
    seg_postprocess: bool = True

    def __post_init__(self):
        for stage in STAGES:
            key = getattr(self, stage.field)
            if key is not None:
                stage.technique(key)  # valida
        if self.segmentation is not None and self.segmentation not in SEGMENTATION_METHODS:
            raise ValueError(
                f"segmentação {self.segmentation!r} inválida; válidas: {list(SEGMENTATION_METHODS)}"
            )

    @classmethod
    def raw(cls) -> "PipelineConfig":
        """Cenário A0: nenhuma etapa de pré-processamento."""
        return cls.only()

    @classmethod
    def only(cls, *stage_ids: str) -> "PipelineConfig":
        """Liga só as etapas indicadas, com a técnica padrão (ex.: `only("P1", "P2")` = A2)."""
        _check(stage_ids)
        return cls(**{s.field: (s.default if s.id in stage_ids else None) for s in STAGES})

    def without(self, *stage_ids: str) -> "PipelineConfig":
        """Desliga as etapas indicadas (ex.: `PipelineConfig().without("P3")` = L3)."""
        _check(stage_ids)
        return replace(self, **{_BY_ID[sid].field: None for sid in stage_ids})

    def using(self, **techniques: str | None) -> "PipelineConfig":
        """Troca a técnica de etapas (ex.: `PipelineConfig().using(P4="equalize")` = T4)."""
        _check(tuple(techniques))
        return replace(self, **{_BY_ID[sid].field: key for sid, key in techniques.items()})

    def technique(self, stage_id: str) -> str | None:
        return getattr(self, _BY_ID[stage_id].field)

    def enabled(self, stage_id: str) -> bool:
        return self.technique(stage_id) is not None

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        return tuple(sid for sid in STAGE_IDS if self.enabled(sid))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"campos desconhecidos na configuração: {sorted(unknown)}")
        return cls(**data)


def _check(stage_ids: tuple[str, ...]) -> None:
    unknown = set(stage_ids) - set(STAGE_IDS)
    if unknown:
        raise ValueError(f"etapas desconhecidas: {sorted(unknown)}; válidas: {STAGE_IDS}")


@dataclass(frozen=True)
class StageOutput:
    id: str
    name: str
    technique: str | None  # chave da técnica usada; None = etapa desligada
    technique_label: str | None
    image: np.ndarray  # imagem após a etapa (igual à anterior se a etapa estiver desligada)
    seconds: float
    mask: np.ndarray | None = None  # máscara produzida pela etapa (vinheta, pelos), se houver

    @property
    def enabled(self) -> bool:
        return self.technique is not None


@dataclass(frozen=True)
class PipelineResult:
    config: PipelineConfig
    input: np.ndarray
    stages: tuple[StageOutput, ...]
    image: np.ndarray  # imagem pré-processada final (entrada da CNN e das características)
    roi: np.ndarray  # região útil (False sobre a vinheta)
    lesion_mask: np.ndarray | None  # saída da etapa S
    vignette_detected: bool
    timings: dict[str, float] = field(default_factory=dict)

    def stage(self, stage_id: str) -> StageOutput:
        return next(s for s in self.stages if s.id == stage_id)


def run(image: np.ndarray, config: PipelineConfig | None = None) -> PipelineResult:
    """Executa o pipeline sobre uma imagem RGB uint8 (H, W, 3)."""
    config = config or PipelineConfig()
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"esperada imagem RGB uint8 (H, W, 3); recebida {image.shape} {image.dtype}"
        )

    current = image
    roi = preprocess.full_roi(image)
    vignette_detected = False
    outputs: list[StageOutput] = []

    for stage in STAGES:
        key = config.technique(stage.id)
        if key is None:
            outputs.append(StageOutput(stage.id, stage.name, None, None, current, 0.0))
            continue
        technique = stage.technique(key)
        t0 = time.perf_counter()
        current, mask = technique.fn(current, roi)
        seconds = time.perf_counter() - t0
        if stage.id == "P1":  # a vinheta define a região útil das etapas seguintes
            roi, vignette_detected = ~mask, bool(mask.any())
        outputs.append(
            StageOutput(stage.id, stage.name, key, technique.label, current, seconds, mask)
        )

    lesion_mask = None
    timings = {s.id: s.seconds for s in outputs}
    if config.segmentation is not None:
        t0 = time.perf_counter()
        lesion_mask = segment(current, roi, config.segmentation, config.seg_postprocess)
        timings["S"] = time.perf_counter() - t0

    return PipelineResult(
        config=config,
        input=image,
        stages=tuple(outputs),
        image=current,
        roi=roi,
        lesion_mask=lesion_mask,
        vignette_detected=vignette_detected,
        timings=timings,
    )
