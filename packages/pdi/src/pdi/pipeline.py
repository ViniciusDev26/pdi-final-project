"""Pipeline configurável: pré-processamento (P1–P5) seguido da segmentação (S).

Cada etapa de pré-processamento pode ser ligada ou desligada pela `PipelineConfig`, o que
permite montar os cenários acumulativos (A) e de remoção (L) do plano experimental
(docs/PDI_Etapa2_Plano_Experimental.md). `run` devolve todas as imagens intermediárias e
o tempo de cada etapa, usados pelo relatório, pelo simulador e pela página interativa.

Treino e API devem chamar sempre `run`, para que a inferência aplique exatamente o mesmo
processamento usado no treino.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from pdi import preprocess
from pdi.segment import SegmentationMethod, segment

# ordem fixa das etapas de pré-processamento: (ID do plano, campo da config, nome)
PREPROCESS_STAGES: tuple[tuple[str, str, str], ...] = (
    ("P1", "vignette", "Recorte da vinheta"),
    ("P2", "hair", "Remoção de pelos"),
    ("P3", "color", "Normalização de cor"),
    ("P4", "contrast", "Realce de contraste (CLAHE)"),
    ("P5", "denoise", "Filtragem (mediana)"),
)
STAGE_IDS = tuple(sid for sid, _, _ in PREPROCESS_STAGES)
_FIELD = {sid: f for sid, f, _ in PREPROCESS_STAGES}


@dataclass(frozen=True)
class PipelineConfig:
    vignette: bool = True
    hair: bool = True
    color: bool = True
    contrast: bool = True
    denoise: bool = True
    segmentation: SegmentationMethod | None = "otsu"  # None desliga a etapa S
    seg_postprocess: bool = True

    @classmethod
    def raw(cls) -> "PipelineConfig":
        """Cenário A0: nenhuma etapa de pré-processamento."""
        return cls.only()

    @classmethod
    def only(cls, *stage_ids: str) -> "PipelineConfig":
        """Liga apenas as etapas indicadas (ex.: `only("P1", "P2")` = cenário A2)."""
        _check(stage_ids)
        return cls(**{f: sid in stage_ids for sid, f, _ in PREPROCESS_STAGES})

    def without(self, *stage_ids: str) -> "PipelineConfig":
        """Desliga as etapas indicadas (ex.: `PipelineConfig().without("P3")` = cenário L3)."""
        _check(stage_ids)
        return replace(self, **{_FIELD[sid]: False for sid in stage_ids})

    def enabled(self, stage_id: str) -> bool:
        return getattr(self, _FIELD[stage_id])

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        return tuple(sid for sid in STAGE_IDS if self.enabled(sid))

    def to_dict(self) -> dict:
        return asdict(self)


def _check(stage_ids: tuple[str, ...]) -> None:
    unknown = set(stage_ids) - set(STAGE_IDS)
    if unknown:
        raise ValueError(f"etapas desconhecidas: {sorted(unknown)}; válidas: {STAGE_IDS}")


@dataclass(frozen=True)
class StageOutput:
    id: str
    name: str
    enabled: bool
    image: np.ndarray  # imagem após a etapa (igual à anterior se a etapa estiver desligada)
    seconds: float
    mask: np.ndarray | None = None  # máscara produzida pela etapa (vinheta, pelos), se houver


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
    stages: list[StageOutput] = []

    def step(sid: str, name: str, fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]):
        nonlocal current
        if not config.enabled(sid):
            stages.append(StageOutput(sid, name, False, current, 0.0))
            return
        t0 = time.perf_counter()
        current, mask = fn(current)
        stages.append(StageOutput(sid, name, True, current, time.perf_counter() - t0, mask))

    def p1(img):
        nonlocal roi, vignette_detected
        res = preprocess.remove_vignette(img)
        roi, vignette_detected = ~res.mask, res.detected
        return res.image, res.mask

    names = {sid: name for sid, _, name in PREPROCESS_STAGES}
    step("P1", names["P1"], p1)
    step("P2", names["P2"], lambda img: preprocess.remove_hair(img, roi))
    step("P3", names["P3"], lambda img: (preprocess.shades_of_gray(img, roi), None))
    step("P4", names["P4"], lambda img: (preprocess.clahe(img, roi), None))
    step("P5", names["P5"], lambda img: (preprocess.denoise(img, roi), None))

    lesion_mask = None
    timings = {s.id: s.seconds for s in stages}
    if config.segmentation is not None:
        t0 = time.perf_counter()
        lesion_mask = segment(current, roi, config.segmentation, config.seg_postprocess)
        timings["S"] = time.perf_counter() - t0

    return PipelineResult(
        config=config,
        input=image,
        stages=tuple(stages),
        image=current,
        roi=roi,
        lesion_mask=lesion_mask,
        vignette_detected=vignette_detected,
        timings=timings,
    )
