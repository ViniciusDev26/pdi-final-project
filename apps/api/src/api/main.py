import json
import math
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from api import annotations
from api.imaging import png_b64
from api.predictor import DummyPredictor, Predictor
from pdi import SEGMENTATION_METHODS, STAGES, PipelineConfig, PipelineResult, load_rgb, run
from pdi import features as pdi_features
from pdi.visualize import overlay_contour
from training.compare import MAX_COMBINATIONS, Combination, compare, figure
from training.scenarios import presets as scenario_presets

app = FastAPI(title="PDI Melanoma API", version="0.2.0")
predictor: Predictor = DummyPredictor()
app.include_router(annotations.router)


# ---------------------------------------------------------------- esquemas


class PredictResponse(BaseModel):
    label: str
    probability_malignant: float
    threshold: float
    model: str
    model_version: str


class TechniqueOut(BaseModel):
    key: str
    label: str


class StageOptionsOut(BaseModel):
    id: str
    field: str  # chave correspondente na configuração
    name: str
    default: str
    techniques: list[TechniqueOut]


class PresetOut(BaseModel):
    id: str
    group: str
    description: str
    config: dict[str, Any]


class OptionsOut(BaseModel):
    stages: list[StageOptionsOut]
    segmentation: list[TechniqueOut]
    default_config: dict[str, Any]
    presets: list[PresetOut]
    max_combinations: int
    feature_groups: dict[str, list[str]]  # grupos da lista fechada de características (F)


class StageResultOut(BaseModel):
    id: str
    name: str
    technique: str | None
    technique_label: str | None
    seconds: float
    image: str  # PNG base64
    mask: str | None  # PNG base64 da máscara produzida pela etapa (vinheta, pelos)


class PipelineOut(BaseModel):
    config: dict[str, Any]
    vignette_detected: bool
    timings: dict[str, float]
    total_seconds: float
    original_shape: list[int]  # imagem recebida, antes da normalização P0
    input: str  # imagem após P0
    stages: list[StageResultOut]
    final: str  # imagem pré-processada final
    lesion_mask: str | None
    overlay: str | None  # imagem final com o contorno da lesão
    lesion_area_fraction: float | None
    features: dict[str, float | None]  # etapa F; None quando não há lesão segmentada


class CombinationOut(BaseModel):
    label: str
    result: PipelineOut


class CompareOut(BaseModel):
    combinations: list[CombinationOut]
    mask_dice: list[list[float | None]]
    figure: str  # PNG base64 da grade comparativa


# ---------------------------------------------------------------- utilidades


async def _read_image(file: UploadFile) -> np.ndarray:
    try:
        return load_rgb(await file.read())
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Arquivo não é uma imagem válida") from exc


def _parse_config(data: Any) -> PipelineConfig:
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Configuração deve ser um objeto JSON")
    try:
        return PipelineConfig.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_json(raw: str, what: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{what}: JSON inválido") from exc


def _pipeline_out(result: PipelineResult) -> PipelineOut:
    mask = result.lesion_mask
    return PipelineOut(
        config=result.config.to_dict(),
        vignette_detected=result.vignette_detected,
        timings=result.timings,
        total_seconds=sum(result.timings.values()),
        original_shape=list(result.original_shape),
        input=png_b64(result.input),
        stages=[
            StageResultOut(
                id=s.id,
                name=s.name,
                technique=s.technique,
                technique_label=s.technique_label,
                seconds=s.seconds,
                image=png_b64(s.image),
                mask=png_b64(s.mask) if s.mask is not None else None,
            )
            for s in result.stages
        ],
        final=png_b64(result.image),
        lesion_mask=png_b64(mask) if mask is not None else None,
        overlay=png_b64(overlay_contour(result.image, mask)) if mask is not None else None,
        lesion_area_fraction=float(mask.mean()) if mask is not None else None,
        features={
            k: None if math.isnan(v) else v
            for k, v in pdi_features.extract(result.image, mask, result.roi).items()
        },
    )


# ---------------------------------------------------------------- rotas


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": predictor.name}


@app.post("/predict")
async def predict(file: UploadFile) -> PredictResponse:
    image = await _read_image(file)
    result = predictor.predict(image)
    return PredictResponse(
        label=result.label,
        probability_malignant=result.probability_malignant,
        threshold=result.threshold,
        model=predictor.name,
        model_version=predictor.version,
    )


@app.get("/pipeline/options")
def pipeline_options() -> OptionsOut:
    """Etapas, técnicas e presets (cenários congelados) para montar o simulador."""
    return OptionsOut(
        stages=[
            StageOptionsOut(
                id=s.id,
                field=s.field,
                name=s.name,
                default=s.default,
                techniques=[TechniqueOut(key=t.key, label=t.label) for t in s.techniques],
            )
            for s in STAGES
        ],
        segmentation=[TechniqueOut(key=k, label=v) for k, v in SEGMENTATION_METHODS.items()],
        default_config=PipelineConfig().to_dict(),
        presets=[
            PresetOut(
                id=s.id, group=s.group, description=s.description, config=s.pipeline.to_dict()
            )
            for s in scenario_presets()
        ],
        max_combinations=MAX_COMBINATIONS,
        feature_groups={g: list(names) for g, names in pdi_features.FEATURE_GROUPS.items()},
    )


@app.post("/pipeline")
async def pipeline(file: UploadFile, config: Annotated[str, Form()] = "{}") -> PipelineOut:
    """Executa o pipeline com a configuração dada e devolve todas as etapas intermediárias."""
    cfg = _parse_config(_parse_json(config, "config"))
    return _pipeline_out(run(await _read_image(file), cfg))


@app.post("/compare")
async def compare_route(file: UploadFile, combinations: Annotated[str, Form()]) -> CompareOut:
    """Simulador: executa de 1 a 4 combinações sobre a mesma imagem e compara.

    `combinations` é uma lista JSON de objetos `{"label": str, "config": {...}}`.
    """
    data = _parse_json(combinations, "combinations")
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_COMBINATIONS:
        raise HTTPException(
            status_code=422, detail=f"Informe de 1 a {MAX_COMBINATIONS} combinações"
        )
    combos = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="Cada combinação deve ser um objeto")
        label = str(item.get("label") or f"Combinação {i}")
        combos.append(Combination(label, _parse_config(item.get("config", {}))))

    cmp = compare(await _read_image(file), combos)
    dice = [[None if np.isnan(v) else float(v) for v in row] for row in cmp.mask_dice]
    return CompareOut(
        combinations=[
            CombinationOut(label=c.label, result=_pipeline_out(r))
            for c, r in zip(cmp.combinations, cmp.results, strict=True)
        ],
        mask_dice=dice,
        figure=png_b64(figure(cmp)),
    )
