import json
import math
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from api import annotations, dataset_routes, experiments
from api.imaging import png_b64
from api.predictor import triage_predictor
from pdi import (
    SEGMENTATION_METHODS,
    STAGES,
    PipelineConfig,
    PipelineResult,
    load_rgb,
    preprocess,
    run,
)
from pdi import features as pdi_features
from pdi.visualize import overlay_contour, overlay_heatmap
from training import inference
from training.compare import MAX_COMBINATIONS, Combination, compare, figure
from training.scenarios import SCENARIOS
from training.scenarios import presets as scenario_presets

# indicadores ABCD mostrados na Triagem (subconjunto das características F, seção 3.1)
ABCD_FEATURES = (
    "assimetria_eixo_maior",
    "assimetria_eixo_menor",
    "compacidade",
    "variacao_raio",
    "n_cores",
    "delta_e_pele",
    "diametro_max_px",
    "area_relativa",
)
RECOMMENDATIONS = {
    "malignant": "Encaminhar ao dermatologista",
    "benign": "Manter acompanhamento de rotina",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    experiments.queue.start()
    yield


app = FastAPI(title="PDI Melanoma API", version="0.4.0", lifespan=lifespan)
app.include_router(annotations.router)
app.include_router(experiments.router)
app.include_router(dataset_routes.router)


# ---------------------------------------------------------------- esquemas


class PredictResponse(BaseModel):
    label: str
    probability_malignant: float
    threshold: float
    model: str
    model_version: str
    model_description: str
    recommendation: str
    image: str  # imagem após P0 (300 × 300)
    gradcam: str | None  # Grad-CAM sobreposto (só modelos CNN)
    lesion_overlay: str | None  # contorno da segmentação do pipeline tradicional (A5)
    vignette_detected: bool
    abcd: dict[str, float | None]  # indicadores ABCD do pipeline tradicional (A5)


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


class ModelProbabilityOut(BaseModel):
    id: str
    description: str
    kind: str  # tradicional ou cnn
    probability_malignant: float
    threshold: float
    label: str
    auc_test: float | None


class CombinationOut(BaseModel):
    label: str
    result: PipelineOut
    # modelos treinados exatamente sobre o pipeline desta combinação (seção 6.2)
    models: list[ModelProbabilityOut]


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


def _clean(feats: dict) -> dict[str, float | None]:
    return {k: None if math.isnan(v) else float(v) for k, v in feats.items()}


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
        features=_clean(pdi_features.extract(result.image, mask, result.roi)),
    )


def _model_probabilities(image: np.ndarray, out: PipelineOut) -> list[ModelProbabilityOut]:
    """Probabilidade de cada modelo treinado sobre o pipeline desta combinação."""
    config = PipelineConfig.from_dict(out.config)
    rows = []
    for m in inference.matching(config):
        if m.kind == "cnn":
            p = inference.predict(m, image)
        else:
            p = inference.predict_traditional(m, out.features)
        rows.append(
            ModelProbabilityOut(
                id=m.sid,
                description=m.description,
                kind=m.kind,
                probability_malignant=p,
                threshold=m.threshold,
                label=m.label(p),
                auc_test=m.auc_test,
            )
        )
    return rows


# ---------------------------------------------------------------- rotas


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": triage_predictor().name}


@app.post("/predict")
async def predict(file: UploadFile) -> PredictResponse:
    """Triagem: probabilidade do modelo escolhido (seção 2), Grad-CAM e indicadores ABCD."""
    image = preprocess.normalize_input(await _read_image(file))
    predictor = triage_predictor()
    result = predictor.predict(image)
    trad = run(image, SCENARIOS["A5"].pipeline)
    feats = _clean(pdi_features.extract(trad.image, trad.lesion_mask, trad.roi))
    mask = trad.lesion_mask
    return PredictResponse(
        label=result.label,
        probability_malignant=result.probability_malignant,
        threshold=result.threshold,
        model=predictor.name,
        model_version=predictor.version,
        model_description=predictor.description,
        recommendation=RECOMMENDATIONS[result.label],
        image=png_b64(image),
        gradcam=None if result.gradcam is None else png_b64(overlay_heatmap(image, result.gradcam)),
        lesion_overlay=png_b64(overlay_contour(image, mask))
        if mask is not None and mask.any()
        else None,
        vignette_detected=trad.vignette_detected,
        abcd={k: feats[k] for k in ABCD_FEATURES},
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

    raw = await _read_image(file)
    cmp = compare(raw, combos)
    image = preprocess.normalize_input(raw)
    dice = [[None if np.isnan(v) else float(v) for v in row] for row in cmp.mask_dice]
    outs = [_pipeline_out(r) for r in cmp.results]
    return CompareOut(
        combinations=[
            CombinationOut(label=c.label, result=o, models=_model_probabilities(image, o))
            for c, o in zip(cmp.combinations, outs, strict=True)
        ],
        mask_dice=dice,
        figure=png_b64(figure(cmp)),
    )
