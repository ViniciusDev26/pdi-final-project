from fastapi import FastAPI, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from api.predictor import DummyPredictor, Predictor
from pdi import load_rgb

app = FastAPI(title="PDI Melanoma API", version="0.1.0")
predictor: Predictor = DummyPredictor()


class PredictResponse(BaseModel):
    label: str
    probability_malignant: float
    threshold: float
    model: str
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": predictor.name}


@app.post("/predict")
async def predict(file: UploadFile) -> PredictResponse:
    try:
        image = load_rgb(await file.read())
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Arquivo não é uma imagem válida") from exc
    result = predictor.predict(image)
    return PredictResponse(
        label=result.label,
        probability_malignant=result.probability_malignant,
        threshold=result.threshold,
        model=predictor.name,
        model_version=predictor.version,
    )
