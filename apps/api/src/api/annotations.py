"""Rotas da página Anotação (plano experimental, seção 5.8)."""

from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel

from api.imaging import png_b64
from pdi import load_rgb, preprocess
from training import annotations as ann
from training import config as cfg

router = APIRouter(prefix="/annotations", tags=["anotação"])


class ItemOut(BaseModel):
    name: str
    path: str
    label: str
    vinheta: bool
    responsavel: str
    compartilhada: bool
    annotators: list[str]
    done: dict[str, bool]  # anotador → já salvou a máscara


class AnnotationListOut(BaseModel):
    conjunto: str
    annotators: list[str]
    items: list[ItemOut]
    progress: list[dict[str, Any]]
    inter_annotator: list[dict[str, Any]]  # Dice entre anotadores (compartilhadas do teste)
    inter_annotator_mean: float | None


class AnnotationImageOut(BaseModel):
    name: str
    image: str  # PNG base64 da imagem após P0 (300 × 300)
    mask: str | None  # máscara já salva por este anotador


def _item(conjunto: str, name: str) -> ann.Item:
    try:
        return ann.find(conjunto, name, cfg.EXPERIMENTS_DIR)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _check_set(conjunto: str) -> None:
    if conjunto not in ann.SETS:
        raise HTTPException(status_code=404, detail=f"conjunto inválido: {conjunto}")


@router.get("/{conjunto}")
def list_annotations(conjunto: str) -> AnnotationListOut:
    _check_set(conjunto)
    items = ann.items(conjunto, cfg.EXPERIMENTS_DIR)
    inter = ann.inter_annotator_dice(cfg.EXPERIMENTS_DIR, cfg.MASKS_DIR)
    return AnnotationListOut(
        conjunto=conjunto,
        annotators=list(cfg.ANNOTATORS),
        items=[
            ItemOut(
                name=i.name,
                path=i.path,
                label=i.label,
                vinheta=i.vinheta,
                responsavel=i.responsavel,
                compartilhada=i.compartilhada,
                annotators=list(i.annotators),
                done={
                    a: ann.mask_path(conjunto, a, i.name, cfg.MASKS_DIR).exists()
                    for a in i.annotators
                },
            )
            for i in items
        ],
        progress=ann.progress(cfg.EXPERIMENTS_DIR, cfg.MASKS_DIR).to_dict("records"),
        inter_annotator=inter.to_dict("records"),
        inter_annotator_mean=float(inter["dice"].mean()) if len(inter) else None,
    )


@router.get("/{conjunto}/{name}")
def annotation_image(conjunto: str, name: str, annotator: str) -> AnnotationImageOut:
    item = _item(conjunto, name)
    path = cfg.DATASET_DIR / item.path
    if not path.exists():
        raise HTTPException(status_code=503, detail="Dataset não disponível nesta máquina")
    image = preprocess.normalize_input(load_rgb(path))
    mask = ann.load_mask(conjunto, annotator, name, cfg.MASKS_DIR)
    return AnnotationImageOut(
        name=name, image=png_b64(image), mask=png_b64(mask) if mask is not None else None
    )


def polygon_mask(points: list[list[float]], size: int = preprocess.INPUT_SIZE) -> np.ndarray:
    """Rasteriza o contorno desenhado (coordenadas na imagem 300 × 300) numa máscara."""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


@router.put("/{conjunto}/{name}")
def save_annotation(
    conjunto: str,
    name: str,
    annotator: Annotated[str, Form()],
    points: Annotated[str, Form()],
) -> dict[str, Any]:
    """Salva a máscara a partir do polígono `points` (JSON: lista de [x, y], ≥ 3 pontos)."""
    import json

    item = _item(conjunto, name)
    try:
        pts = json.loads(points)
        valid = isinstance(pts, list) and len(pts) >= 3 and all(len(p) == 2 for p in pts)
    except (json.JSONDecodeError, TypeError):
        valid = False
    if not valid:
        raise HTTPException(status_code=422, detail="Informe um contorno com pelo menos 3 pontos")
    try:
        path = ann.save_mask(item, annotator, polygon_mask(pts), cfg.MASKS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"saved": str(path.relative_to(cfg.EXPERIMENTS_DIR.parent))}
