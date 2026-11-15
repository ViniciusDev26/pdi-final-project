"""Rotas da página Base de dados (plano experimental, seção 7.1).

O resumo vem de `experiments/vinheta.csv` (versionado: caminho, classe, divisão e vinheta de
cada imagem após a remoção de duplicatas), então funciona no modo leitura, sem o dataset.
As imagens em si só existem com o dataset baixado.
"""

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from training import config as cfg

router = APIRouter(prefix="/dataset", tags=["base de dados"])

# Melanoma Skin Cancer Dataset of 10000 Images (Kaggle): 9.605 de treino + 1.000 de teste
ORIGINAL_IMAGES = 10_605
SPLIT_ORDER = ("train", "val", "test")
MAX_LIST = 60


def _table() -> pd.DataFrame:
    path = cfg.EXPERIMENTS_DIR / "vinheta.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="experiments/vinheta.csv ausente")
    return pd.read_csv(path)


def _available() -> bool:
    return cfg.DATASET_DIR.is_dir()


@router.get("/summary")
def summary() -> dict[str, Any]:
    df = _table()
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    vignette = df.groupby(["split", "label"])["vinheta"].mean().unstack()
    return {
        "total": len(df),
        "imagens_originais": ORIGINAL_IMAGES,
        "duplicatas_removidas": ORIGINAL_IMAGES - len(df),
        "contagem": {s: counts.loc[s].to_dict() for s in SPLIT_ORDER if s in counts.index},
        "vinheta": {
            s: {k: float(v) for k, v in vignette.loc[s].items()}
            for s in SPLIT_ORDER
            if s in vignette.index
        },
        "vinheta_geral": {k: float(v) for k, v in df.groupby("label")["vinheta"].mean().items()},
        "dataset_disponivel": _available(),
    }


@router.get("/images")
def images(
    split: str | None = None,
    label: str | None = None,
    vinheta: bool | None = None,
    limit: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    """Amostra de imagens da base com os filtros dados (sorteio reprodutível por `seed`)."""
    df = _table()
    if split:
        df = df[df["split"] == split]
    if label:
        df = df[df["label"] == label]
    if vinheta is not None:
        df = df[df["vinheta"] == vinheta]
    n = min(max(limit, 1), MAX_LIST, len(df))
    sample = df.sample(n, random_state=seed) if n else df
    return {
        "total": len(df),
        "dataset_disponivel": _available(),
        "items": sample[["path", "label", "split", "vinheta"]].to_dict("records"),
    }


@router.get("/image")
def image(path: str) -> Response:
    """Arquivo original de uma imagem da base (`path` relativo ao dataset)."""
    if path not in set(_table()["path"]):
        raise HTTPException(status_code=404, detail="imagem fora da base")
    file = cfg.DATASET_DIR / path
    if not file.exists():
        raise HTTPException(status_code=503, detail="Dataset não disponível nesta máquina")
    return Response(file.read_bytes(), media_type="image/jpeg")
