"""Recursos compartilhados entre as páginas do sistema web."""

import base64

import httpx
import pandas as pd
import streamlit as st

from web.api_client import ApiClient


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def b64_image(data: str) -> bytes:
    """Decodifica uma imagem PNG em base64 vinda da API."""
    return base64.b64decode(data)


def api_error_message(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", str(exc))
        except ValueError:
            detail = str(exc)
        return f"A API recusou o pedido: {detail}"
    return "Não foi possível conectar à API."


def features_table(combinations: list[dict], groups: dict[str, list[str]]) -> pd.DataFrame:
    """Tabela das características (F): uma linha por característica, uma coluna por combinação.

    O índice tem dois níveis (grupo, característica); `None` vira NaN (sem lesão segmentada).
    """
    index = pd.MultiIndex.from_tuples(
        [(g, name) for g, names in groups.items() for name in names],
        names=["Grupo", "Característica"],
    )
    data = {
        c["label"]: [c["result"]["features"].get(name) for _, name in index] for c in combinations
    }
    return pd.DataFrame(data, index=index, dtype=float)


# nome curto da técnica de cada modelo, exibido ao lado do ID (ex.: "C1 (SVM)")
MODEL_NAMES = {
    "A0": "RF, imagem bruta",
    "A5": "RF, pipeline completo",
    "C1": "SVM",
    "C2": "Random Forest",
    "C3": "RF só forma",
    "C4": "RF só cor",
    "C5": "RF só textura",
    "C6": "CNN, imagem bruta",
    "C7": "CNN, pré-processada",
    "V3": "CNN, só a lesão",
}


def model_label(model: dict) -> str:
    """ "C1 (SVM)"; cai na descrição do cenário se o ID não tiver nome curto."""
    return f"{model['id']} ({MODEL_NAMES.get(model['id'], model['description'])})"
