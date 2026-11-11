"""Recursos compartilhados entre as páginas do sistema web."""

import base64

import httpx
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
