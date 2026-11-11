from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from web.api_client import ApiClient

APP = Path(__file__).parents[1] / "src" / "web" / "app.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    # cache_data/cache_resource persistem entre testes no mesmo processo
    st.cache_data.clear()
    st.cache_resource.clear()


def _client(handler) -> ApiClient:
    return ApiClient(base_url="http://api", transport=httpx.MockTransport(handler))


def test_health_returns_none_when_api_is_down():
    def down(request):
        raise httpx.ConnectError("down", request=request)

    assert _client(down).health() is None


def test_predict_sends_file_and_returns_json():
    def handler(request):
        assert request.url.path == "/predict"
        assert b"conteudo" in request.content
        return httpx.Response(200, json={"label": "benign"})

    assert _client(handler).predict(b"conteudo", "a.jpg", "image/jpeg") == {"label": "benign"}


def test_app_renders_without_api(monkeypatch):
    monkeypatch.setenv("PDI_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP)).run(timeout=10)
    assert not at.exception
    assert at.sidebar.error[0].value == "API indisponível"


# ---------------------------------------------------------------- simulador

SIMULADOR = Path(__file__).parents[1] / "src" / "web" / "views" / "simulador.py"

FULL = {
    "vignette": "fill",
    "hair": "dullrazor",
    "color": "shades_of_gray",
    "contrast": "clahe",
    "denoise": "median",
    "segmentation": "otsu",
    "seg_postprocess": True,
}
RAW = {**FULL, "vignette": None, "hair": None, "color": None, "contrast": None, "denoise": None}
OPTIONS = {
    "stages": [
        {
            "id": "P1",
            "field": "vignette",
            "name": "Vinheta",
            "default": "fill",
            "techniques": [{"key": "fill", "label": "Preenchimento"}],
        },
        {
            "id": "P2",
            "field": "hair",
            "name": "Pelos",
            "default": "dullrazor",
            "techniques": [{"key": "dullrazor", "label": "DullRazor"}],
        },
        {
            "id": "P3",
            "field": "color",
            "name": "Cor",
            "default": "shades_of_gray",
            "techniques": [{"key": "shades_of_gray", "label": "SoG"}],
        },
        {
            "id": "P4",
            "field": "contrast",
            "name": "Contraste",
            "default": "clahe",
            "techniques": [{"key": "clahe", "label": "CLAHE"}, {"key": "equalize", "label": "Eq"}],
        },
        {
            "id": "P5",
            "field": "denoise",
            "name": "Filtro",
            "default": "median",
            "techniques": [{"key": "median", "label": "Mediana"}],
        },
    ],
    "segmentation": [{"key": "otsu", "label": "Otsu"}, {"key": "adaptive", "label": "Adapt."}],
    "default_config": FULL,
    "presets": [
        {"id": "A0", "group": "A", "description": "Bruta", "config": RAW},
        {"id": "A5", "group": "A", "description": "Completo", "config": FULL},
        {"id": "T4", "group": "T", "description": "Eq", "config": {**FULL, "contrast": "equalize"}},
        {"id": "L1", "group": "L", "description": "Sem P1", "config": {**FULL, "vignette": None}},
    ],
    "max_combinations": 4,
}


def test_client_pipeline_and_compare_send_json_config():
    import json

    seen = {}

    def handler(request):
        seen[request.url.path] = request.content
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    client.run_pipeline(b"img", "a.png", "image/png", {"contrast": "equalize"})
    client.compare(b"img", "a.png", "image/png", [{"label": "x", "config": {}}])
    assert b'"contrast": "equalize"' in seen["/pipeline"]
    assert json.dumps([{"label": "x", "config": {}}]).encode() in seen["/compare"]


def test_simulador_renders_presets_without_image(monkeypatch):
    monkeypatch.setattr(ApiClient, "pipeline_options", lambda self: OPTIONS)
    at = AppTest.from_file(str(SIMULADOR)).run(timeout=10)
    assert not at.exception
    assert at.info[0].value.startswith("Envie uma imagem")
    # padrão: duas colunas, A0 (sem pré-processamento) × A5 (completo)
    assert at.session_state["sim_c0_preset"] == "A0"
    assert at.session_state["sim_c1_preset"] == "A5"
    assert at.session_state["sim_c0_P4"] == "off"
    assert at.session_state["sim_c1_P4"] == "clahe"


def test_simulador_preset_fills_techniques_and_manual_change_marks_custom(monkeypatch):
    monkeypatch.setattr(ApiClient, "pipeline_options", lambda self: OPTIONS)
    at = AppTest.from_file(str(SIMULADOR)).run(timeout=10)
    preset = next(s for s in at.selectbox if s.key == "sim_c0_preset")
    preset.set_value("T4").run()
    assert at.session_state["sim_c0_P4"] == "equalize"
    assert at.session_state["sim_c0_P1"] == "fill"
    p4 = next(s for s in at.selectbox if s.key == "sim_c0_P4")
    p4.set_value("clahe").run()
    assert at.session_state["sim_c0_preset"] == "custom"


def test_simulador_shows_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("PDI_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(SIMULADOR)).run(timeout=10)
    assert not at.exception
    assert at.error[0].value == "Não foi possível conectar à API."
