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
    "feature_groups": {"forma": ["solidez"], "cor": ["n_cores"]},
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


def test_features_table_has_one_column_per_combination():
    import math

    from web.common import features_table

    combos = [
        {"label": "A0", "result": {"features": {"solidez": 0.9, "n_cores": 2.0}}},
        {"label": "A5", "result": {"features": {"solidez": None, "n_cores": None}}},
    ]
    table = features_table(combos, OPTIONS["feature_groups"])
    assert list(table.columns) == ["A0", "A5"]
    assert table.loc[("forma", "solidez"), "A0"] == 0.9
    assert math.isnan(table.loc[("cor", "n_cores"), "A5"])


ANOTACAO = APP.parent / "views" / "anotacao.py"


def _ann_listing(conjunto):
    return {
        "conjunto": conjunto,
        "annotators": ["Renan", "Guilherme", "Carlos"],
        "items": [
            {
                "name": "melanoma_1",
                "path": "test/benign/melanoma_1.jpg",
                "label": "benign",
                "vinheta": False,
                "responsavel": "Renan",
                "compartilhada": False,
                "annotators": ["Renan"],
                "done": {"Renan": False},
            }
        ],
        "progress": [{"conjunto": conjunto, "anotador": "Renan", "total": 1, "feitas": 0}],
        "inter_annotator": [],
        "inter_annotator_mean": None,
    }


def test_anotacao_renders_pending_image(monkeypatch):
    import base64
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (300, 300), (200, 160, 140)).save(buf, format="PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    monkeypatch.setattr(ApiClient, "annotations", lambda self, c: _ann_listing(c))
    monkeypatch.setattr(
        ApiClient,
        "annotation_image",
        lambda self, c, n, a: {"name": n, "image": image, "mask": None},
    )
    at = AppTest.from_file(str(ANOTACAO)).run(timeout=10)
    assert not at.exception
    assert at.session_state["ann_annotator"] == "Renan"
    save = next(b for b in at.button if b.label == "Salvar máscara")
    assert save.disabled  # sem contorno ainda


def test_anotacao_shows_message_for_annotator_without_images(monkeypatch):
    monkeypatch.setattr(ApiClient, "annotations", lambda self, c: _ann_listing(c))
    at = AppTest.from_file(str(ANOTACAO)).run(timeout=10)
    next(s for s in at.selectbox if s.key == "ann_annotator").set_value("Carlos").run()
    assert at.info[0].value.startswith("Nenhuma imagem atribuída")


EXPERIMENTOS = APP.parent / "views" / "experimentos.py"
RESULTADOS = APP.parent / "views" / "resultados.py"


def _scenarios(self):
    return [
        {
            "id": "A0",
            "group": "A",
            "description": "Bruta",
            "kind": "ablation",
            "model": "rf",
            "estimate": "≈ 1 min",
            "uses_gpu": False,
            "depends": [],
            "config": {},
            "status": {"state": "executando", "progress": 0.5, "message": "treinando"},
            "has_results": False,
        }
    ]


def test_experimentos_shows_board_and_queues(monkeypatch):
    seen = {}
    monkeypatch.setattr(ApiClient, "scenarios", _scenarios)
    monkeypatch.setattr(ApiClient, "jobs", lambda self: {"executando": "A0", "fila": []})
    monkeypatch.setattr(ApiClient, "scenario_log", lambda self, sid, tail=200: "linha 1")
    monkeypatch.setattr(
        ApiClient,
        "enqueue",
        lambda self, ids: seen.setdefault("ids", ids) and {"adicionados": ids},
    )
    at = AppTest.from_file(str(EXPERIMENTOS)).run(timeout=10)
    assert not at.exception
    assert at.metric[1].value == "A0"
    next(b for b in at.button if b.label == "Executar todos").click().run()
    assert seen["ids"] == ["todos"]


def test_resultados_renders_empty_report(monkeypatch):
    empty = {
        "cenarios": [{"id": "A0", "grupo": "A", "descricao": "Bruta", "estado": "nao_executado"}],
        "impacto_etapas": [],
        "tecnicas": [],
        "modelos": [],
        "triagem": None,
        "segmentacao": [],
        "vies": {},
        "local": {},
        "tempo": {},
    }
    monkeypatch.setattr(ApiClient, "report", lambda self: empty)
    monkeypatch.setattr(ApiClient, "figures", lambda self: {})
    at = AppTest.from_file(str(RESULTADOS)).run(timeout=10)
    assert not at.exception
    assert at.info[0].value.startswith("Nenhum cenário executado")
