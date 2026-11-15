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
        "items": [
            {
                "name": "melanoma_1",
                "path": "test/benign/melanoma_1.jpg",
                "label": "benign",
                "vinheta": False,
                "done": False,
            }
        ],
        "progress": [{"conjunto": conjunto, "total": 1, "feitas": 0}],
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
        lambda self, c, n: {"name": n, "image": image, "mask": None},
    )
    at = AppTest.from_file(str(ANOTACAO)).run(timeout=10)
    assert not at.exception
    save = next(b for b in at.button if b.label == "Salvar máscara")
    assert save.disabled  # sem contorno ainda


def test_anotacao_shows_message_for_empty_list(monkeypatch):
    monkeypatch.setattr(ApiClient, "annotations", lambda self, c: {**_ann_listing(c), "items": []})
    at = AppTest.from_file(str(ANOTACAO)).run(timeout=10)
    assert at.info[0].value.startswith("A lista de anotação")


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
    monkeypatch.setattr(
        ApiClient, "visuals", lambda self: {"index": None, "notas": {}, "causas": []}
    )
    at = AppTest.from_file(str(RESULTADOS)).run(timeout=10)
    assert not at.exception
    assert at.info[0].value.startswith("Nenhum cenário executado")


def _png_b64() -> str:
    import base64
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (30, 30), (200, 160, 140)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_resultados_error_analysis_saves_note(monkeypatch):
    import base64

    empty = {
        "cenarios": [],
        "impacto_etapas": [],
        "tecnicas": [],
        "modelos": [],
        "triagem": None,
        "segmentacao": [],
        "vies": {},
        "local": {},
        "tempo": {},
    }
    case = {
        "path": "test/benign/melanoma_4.jpg",
        "imagem": "melanoma_4",
        "tipo": "FP",
        "y": 0,
        "p": 0.97,
        "figura": "erros/melanoma_4.png",
    }
    index = {
        "gerado_em": "2026-11-15T00:00:00",
        "commit": "abc",
        "segmentacao": {"pendente": "sem máscaras"},
        "erros": {"modelo": "C6", "figuras": {}, "casos": [case]},
        "exemplos": {"presets": ["A0"], "figuras": {}, "imagens": []},
    }
    saved = {}
    monkeypatch.setattr(ApiClient, "report", lambda self: empty)
    monkeypatch.setattr(ApiClient, "figures", lambda self: {})
    monkeypatch.setattr(
        ApiClient,
        "visuals",
        lambda self: {"index": index, "notas": {}, "causas": ["pelos", "vinheta"]},
    )
    monkeypatch.setattr(ApiClient, "visual", lambda self, name: base64.b64decode(_png_b64()))
    monkeypatch.setattr(
        ApiClient,
        "save_error_note",
        lambda self, path, causa, nota: saved.update(path=path, causa=causa) or saved,
    )
    at = AppTest.from_file(str(RESULTADOS)).run(timeout=10)
    assert not at.exception
    causa = next(s for s in at.selectbox if s.label == "Causa provável")
    causa.set_value("vinheta")
    submit = next(b for b in at.button if b.label == "Salvar anotação")
    submit.click().run()
    assert saved == {"path": "test/benign/melanoma_4.jpg", "causa": "vinheta"}


TRIAGEM = APP.parent / "views" / "triagem.py"
BASE = APP.parent / "views" / "base_dados.py"


def test_triagem_shows_probability_gradcam_and_abcd(monkeypatch):
    img = _png_b64()
    result = {
        "label": "malignant",
        "probability_malignant": 0.97,
        "threshold": 0.933,
        "model": "C6",
        "model_version": "abc",
        "model_description": "CNN, imagem bruta",
        "recommendation": "Encaminhar ao dermatologista",
        "image": img,
        "gradcam": img,
        "lesion_overlay": img,
        "vignette_detected": True,
        "abcd": {"n_cores": 3.0, "assimetria_eixo_maior": 0.2},
    }
    monkeypatch.setattr(ApiClient, "predict", lambda self, *a: result)
    at = AppTest.from_file(str(TRIAGEM))
    at.run(timeout=10)
    assert not at.exception  # sem upload, a página só mostra o seletor de arquivo


def test_base_dados_without_dataset(monkeypatch):
    summary = {
        "total": 10583,
        "imagens_originais": 10605,
        "duplicatas_removidas": 22,
        "contagem": {"train": {"benign": 4000, "malignant": 4000}},
        "vinheta": {"train": {"benign": 0.01, "malignant": 0.3}},
        "vinheta_geral": {"benign": 0.01, "malignant": 0.3},
        "dataset_disponivel": False,
    }
    monkeypatch.setattr(ApiClient, "dataset_summary", lambda self: summary)
    at = AppTest.from_file(str(BASE)).run(timeout=10)
    assert not at.exception
    assert at.metric[2].value == "22"
    assert "modo leitura" in at.info[0].value


def test_base_dados_lists_images(monkeypatch):
    import base64

    summary = {
        "total": 2,
        "imagens_originais": 10605,
        "duplicatas_removidas": 10603,
        "contagem": {"train": {"benign": 1, "malignant": 1}},
        "vinheta": {"train": {"benign": 0.0, "malignant": 1.0}},
        "vinheta_geral": {"benign": 0.0, "malignant": 1.0},
        "dataset_disponivel": True,
    }
    items = [
        {
            "path": "train/benign/melanoma_1.jpg",
            "label": "benign",
            "split": "train",
            "vinheta": False,
        }
    ]
    monkeypatch.setattr(ApiClient, "dataset_summary", lambda self: summary)
    monkeypatch.setattr(ApiClient, "dataset_images", lambda self, **f: {"total": 1, "items": items})
    monkeypatch.setattr(ApiClient, "dataset_image", lambda self, path: base64.b64decode(_png_b64()))
    at = AppTest.from_file(str(BASE)).run(timeout=10)
    assert not at.exception
    assert any(b.label == "Simulador" for b in at.button)
