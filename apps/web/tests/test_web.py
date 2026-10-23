from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from web.api_client import ApiClient

APP = Path(__file__).parents[1] / "src" / "web" / "app.py"


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
