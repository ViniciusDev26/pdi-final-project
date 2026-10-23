from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok", "model": "dummy"}


def test_predict_returns_label_and_probability():
    resp = client.post("/predict", files={"file": ("lesao.jpg", _jpeg((20, 20, 20)), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "malignant"
    assert 0.0 <= body["probability_malignant"] <= 1.0


def test_predict_rejects_non_image():
    resp = client.post("/predict", files={"file": ("x.txt", b"nao sou imagem", "text/plain")})
    assert resp.status_code == 422
