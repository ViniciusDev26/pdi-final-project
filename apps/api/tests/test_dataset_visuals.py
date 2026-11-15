import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from training import config as cfg
from training import visuals

client = TestClient(app)


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    exp, data, res = tmp_path / "experiments", tmp_path / "dataset", tmp_path / "results"
    exp.mkdir()
    rows = [
        ("train/benign/melanoma_1.jpg", "benign", "train", False),
        ("train/malignant/melanoma_2.jpg", "malignant", "train", True),
        ("val/malignant/melanoma_3.jpg", "malignant", "val", False),
        ("test/benign/melanoma_4.jpg", "benign", "test", False),
    ]
    pd.DataFrame(rows, columns=["path", "label", "split", "vinheta"]).to_csv(
        exp / "vinheta.csv", index=False
    )
    monkeypatch.setattr(cfg, "EXPERIMENTS_DIR", exp)
    monkeypatch.setattr(cfg, "DATASET_DIR", data)
    monkeypatch.setattr(cfg, "RESULTS_DIR", res)
    return rows, data


def test_summary_works_without_dataset():
    body = client.get("/dataset/summary").json()
    assert body["total"] == 4
    assert body["duplicatas_removidas"] == 10_605 - 4
    assert body["contagem"]["train"] == {"benign": 1, "malignant": 1}
    assert body["vinheta_geral"]["malignant"] == 0.5
    assert body["dataset_disponivel"] is False
    assert (
        client.get("/dataset/image", params={"path": "train/benign/melanoma_1.jpg"}).status_code
        == 503
    )


def test_images_filters_and_serves_files(_dirs):
    rows, data = _dirs
    for path, *_ in rows:
        (data / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 300), (200, 160, 140)).save(data / path)
    body = client.get("/dataset/images", params={"label": "malignant"}).json()
    assert body["total"] == 2 and {i["label"] for i in body["items"]} == {"malignant"}
    body = client.get("/dataset/images", params={"vinheta": True}).json()
    assert [i["path"] for i in body["items"]] == ["train/malignant/melanoma_2.jpg"]
    resp = client.get("/dataset/image", params={"path": "test/benign/melanoma_4.jpg"})
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
    # só caminhos da base: nada de ler arquivos arbitrários
    assert client.get("/dataset/image", params={"path": "../../etc/passwd"}).status_code == 404


def test_visuals_index_and_error_notes():
    assert client.get("/results/visuals").json()["index"] is None
    out = visuals.out_dir()
    (out / "erros").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(out / "erros" / "melanoma_4.png")
    index = {
        "gerado_em": "2026-11-15T00:00:00+00:00",
        "commit": "abc",
        "erros": {
            "modelo": "C6",
            "figuras": {},
            "casos": [{"path": "test/benign/melanoma_4.jpg", "figura": "erros/melanoma_4.png"}],
        },
    }
    (out / "index.json").write_text(json.dumps(index))
    body = client.get("/results/visuals").json()
    assert body["index"]["commit"] == "abc" and "pelos" in body["causas"]
    assert client.get("/results/visuals/erros/melanoma_4.png").status_code == 200
    assert client.get("/results/visuals/../index.json").status_code == 404

    data = {"path": "test/benign/melanoma_4.jpg", "causa": "vinheta", "nota": "borda escura"}
    assert client.put("/results/errors", data=data).status_code == 200
    notes = client.get("/results/visuals").json()["notas"]
    assert notes["test/benign/melanoma_4.jpg"]["causa"] == "vinheta"
    assert client.put("/results/errors", data={**data, "causa": "xyz"}).status_code == 422
    assert client.put("/results/errors", data={**data, "path": "outra.jpg"}).status_code == 404


def test_generate_visuals_needs_dataset():
    assert client.post("/results/visuals").status_code == 503
