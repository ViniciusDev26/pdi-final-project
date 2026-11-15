import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from training import config as cfg

client = TestClient(app)


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    exp, data = tmp_path / "experiments", tmp_path / "dataset"
    exp.mkdir()
    rows = [
        ("test/benign/melanoma_1.jpg", "benign", False),
        ("test/malignant/melanoma_2.jpg", "malignant", True),
    ]
    cols = ["path", "label", "vinheta"]
    pd.DataFrame(rows, columns=cols).to_csv(exp / "anotacao_teste_ids.csv", index=False)
    pd.DataFrame([], columns=cols).to_csv(exp / "anotacao_val_ids.csv", index=False)
    for path, *_ in rows:
        (data / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 300), (200, 160, 140)).save(data / path)
    monkeypatch.setattr(cfg, "EXPERIMENTS_DIR", exp)
    monkeypatch.setattr(cfg, "MASKS_DIR", exp / "mascaras")
    monkeypatch.setattr(cfg, "DATASET_DIR", data)


SQUARE = json.dumps([[100, 100], [200, 100], [200, 200], [100, 200]])


def test_list_and_image_before_annotating():
    body = client.get("/annotations/teste").json()
    assert [i["name"] for i in body["items"]] == ["melanoma_1", "melanoma_2"]
    assert body["items"][1]["done"] is False
    img = client.get("/annotations/teste/melanoma_2").json()
    assert img["image"] and img["mask"] is None


def test_save_polygon():
    resp = client.put("/annotations/teste/melanoma_1", data={"points": SQUARE})
    assert resp.status_code == 200
    saved = cfg.MASKS_DIR / "teste" / "melanoma_1.png"
    mask = np.asarray(Image.open(saved)) > 0
    assert 0.1 < mask.mean() < 0.13
    body = client.get("/annotations/teste").json()
    assert body["items"][0]["done"] is True
    assert {"conjunto": "teste", "total": 2, "feitas": 1} in body["progress"]
    assert client.get("/annotations/teste/melanoma_1").json()["mask"]
    log = pd.read_csv(cfg.MASKS_DIR / "log.csv")
    assert list(log["imagem"]) == ["melanoma_1"]


def test_rejects_bad_polygon_and_unknown_items():
    resp = client.put("/annotations/teste/melanoma_2", data={"points": "[[1, 2]]"})
    assert resp.status_code == 422
    assert client.put("/annotations/teste/melanoma_9", data={"points": SQUARE}).status_code == 404
    assert client.get("/annotations/xyz").status_code == 404
