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
        ("test/benign/melanoma_1.jpg", "benign", False, "Renan", True),
        ("test/malignant/melanoma_2.jpg", "malignant", True, "Carlos", False),
    ]
    cols = ["path", "label", "vinheta", "responsavel", "compartilhada"]
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
    assert body["items"][0]["annotators"] == list(cfg.ANNOTATORS)  # compartilhada
    assert body["items"][1]["done"] == {"Carlos": False}
    img = client.get("/annotations/teste/melanoma_2", params={"annotator": "Carlos"}).json()
    assert img["image"] and img["mask"] is None


def test_save_polygon_and_inter_annotator_dice():
    for annotator in ("Renan", "Guilherme"):
        resp = client.put(
            "/annotations/teste/melanoma_1", data={"annotator": annotator, "points": SQUARE}
        )
        assert resp.status_code == 200
    saved = cfg.MASKS_DIR / "teste" / "Renan" / "melanoma_1.png"
    mask = np.asarray(Image.open(saved)) > 0
    assert 0.1 < mask.mean() < 0.13
    body = client.get("/annotations/teste").json()
    assert body["inter_annotator_mean"] == 1.0
    assert body["items"][0]["done"]["Renan"] is True
    log = pd.read_csv(cfg.MASKS_DIR / "anotadores.csv")
    assert list(log["anotador"]) == ["Renan", "Guilherme"]


def test_rejects_wrong_annotator_and_bad_polygon():
    resp = client.put(
        "/annotations/teste/melanoma_2", data={"annotator": "Renan", "points": SQUARE}
    )
    assert resp.status_code == 422
    resp = client.put(
        "/annotations/teste/melanoma_2", data={"annotator": "Carlos", "points": "[[1, 2]]"}
    )
    assert resp.status_code == 422
    assert client.get("/annotations/xyz").status_code == 404
