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


# ---------------------------------------------------------------- simulador


def _lesion_jpeg() -> bytes:
    import numpy as np

    img = np.full((120, 120, 3), (200, 160, 140), dtype=np.uint8)
    yy, xx = np.mgrid[:120, :120]
    img[(yy - 60) ** 2 + (xx - 60) ** 2 < 30**2] = (90, 55, 40)
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _file():
    return {"file": ("lesao.png", _lesion_jpeg(), "image/png")}


def test_options_lists_stages_techniques_and_presets():
    body = client.get("/pipeline/options").json()
    assert [s["id"] for s in body["stages"]] == ["P1", "P2", "P3", "P4", "P5"]
    p4 = next(s for s in body["stages"] if s["id"] == "P4")
    assert p4["default"] == "clahe"
    assert {t["key"] for t in p4["techniques"]} == {"clahe", "equalize"}
    presets = {p["id"]: p for p in body["presets"]}
    assert presets["A0"]["config"]["contrast"] is None
    assert presets["T4"]["config"]["contrast"] == "equalize"
    assert body["max_combinations"] == 4
    assert set(body["feature_groups"]) == {"forma", "cor", "textura"}


def test_pipeline_returns_every_stage_and_mask():
    resp = client.post("/pipeline", files=_file(), data={"config": '{"contrast": "equalize"}'})
    assert resp.status_code == 200
    body = resp.json()
    assert [s["id"] for s in body["stages"]] == ["P1", "P2", "P3", "P4", "P5"]
    assert body["stages"][3]["technique"] == "equalize"
    assert body["lesion_mask"] and body["overlay"]
    assert 0.1 < body["lesion_area_fraction"] < 0.3


def test_pipeline_normalizes_input_and_returns_features():
    body = client.post("/pipeline", files=_file()).json()
    assert body["original_shape"] == [120, 120, 3]  # P0 leva a imagem para 300 × 300
    feats = body["features"]
    assert feats["n_cores"] >= 1 and 0 < feats["solidez"] <= 1
    assert feats["assimetria_eixo_maior"] < 0.1  # lesão circular


def test_pipeline_features_are_null_without_segmentation():
    resp = client.post("/pipeline", files=_file(), data={"config": '{"segmentation": null}'})
    assert all(v is None for v in resp.json()["features"].values())


def test_pipeline_defaults_to_full_config():
    body = client.post("/pipeline", files=_file()).json()
    assert body["config"]["contrast"] == "clahe"


def test_pipeline_rejects_invalid_config():
    for cfg in ('{"contrast": "sharpen"}', '{"P9": 1}', "nao-json", "[1]"):
        resp = client.post("/pipeline", files=_file(), data={"config": cfg})
        assert resp.status_code == 422, cfg


def test_compare_returns_dice_matrix_and_figure():
    combos = (
        '[{"label": "sem pré", "config": {"vignette": null, "hair": null, "color": null,'
        ' "contrast": null, "denoise": null}}, {"label": "completo", "config": {}}]'
    )
    resp = client.post("/compare", files=_file(), data={"combinations": combos})
    assert resp.status_code == 200
    body = resp.json()
    assert [c["label"] for c in body["combinations"]] == ["sem pré", "completo"]
    assert body["mask_dice"][0][0] == 1.0
    assert 0.8 < body["mask_dice"][0][1] <= 1.0
    assert body["figure"]


def test_compare_limits_number_of_combinations():
    too_many = "[" + ",".join(['{"config": {}}'] * 5) + "]"
    for combos in ("[]", too_many):
        resp = client.post("/compare", files=_file(), data={"combinations": combos})
        assert resp.status_code == 422
