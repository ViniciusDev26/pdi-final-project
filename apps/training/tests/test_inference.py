import numpy as np
import pytest

from pdi import PipelineConfig, features
from training import config as cfg
from training import inference
from training.scenarios import SCENARIOS


def _has_lfs_models() -> bool:
    path = cfg.RESULTS_DIR / "C3" / "model.joblib"
    return path.exists() and path.stat().st_size > 10_000  # não é ponteiro do LFS


def test_input_config_uses_raw_image_for_c6():
    assert inference.input_config(SCENARIOS["C6"]) == PipelineConfig.raw()
    assert inference.input_config(SCENARIOS["C7"]) == SCENARIOS["A5"].pipeline


def test_feature_vector_follows_groups_and_maps_none_to_nan():
    feats = dict.fromkeys(features.FEATURE_NAMES, 1.0) | {"n_cores": None}
    assert inference.feature_vector(feats, None).shape == (1, len(features.FEATURE_NAMES))
    color = inference.feature_vector(feats, ("cor",))
    assert color.shape == (1, len(features.FEATURE_GROUPS["cor"]))
    assert np.isnan(color).sum() == 1


def test_no_models_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path)
    assert inference.available() == []
    with pytest.raises(KeyError):
        inference.get("C6")


@pytest.mark.skipif(not _has_lfs_models(), reason="modelos do Git LFS ausentes")
def test_matching_and_feature_subsets():
    raw = {m.sid for m in inference.matching(PipelineConfig.raw())}
    assert raw == {"A0", "C6"}
    full = {m.sid for m in inference.matching(SCENARIOS["A5"].pipeline)}
    assert {"A5", "C1", "C2", "C3", "C4", "C5", "C7", "V3"} <= full
    assert "V1" not in {m.sid for m in inference.available()}
    # C4 (só cor) recebe só as características de cor e devolve uma probabilidade válida
    feats = dict.fromkeys(features.FEATURE_NAMES, 0.5)
    p = inference.predict_traditional(inference.get("C4"), feats)
    assert 0.0 <= p <= 1.0
