import numpy as np
import pytest

from training import config as cfg
from training import evaluation, status
from training.scenarios import GROUPS, SCENARIOS, presets


def test_threshold_reaches_target_sensitivity_on_validation():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)
    t = evaluation.choose_threshold(y, p)
    sens = ((p >= t) & (y == 1)).sum() / y.sum()
    assert sens >= 0.90
    assert ((p >= t + 1e-9) & (y == 1)).sum() / y.sum() < 0.90  # o maior limiar possível


def test_evaluate_has_ci_around_point_and_confusion_matrix():
    y = np.array([0, 0, 0, 1, 1, 1] * 20)
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9] * 20)
    res = evaluation.evaluate(y, p, threshold=0.5, n_boot=200)
    for m in evaluation.METRICS:
        lo, hi = res[m]["ic95"]
        assert lo <= res[m]["valor"] <= hi
    assert res["matriz_confusao"] == [[40, 20], [20, 40]]
    diff = evaluation.paired_difference(y, p, p, n_boot=50)
    assert diff["diferenca"] == 0 and diff["ic95"] == [0, 0]


def test_status_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path)
    assert status.read("A0") is None
    status.write("A0", state="na_fila", progress=0.0)
    status.write("A0", state="executando", progress=0.5)
    s = status.read("A0")
    assert s["state"] == "executando" and s["progress"] == 0.5 and s["id"] == "A0"
    with pytest.raises(ValueError):
        status.write("A0", state="voando")


def test_scenarios_cover_every_planned_id():
    expected = (
        [f"A{i}" for i in range(6)]
        + [f"L{i}" for i in range(1, 6)]
        + [f"T{i}" for i in range(1, 6)]
        + [f"S{i}" for i in range(1, 5)]
        + [f"C{i}" for i in range(1, 8)]
        + [f"V{i}" for i in range(1, 6)]
    )
    assert sorted(SCENARIOS) == sorted(expected)
    assert {s.group for s in SCENARIOS.values()} == set(GROUPS)
    assert all(s.pipeline is not None for s in presets())
    for s in SCENARIOS.values():
        assert all(d in SCENARIOS for d in s.depends)
