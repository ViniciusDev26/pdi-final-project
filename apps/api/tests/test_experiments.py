import sys

import pytest
from fastapi.testclient import TestClient

from api import experiments
from api.jobs import JobQueue
from api.main import app
from training import config as cfg
from training import status

FAKE = (
    "import json, sys; from training import status; sid = sys.argv[1];"
    "status.write(sid, state='concluido', progress=1.0);"
    "(status.scenario_dir(sid) / 'metrics.json').write_text(json.dumps({'id': sid}));"
    "print('ok', sid)"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("PDI_RESULTS_DIR", str(tmp_path))
    q = JobQueue(command=lambda sid: [sys.executable, "-c", FAKE, sid])
    monkeypatch.setattr(experiments, "queue", q)
    q.start()
    with TestClient(app) as c:
        yield c


def test_scenarios_list_every_group(client):
    body = client.get("/scenarios").json()
    assert {s["group"] for s in body} == {"A", "L", "T", "S", "C", "V"}
    c7 = next(s for s in body if s["id"] == "C7")
    assert c7["uses_gpu"] and c7["status"]["state"] == "nao_executado"


def test_enqueue_runs_in_separate_process_and_writes_log(client):
    resp = client.post("/jobs", json={"ids": ["A0", "A1"]})
    assert resp.status_code == 200 and resp.json()["adicionados"] == ["A0", "A1"]
    assert experiments.queue.wait_idle()
    assert status.read("A0")["state"] == "concluido"
    assert client.get("/results/A1").json() == {"id": "A1"}
    assert "ok A1" in client.get("/results/A1/log").text


def test_enqueue_rejects_unknown_and_cancel_missing(client):
    assert client.post("/jobs", json={"ids": ["Z9"]}).status_code == 422
    assert client.delete("/jobs/A0").status_code == 404
    assert client.get("/results/A3").status_code == 404


def test_recover_marks_stale_jobs_as_interrupted(client):
    status.write("L2", state="executando", progress=0.3)
    experiments.queue.recover()
    assert status.read("L2")["state"] == "interrompido"


def test_report_and_figures_work_without_results(client):
    body = client.get("/results/report").json()
    assert len(body["cenarios"]) == 32 and body["triagem"] is None
    for name in client.get("/results/figures").json():
        resp = client.get(f"/results/figures/{name}.png")
        assert resp.status_code == 200 and resp.content[:4] == b"\x89PNG"


def test_queue_survives_restart_and_adopts_live_process(client, tmp_path):
    import json
    import subprocess

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    status.write("C1", state="executando", progress=0.2, pid=sleeper.pid)
    (tmp_path / "_fila.json").write_text(json.dumps({"fila": ["A2"], "executando": "C1"}))
    q = JobQueue(command=lambda sid: [sys.executable, "-c", FAKE, sid])
    q.recover()
    assert q.snapshot() == {"executando": "C1", "fila": ["A2"]}
    assert status.read("A2")["state"] == "na_fila"
    q.start()
    sleeper.wait()
    assert q.wait_idle(timeout=15)
    # o processo readotado terminou sem concluir; o próximo da fila rodou
    assert status.read("C1")["state"] == "interrompido"
    assert status.read("A2")["state"] == "concluido"
    assert json.loads((tmp_path / "_fila.json").read_text())["fila"] == []
