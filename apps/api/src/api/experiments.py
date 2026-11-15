"""Rotas das páginas Experimentos e Resultados (plano experimental, seção 7.1)."""

import json
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from api.jobs import JobQueue
from training import evaluation, report, status, visuals
from training.scenarios import SCENARIOS

router = APIRouter(tags=["experimentos"])
queue = JobQueue()


class ScenarioOut(BaseModel):
    id: str
    group: str
    description: str
    kind: str
    model: str | None
    estimate: str
    uses_gpu: bool
    depends: list[str]
    config: dict[str, Any] | None
    status: dict[str, Any]
    has_results: bool


class JobsIn(BaseModel):
    ids: list[str]  # IDs ou ["todos"]


@router.get("/scenarios")
def list_scenarios() -> list[ScenarioOut]:
    return [
        ScenarioOut(
            id=s.id,
            group=s.group,
            description=s.description,
            kind=s.kind,
            model=s.model,
            estimate=s.estimate,
            uses_gpu=s.uses_gpu,
            depends=list(s.depends),
            config=s.pipeline.to_dict() if s.pipeline else None,
            status=status.read(s.id) or {"state": "nao_executado"},
            has_results=(status.scenario_dir(s.id) / "metrics.json").exists(),
        )
        for s in SCENARIOS.values()
    ]


@router.get("/jobs")
def jobs() -> dict[str, Any]:
    return queue.snapshot()


@router.post("/jobs")
def enqueue(body: JobsIn) -> dict[str, Any]:
    ids = list(SCENARIOS) if body.ids == ["todos"] else body.ids
    try:
        added = queue.enqueue(ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"adicionados": added, **queue.snapshot()}


@router.delete("/jobs/{sid}")
def cancel(sid: str) -> dict[str, Any]:
    if not queue.cancel(sid):
        raise HTTPException(status_code=404, detail=f"{sid} não está na fila nem executando")
    return queue.snapshot()


@router.get("/results/report")
def results_report() -> dict[str, Any]:
    return evaluation.to_jsonable(report.build())


@router.get("/results/figures")
def list_figures() -> dict[str, str]:
    return report.FIGURES


@router.get("/results/figures/{name}.png")
def figure(name: str) -> Response:
    try:
        return Response(report.figure(name), media_type="image/png")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"figura desconhecida: {name}") from exc


@router.get("/results/visuals")
def visuals_index() -> dict[str, Any]:
    """Saídas visuais da seção 5.6 já geradas, com as anotações de causa dos erros."""
    return {
        "index": visuals.load_index(),
        "notas": visuals.load_notes(),
        "causas": list(visuals.ERROR_CAUSES),
    }


@router.post("/results/visuals")
def visuals_generate() -> dict[str, Any]:
    """Gera de novo as saídas visuais (precisa do dataset; leva cerca de 1 minuto)."""
    try:
        return {"index": visuals.generate()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/results/visuals/{name:path}")
def visuals_file(name: str) -> Response:
    base = visuals.out_dir().resolve()
    path = (base / name).resolve()
    if path.suffix != ".png" or base not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail=f"figura não encontrada: {name}")
    return Response(path.read_bytes(), media_type="image/png")


@router.put("/results/errors")
def error_note(
    path: Annotated[str, Form()], causa: Annotated[str, Form()], nota: Annotated[str, Form()] = ""
) -> dict[str, Any]:
    """Anotação da equipe sobre a causa provável de um erro do modelo (seção 5.6)."""
    index = visuals.load_index() or {}
    known = {c["path"] for c in index.get("erros", {}).get("casos", [])}
    if path not in known:
        raise HTTPException(status_code=404, detail="imagem fora da análise de erros")
    try:
        return visuals.save_error_note(path, causa, nota)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _scenario(sid: str) -> None:
    if sid not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"cenário desconhecido: {sid}")


@router.get("/results/{sid}")
def results(sid: str) -> dict[str, Any]:
    _scenario(sid)
    path = status.scenario_dir(sid) / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{sid} ainda não foi executado")
    return json.loads(path.read_text())


@router.get("/results/{sid}/log", response_class=PlainTextResponse)
def log(sid: str, tail: int = 200) -> str:
    _scenario(sid)
    path = status.scenario_dir(sid) / "log.txt"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-tail:])
