"""Estado de execução de um cenário: `results/<ID>/status.json` (plano, seção 7.1).

Escrito pela fila da API (na fila) e pelo próprio executor (executando, progresso,
concluído, erro). A página Experimentos lê pela API.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from training import config as cfg

STATES = ("na_fila", "executando", "concluido", "erro", "interrompido")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def scenario_dir(sid: str) -> Path:
    return cfg.RESULTS_DIR / sid


def read(sid: str) -> dict | None:
    path = scenario_dir(sid) / "status.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(sid: str, **fields) -> dict:
    """Atualiza campos do status (escrita atômica)."""
    d = scenario_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    data = (read(sid) or {"id": sid}) | fields | {"atualizado_em": now()}
    if data.get("state") not in STATES:
        raise ValueError(f"estado inválido: {data.get('state')!r}")
    tmp = d / f"status.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(d / "status.json")
    return data
