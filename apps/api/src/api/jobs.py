"""Fila de execução de cenários (plano experimental, seção 7.1).

Um cenário por vez, cada um num processo separado (`python -m training.run_scenario <ID>`),
para não disputar a GPU nem travar a API. O estado de cada cenário fica em
`results/<ID>/status.json` e a saída do processo em `results/<ID>/log.txt`.

A fila sobrevive a reinícios da API (treinos de CNN levam horas): ela é gravada em
`results/_fila.json`, o processo do cenário roda em sessão própria (não morre com a API) e
registra o PID no status; ao subir, a API readota o processo ainda vivo, espera ele
terminar e continua a fila.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable

from training import config as cfg
from training import status
from training.scenarios import SCENARIOS

CommandFactory = Callable[[str], list[str]]
POLL_SECONDS = 2.0


def default_command(sid: str) -> list[str]:
    return [sys.executable, "-m", "training.run_scenario", sid]


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # processo zumbi (já terminou, aguardando o pai) não conta como vivo
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


class JobQueue:
    def __init__(self, command: CommandFactory = default_command):
        self._command = command
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._process: subprocess.Popen | None = None
        self._adopted_pid: int | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ persistência

    @staticmethod
    def _file():
        return cfg.RESULTS_DIR / "_fila.json"

    def _save(self) -> None:
        path = self._file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fila": list(self._queue), "executando": self._running}))
        tmp.replace(path)

    # ------------------------------------------------------------ ciclo de vida

    def start(self) -> None:
        self.recover()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="fila-cenarios", daemon=True)
            self._thread.start()

    def recover(self) -> None:
        """Retoma a fila gravada e readota um cenário cujo processo ainda está vivo."""
        saved = {"fila": []}
        try:
            saved = json.loads(self._file().read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        with self._lock:
            for sid in SCENARIOS:
                st = status.read(sid) or {}
                if st.get("state") != "executando" or sid == self._running:
                    continue
                if self._running is None and pid_alive(st.get("pid")):
                    self._running, self._adopted_pid = sid, st["pid"]
                else:
                    status.write(sid, state="interrompido", message="a API foi reiniciada")
            for sid in saved.get("fila", []):
                if sid in SCENARIOS and sid not in self._queue and sid != self._running:
                    self._queue.append(sid)
                    status.write(sid, state="na_fila", progress=0.0, message="na fila")
            for sid in SCENARIOS:  # na fila sem estar na fila gravada: perdido
                st = status.read(sid) or {}
                if st.get("state") == "na_fila" and sid not in self._queue:
                    status.write(sid, state="interrompido", message="a API foi reiniciada")
            self._save()
        self._wake.set()

    # ------------------------------------------------------------ operações

    def enqueue(self, ids: list[str]) -> list[str]:
        unknown = [i for i in ids if i not in SCENARIOS]
        if unknown:
            raise ValueError(f"cenários desconhecidos: {unknown}")
        added = []
        with self._lock:
            for sid in ids:
                if sid == self._running or sid in self._queue:
                    continue
                self._queue.append(sid)
                status.write(sid, state="na_fila", progress=0.0, message="na fila", erro=None)
                added.append(sid)
            self._save()
        self._wake.set()
        return added

    def cancel(self, sid: str) -> bool:
        with self._lock:
            if sid in self._queue:
                self._queue.remove(sid)
                self._save()
                status.write(sid, state="interrompido", message="removido da fila")
                return True
            if sid == self._running:
                if self._process is not None:
                    self._process.terminate()
                elif self._adopted_pid:
                    os.kill(self._adopted_pid, 15)
                return True
        return False

    def snapshot(self) -> dict:
        with self._lock:
            return {"executando": self._running, "fila": list(self._queue)}

    # ------------------------------------------------------------ execução

    def _next(self) -> str | None:
        with self._lock:
            if self._running is not None:  # cenário readotado
                return self._running
            if not self._queue:
                return None
            self._running = self._queue.popleft()
            self._save()
            return self._running

    def _loop(self) -> None:
        while True:
            sid = self._next()
            if sid is None:
                self._wake.wait()
                self._wake.clear()
                continue
            if self._adopted_pid:
                self._wait_adopted(sid)
            else:
                self._execute(sid)

    def _finish(self) -> None:
        with self._lock:
            self._process, self._running, self._adopted_pid = None, None, None
            self._save()

    def _wait_adopted(self, sid: str) -> None:
        while pid_alive(self._adopted_pid):
            time.sleep(POLL_SECONDS)
        final = status.read(sid) or {}
        if final.get("state") == "executando":
            status.write(sid, state="interrompido", message="o processo terminou sem concluir")
        self._finish()

    def _execute(self, sid: str) -> None:
        out = status.scenario_dir(sid)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "log.txt").open("w") as log:
            proc = subprocess.Popen(
                self._command(sid),
                cwd=cfg.REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # sobrevive a um reinício da API
            )
            status.write(
                sid, state="executando", progress=0.0, message="iniciando processo", pid=proc.pid
            )
            with self._lock:
                self._process = proc
            code = proc.wait()
        final = status.read(sid) or {}
        if code != 0 and final.get("state") != "erro":
            message = "cancelado" if code < 0 else f"processo terminou com código {code}"
            status.write(sid, state="interrompido" if code < 0 else "erro", message=message)
        self._finish()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Para testes: espera a fila esvaziar."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            snap = self.snapshot()
            if snap["executando"] is None and not snap["fila"]:
                return True
            time.sleep(0.05)
        return False
