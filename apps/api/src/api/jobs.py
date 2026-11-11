"""Fila de execução de cenários (plano experimental, seção 7.1).

Um cenário por vez, cada um num processo separado (`python -m training.run_scenario <ID>`),
para não disputar a GPU nem travar a API. O estado de cada cenário fica em
`results/<ID>/status.json` e a saída do processo em `results/<ID>/log.txt`.
"""

import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable

from training import config as cfg
from training import status
from training.scenarios import SCENARIOS

CommandFactory = Callable[[str], list[str]]


def default_command(sid: str) -> list[str]:
    return [sys.executable, "-m", "training.run_scenario", sid]


class JobQueue:
    def __init__(self, command: CommandFactory = default_command):
        self._command = command
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ ciclo de vida

    def start(self) -> None:
        self.recover()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="fila-cenarios", daemon=True)
            self._thread.start()

    def recover(self) -> None:
        """Cenários que ficaram na fila ou executando quando a API parou: interrompidos."""
        for sid in SCENARIOS:
            st = status.read(sid)
            if st and st.get("state") in ("na_fila", "executando") and sid != self._running:
                if sid not in self._queue:
                    status.write(sid, state="interrompido", message="a API foi reiniciada")

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
        self._wake.set()
        return added

    def cancel(self, sid: str) -> bool:
        with self._lock:
            if sid in self._queue:
                self._queue.remove(sid)
                status.write(sid, state="interrompido", message="removido da fila")
                return True
            if sid == self._running and self._process is not None:
                self._process.terminate()
                return True
        return False

    def snapshot(self) -> dict:
        with self._lock:
            return {"executando": self._running, "fila": list(self._queue)}

    # ------------------------------------------------------------ execução

    def _next(self) -> str | None:
        with self._lock:
            if not self._queue:
                return None
            self._running = self._queue.popleft()
            return self._running

    def _loop(self) -> None:
        while True:
            sid = self._next()
            if sid is None:
                self._wake.wait()
                self._wake.clear()
                continue
            self._execute(sid)

    def _execute(self, sid: str) -> None:
        out = status.scenario_dir(sid)
        out.mkdir(parents=True, exist_ok=True)
        status.write(sid, state="executando", progress=0.0, message="iniciando processo")
        with (out / "log.txt").open("w") as log:
            proc = subprocess.Popen(
                self._command(sid),
                cwd=cfg.REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            with self._lock:
                self._process = proc
            code = proc.wait()
        with self._lock:
            self._process, self._running = None, None
        final = status.read(sid) or {}
        if code != 0 and final.get("state") not in ("erro",):
            message = "cancelado" if code < 0 else f"processo terminou com código {code}"
            status.write(sid, state="interrompido" if code < 0 else "erro", message=message)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Para testes: espera a fila esvaziar."""
        import time

        end = time.monotonic() + timeout
        while time.monotonic() < end:
            snap = self.snapshot()
            if snap["executando"] is None and not snap["fila"]:
                return True
            time.sleep(0.05)
        return False
