import json
import os

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ApiClient:
    """Cliente HTTP da API; o front não acessa o pipeline nem os modelos diretamente."""

    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(
            base_url=base_url or os.environ.get("PDI_API_URL", DEFAULT_API_URL),
            transport=transport,
            timeout=60.0,
        )

    def health(self) -> dict | None:
        """Retorna o status da API, ou None se ela estiver fora do ar."""
        try:
            resp = self._http.get("/health", timeout=3.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        return resp.json()

    def predict(self, image: bytes, filename: str, content_type: str) -> dict:
        resp = self._http.post("/predict", files={"file": (filename, image, content_type)})
        resp.raise_for_status()
        return resp.json()

    def pipeline_options(self) -> dict:
        """Etapas, técnicas, presets (cenários) e limite de combinações do simulador."""
        resp = self._http.get("/pipeline/options")
        resp.raise_for_status()
        return resp.json()

    def run_pipeline(self, image: bytes, filename: str, content_type: str, config: dict) -> dict:
        resp = self._http.post(
            "/pipeline",
            files={"file": (filename, image, content_type)},
            data={"config": json.dumps(config)},
        )
        resp.raise_for_status()
        return resp.json()

    def compare(
        self, image: bytes, filename: str, content_type: str, combinations: list[dict]
    ) -> dict:
        """`combinations`: lista de `{"label": str, "config": dict}`."""
        resp = self._http.post(
            "/compare",
            files={"file": (filename, image, content_type)},
            data={"combinations": json.dumps(combinations)},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------ anotação

    def annotations(self, conjunto: str) -> dict:
        resp = self._http.get(f"/annotations/{conjunto}")
        resp.raise_for_status()
        return resp.json()

    def annotation_image(self, conjunto: str, name: str, annotator: str) -> dict:
        resp = self._http.get(f"/annotations/{conjunto}/{name}", params={"annotator": annotator})
        resp.raise_for_status()
        return resp.json()

    def save_annotation(
        self, conjunto: str, name: str, annotator: str, points: list[list[float]]
    ) -> dict:
        resp = self._http.put(
            f"/annotations/{conjunto}/{name}",
            data={"annotator": annotator, "points": json.dumps(points)},
        )
        resp.raise_for_status()
        return resp.json()
