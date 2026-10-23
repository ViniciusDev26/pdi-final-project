import os

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ApiClient:
    """Cliente HTTP da API de inferência; o front não acessa o modelo diretamente."""

    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(
            base_url=base_url or os.environ.get("PDI_API_URL", DEFAULT_API_URL),
            transport=transport,
            timeout=30.0,
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
