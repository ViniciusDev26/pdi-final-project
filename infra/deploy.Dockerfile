# Imagem da API para o deploy público (modo leitura): igual a python.Dockerfile, mas com os
# modelos e resultados oficiais (results/, Git LFS) e as listas e máscaras (experiments/)
# dentro da imagem. Build a partir da raiz do repo (ignore: deploy.Dockerfile.dockerignore).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/
COPY apps/training/pyproject.toml apps/training/
COPY apps/web/pyproject.toml apps/web/
COPY packages/pdi/pyproject.toml packages/pdi/
RUN uv sync --frozen --no-dev --package api --no-install-workspace

COPY apps/ apps/
COPY packages/ packages/
RUN uv sync --frozen --no-dev --package api

COPY experiments/ experiments/
COPY results/ results/
# falha o build se os modelos vierem como ponteiros do Git LFS (clone sem `git lfs pull`)
RUN python -c "import pathlib,sys; s=pathlib.Path('results/C6/model.pt').stat().st_size; sys.exit(0 if s > 1_000_000 else 'results/C6/model.pt é ponteiro do LFS: ative o Git LFS no clone')"
RUN mkdir -p data && useradd --uid 1000 --no-create-home app && chown -R app results experiments
USER app
