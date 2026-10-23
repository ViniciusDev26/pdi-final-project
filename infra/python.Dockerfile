# Imagem de um app do workspace uv. Build a partir da raiz do repo:
#   docker build -f infra/python.Dockerfile --build-arg APP=api .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ARG APP
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Dependências primeiro (camada em cache enquanto o uv.lock não mudar).
# Todos os pyproject.toml são copiados porque o uv precisa enxergar o workspace inteiro.
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/
COPY apps/training/pyproject.toml apps/training/
COPY apps/web/pyproject.toml apps/web/
COPY packages/pdi/pyproject.toml packages/pdi/
RUN uv sync --frozen --no-dev --package ${APP} --no-install-workspace

COPY apps/ apps/
COPY packages/ packages/
RUN uv sync --frozen --no-dev --package ${APP}
