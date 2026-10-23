set dotenv-load

kaggle_dataset := "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"

# Lista os comandos disponíveis
default:
    @just --list

# Instala as ferramentas (mise) e as dependências de todo o workspace
setup:
    mise install
    uv sync --all-packages

# Sincroniza as dependências após mudanças nos pyproject.toml
sync:
    uv sync --all-packages

# Baixa o dataset do Kaggle para data/ (requer token do Kaggle)
data-download:
    uvx kaggle datasets download -d {{kaggle_dataset}} -p data --unzip

# Verifica o dataset, remove duplicatas e gera data/splits.csv
data-prepare *args:
    uv run --package training python -m training.prepare_data {{args}}

# Sobe a API em modo desenvolvimento (http://localhost:8000/docs)
api:
    uv run --package api uvicorn api.main:app --reload

# Sobe o front Streamlit (http://localhost:8501); requer a API rodando
web:
    uv run --package web streamlit run apps/web/src/web/app.py

# Sobe API e front juntos via Docker Compose
up:
    docker compose -f infra/docker-compose.yml up --build

# Roda todos os testes
test *args:
    uv run pytest {{args}}

# Verifica lint e formatação
lint:
    uv run ruff check .
    uv run ruff format --check .

# Corrige lint e formata o código
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Regera o PDF da Etapa 1
docs-etapa1:
    python3 docs/gerar_etapa1.py
