# PDI — Triagem de melanoma em imagens dermatoscópicas

Projeto final da disciplina de Processamento Digital de Imagens (Unifor).
Classifica lesões de pele como benignas ou malignas (melanoma), comparando
técnicas tradicionais de PDI com redes neurais. A proposta completa está em
[`docs/PDI_Etapa1_Equipe_e_Proposta.pdf`](docs/PDI_Etapa1_Equipe_e_Proposta.pdf).

## Entregas

| Etapa | Documento | Enunciado |
|---|---|---|
| 1 — Equipe, público-alvo e proposta | [PDI_Etapa1_Equipe_e_Proposta.pdf](docs/PDI_Etapa1_Equipe_e_Proposta.pdf) | [PDF da professora](<[PDI] - Etapa 1 - D. Equipe e Proposta.docx (1).pdf>) |

O PDF da Etapa 1 é gerado a partir de [`docs/gerar_etapa1.py`](docs/gerar_etapa1.py):
edite o texto no script e rode `just docs-etapa1`.

## Estrutura

```
apps/
  api/        FastAPI de inferência (/health, /predict)
  training/   preparação de dados, treinamento e avaliação
  web/        interface Streamlit (consome a API por HTTP)
packages/
  pdi/        biblioteca de PDI compartilhada (pré-processamento, segmentação, features)
infra/        Dockerfile e docker-compose (API + web)
data/         dataset e splits locais (fora do git)
models/       modelos exportados pelo treino (fora do git)
docs/         documentos das etapas
```

É um monorepo com **uv workspaces**: um único `.venv` e um único `uv.lock` na
raiz. `apps/api` e `apps/training` dependem de `packages/pdi`, de modo que o
treino e a API usam exatamente o mesmo pré-processamento.

## Setup

Pré-requisito: [mise](https://mise.jdx.dev) instalado e ativado no shell.

```sh
mise install        # instala Python 3.12, uv e just nas versões do mise.toml
just setup          # instala as dependências de todo o workspace
```

### Dataset

[Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
(Kaggle, licença CC0). Com um token do Kaggle configurado
(`~/.kaggle/kaggle.json` ou a variável `KAGGLE_API_TOKEN`):

```sh
just data-download  # baixa e extrai em data/melanoma_cancer_dataset/
just data-prepare   # remove duplicatas e gera data/splits.csv
```

Sem token, baixe o zip pela página do Kaggle e extraia em `data/`, de forma
que exista `data/melanoma_cancer_dataset/{train,test}/{benign,malignant}/`.

O `data-prepare` remove os 22 pares de imagens duplicadas (mantendo a cópia do
teste) e separa 15% do treino para validação, estratificado e com semente fixa.
**Todos os modelos devem usar o `data/splits.csv`**, para que as comparações
sejam justas.

## Comandos

| Comando | O que faz |
|---|---|
| `just` | lista os comandos |
| `just api` | sobe a API em http://localhost:8000 (docs em `/docs`) |
| `just web` | sobe o front em http://localhost:8501 (com a API rodando) |
| `just up` | sobe API + front via Docker Compose |
| `just test` | roda os testes |
| `just lint` / `just fmt` | verifica / corrige lint e formatação (ruff) |
| `just sync` | sincroniza dependências após editar algum `pyproject.toml` |
| `just docs-etapa1` | regera o PDF da Etapa 1 |

Para adicionar uma dependência a um app: `uv add --package training <pacote>`.

## Treinamento local

O treino roda localmente. `training.device.get_device()` escolhe a GPU da Apple
(MPS), CUDA ou CPU automaticamente.

## Contrato entre treino e API

A API depende apenas do protocolo `Predictor` (`apps/api/src/api/predictor.py`).
Enquanto não houver modelo treinado, ela usa o `DummyPredictor`, **sem valor
diagnóstico**, para que front e API possam ser desenvolvidos em paralelo.
Os modelos treinados serão exportados para `models/` (fora do git).
