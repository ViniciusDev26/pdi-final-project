# PDI — Triagem de melanoma em imagens dermatoscópicas

Projeto final da disciplina de Processamento Digital de Imagens (Unifor).
Classifica lesões de pele como benignas ou malignas (melanoma), comparando
técnicas tradicionais de PDI com redes neurais. A proposta completa está em
[`docs/PDI_Etapa1_Equipe_e_Proposta.pdf`](docs/PDI_Etapa1_Equipe_e_Proposta.pdf).

## Entregas

| Etapa | Documento | Apresentação | Enunciado |
|---|---|---|---|
| 1 — Equipe, público-alvo e proposta | [PDI_Etapa1_Equipe_e_Proposta.pdf](docs/PDI_Etapa1_Equipe_e_Proposta.pdf) | [PDI_Etapa1_Apresentacao.pdf](docs/PDI_Etapa1_Apresentacao.pdf) | [Instruções da professora](<docs/[PDI] - Etapa 1 - instruções.pdf>) |

O PDF da Etapa 1 é gerado a partir de [`docs/gerar_etapa1.py`](docs/gerar_etapa1.py):
edite o texto no script e rode `just docs-etapa1`.

## Estrutura

```
apps/
  api/        FastAPI: pipeline, simulador, anotação, fila de cenários, resultados
  training/   cenários congelados, treino (RF, SVM, CNN), avaliação e relatório
  web/        sistema web Streamlit (Triagem, Simulador, Experimentos, Resultados, Anotação)
packages/
  pdi/        biblioteca de PDI compartilhada (P0–P5, segmentação, características, artefatos)
infra/        Dockerfile, docker-compose (CPU) e override com GPU
experiments/  listas fixas de IDs e máscaras anotadas (versionadas)
data/         dataset e splits locais (fora do git)
results/      saídas dos cenários: métricas, predições, modelos e logs (fora do git)
models/       modelos exportados (fora do git)
docs/         documentos das etapas e plano experimental
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
| `just up` | sobe API + front via Docker Compose (só CPU) |
| `just up-gpu` | sobe o sistema no servidor de experimentos, com a GPU no contêiner da API |
| `just experimento <ID>` | executa um cenário (atalho; a página Experimentos faz o mesmo) |
| `just experimentos` | executa todos os cenários |
| `just test` | roda os testes |
| `just lint` / `just fmt` | verifica / corrige lint e formatação (ruff) |
| `just sync` | sincroniza dependências após editar algum `pyproject.toml` |
| `just docs-etapa1` | regera o PDF da Etapa 1 |

Para adicionar uma dependência a um app: `uv add --package training <pacote>`.

## Experimentos

O plano congelado está em
[`docs/PDI_Etapa2_Plano_Experimental.md`](docs/PDI_Etapa2_Plano_Experimental.md). Todos os
cenários (A, L, T, S, C, V) são executados pela página **Experimentos** do sistema web: a
API os coloca numa fila e roda um por vez num processo separado, gravando em
`results/<ID>/` o `metrics.json` (métricas com IC 95%, commit e ambiente), as predições, o
modelo e o log. A página **Resultados** consolida tudo (impacto por etapa, ROC, viés,
figuras para os slides). Cenários S e V4 dependem das máscaras desenhadas na página
**Anotação**.

**Números oficiais só do servidor de experimentos** (Linux, RTX 4060 Ti), via
`just up-gpu`. A GPU é repassada ao contêiner por CDI (`nvidia-ctk cdi generate`). Em
outras máquinas o mesmo código roda; `training.device.get_device()` escolhe CUDA, MPS ou
CPU.

## Contrato entre treino e API

A API depende apenas do protocolo `Predictor` (`apps/api/src/api/predictor.py`).
Enquanto não houver modelo treinado, ela usa o `DummyPredictor`, **sem valor
diagnóstico**, para que front e API possam ser desenvolvidos em paralelo.
Os modelos treinados serão exportados para `models/` (fora do git).
