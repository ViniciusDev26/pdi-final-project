# PDI — Triagem de melanoma em imagens dermatoscópicas

Projeto final da disciplina de Processamento Digital de Imagens (Unifor).
Classifica lesões de pele como benignas ou malignas (melanoma), comparando
técnicas tradicionais de PDI com redes neurais. A proposta completa está em
[`docs/PDI_Etapa1_Equipe_e_Proposta.pdf`](docs/PDI_Etapa1_Equipe_e_Proposta.pdf).

## Como avaliar em 3 comandos (modo leitura)

Não precisa de GPU nem do dataset: os modelos treinados e os resultados oficiais já vêm
junto. Só precisa do [Docker](https://docs.docker.com/get-docker/).

**Pelo pacote .zip (entrega no AVA):** descompacte, entre na pasta e rode

```sh
docker compose -f infra/docker-compose.yml up --build
```

**Pelo Git:** os modelos ficam no [Git LFS](https://git-lfs.com), então rode `git lfs install`
antes de clonar o repositório; depois, o mesmo `docker compose` acima.

Depois abra http://localhost:8501. A primeira construção baixa as dependências e leva
alguns minutos.

| Página | O que dá para ver sem o dataset |
|---|---|
| **Triagem** | envie uma imagem dermatoscópica: probabilidade do modelo C6, Grad-CAM, lesão segmentada, indicadores ABCD e recomendação |
| **Simulador** | envie uma imagem e compare combinações de técnicas etapa a etapa; quando a combinação é a de um cenário treinado, mostra a probabilidade de cada modelo |
| **Resultados** | todos os cenários com IC 95%, impacto por etapa, modelos, segmentação, viés e as saídas visuais (análise de erros, segmentação, grades do simulador) |
| **Base de dados** | classes por divisão, duplicatas e prevalência da vinheta (as imagens exigem o dataset) |
| **Experimentos** / **Anotação** | estado dos cenários e máscaras salvas; executar cenários e anotar exigem o dataset |

Para refazer tudo (modo completo), veja [Setup](#setup) e [Experimentos](#experimentos).

## Entregas

| Etapa | Documento | Apresentação | Enunciado |
|---|---|---|---|
| 1 — Equipe, público-alvo e proposta | [PDI_Etapa1_Equipe_e_Proposta.pdf](docs/PDI_Etapa1_Equipe_e_Proposta.pdf) | [PDI_Etapa1_Apresentacao.pdf](docs/PDI_Etapa1_Apresentacao.pdf) | [Instruções da professora](<docs/[PDI] - Etapa 1 - instruções.pdf>) |
| 2 — Implementação e avaliação | [Plano experimental](docs/PDI_Etapa2_Plano_Experimental.md) · [Status dos objetivos](docs/PDI_Etapa2_Status_Objetivos.md) | slides (deck do claude.ai, exportado em PDF) | [Instruções da professora](<docs/[PDI] - Etapa 2 - instruções.pdf>) |

O PDF da Etapa 1 é gerado a partir de [`docs/gerar_etapa1.py`](docs/gerar_etapa1.py):
edite o texto no script e rode `just docs-etapa1`.

## Estrutura

```
apps/
  api/        FastAPI: pipeline, simulador, anotação, fila de cenários, resultados
  training/   cenários congelados, treino (RF, SVM, CNN), avaliação e relatório
  web/        sistema web Streamlit (Triagem, Simulador, Experimentos, Resultados, Base de
              dados, Anotação)
packages/
  pdi/        biblioteca de PDI compartilhada (P0–P5, segmentação, características, artefatos)
infra/        Dockerfile, docker-compose (CPU) e override com GPU
experiments/  listas fixas de IDs, máscaras de referência e anotação dos erros (versionadas)
data/         dataset e splits locais (fora do git)
results/      resultados oficiais por cenário (métricas, predições, modelos no Git LFS) e
              saídas visuais da seção 5.6 em results/visuais/
models/       modelos exportados (fora do git)
docs/         documentos das etapas e plano experimental
```

É um monorepo com **uv workspaces**: um único `.venv` e um único `uv.lock` na
raiz. `apps/api` e `apps/training` dependem de `packages/pdi`, de modo que o
treino e a API usam exatamente o mesmo pré-processamento.

## Setup

Pré-requisitos: [mise](https://mise.jdx.dev) instalado e ativado no shell e
[Git LFS](https://git-lfs.com) (os modelos treinados ficam no LFS):

```sh
git lfs install     # uma vez por máquina, antes do clone (ou `git lfs pull` depois)
```

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
**Anotação** (80 máscaras de referência, já salvas em `experiments/mascaras/`).

### Resultados oficiais

Ficam versionados em `results/<ID>/` (commit de execução `6ee6538`, desktop com RTX 4060
Ti via Docker): `metrics.json` com as métricas e IC 95%, `predictions.csv`, `history.csv`
(CNN) e o modelo treinado (`model.joblib` ou `model.pt`, no Git LFS). AUC no teste:

| Cenário | Modelo | AUC (IC 95%) |
|---|---|---|
| C7 | CNN, imagem pré-processada | 0,977 (0,969–0,985) |
| C6 | CNN, imagem bruta | 0,976 (0,968–0,984) |
| C1 | características F + SVM | 0,949 (0,936–0,960) |
| C2 | características F + Random Forest | 0,945 (0,931–0,956) |
| C8 | regra ABCD da dermatoscopia (TDS), 100% PDI, sem aprendizado | 0,781 (0,751–0,810) |

O modelo da Triagem, pela regra do plano (maior AUC na validação), é o **C6**. A página
**Resultados** mostra todos os cenários, o impacto por etapa e a análise de viés. Na
segmentação (S1–S4, executados no commit `cbaeb6a`), a padrão (S3: Otsu + morfologia +
componente central) tem Dice 0,70 no teste. As máscaras de referência têm anotador único e
não foram revisadas por dermatologista (plano, seção 5.8). As saídas visuais
(`results/visuais/`, botão "Gerar de novo" na aba Saídas visuais) precisam do dataset para
serem refeitas, mas ficam versionadas para o modo leitura.

**Números oficiais só do servidor de experimentos** (Linux, RTX 4060 Ti), via
`just up-gpu`. A GPU é repassada ao contêiner por CDI (`nvidia-ctk cdi generate`). Em
outras máquinas o mesmo código roda; `training.device.get_device()` escolhe CUDA, MPS ou
CPU.

## Contrato entre treino e API

A API depende apenas do protocolo `Predictor` (`apps/api/src/api/predictor.py`). A
Triagem usa o modelo escolhido pela regra do plano (cenário C com maior AUC na validação,
hoje o C6), carregado de `results/<ID>/`. Se os modelos não estiverem presentes (clone sem
Git LFS), a API cai no `DummyPredictor`, **sem valor diagnóstico**, e a página avisa. O
Simulador usa `training.inference`: cada modelo só é aplicado à combinação de técnicas com
que foi treinado.
