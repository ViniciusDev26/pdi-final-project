# CLAUDE.md

Projeto final de PDI (Unifor): triagem de melanoma em imagens dermatoscópicas,
comparando PDI tradicional com CNN. Visão geral e comandos no [README](README.md).

## Leia antes de mudar qualquer coisa

- **Plano congelado:** [`docs/PDI_Etapa2_Plano_Experimental.md`](docs/PDI_Etapa2_Plano_Experimental.md)
  define cenários, métricas, protocolo de treino, páginas e entrega. IDs de cenários e
  itens congelados (seção 2) não mudam sem registrar no histórico do plano.
- **Enunciado da Etapa 2:** `docs/[PDI] - Etapa 2 - instruções.pdf`.

## Regras do projeto

- **Tudo pelo sistema web.** Toda funcionalidade precisa de rota na API e página no
  Streamlit; o front só fala com a API por HTTP. Comandos `just` são só atalhos.
- **Teste intocado.** Nenhum ajuste (limiar, hiperparâmetro, parâmetro de PDI) usa o
  conjunto de teste; só treino e validação. Ao calibrar uma constante, registre a
  evidência num comentário junto a ela.
- **"Simulador" = comparador de combinações** de técnicas sobre a mesma imagem (com/sem
  pré-processamento, técnica X × Y, tradicional × CNN), não injeção de artefatos.
- **Números oficiais só do desktop** (Linux, RTX 4060 Ti), via Docker com GPU.
- Probabilidade no simulador só para combinações com modelo treinado.
- Textos, commits e documentação em português.

## Desenvolvimento

- Monorepo uv workspaces: `packages/pdi` (PDI), `apps/training` (cenários, comparador,
  treino), `apps/api` (FastAPI), `apps/web` (Streamlit, páginas em `views/` — não use
  `pages/`, que ativa a navegação automática antiga do Streamlit).
- Clone com Git LFS (`git lfs install` antes do clone, ou `git lfs pull` depois): os
  modelos de `results/<ID>/` estão no LFS.
- `just setup` na primeira vez (o `.venv` precisa das dependências de dev; sem isso o
  `uv run` tenta baixá-las e fica parado se não houver rede).
- `just test`, `just lint`, `just fmt` (ruff, linha de 100). `just api` + `just web`
  para rodar o sistema.
- Testes do Streamlit: limpe `st.cache_data`/`st.cache_resource` entre testes (fixture
  em `apps/web/tests/test_web.py`).
- No Streamlit, `selectbox` com valor `None` aparece vazio; use um valor sentinela
  (ex.: `"off"`) e converta ao montar a configuração.

## Estado (2026-11-10, desktop)

Pronto (M1–M4 implementados e executados):
- `pdi`: P0 + P1–P5 + S configuráveis; P2 corrigido e calibrado (`training.calibrate_p2`);
  características F (38 valores, `pdi.features`); artefatos sintéticos (`pdi.artifacts`).
- `experiments/`: listas fixas geradas por `training.experiment_lists` (não refazer).
- Engine: `training.run_scenario` (32 cenários), `training.report`, cache do pipeline por
  configuração em `results/_cache/` (fora do git; invalidado quando o código de `pdi` muda).
- CNN: `training.cnn` (protocolo 5.7, Grad-CAM); ~14–16 s por época na 4060 Ti.
- API: pipeline/simulador, `/annotations`, `/scenarios`, `/jobs` (fila persistente em
  `results/_fila.json`, sobrevive a reinício), `/results`.
- Web: Triagem (ainda com o dummy), Simulador, Experimentos, Resultados, Anotação.
- Docker: `just up-gpu` (override `infra/docker-compose.gpu.yml`, GPU por CDI).

Resultados oficiais (commit `6ee6538`, Docker com GPU) versionados em `results/<ID>/`:
`metrics.json`, `predictions.csv`, `history.csv` e o modelo (`model.joblib` comprimido ou
`model.pt`, no Git LFS). Para refazer um cenário: página Experimentos (ou
`just experimento <ID>`), depois commitar os arquivos novos de `results/<ID>/`.
- AUC no teste: C7 0,977, C6 0,976, V3 0,954, C1 0,949, C2/A5 0,945, A0 0,960.
- Modelo da Triagem pela regra da seção 2: **C6** (AUC val 0,989, limiar 0,933).
- V1: vinheta sozinha dá AUC 0,70. V5: vinheta artificial em 50 benignas faz 46 virarem
  malignas no C6 e 31 no C7; no RF com P1 (C2) a variação é ≈ 0.
- S1–S4 e V4 falharam de propósito: aguardam as máscaras anotadas.

Próximo:
- Equipe: decidir sobre a ablação (P1 e P3 pioram o RF com significância; histórico do
  plano) e anotar as 80 máscaras (página Anotação) → rodar S1–S4 e V4.
- **M5:** Triagem com o C6 (substituir o `DummyPredictor`, com Grad-CAM e indicadores ABCD
  do pipeline tradicional), página Base de dados, probabilidade no simulador para
  combinações com modelo treinado, saídas visuais da seção 5.6.
- **M6/M7:** modo leitura, README "como avaliar em 3 comandos", tag/release, slides.

Problemas conhecidos:
- Limiarização adaptativa (S2/S4) fragmenta lesões grandes.
- C7 (com P1) ainda reage à vinheta artificial (V5): o preenchimento uniforme da P1 parece
  ser, ele próprio, um sinal que a CNN aprendeu; discutir na Q3.
