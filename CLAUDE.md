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

## Estado (2026-11-15, desktop)

Pronto (M1–M5 e o código do M6):
- `pdi`: P0 + P1–P5 + S configuráveis; P2 corrigido e calibrado (`training.calibrate_p2`);
  características F (38 valores, `pdi.features`); artefatos sintéticos (`pdi.artifacts`).
- `experiments/`: listas fixas geradas por `training.experiment_lists` (não refazer) e as 80
  máscaras de referência em `experiments/mascaras/<conjunto>/` (anotador único, desenhadas
  por IA a pedido da equipe; plano 5.8). Não existe mais divisão por anotador.
- Engine: `training.run_scenario` (32 cenários), `training.report`, cache do pipeline por
  configuração em `results/_cache/` (fora do git; invalidado quando o código de `pdi` muda).
- CNN: `training.cnn` (protocolo 5.7, Grad-CAM); ~14–16 s por época na 4060 Ti.
- API: pipeline/simulador, `/annotations`, `/scenarios`, `/jobs` (fila persistente em
  `results/_fila.json`, sobrevive a reinício), `/results`.
- Web: Triagem (modelo C6, Grad-CAM, ABCD), Simulador (probabilidade só para combinações
  iguais ao pipeline de um modelo: `training.inference`), Experimentos, Resultados (com a aba
  Saídas visuais), Base de dados, Anotação.
- Saídas visuais da seção 5.6: `training.visuals` → `results/visuais/` (versionado, serve o
  modo leitura); causa dos erros anotada em `experiments/analise_erros.csv`.
- Modo leitura testado (clone limpo, Docker só CPU, sem dataset): README "Como avaliar em 3
  comandos".
- Docker: `just up-gpu` (override `infra/docker-compose.gpu.yml`, GPU por CDI).

Resultados oficiais (commit `6ee6538`, Docker com GPU) versionados em `results/<ID>/`:
`metrics.json`, `predictions.csv`, `history.csv` e o modelo (`model.joblib` comprimido ou
`model.pt`, no Git LFS). Para refazer um cenário: página Experimentos (ou
`just experimento <ID>`), depois commitar os arquivos novos de `results/<ID>/`.
- AUC no teste: C7 0,977, C6 0,976, V3 0,954, C1 0,949, C2/A5 0,945, A0 0,960.
- Ablação: P1 e P3 pioram o RF com significância; decisão da equipe (2026-11-15): manter
  o A5 e apresentar como achado (P1 remove o atalho da vinheta; P3 é hipótese). Histórico do plano.
- Modelo da Triagem pela regra da seção 2: **C6** (AUC val 0,989, limiar 0,933).
- V1: vinheta sozinha dá AUC 0,70. V5: vinheta artificial em 50 benignas faz 46 virarem
  malignas no C6 e 31 no C7; no RF com P1 (C2) a variação é ≈ 0.
- Achado exploratório: CNN detecta só ~38% dos 100 melanomas do início/fim da pasta de teste
  (pouca vinheta) e ~100% dos demais; imagens da demo em `docs/demo/` (não use as primeiras
  da pasta para demonstrar).
- Segmentação (S1–S4 e V4 executados no commit `cbaeb6a`; Dice no teste, 60 máscaras): S3 (Otsu + morfologia, padrão) 0,70, S4 0,55,
  S1 0,54, S2 0,34. V4: fração do Grad-CAM dentro da lesão 0,40 no C6 e 0,56 no C7 (a lesão
  ocupa 21% da imagem em média).

Próximo:
- Status dos objetivos da Etapa 1: `docs/PDI_Etapa2_Status_Objetivos.md` (5 cumpridos, 1
  parcial: segmentação). Slides da apresentação: deck no claude.ai (exporta PDF/PPTX).
- Falta: vídeo do plano B, ensaios cronometrados, formato de entrega no AVA; tag/release
  dispensados pela equipe por ora.

Problemas conhecidos:
- Limiarização adaptativa (S2/S4) fragmenta lesões grandes.
- C7 (com P1) ainda reage à vinheta artificial (V5): o preenchimento uniforme da P1 parece
  ser, ele próprio, um sinal que a CNN aprendeu; discutir na Q3.
