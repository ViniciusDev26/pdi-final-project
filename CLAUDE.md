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
- `just setup` na primeira vez (o `.venv` precisa das dependências de dev; sem isso o
  `uv run` tenta baixá-las e fica parado se não houver rede).
- `just test`, `just lint`, `just fmt` (ruff, linha de 100). `just api` + `just web`
  para rodar o sistema.
- Testes do Streamlit: limpe `st.cache_data`/`st.cache_resource` entre testes (fixture
  em `apps/web/tests/test_web.py`).
- No Streamlit, `selectbox` com valor `None` aparece vazio; use um valor sentinela
  (ex.: `"off"`) e converta ao montar a configuração.

## Estado (2026-11-10, desktop)

Pronto:
- Pipeline P0 (sempre ligada) + P1–P5 + S configurável por técnica (`pdi.pipeline`), com
  técnicas alternativas da seção 6.1; cenários A, L, T, S em `training.scenarios`.
- Características F (`pdi.features`, lista fechada da seção 3.1: 38 valores em 3 grupos);
  NaN quando a segmentação vem vazia (≈1% numa amostra de 200 imagens de treino).
- API: `/pipeline/options`, `/pipeline`, `/compare` (com as características F);
  `/predict` ainda usa o `DummyPredictor` (sem valor diagnóstico).
- Web: Triagem (com o dummy) e Simulador (modo "uma imagem", com aba Características,
  sem probabilidade por enquanto).

Próximo (marcos da seção 11 do plano):
- **M1 (restante):** correção do P2, página Anotação.
- **M3:** engine de cenários + fila na API + páginas Experimentos e Resultados;
  cenários tradicionais.
- **M4:** CNN (EfficientNet-B0, protocolo da seção 5.7) no desktop; Docker Compose com
  GPU (arquivo de override separado, para o compose continuar funcionando sem NVIDIA).

Problemas conhecidos:
- DullRazor (P2) apaga glóbulos escuros dentro da lesão; aceitar só estruturas finas e
  alongadas como pelo.
- Limiarização adaptativa (S2/S4) fragmenta lesões grandes.
