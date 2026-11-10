# Etapa 2 — Plano experimental (congelado)

> **Status: CONGELADO.** Este documento fixa *o que* será medido e *como* será
> comparado antes de qualquer resultado existir. Mudanças só por decisão da equipe,
> registradas no [histórico de alterações](#histórico-de-alterações) com justificativa —
> o enunciado da Etapa 2 exige justificar mudanças em relação ao escopo da Etapa 1.

Enunciado: [`[PDI] - Etapa 2 - instruções.pdf`](<[PDI] - Etapa 2 - instruções.pdf>) ·
Proposta aprovada: [`PDI_Etapa1_Equipe_e_Proposta.pdf`](PDI_Etapa1_Equipe_e_Proposta.pdf)

## 1. Objetivo do plano

Responder às três questões científicas da Etapa 1 com experimentos controlados:

| # | Questão | Experimento que responde |
|---|---|---|
| Q1 | A abordagem tradicional (ABCD + cor + textura) chega perto da CNN? | Cenários **C** (seção 5.3) |
| Q2 | Quanto cada etapa de pré-processamento contribui para segmentação e classificação? | Cenários **A** e **L** (seções 5.1 e 5.2) + simulador (seção 6) |
| Q3 | Os modelos decidem pela lesão ou por artefatos (pelos, réguas, vinheta)? | Cenários **V** (seção 5.4) + Grad-CAM + simulador |

Regra central: **cada etapa do pipeline tem um cenário de teste próprio**, com a
métrica local da etapa *e* o impacto no resultado final (segmentação e classificação),
sempre comparado com o mesmo ponto de referência.

## 2. Itens congelados

Tudo abaixo é fixo para todos os experimentos. Se algum mudar, **todos** os cenários
são reexecutados.

| Item | Valor |
|---|---|
| Base | Melanoma Skin Cancer Dataset of 10000 Images (Kaggle, CC0) |
| Divisão | `data/splits.csv` (gerado por `just data-prepare`): treino / validação 15% estratificada / teste original; 22 duplicatas removidas |
| Semente | `42` (`training.config.SEED`) em toda fonte de aleatoriedade |
| Resolução de trabalho | 300 × 300 (original da base) |
| Subconjunto anotado | 60 imagens do teste (30 benignas, 30 malignas; metade com vinheta, metade sem), lista versionada em `experiments/anotacao_ids.csv` |
| Subconjunto "sem vinheta" | imagens do teste sem vinheta segundo o detector da etapa P1, lista versionada em `experiments/teste_sem_vinheta.csv` |
| Ajuste de hiperparâmetros | modelos tradicionais: validação cruzada estratificada 5-fold **somente no treino**; CNN: early stopping na validação |
| Limiar de decisão | escolhido **na validação** para sensibilidade ≥ 0,90; o teste nunca é usado para escolher nada |
| Métrica principal de comparação | **AUC-ROC no teste** (independe do limiar) |
| Métricas secundárias | sensibilidade, especificidade, precisão, F1, acurácia balanceada, matriz de confusão (no limiar fixado), tempo por imagem |
| Métrica de segmentação | Dice e IoU no subconjunto anotado |
| Incerteza | intervalo de confiança 95% por bootstrap (1.000 reamostragens do teste) em toda métrica reportada |

> **Ajuste em relação à Etapa 1:** a proposta citava validação cruzada no treino; o
> `splits.csv` também separa 15% de validação fixa. Os dois convivem: a CV ajusta os
> modelos tradicionais e a validação fixa escolhe o limiar e faz o early stopping da CNN,
> que é cara demais para 5 folds.

## 3. Pipeline e etapas avaliadas

O pré-processamento (P1–P5) é a primeira parte do pipeline. Fluxo completo:

```
imagem → P1 → P2 → P3 → P4 → P5 → S → F → M (SVM / Random Forest)
         └─ pré-processamento ─┘
imagem → P1 → P2 → P3 → P4 → P5 ──────→ M (CNN)
```

- Nos cenários A e L, "pipeline completo" = P1–P5 ligadas; S, F e M ficam fixos, de
  modo que a diferença de resultado é atribuível só à etapa variada.
- A CNN recebe a imagem pré-processada diretamente (sem S e F), exceto no cenário V3,
  que usa a máscara de S.
- Treino e API usam o mesmo código de `packages/pdi`: a imagem enviada ao protótipo passa
  exatamente pelas mesmas etapas usadas no treino.

Cada etapa é uma função pura em `packages/pdi` que pode ser ligada/desligada por
configuração. A ordem é fixa:

| ID | Etapa | Técnica | Métrica local da etapa (isolada) |
|---|---|---|---|
| P1 | Recorte da vinheta | limiarização da borda escura + máscara circular | taxa de detecção da vinheta; área recortada |
| P2 | Remoção de pelos | DullRazor (black-hat + limiar + inpainting) | PSNR/SSIM na restauração de pelos sintéticos (simulador) |
| P3 | Normalização de cor | Shades of Gray (p = 6) | dispersão da cor média da pele entre imagens; ΔE sob dominante de cor sintética |
| P4 | Realce de contraste | CLAHE no canal L (CIELab) | contraste RMS; gradiente médio na borda da lesão |
| P5 | Filtragem | mediana / gaussiana | estabilidade da máscara sob ruído sintético |
| S  | Segmentação | limiarização (Otsu global ou adaptativa) + morfologia (abertura/fechamento) + maior componente conexo | **Dice / IoU** no subconjunto anotado |
| F  | Características | forma (A, D), borda (B: gradiente Sobel e bordas Canny ao longo do contorno), cor (C: estatísticas em HSV e CIELab), textura (GLCM, LBP) | importância por permutação; AUC por grupo |

Cada etapa gera uma figura "antes → depois" com as mesmas 6 imagens de exemplo (lista
fixa em `experiments/exemplos_ids.csv`), usada na apresentação para mostrar a técnica
funcionando.
| M  | Classificação | SVM, Random Forest, CNN (transfer learning) | métricas da seção 2 |

## 4. Como o impacto de uma etapa é medido

Para cada etapa `Pk` existem duas leituras complementares:

- **Ganho incremental (cenários A):** liga as etapas uma a uma, na ordem.
  Impacto de `Pk` = métrica(A_k) − métrica(A_{k−1}).
- **Perda na remoção (cenários L):** parte do pipeline completo e desliga só `Pk`.
  Impacto de `Pk` = métrica(completo) − métrica(completo sem Pk).

Uma etapa só é considerada **útil** se o impacto for positivo nas duas leituras e o
IC 95% da diferença (bootstrap pareado) não contiver zero. Caso contrário, é reportada
como "sem efeito mensurável" — o que também é um resultado válido para a apresentação.

O impacto é medido em três níveis, sempre na mesma tabela:

1. **Local:** a métrica própria da etapa (seção 3).
2. **Segmentação:** Dice/IoU.
3. **Final:** AUC-ROC e sensibilidade do Random Forest (classificador de referência para
   ablação, por ser rápido e estável), no teste completo e no subconjunto sem vinheta.

## 5. Matriz de cenários

IDs são permanentes: um cenário nunca é renomeado nem reaproveitado.

### 5.1 Cenários acumulativos (A)

| ID | P1 vinheta | P2 pelos | P3 cor | P4 CLAHE | P5 filtro |
|---|:-:|:-:|:-:|:-:|:-:|
| A0 (imagem bruta) | – | – | – | – | – |
| A1 | ✓ | – | – | – | – |
| A2 | ✓ | ✓ | – | – | – |
| A3 | ✓ | ✓ | ✓ | – | – |
| A4 | ✓ | ✓ | ✓ | ✓ | – |
| A5 (completo) | ✓ | ✓ | ✓ | ✓ | ✓ |

### 5.2 Cenários de remoção (L)

| ID | Pipeline |
|---|---|
| L1 | completo sem P1 (vinheta) |
| L2 | completo sem P2 (pelos) |
| L3 | completo sem P3 (cor) |
| L4 | completo sem P4 (CLAHE) |
| L5 | completo sem P5 (filtro) |

### 5.3 Cenários de comparação de modelos (C) — Q1

Todos com o pipeline vencedor da ablação (A5, salvo se a ablação mostrar que alguma
etapa atrapalha — nesse caso, justificar e registrar no histórico).

| ID | Entrada | Modelo |
|---|---|---|
| C1 | características F (todas) | SVM (RBF) |
| C2 | características F (todas) | Random Forest |
| C3 | características só de forma (A, B, D) | Random Forest |
| C4 | características só de cor (C) | Random Forest |
| C5 | características só de textura | Random Forest |
| C6 | imagem bruta (A0) | CNN |
| C7 | imagem pré-processada (A5) | CNN |

### 5.4 Cenários de viés e artefatos (V) — Q3

| ID | Teste | Leitura esperada |
|---|---|---|
| V1 | Classificador treinado **só** com o indicador "tem vinheta" | Mede o "atalho": se a AUC for alta, a base permite acertar sem olhar a lesão |
| V2 | C2 e C6/C7 avaliados separadamente nas imagens **com** e **sem** vinheta | Diferença de AUC = quanto o modelo depende do artefato |
| V3 | CNN com a pele fora da máscara de segmentação zerada | Se o desempenho se mantém, o modelo usa a lesão |
| V4 | Grad-CAM nas 60 imagens anotadas: fração da ativação dentro da máscara manual | Quantifica "onde o modelo olha" |

### 5.5 Cenários de segmentação (S)

Todos sobre o pipeline A5, avaliados por Dice/IoU no subconjunto anotado.

| ID | Método |
|---|---|
| S1 | Otsu global, sem pós-processamento |
| S2 | limiarização adaptativa (gaussiana), sem pós-processamento |
| S3 | Otsu global + morfologia + maior componente conexo (padrão do pipeline) |
| S4 | adaptativa + morfologia + maior componente conexo |

### 5.6 Saídas visuais obrigatórias

Além das tabelas, cada rodada de `just relatorio` gera:

- curvas ROC dos cenários C sobrepostas e matrizes de confusão no limiar fixado;
- sobreposição da máscara automática com a manual (melhores e piores casos de Dice);
- **análise qualitativa de erros:** os 10 falsos negativos e os 10 falsos positivos de
  maior confiança do melhor modelo, com imagem, máscara, Grad-CAM e uma anotação da equipe
  sobre a causa provável (pelo, vinheta, baixo contraste, lesão atípica…);
- gráfico de barras do impacto por etapa (cenários A e L) e as curvas do simulador;
- tempo médio de processamento por imagem, por etapa.

## 6. Simulador de artefatos (testes de estresse)

Como a base não diz quais imagens têm pelos, dominante de cor etc., o simulador
**injeta artefatos controlados** em imagens limpas e mede, para cada intensidade, o
resultado com e sem a etapa que deveria corrigir o artefato. Isso isola o efeito de cada
etapa de uma forma que a ablação na base real não consegue.

| Artefato sintético | Parâmetro de intensidade | Etapa avaliada | Saídas medidas |
|---|---|---|---|
| Pelos (curvas escuras aleatórias) | nº de fios: 0, 10, 25, 50, 100 | P2 | PSNR/SSIM vs original, Dice, Δ probabilidade |
| Dominante de cor (ganho por canal RGB) | desvio: 0, 10%, 20%, 30%, 40% | P3 | ΔE da pele, variação das características de cor, Δ probabilidade |
| Baixo contraste (compressão do histograma) | fator: 1,0; 0,8; 0,6; 0,4 | P4 | gradiente na borda, Dice |
| Ruído gaussiano | σ: 0, 5, 10, 20 | P5 | Dice, Δ probabilidade |
| Vinheta artificial (borda escura circular) | raio: sem, 0,95, 0,85, 0,75 | P1 | Δ probabilidade em lesões benignas (deveria ser ≈ 0) |

- Imagens de origem: 100 imagens do teste (50/50), sem vinheta, com pouco pelo (lista
  fixa em `experiments/simulador_ids.csv`); semente fixa por imagem e intensidade.
- Resultado: uma curva "métrica × intensidade" por artefato, com duas linhas
  (etapa ligada / desligada). Essas curvas são a principal evidência visual da Q2.
- O teste de vinheta artificial é o mais importante para a Q3: se colocar uma borda
  escura numa lesão benigna aumentar a probabilidade de melanoma, o modelo aprendeu o
  atalho.

### Modo interativo (demonstração)

A mesma engine alimenta uma página do protótipo (Streamlit), usada na apresentação:

- upload de imagem ou escolha de uma imagem de exemplo;
- interruptores para ligar/desligar cada etapa P1–P5;
- controles deslizantes para injetar cada artefato do simulador;
- exibição lado a lado de cada etapa intermediária, da máscara, das características ABCD
  e da probabilidade, atualizadas a cada mudança.

## 7. Organização no código

```
packages/pdi/src/pdi/
  preprocess.py     P1–P5, cada etapa uma função pura
  segment.py        S
  features.py       F
  pipeline.py       PipelineConfig (quais etapas ligadas) + run(image, config) -> resultado com intermediários
  artifacts.py      geradores de artefatos sintéticos do simulador
  metrics.py        Dice, IoU, PSNR, SSIM, ΔE, contraste
apps/training/src/training/
  scenarios.py      definição congelada dos cenários A, L, C, V (IDs da seção 5)
  run_scenario.py   executa um cenário -> results/<ID>/metrics.json + figuras
  simulate.py       executa o simulador -> results/sim/<artefato>.csv + curvas
  report.py         consolida results/ em tabelas e gráficos para a apresentação
experiments/        listas fixas de IDs (anotação, sem vinheta, simulador)
results/            saídas dos experimentos (fora do git, reproduzíveis)
```

Comandos previstos: `just experimento <ID>`, `just experimentos` (todos),
`just simulador`, `just relatorio`.

Cada `metrics.json` registra também o commit do git e a configuração usada, para que
qualquer número da apresentação seja rastreável até o código que o gerou.

## 8. Mapa da apresentação (15 min)

O enunciado lista 9 tópicos obrigatórios. Cada um tem uma fonte definida e um tempo
máximo; a soma fica abaixo dos 15 minutos para sobrar margem.

| # | Tópico obrigatório | Conteúdo | Fonte | Tempo |
|---|---|---|---|---|
| 1 | Problema social e público-alvo | APS de Fortaleza, dados do INCA/SBD, escassez de dermatologistas | Etapa 1 (resumida em 1 slide) | 1 min |
| 2 | Objetivo geral e específicos | objetivo geral + tabela dos 6 objetivos específicos com **status** (cumprido / parcial / não cumprido) e o slide/evidência que comprova cada um | Etapa 1 + seção 9 | 1 min |
| 3 | Arquitetura e ferramentas | diagrama: imagem → web (Streamlit) → API (FastAPI) → `pdi` (pipeline) → modelo; treino offline gerando `models/`; ferramentas: Python, OpenCV, scikit-image, scikit-learn, PyTorch, FastAPI, Streamlit, Docker, uv | README + diagrama | 1 min |
| 4 | Técnicas de PDI implementadas | uma figura "antes → depois" por etapa (P1–P5, S, F) com uma frase sobre o que ela faz | seção 3 | 2,5 min |
| 5 | Demonstração do protótipo | página interativa: imagem real → etapas → máscara → indicadores ABCD → probabilidade; ligar/desligar etapa e injetar vinheta artificial ao vivo | seção 6 | 3 min |
| 6 | Resultados visuais e métricas | impacto por etapa (A/L) e curvas do simulador (Q2); tabela e ROC tradicional × CNN (Q1); viés da vinheta e Grad-CAM (Q3); Dice/IoU; erros típicos | seções 5 e 5.6 | 3,5 min |
| 7 | Limitações | ver seção 8.1 | — | 1 min |
| 8 | Contribuição para o público-alvo | ver seção 8.2 | — | 0,5 min |
| 9 | Conclusões e melhorias | respostas diretas a Q1, Q2 e Q3 + ver seção 8.3 | — | 1 min |
|   | **Total** | | | **14,5 min** |

**Plano B da demonstração:** vídeo gravado da demo (≤ 3 min) e 6 imagens de exemplo já
carregadas localmente, para o caso de falha de rede, de máquina ou de tempo.

### 8.1 Limitações a apresentar (lista inicial, completar com os resultados)

- Base única, de origem não documentada e sem dados dos pacientes (idade, sexo, tom de
  pele): o desempenho na população brasileira miscigenada não é garantido.
- Viés de aquisição (vinheta) presente na própria base; o resultado é reportado com e sem
  esse subconjunto.
- Tarefa binária: o protótipo não reconhece outros tipos de lesão (carcinoma
  basocelular, queratoses etc.).
- Exige imagem dermatoscópica; não funciona com foto comum de celular.
- Máscaras de referência anotadas pela equipe (não por dermatologista) e em apenas 60
  imagens.
- Sem validação clínica: ferramenta de apoio à triagem, não diagnóstico.

### 8.2 Contribuição para o público-alvo

Traduzir as métricas para a rotina da UBS, no limiar escolhido: "a cada 100 lesões
examinadas com a proporção de melanomas da base de teste, o protótipo encaminharia X,
deixaria de encaminhar Y melanomas e evitaria Z encaminhamentos desnecessários". Mostrar
também que o profissional recebe as **evidências visuais** (máscara, indicadores ABCD,
Grad-CAM), e não só um rótulo.

### 8.3 Possibilidades de melhoria (lista inicial)

- Treinar e validar em bases maiores e multicêntricas (ISIC) e com imagens de pacientes
  brasileiros.
- Classificação multiclasse e estimativa de incerteza.
- Versão móvel com dermatoscópio acoplado ao celular.
- Avaliação com dermatologistas e profissionais da APS em um estudo piloto.

## 9. Divisão de trabalho

| Frente | Responsável | Entregáveis |
|---|---|---|
| P1–P5, S, F, simulador de artefatos, anotação das 60 máscaras | Renan (anotação dividida entre os três) | `pdi/*`, `experiments/anotacao_ids.csv`, máscaras |
| Cenários A, L, C, V; CNN; Grad-CAM; relatório de métricas | Guilherme | `training/*`, `results/`, tabelas e gráficos |
| Engine de cenários, API com intermediários, página interativa, README de execução | Carlos | `pipeline.py`, `api`, `web`, `justfile` |
| Apresentação (15 min) | todos | slides seguindo os 9 tópicos do enunciado |

## 10. Critérios de pronto

- [ ] Todos os cenários A, L, C, V, S executados com `metrics.json` em `results/`
- [ ] Curvas do simulador para os 5 artefatos
- [ ] Tabela de impacto por etapa (local / segmentação / final) com IC 95%
- [ ] Saídas visuais da seção 5.6 geradas, incluindo a análise de erros anotada
- [ ] Protótipo rodando com a página interativa + vídeo de plano B gravado
- [ ] README permite que alguém de fora rode tudo do zero
- [ ] Status de cada objetivo específico da Etapa 1 levantado, com evidência
- [ ] Apresentação cobre os 9 tópicos da seção 8 em no máximo 15 min (ensaiada e cronometrada)

## Histórico de alterações

| Data | Alteração | Justificativa |
|---|---|---|
| 2026-11-10 | Versão inicial congelada | — |
| 2026-11-10 | Incluídos Sobel/Canny (F) e limiarização adaptativa (S, cenários S1–S4); saídas visuais obrigatórias (5.6); mapa da apresentação com limitações, contribuição e melhorias (seção 8) | Cobrir técnicas prometidas na Etapa 1 e os 9 tópicos obrigatórios da apresentação da Etapa 2 |
