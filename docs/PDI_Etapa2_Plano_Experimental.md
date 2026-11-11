# Etapa 2 — Plano experimental (congelado)

> **Status: CONGELADO.** Este documento fixa *o que* será medido e *como* será
> comparado antes de qualquer resultado existir. Mudanças só por decisão da equipe,
> registradas no [histórico de alterações](#histórico-de-alterações) com justificativa —
> o enunciado da Etapa 2 exige justificar mudanças em relação ao escopo da Etapa 1.

Enunciado: [`[PDI] - Etapa 2 - instruções.pdf`](<[PDI] - Etapa 2 - instruções.pdf>) ·
Proposta aprovada: [`PDI_Etapa1_Equipe_e_Proposta.pdf`](PDI_Etapa1_Equipe_e_Proposta.pdf)

**Pendências que dependem de informação externa** (preencher assim que houver):

| Pendência | Onde impacta |
|---|---|
| Prazo de entrega no AVA (código e apresentação) | cronograma (seção 11) |
| Formato de entrega aceito pela professora (link do GitHub, zip ou ambos) | entrega (seção 7.3) |

## 1. Objetivo do plano

Responder às três questões científicas da Etapa 1 com experimentos controlados:

| # | Questão | Experimento que responde |
|---|---|---|
| Q1 | A abordagem tradicional (ABCD + cor + textura) chega perto da CNN? | Cenários **C** (seção 5.3) |
| Q2 | Quanto cada etapa de pré-processamento contribui para segmentação e classificação? | Cenários **A**, **L** e **T** (seções 5 e 6.3) + simulador (seção 6) |
| Q3 | Os modelos decidem pela lesão ou por artefatos (pelos, vinheta)? | Cenários **V** (seção 5.4) + Grad-CAM; réguas e marcações de caneta só qualitativamente (seção 5.4) |

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
| Resolução de trabalho | 300 × 300 (original da base); imagens de fora da base passam pela etapa P0 (seção 3) |
| **Regra do teste intocado** | o conjunto de teste **nunca** é usado para escolher nada: nem limiar de decisão, nem hiperparâmetro de modelo, nem **parâmetro de PDI** (limiares, tamanhos de kernel, cortes da vinheta etc.). Todo ajuste usa só treino e validação |
| Ajuste de parâmetros de PDI | feito em treino/validação; para a segmentação, com as máscaras anotadas da validação. O valor escolhido e a evidência ficam registrados no código (comentário junto à constante) |
| Máscaras anotadas — ajuste | 20 imagens da **validação** (10 benignas, 10 malignas; metade com vinheta), lista em `experiments/anotacao_val_ids.csv` |
| Máscaras anotadas — avaliação | 60 imagens do **teste** (30 benignas, 30 malignas; metade com vinheta), lista em `experiments/anotacao_teste_ids.csv`; usadas **só** na avaliação final de Dice/IoU e no V4 |
| Subconjunto "sem vinheta" | imagens do teste sem vinheta segundo o detector da etapa P1, lista versionada em `experiments/teste_sem_vinheta.csv` |
| Ajuste de hiperparâmetros | modelos tradicionais: validação cruzada estratificada 5-fold **somente no treino**; CNN: early stopping na validação (protocolo na seção 5.7) |
| Limiar de decisão | escolhido **na validação** para sensibilidade ≥ 0,90 |
| Modelo da Triagem | o modelo C (seção 5.3) com **maior AUC na validação**, com o limiar acima; a escolha é registrada no histórico. Os indicadores ABCD exibidos vêm sempre do pipeline tradicional, qualquer que seja o modelo escolhido |
| Métrica principal de comparação | **AUC-ROC no teste** (independe do limiar) |
| Métricas secundárias | sensibilidade, especificidade, precisão, F1, acurácia balanceada, matriz de confusão (no limiar fixado), tempo por imagem |
| Métrica de segmentação | Dice e IoU nas 60 máscaras do teste; Dice entre anotadores como referência (seção 5.8) |
| Incerteza | intervalo de confiança 95% por bootstrap (1.000 reamostragens do teste) em toda métrica reportada |

> **Ajuste em relação à Etapa 1:** a proposta citava validação cruzada no treino; o
> `splits.csv` também separa 15% de validação fixa. Os dois convivem: a CV ajusta os
> modelos tradicionais e a validação fixa escolhe o limiar, os parâmetros de PDI e faz o
> early stopping da CNN, que é cara demais para 5 folds.

> **Registro:** o corte de área da vinheta (3% da imagem, etapa P1) foi calibrado
> comparando a taxa de detecção com a análise exploratória da Etapa 1. Com o código
> atual, a taxa no **treino** é 29,2% das malignas e 1,1% das benignas (validação: 28,3%
> e 1,1%), coerente com a análise (≈31% e 0,6%); a escolha se sustenta sem usar o teste.

## 3. Pipeline e etapas avaliadas

O pré-processamento (P1–P5) é a primeira parte do pipeline. Fluxo completo:

```
imagem → P0 → P1 → P2 → P3 → P4 → P5 → S → F → M (SVM / Random Forest)
              └── pré-processamento ──┘
imagem → P0 → P1 → P2 → P3 → P4 → P5 ──────→ M (CNN)
```

- **P0 (normalização da entrada)** está sempre ligada e não participa da ablação: recorta
  o quadrado central e redimensiona para 300 × 300 (interpolação por área). Nas imagens da
  base é uma operação nula; existe para que imagens de outros dermatoscópios, enviadas na
  Triagem, cheguem na escala para a qual os kernels foram dimensionados.
- Nos cenários A e L, "pipeline completo" = P1–P5 ligadas; S, F e M ficam fixos, de
  modo que a diferença de resultado é atribuível só à etapa variada.
- A CNN recebe a imagem pré-processada diretamente (sem S e F), exceto no cenário V3,
  que usa a máscara de S.
- Treino e API usam o mesmo código de `packages/pdi`: a imagem enviada ao protótipo passa
  exatamente pelas mesmas etapas usadas no treino.

Cada etapa é uma função pura em `packages/pdi`, configurável: pode ser desligada ou
trocada por uma técnica alternativa (seção 6.1). A ordem é fixa. A tabela mostra a
técnica **padrão** de cada etapa:

| ID | Etapa | Técnica padrão | Métrica local da etapa (isolada) |
|---|---|---|---|
| P0 | Normalização da entrada | recorte do quadrado central + redimensionamento para 300 × 300 | — (sempre ligada) |
| P1 | Recorte da vinheta | limiarização da borda escura + máscara circular | taxa de detecção da vinheta; área recortada |
| P2 | Remoção de pelos | DullRazor (black-hat + limiar + inpainting) | PSNR/SSIM na restauração de pelos sintéticos (seção 6.4) |
| P3 | Normalização de cor | Shades of Gray (p = 6) | dispersão da cor média da pele entre imagens; ΔE sob dominante de cor sintética |
| P4 | Realce de contraste | CLAHE no canal L (CIELab) | contraste RMS; gradiente médio na borda da lesão |
| P5 | Filtragem | mediana / gaussiana | estabilidade da máscara sob ruído sintético |
| S  | Segmentação | limiarização (Otsu global ou adaptativa) + morfologia (abertura/fechamento) + componente central | **Dice / IoU** nas máscaras anotadas |
| F  | Características | lista fechada na seção 3.1 | importância por permutação; AUC por grupo |
| M  | Classificação | SVM, Random Forest, CNN (transfer learning) | métricas da seção 2 |

Cada etapa gera uma figura "antes → depois" com as mesmas 6 imagens de exemplo (lista
fixa em `experiments/exemplos_ids.csv`), usada na apresentação para mostrar a técnica
funcionando.

> **Correção do P2 (resolvido):** na implementação inicial, o DullRazor tratava pontos e
> glóbulos escuros *dentro* da lesão como pelos e apagava 41% dos pixels da lesão. A
> correção aceita como pelo só componentes finos e alongados (comprimento² / área ≥ 4) e
> sobe o limiar do black-hat de 12 para 40; com isso 2,6% da lesão é alterada (treino),
> mantendo 95% de cobertura dos pelos sintéticos. Calibrado só em treino/validação
> (`training.calibrate_p2`); a evidência está junto às constantes em `pdi/preprocess.py`.

### 3.1 Lista de características (F)

Todas calculadas dentro da máscara da lesão (S) sobre a imagem pré-processada. A lista é
fechada: incluir ou remover características é uma alteração registrada no histórico.

| Grupo | Critério | Características |
|---|---|---|
| Forma | A — assimetria | sobreposição da máscara com sua reflexão em torno de cada um dos dois eixos principais (2 valores, 0 = simétrica) |
| Forma | B — borda | compacidade (perímetro² / 4π·área); solidez (área / área do fecho convexo); dimensão fractal do contorno (box-counting); variação do raio (desvio-padrão / média da distância centroide–contorno) |
| Borda | B — nitidez | gradiente médio de Sobel ao longo do contorno; densidade de bordas de Canny numa faixa de 5 px em torno do contorno |
| Forma | D — dimensão | área da lesão relativa à imagem; maior diâmetro em pixels |
| Cor | C — cor | média e desvio-padrão de L, a, b (CIELab) e de H, S, V dentro da lesão (12 valores); número de cores dermatoscópicas presentes entre 6 de referência (branco, vermelho, marrom-claro, marrom-escuro, azul-acinzentado, preto), contando uma cor quando ocupa ≥ 5% da lesão; contraste de cor lesão × pele (ΔE entre as médias da lesão e de uma faixa de pele ao redor) |
| Textura | — | GLCM no canal L (contraste, homogeneidade, energia, correlação; média de 4 direções, distância 1); histograma LBP uniforme (P = 8, R = 1, 10 posições) |

São cerca de 45 valores por imagem. Os grupos usados nos cenários C3–C5 são: forma =
A, B e D (forma + borda); cor = C; textura = GLCM + LBP.

> **Limitação do critério D:** a base não informa a escala (mm por pixel) nem a
> ampliação do dermatoscópio. O "diâmetro > 6 mm" da regra ABCD não pode ser medido; D é
> representado por medidas relativas à imagem.

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

> **Comparações múltiplas:** são mais de 30 cenários. As comparações entre etapas e
> técnicas (A, L, T, S) são tratadas como **exploratórias**: não se aplica correção para
> comparações múltiplas, e só são apresentadas como conclusão as diferenças que passam no
> critério acima. Diferenças pequenas são reportadas como tais, sem interpretação causal.

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
| V4 | Grad-CAM nas 60 imagens anotadas do teste: fração da ativação dentro da máscara manual | Quantifica "onde o modelo olha" |
| V5 | Vinheta artificial inserida em 50 benignas do teste sem vinheta (lista fixa em `experiments/v5_ids.csv`); variação da probabilidade de melanoma, com e sem P1 | Se a probabilidade sobe só por causa da borda escura, o modelo aprendeu o atalho; com P1 ligada, a variação deveria ser ≈ 0 |

**Réguas e marcações de caneta** não têm cenário quantitativo: a base não as identifica
e não há etapa que as remova. São tratadas só na análise qualitativa dos erros e do
Grad-CAM (seção 5.6), e citadas nas limitações.

### 5.5 Cenários de segmentação (S)

Todos sobre o pipeline A5, avaliados por Dice/IoU nas máscaras do teste.

| ID | Método |
|---|---|
| S1 | Otsu global, sem pós-processamento |
| S2 | limiarização adaptativa (gaussiana), sem pós-processamento |
| S3 | Otsu global + morfologia + componente central (padrão do pipeline) |
| S4 | adaptativa + morfologia + componente central |

### 5.6 Saídas visuais obrigatórias

Além das tabelas, a página Resultados gera e exporta:

- curvas ROC dos cenários C sobrepostas e matrizes de confusão no limiar fixado;
- sobreposição da máscara automática com a manual (melhores e piores casos de Dice);
- **análise qualitativa de erros:** os 10 falsos negativos e os 10 falsos positivos de
  maior confiança do melhor modelo, com imagem, máscara, Grad-CAM e uma anotação da equipe
  sobre a causa provável (pelo, vinheta, régua, baixo contraste, lesão atípica…);
- gráfico de barras do impacto por etapa (cenários A e L) e das técnicas alternativas
  (cenários T);
- grades do simulador para as 6 imagens de exemplo (seção 6.2);
- tempo médio de processamento por imagem, por etapa.

### 5.7 Protocolo de treino (congelado)

O mesmo protocolo vale para todos os cenários do mesmo tipo de modelo, para que as
diferenças venham só da entrada.

**Modelos tradicionais (C1–C5 e o Random Forest de referência da ablação)**

- Padronização das características (média 0, desvio 1) ajustada **só no treino**.
- Busca em grade com validação cruzada estratificada 5-fold no treino, métrica AUC:
  - SVM (RBF): C ∈ {0,1; 1; 10; 100}, γ ∈ {scale; 0,001; 0,01; 0,1}; probabilidades por
    calibração de Platt;
  - Random Forest: 500 árvores; max_features ∈ {sqrt; 0,3}; min_samples_leaf ∈ {1; 5};
    max_depth ∈ {sem limite; 20}.
- Modelo final reajustado em todo o treino com os melhores hiperparâmetros; limiar
  escolhido na validação.

**CNN (C6, C7, V3)**

- Arquitetura: **EfficientNet-B0** pré-treinada na ImageNet (torchvision), cabeça
  substituída por uma camada linear de 1 saída.
- Entrada: 224 × 224 (redimensionada de 300 × 300), normalização da ImageNet.
- Aumento de dados **só geométrico**: espelhamento horizontal e vertical, rotações
  múltiplas de 90°, recorte aleatório com escala 0,8–1,0. Sem aumento de cor, para não
  confundir a comparação com a normalização de cor (P3).
- Otimização: AdamW, taxa de aprendizado 3 × 10⁻⁴, decaimento de peso 10⁻⁴, agenda
  cosseno, batch 32, precisão mista, até 30 épocas, early stopping com paciência 5 pela
  AUC na validação.
- Reprodutibilidade: semente 42, `cudnn.deterministic = True`, ambiente registrado no
  `metrics.json` (seção 7.1).

### 5.8 Protocolo de anotação das máscaras

- **Ferramenta:** página **Anotação** do sistema web (seção 7.1): o anotador desenha o
  contorno da lesão sobre a imagem e salva. Não se usa ferramenta externa.
- **O que anotar:** toda a área pigmentada da lesão, inclusive a periferia mais clara;
  pelos sobre a lesão são considerados parte dela; vinheta, régua e marcações ficam de
  fora.
- **Formato:** PNG binário 300 × 300, um por imagem e anotador, em
  `experiments/mascaras/{val,teste}/<anotador>/<arquivo>.png`, **versionado no git**
  (arquivos pequenos); cada salvamento é registrado em `experiments/mascaras/anotadores.csv`.
  A máscara de referência é a do responsável pela imagem (coluna `responsavel` da lista);
  nas compartilhadas, as dos outros dois servem só ao Dice entre anotadores.
- **Divisão:** cada integrante anota 1/3 das imagens. **10 imagens do teste são anotadas
  pelos três**: o Dice entre anotadores é reportado como referência — uma segmentação
  automática com Dice próximo ao dos anotadores entre si está no limite do que a anotação
  permite medir.
- As máscaras do teste não são abertas nem usadas durante o ajuste do pipeline.

## 6. Simulador: comparador de técnicas e combinações

O simulador compara o resultado de **diferentes combinações de técnicas** sobre a mesma
imagem ou sobre a base inteira: com e sem pré-processamento, técnica X ou técnica Y em
cada etapa, e abordagem tradicional × CNN. É ao mesmo tempo a ferramenta de exploração
da equipe, a fonte das evidências da Q1 e da Q2 e a peça central da demonstração.

### 6.1 O que pode variar

Cada etapa aceita "nenhuma" ou uma das técnicas listadas (padrão em **negrito**). Todas
as alternativas constam da tabela de técnicas da Etapa 1.

| Etapa | Opções |
|---|---|
| P1 Vinheta | nenhuma · **preenchimento com a cor da pele** |
| P2 Pelos | nenhuma · **DullRazor (black-hat + inpainting)** · fechamento morfológico |
| P3 Cor | nenhuma · **Shades of Gray (p = 6)** · Gray World (p = 1) · White Patch (p = ∞) |
| P4 Contraste | nenhum · **CLAHE (canal L)** · equalização global de histograma |
| P5 Filtro | nenhum · **mediana** · gaussiano |
| S Limiarização | **Otsu** · adaptativa |
| S Pós-processamento | sem · **morfologia + componente central** |
| Classificador | SVM · **Random Forest** · CNN |
| Entrada da CNN | imagem bruta · imagem pré-processada · só a lesão (fundo zerado pela máscara) |

Só na parte tradicional são 2 × 3 × 4 × 3 × 3 × 2 × 2 = **864 pipelines**, vezes 2
classificadores. Por isso o simulador tem dois modos.

### 6.2 Modo "uma imagem" (exploração e demonstração)

- Qualquer combinação pode ser montada e executada sobre uma imagem, mostrando cada etapa
  intermediária, a máscara e os indicadores ABCD.
- **Comparação lado a lado** de 2 a 4 combinações sobre a mesma imagem, com as
  diferenças calculadas: Dice entre as máscaras, variação de cada indicador ABCD e
  variação da probabilidade. Exemplos: sem × com pré-processamento; CLAHE × equalização
  global; Otsu × adaptativa; Random Forest × CNN.
- **A probabilidade só aparece para combinações com modelo treinado** (os cenários da
  seção 5 e 6.3, cada um com seu modelo salvo). Um classificador treinado com as
  características de um pipeline não vale para as de outro; aplicar mesmo assim mostraria
  números enganosos. Combinações livres mostram imagens, máscara e indicadores, sem
  probabilidade.
- A CNN é exibida pelos modelos C6 (imagem bruta), C7 (pré-processada) e V3 (só a
  lesão), cada um sobre a entrada com que foi treinado.
- Imagem de entrada: upload ou escolha de uma imagem da base (página Base de dados).
- Disponível no sistema web (seção 7.1), com botão para baixar a grade em figura para
  os slides.

### 6.3 Modo "base inteira" (evidência quantitativa)

- Executa o conjunto congelado de cenários (seção 5 e cenários T abaixo), cada um com o
  próprio classificador treinado, e gera a tabela comparativa com IC 95%.
- **Com × sem pré-processamento:** A0 × A5 na abordagem tradicional e C6 × C7 na CNN.
- **Técnica X × técnica Y:** cenários T, que partem de A5 e trocam a técnica de uma única
  etapa, para que a diferença seja atribuível só a essa troca:

| ID | Troca em relação a A5 |
|---|---|
| T1 | P2 = fechamento morfológico |
| T2 | P3 = Gray World |
| T3 | P3 = White Patch |
| T4 | P4 = equalização global de histograma |
| T5 | P5 = gaussiano |

- **Tradicional × CNN:** cenários C (seção 5.3).
- A grade completa de 864 combinações **não** é executada na base: além do custo, testar
  centenas de combinações e escolher a melhor no teste superestimaria o resultado. Uma
  combinação nova só entra como cenário novo, com ID próprio, registrada no histórico.

### 6.4 Artefatos sintéticos (apoio)

Imagens com artefatos inseridos de forma controlada servem a duas medições, não ao
simulador:

- métricas locais das etapas (seção 3): pelos sintéticos para o PSNR/SSIM de P2 e
  dominante de cor sintética para o ΔE de P3;
- o cenário V5 (vinheta artificial em lesões benignas).

## 7. Sistema web, código e entrega

### 7.1 Tudo pelo sistema web

**Regra: toda funcionalidade do projeto é acessível pelo sistema web.** Nada depende de
rodar comando no terminal; comandos `just` podem existir como atalho para a equipe, mas
chamam o mesmo código que a interface web.

Arquitetura: o front (Streamlit) só conversa com a API (FastAPI) por HTTP; a API usa
`packages/pdi` e `apps/training`.

```
navegador → web (Streamlit) ──HTTP──► API (FastAPI) ──► pdi (pipeline)
                                          │           └► training (cenários, modelos)
                                          └► results/ · models/ · data/ · experiments/
```

Páginas do sistema web:

| Página | Para quem | O que faz |
|---|---|---|
| **Triagem** | público-alvo (profissional da APS) | envia uma imagem e recebe: lesão segmentada, indicadores ABCD, probabilidade, Grad-CAM e a recomendação (encaminhar / acompanhar), com o aviso de que não substitui diagnóstico |
| **Simulador — uma imagem** | equipe e banca | monta combinações de técnicas (seção 6.1), mostra cada etapa e compara 2 a 4 combinações lado a lado (seção 6.2); baixa a grade em PNG |
| **Experimentos** | equipe | lista os cenários congelados (A, L, T, C, V, S) com status; botão **Executar** para um cenário ou para todos; progresso e log em tempo real; reexecução |
| **Resultados — base inteira** | equipe e banca | tabela comparativa com IC 95%, filtro com/sem vinheta, impacto por etapa (A/L/T), ROC e matrizes de confusão (C), Dice/IoU (S), V1–V5, Grad-CAM e análise de erros (seção 5.6); exporta figuras e tabelas; baixa e carrega o pacote de modelos e resultados (seção 7.3) |
| **Base de dados** | equipe e banca | distribuição das classes e divisões, duplicatas removidas, prevalência da vinheta por classe, exemplos por classe; envia uma imagem da base para o Simulador |
| **Anotação** | equipe | desenha e salva as máscaras de referência das listas de anotação (seção 5.8), com progresso por anotador e Dice entre anotadores |

Execução de experimentos pela web:

- A API recebe o pedido, enfileira o cenário e o executa em um processo separado (um por
  vez, para não disputar a GPU); o estado fica em `results/<ID>/status.json`
  (na fila / executando / concluído / erro, com progresso e log), e a página consulta a
  API periodicamente.
- Cenários tradicionais levam minutos; os de CNN podem levar horas. A página mostra o
  tempo estimado antes de executar.

Onde o sistema roda:

- **Servidor de experimentos: desktop da equipe — Linux Omarchy (Arch), Intel i9, 32 GB
  de RAM, GPU NVIDIA RTX 4060 Ti (CUDA).** API, web e fila de experimentos rodam nele via
  Docker Compose, com a GPU repassada ao contêiner da API; a equipe acessa o sistema pelo
  navegador, de qualquer máquina da rede. O treino da CNN e todos os cenários executam
  ali, para que todos os resultados venham da mesma máquina e do mesmo ambiente.
- **Máquinas de desenvolvimento:** desenvolvem e testam o código. O
  `training.device.get_device()` escolhe CUDA, MPS ou CPU automaticamente, então o mesmo
  código roda em qualquer uma; mas **números oficiais só vêm do desktop**.
- **Demonstração:** o sistema roda no desktop e é acessado de um notebook ou, se não
  houver rede confiável no local da apresentação, o pacote de modelos e resultados
  (seção 7.3) é carregado no notebook, que roda o modo de leitura.
- **Reprodutibilidade na GPU:** semente fixa e `torch.backends.cudnn.deterministic = True`
  nos cenários de CNN; cada `metrics.json` registra a GPU, o driver e as versões de CUDA e
  PyTorch usados.

### 7.2 Organização do código

```
packages/pdi/src/pdi/
  preprocess.py     P0–P5, cada etapa uma função pura
  segment.py        S
  features.py       F (lista da seção 3.1)
  pipeline.py       PipelineConfig (técnica de cada etapa ou "nenhuma") + run(image, config) -> resultado com intermediários
  artifacts.py      artefatos sintéticos (seção 6.4)
  metrics.py        Dice, IoU, PSNR, SSIM, ΔE, contraste
  visualize.py      contorno, legendas e grades para figuras
apps/training/src/training/
  scenarios.py      definição congelada dos cenários A, L, T, C, V, S (IDs das seções 5 e 6.3)
  run_scenario.py   executa um cenário -> results/<ID>/metrics.json + modelo + figuras
  compare.py        simulador, modo "uma imagem": executa e compara combinações
  report.py         simulador, modo "base inteira": consolida results/ em tabelas e gráficos
apps/api/src/api/
  main.py           rotas: /predict, /pipeline, /compare, /scenarios, /jobs, /results, /dataset, /annotations
  jobs.py           fila de execução de cenários (um processo por vez) + status.json
apps/web/src/web/
  app.py            roteador (st.navigation)
  views/            Triagem, Simulador, Experimentos, Resultados, Base de dados, Anotação
experiments/        listas fixas de IDs e máscaras anotadas (versionados)
results/            saídas dos experimentos (fora do git, reproduzíveis)
models/             modelos treinados (fora do git, distribuídos no pacote da seção 7.3)
```

Atalhos de terminal para a equipe (opcionais; chamam as mesmas funções que a API):
`just experimento <ID>`, `just experimentos`.

Cada `metrics.json` registra também o commit do git e a configuração usada, para que
qualquer número da apresentação seja rastreável até o código que o gerou.

### 7.3 Entrega do código no AVA

O enunciado pede "código-fonte e materiais necessários para compreender e executar a
solução". A professora não terá a GPU nem, necessariamente, o dataset; por isso há dois
modos de execução.

- **Modo leitura (para avaliar):** `docker compose up` em qualquer máquina, só com CPU.
  Carrega os modelos e resultados prontos; Triagem, Simulador e Resultados funcionam sem
  o dataset. É o caminho descrito em primeiro lugar no README.
- **Modo completo (para reproduzir):** com o dataset (`just data-download`) e, de
  preferência, GPU NVIDIA; a página Experimentos refaz todos os cenários.
- **O que é entregue:**
  - código: tag `etapa2-entrega` no GitHub; no AVA, o link e/ou um zip do repositório
    nessa tag, **sem** `data/`, `models/` e `results/` (formato a confirmar — ver
    pendências no topo);
  - pacote de modelos e resultados: arquivo anexado ao release `etapa2-entrega` do GitHub
    (`modelos-e-resultados.zip`), baixado pelo README ou carregado pela página Resultados;
  - material da apresentação: slides em PDF.
- **README** com, no topo, a seção "Como avaliar em 3 comandos" (modo leitura), depois a
  reprodução completa e o mapa do repositório.

## 8. Mapa da apresentação (15 min)

O enunciado lista 9 tópicos obrigatórios. Cada um tem uma fonte definida e um tempo
máximo; a soma fica abaixo dos 15 minutos para sobrar margem.

| # | Tópico obrigatório | Conteúdo | Fonte | Tempo |
|---|---|---|---|---|
| 1 | Problema social e público-alvo | APS de Fortaleza, dados do INCA/SBD, escassez de dermatologistas | Etapa 1 (resumida em 1 slide) | 1 min |
| 2 | Objetivo geral e específicos | objetivo geral + tabela dos 6 objetivos específicos com **status** (cumprido / parcial / não cumprido) e o slide/evidência que comprova cada um | Etapa 1 + seção 10 | 1 min |
| 3 | Arquitetura e ferramentas | diagrama: imagem → web (Streamlit) → API (FastAPI) → `pdi` (pipeline) → modelo; treino no servidor com GPU gerando `models/`; ferramentas: Python, OpenCV, scikit-image, scikit-learn, PyTorch, FastAPI, Streamlit, Docker, uv | README + diagrama | 1 min |
| 4 | Técnicas de PDI implementadas | uma figura "antes → depois" por etapa (P1–P5, S, F) com uma frase sobre o que ela faz | seção 3 | 2,5 min |
| 5 | Demonstração do protótipo | no sistema web: página Triagem (fluxo do profissional da APS) e Simulador ao vivo, com comparação lado a lado (sem × com pré-processamento, técnica X × Y, tradicional × CNN) | seções 6.2 e 7.1 | 3 min |
| 6 | Resultados visuais e métricas | apresentados pela página Resultados: impacto por etapa (A/L) e técnicas alternativas (T) (Q2); tabela e ROC tradicional × CNN (Q1); viés da vinheta, V5 e Grad-CAM (Q3); Dice/IoU com a referência entre anotadores; erros típicos | seções 5, 5.6, 6.3 e 7.1 | 3,5 min |
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
  esse subconjunto. Réguas e marcações só foram analisadas qualitativamente.
- Parâmetros de PDI calibrados nesta base (imagens de 300 × 300); imagens de outros
  equipamentos passam pela normalização P0, mas não foram avaliadas quantitativamente.
- Critério D da regra ABCD sem escala física (seção 3.1).
- Tarefa binária: o protótipo não reconhece outros tipos de lesão (carcinoma
  basocelular, queratoses etc.).
- Exige imagem dermatoscópica; não funciona com foto comum de celular.
- Máscaras de referência anotadas pela equipe (não por dermatologista), em 80 imagens.
- Comparações entre etapas tratadas como exploratórias (seção 4).
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
| P1–P5 e técnicas alternativas (incl. correção do P2), S, F (seção 3.1), artefatos sintéticos | Renan | `pdi/*` |
| Protocolo de treino (5.7); cenários A, L, T, C, V, S; CNN; Grad-CAM; simulador modo "base inteira" | Guilherme | `training/*`, `results/`, tabelas e gráficos |
| Engine de cenários; sistema web completo (6 páginas da seção 7.1); API, fila de experimentos e resultados; P0; README, release e pacote de entrega (7.3) | Carlos | `pipeline.py`, `compare.py`, `api`, `web`, `justfile`, `README.md` |
| Anotação das 80 máscaras (5.8), 1/3 cada + 10 compartilhadas | todos | `experiments/mascaras/` |
| Apresentação (15 min) | todos | slides seguindo os 9 tópicos do enunciado |

## 10. Critérios de pronto

- [ ] Máscaras anotadas: 20 da validação e 60 do teste, com as 10 compartilhadas pelos três
- [ ] Parâmetros de PDI ajustados só em treino/validação, com a evidência registrada
- [ ] Todos os cenários A, L, T, C, V, S executados com `metrics.json` e modelo em `results/`
- [ ] Simulador funcionando nos dois modos (uma imagem e base inteira)
- [ ] Todas as funcionalidades acessíveis pelo sistema web, inclusive executar cenários,
      ver resultados e anotar (seção 7.1); nenhuma depende só do terminal
- [ ] Modelo da Triagem escolhido pela regra da seção 2 e registrado no histórico
- [ ] Tabela de impacto por etapa (local / segmentação / final) com IC 95%
- [ ] Saídas visuais da seção 5.6 geradas, incluindo a análise de erros anotada
- [ ] Protótipo rodando com a página interativa + vídeo de plano B gravado
- [ ] Modo leitura testado numa máquina sem GPU e sem dataset, só com o README (seção 7.3)
- [ ] Tag e release `etapa2-entrega` publicados com o pacote de modelos e resultados
- [ ] Status de cada objetivo específico da Etapa 1 levantado, com evidência
- [ ] Apresentação cobre os 9 tópicos da seção 8 em no máximo 15 min (ensaiada e cronometrada)

### 10.1 Opcionais (só se houver tempo)

- **Teste externo qualitativo:** 20–30 imagens da ISIC (verificar a licença de cada
  imagem), sem métrica oficial, para discutir a generalização nas limitações.
- **Integração contínua:** GitHub Actions rodando lint e testes a cada push.

## 11. Cronograma

As datas dependem do prazo do AVA (pendência no topo). A ordem e as dependências são
fixas; as durações são **estimativas** de dias de trabalho e serão revistas quando o
prazo for conhecido.

| Marco | Conteúdo | Depende de | Estimativa |
|---|---|---|---|
| M1 | Características F (3.1), P0, correção do P2, página Anotação | — | 3–4 dias |
| M2 | Anotação das 80 máscaras; ajuste da segmentação nas 20 da validação | M1 (página Anotação) | 2 dias da equipe, em paralelo |
| M3 | Engine de cenários + fila + páginas Experimentos e Resultados; cenários tradicionais (A, L, T, S, C1–C5, V1, V2) | M1 | 4–5 dias |
| M4 | CNN no desktop (C6, C7, V3–V5) e Grad-CAM | M3 (engine) | 3–4 dias, incluindo horas de treino |
| M5 | Modelo da Triagem, Base de dados, saídas visuais da seção 5.6 | M3, M4 | 2–3 dias |
| M6 | Entrega: README, modo leitura testado, tag, release, vídeo do plano B | M5 | 1–2 dias |
| M7 | Slides e dois ensaios cronometrados | M5 | 2 dias, em paralelo a M6 |

**Congelamento do código:** 3 dias antes do prazo do AVA, nada além de correções.

## Histórico de alterações

| Data | Alteração | Justificativa |
|---|---|---|
| 2026-11-10 | Versão inicial congelada | — |
| 2026-11-10 | Incluídos Sobel/Canny (F) e limiarização adaptativa (S, cenários S1–S4); saídas visuais obrigatórias (5.6); mapa da apresentação com limitações, contribuição e melhorias (seção 8) | Cobrir técnicas prometidas na Etapa 1 e os 9 tópicos obrigatórios da apresentação da Etapa 2 |
| 2026-11-10 | Simulador redefinido como comparador de técnicas e combinações (seção 6), com técnicas alternativas por etapa (6.1) e cenários T1–T5 (6.3); artefatos sintéticos rebaixados a apoio (6.4); vinheta artificial virou o cenário V5 | Alinhar com o pedido da equipe: comparar a mesma imagem em várias técnicas, com/sem pré-processamento e tradicional × CNN |
| 2026-11-10 | Sistema web como interface única (seção 7.1): páginas Triagem, Simulador, Experimentos, Resultados e Base de dados; execução de cenários pela web via fila na API | Requisito da equipe: nenhuma funcionalidade pode depender só do terminal |
| 2026-11-10 | Experimentos e treino da CNN no desktop com RTX 4060 Ti (CUDA), sistema completo rodando nele via Docker com GPU; Mac apenas para desenvolvimento | GPU NVIDIA disponível na equipe; a GPU da Apple não é acessível dentro do Docker |
| 2026-11-10 | Revisão de lacunas: regra do teste intocado estendida aos parâmetros de PDI; 20 máscaras da validação para ajuste e 60 do teste só para avaliação, com 10 compartilhadas para Dice entre anotadores (5.8) e página Anotação; P0 (normalização da entrada); lista fechada de características (3.1) e limitação do critério D; protocolo de treino congelado (5.7); regra de escolha do modelo da Triagem; comparações entre etapas declaradas exploratórias; réguas só qualitativas na Q3; entrega no AVA com modo leitura e release (7.3); cronograma por marcos (11); opcionais (10.1) | Revisão do plano antes de existir qualquer resultado, para evitar vazamento do teste e lacunas de método e de entrega |
| 2026-11-10 | Correção do P2: limiar do black-hat 12 → 40 e filtro de alongamento ≥ 4, calibrados com pelos sintéticos em 150 imagens de treino e 150 de validação | O DullRazor apagava 41% dos pixels da lesão; agora 2,6%, com 0,5 dB a menos de PSNR que o melhor candidato sem filtro |
| 2026-11-10 | A lista fechada de características (3.1) resulta em 38 valores, não "cerca de 45" | Contagem exata da implementação; nenhuma característica incluída ou removida |
| 2026-11-10 | Listas de `experiments/` geradas (`training.experiment_lists`, semente 42); o teste tem só 4 benignas com vinheta, então a lista de anotação do teste tem 4 (não 15) benignas com vinheta e 15 malignas com vinheta; máscaras guardadas por anotador (`mascaras/<conjunto>/<anotador>/`) | A base não tem benignas com vinheta suficientes no teste; guardar por anotador permite o Dice entre anotadores nas 10 compartilhadas |
