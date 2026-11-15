# Etapa 2 — Status dos objetivos da Etapa 1

Levantamento do que foi cumprido em relação à proposta aprovada na Etapa 1
([`PDI_Etapa1_Equipe_e_Proposta.pdf`](PDI_Etapa1_Equipe_e_Proposta.pdf)), com a evidência
de cada item. Os números são do conjunto de teste (999 imagens: 499 benignas, 500 malignas),
com IC 95% por bootstrap, e vêm de `results/<ID>/metrics.json` (servidor com RTX 4060 Ti,
Docker com GPU). Critério do plano: **cumprido**, **parcial** ou **não cumprido**.

| # | Objetivo específico (Etapa 1) | Status | Evidência |
|---|---|---|---|
| 1 | Investigar e implementar pré-processamento (remoção de pelos, normalização de cor, realce) e avaliar o efeito nas etapas seguintes | **Cumprido** | P0–P5 implementados com técnicas alternativas (T1–T5); métricas locais por etapa e ablação acumulativa (A) e por remoção (L) |
| 2 | Implementar e avaliar a segmentação automática (limiarização + morfologia) contra máscaras de referência | **Parcial** | S1–S4 avaliados em 60 máscaras do teste; a referência não foi anotada manualmente pela equipe, como previsto (plano, seção 5.8) |
| 3 | Extrair características de forma, borda, cor e textura e treinar SVM e Random Forest | **Cumprido** | 38 características (`pdi.features`); C1 (SVM) e C2 (RF) com AUC 0,949 e 0,945 |
| 4 | Treinar uma CNN por transferência de aprendizado e comparar com os tradicionais nas mesmas divisões e métricas | **Cumprido** | EfficientNet-B0 (ImageNet): C6 0,976 e C7 0,977, mesmas divisões, limiar e bootstrap |
| 5 | Analisar interpretabilidade e robustez (Grad-CAM, ablação; lesão × artefatos) | **Cumprido** | Grad-CAM (V4), ablação (A/L/T), atalho da vinheta (V1, V2, V5); réguas e marcações só qualitativamente, na análise de erros |
| 6 | Protótipo web com lesão segmentada, indicadores e probabilidade | **Cumprido** | Sistema web (Streamlit + FastAPI) com 6 páginas; Triagem com o C6, Grad-CAM, ABCD e recomendação |

Objetivo geral (protótipo que segmenta e classifica, comparando PDI tradicional e CNN):
**cumprido**, com a ressalva do objetivo 2.

## Detalhe por objetivo

### 1. Pré-processamento — cumprido

- **Implementado:** P0 (normalização da entrada), P1 vinheta, P2 pelos (DullRazor), P3 cor
  (Shades of Gray), P4 contraste (CLAHE), P5 filtragem (mediana), cada uma com alternativa:
  fechamento morfológico, Gray World, White Patch, equalização global, gaussiano.
- **Métricas locais** (100 imagens do teste): P2 reconstrói pelos sintéticos com PSNR 35,2 dB
  e SSIM 0,96 (após a correção que reduziu de 41% para 2,6% os pixels de lesão apagados);
  P1 detecta vinheta em 30,3% das malignas e 1,1% das benignas.
- **Efeito na classificação (Random Forest):** nenhuma etapa melhora a AUC de forma
  significativa; P1 e P3 pioram (P1: −0,007 incremental, −0,011 na remoção; P3: −0,010 e
  −0,011). A imagem bruta (A0, AUC 0,960) tem AUC maior que o pipeline completo (A5, 0,945). As
  técnicas alternativas (T1–T5) não diferem da padrão. Decisão registrada no plano: manter
  o pipeline e apresentar o resultado como achado (P1 remove o atalho da vinheta).
- Evidência: página Resultados (Impacto por etapa, Técnicas, Métricas locais);
  `results/A*`, `results/L*`, `results/T*`.

### 2. Segmentação — parcial

- **Implementado:** Otsu global e limiarização adaptativa, com e sem morfologia + componente
  central (S1–S4).
- **Dice no teste (60 máscaras):** S3 (Otsu + morfologia, padrão) 0,70 [0,63; 0,77];
  S4 0,55; S1 0,54; S2 0,34. A morfologia ajuda nos dois métodos; a adaptativa fragmenta
  lesões grandes.
- **Por que parcial:** a proposta previa máscaras anotadas manualmente pela equipe, com Dice
  entre anotadores. As 80 máscaras de referência não foram anotadas manualmente pela equipe
  e não há Dice entre anotadores (decisão registrada no plano, seção 5.8 e histórico).
- Evidência: Resultados → Segmentação e Saídas visuais; `results/S*`, `experiments/mascaras/`.

### 3. Características e classificadores tradicionais — cumprido

- 38 características (forma e borda: assimetria, compacidade, solidez, dimensão fractal,
  variação do raio, Sobel, Canny, área, diâmetro; cor: Lab/HSV, número de cores, ΔE; textura:
  GLCM e LBP).
- C1 SVM: AUC 0,949 [0,936; 0,960], sensibilidade 87,6%, especificidade 85,8%.
  C2 RF: AUC 0,945 [0,931; 0,956], sensibilidade 90,2%, especificidade 81,6%.
- Por grupo (RF): cor sozinha 0,930, forma 0,868, textura 0,805.

### 4. CNN e comparação — cumprido

- EfficientNet-B0 pré-treinada, protocolo congelado (seção 5.7 do plano).
- C6 (imagem bruta): AUC 0,976 [0,968; 0,984], sensibilidade 87,4%, especificidade 98,4%.
  C7 (pré-processada): AUC 0,977 [0,969; 0,985]. V3 (só a lesão): 0,954.
- **Resposta à Q1:** a abordagem tradicional chega perto (0,949 × 0,976), mas fica abaixo,
  com intervalos que não se sobrepõem; a maior diferença é a especificidade no mesmo nível de
  sensibilidade (98,4% × 85,8%).

### 5. Interpretabilidade e robustez — cumprido

- **V1:** só o indicador de vinheta já dá AUC 0,70: é um atalho da base.
- **V2:** o RF (C2) tem AUC 0,85 nas imagens com vinheta e 0,94 nas sem.
- **V5:** vinheta artificial em 50 benignas sem vinheta: 46 viram malignas no C6, 31 no C7,
  45 no RF sem P1 (L1) e só 4 no RF com P1 (C2).
- **V4:** fração do Grad-CAM dentro da lesão: 40% no C6 e 56% no C7, contra 21% de área média
  da lesão (acima do acaso, mas com muita atenção fora da lesão).
- **Análise de erros** (10 FN e 8 FP do C6, causa anotada): lesões de aspecto atípico ou
  enganoso, baixo contraste, campo circular escuro não detectado, marcação de caneta, bolha.
- Réguas e marcações: só qualitativas, como previsto no plano.

### 6. Protótipo — cumprido

- Sistema web: Triagem, Simulador, Experimentos, Resultados, Base de dados e Anotação; toda
  funcionalidade tem rota na API.
- Triagem: probabilidade do C6 (escolhido pela maior AUC na validação), Grad-CAM, lesão
  segmentada, indicadores ABCD e recomendação (encaminhar / acompanhar).
- Modo leitura testado num clone limpo, sem GPU e sem dataset (README, "Como avaliar em 3
  comandos").
