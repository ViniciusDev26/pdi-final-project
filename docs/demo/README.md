# Imagens da demonstração

Seis imagens do conjunto de teste (Kaggle, CC0) para a demonstração ao vivo e o plano B.
Envie-as na página **Triagem** (https://pdi.cvinicius.com ou `just up`). O resultado
esperado é o do modelo C6, limiar 0,933 (escolhido na validação).

| Arquivo | Diagnóstico real | Resultado esperado | Probabilidade | O que mostrar |
|---|---|---|---|---|
| `1_maligna_melanoma_10170.jpg` | melanoma | suspeita de melanoma | ≈ 100% | acerto; lesão escura e irregular, com vinheta |
| `2_maligna_melanoma_10164.jpg` | melanoma | suspeita de melanoma | ≈ 100% | acerto sem vinheta; Grad-CAM sobre a lesão |
| `3_benigna_melanoma_10001.jpg` | benigna | sem sinais de malignidade | < 0,1% | acerto; nevo claro e regular |
| `4_benigna_melanoma_9991.jpg` | benigna | sem sinais de malignidade | < 0,1% | acerto mesmo com borda escura do campo |
| `5_maligna_melanoma_10587.jpg` | melanoma | **sem sinais de malignidade (erro)** | < 0,1% | falso negativo: melanoma de aspecto regular, sem vinheta; o Grad-CAM não ativa |
| `6_benigna_melanoma_10094.jpg` | benigna | **suspeita de melanoma (erro)** | ≈ 99,9% | falso positivo: vinheta e marcação de caneta azul |

As imagens 5 e 6 mostram os limites do modelo (slides de Q3 e de erros). Atenção ao escolher
outras imagens: os melanomas do início e do fim da pasta `test/malignant/` (arquivos
`melanoma_10105` a `10154` e `10555` a `10604`) são um grupo em que a CNN acerta só 38%
(plano, histórico de 2026-11-15).
