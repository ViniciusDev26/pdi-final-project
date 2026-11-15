"""Página Triagem: fluxo do profissional da Atenção Primária (plano, seção 7.1).

Mostra a probabilidade do modelo escolhido pela regra da seção 2 (cenário C com maior AUC
na validação), o Grad-CAM (onde a CNN olhou), a lesão segmentada e os indicadores ABCD do
pipeline tradicional, com a recomendação e o aviso de que não substitui o diagnóstico.
"""

import httpx
import pandas as pd
import streamlit as st

from web.common import api_error_message, b64_image, get_client, model_label, score_text

TITLES = {
    "malignant": ("Suspeita de melanoma", st.error),
    "benign": ("Sem sinais de malignidade", st.success),
}
ABCD = {  # característica → (critério, descrição curta)
    "assimetria_eixo_maior": ("A", "Assimetria no eixo maior (0 = simétrica)"),
    "assimetria_eixo_menor": ("A", "Assimetria no eixo menor (0 = simétrica)"),
    "compacidade": ("B", "Compacidade da borda (1 = círculo; maior = mais irregular)"),
    "variacao_raio": ("B", "Variação do raio ao longo da borda"),
    "n_cores": ("C", "Número de cores dermatoscópicas presentes (de 6)"),
    "delta_e_pele": ("C", "Contraste de cor com a pele (ΔE)"),
    "diametro_max_px": ("D", "Diâmetro máximo (pixels; sem escala física)"),
    "area_relativa": ("D", "Área da lesão / área da imagem"),
}

client = get_client()

st.title("Triagem de lesões de pele")
st.caption(
    "Protótipo acadêmico de apoio à triagem. **Não substitui o diagnóstico médico nem a "
    "biópsia.** Use imagens dermatoscópicas; fotos comuns de celular não são suportadas."
)

uploaded = st.file_uploader("Imagem dermatoscópica da lesão", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

with st.spinner("Analisando..."):
    try:
        result = client.predict(uploaded.getvalue(), uploaded.name, uploaded.type)
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
        st.stop()

title, box = TITLES[result["label"]]
prob = result["probability_malignant"]

col_img, col_result = st.columns([3, 2])
with col_result:
    box(f"**{title}**")
    st.metric("Probabilidade de melanoma", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.markdown(f"**Recomendação:** {result['recommendation']}.")
    st.caption(
        f"Modelo {result['model']} ({result['model_description']}) · versão "
        f"{result['model_version']} · limiar {result['threshold']:.3f}, escolhido na validação "
        "para sensibilidade ≥ 90%."
    )
    if result["model"] == "dummy":
        st.warning("Modelo de demonstração: o resultado não tem valor diagnóstico.")
    if result["vignette_detected"]:
        st.info(
            "A imagem tem vinheta (borda escura do dermatoscópio). Na base de treino a vinheta "
            "aparece muito mais em melanomas; interprete o resultado com cautela."
        )

with col_img:
    panels = [("Imagem (após normalização)", result["image"])]
    if result["lesion_overlay"]:
        panels.append(("Lesão segmentada (pipeline tradicional)", result["lesion_overlay"]))
    if result["gradcam"]:
        panels.append(("Grad-CAM: regiões que mais pesaram", result["gradcam"]))
    for col, (caption, image) in zip(st.columns(len(panels)), panels, strict=True):
        col.image(b64_image(image), caption=caption, width="stretch")

st.subheader("Indicadores ABCD")
abcd = result["abcd"]
if all(v is None for v in abcd.values()):
    st.warning("Não foi possível segmentar a lesão; os indicadores ABCD não foram calculados.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {"Critério": crit, "Indicador": desc, "Valor": abcd.get(key)}
                for key, (crit, desc) in ABCD.items()
            ]
        ).style.format({"Valor": "{:.3f}"}, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Indicadores calculados na lesão segmentada pelo pipeline de PDI tradicional; servem "
        "para explicar a imagem e não entram na probabilidade da CNN."
    )

# ---------------------------------------------------------------- PDI tradicional × CNN

st.subheader("Comparação: PDI puro × PDI tradicional × CNN nesta imagem")
comparison = result.get("comparison") or []
if not comparison:
    st.info("Nenhum modelo treinado disponível para comparar.")
else:
    APPROACH = {
        "regra": "PDI puro (regra, sem aprendizado)",
        "tradicional": "PDI tradicional (PDI + classificador)",
        "cnn": "CNN",
    }

    def pct(v):
        return None if v is None else v * 100

    table = pd.DataFrame(
        [
            {
                "Abordagem": APPROACH[m["kind"]],
                "Modelo": model_label(m),
                "Prob. de melanoma": score_text(m),
                "Limiar": score_text(m, "threshold"),
                "Resultado": "melanoma" if m["label"] == "malignant" else "benigna",
                "AUC (teste)": m["auc_test"],
                "Sensib. (teste, %)": pct(m["sensitivity_test"]),
                "Especif. (teste, %)": pct(m["specificity_test"]),
            }
            for m in comparison
        ]
    )
    for kind, label in APPROACH.items():
        sub = [m for m in comparison if m["kind"] == kind]
        if sub:
            positives = sum(m["label"] == "malignant" for m in sub)
            st.caption(f"{label}: {positives} de {len(sub)} modelos indicam melanoma.")
    st.dataframe(
        table.style.format(
            {
                "AUC (teste)": "{:.3f}",
                "Sensib. (teste, %)": "{:.1f}",
                "Especif. (teste, %)": "{:.1f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"O resultado principal acima vem do {result['model']}, escolhido pela maior AUC na "
        "validação (regra do plano). Cada modelo usa o próprio limiar, escolhido na validação "
        "para sensibilidade ≥ 90%. Tradicionais: 38 características (ou só forma, cor ou "
        "textura) extraídas da lesão segmentada pelo pipeline completo. CNN: C6 na imagem "
        "bruta, C7 na imagem pré-processada, V3 só na lesão. As colunas de teste são o "
        "desempenho de cada modelo nas 999 imagens de teste, não nesta imagem. A regra C8 "
        "(TDS) não dá probabilidade: soma pontos dos critérios ABCD com os pesos da literatura "
        "e encaminha a partir de 4,75."
    )
    rule = next((m for m in comparison if m["kind"] == "regra" and m.get("details")), None)
    if rule:
        d = rule["details"]
        structures = ", ".join(d.get("estruturas") or []) or "nenhuma acima do típico"
        st.markdown(
            f"**Regra ABCD (TDS) nesta imagem:** A = {d['A']} eixo(s) assimétrico(s), "
            f"B = {d['B']} de 8 segmentos de borda abrupta, C = {d['C']} cor(es), "
            f"D = {d['D']} estrutura(s) ({structures}) → TDS = 1,3·{d['A']} + 0,1·{d['B']} "
            f"+ 0,5·{d['C']} + 0,5·{d['D']} = **{d['tds']:.2f}**".replace(".", ",")
        )
