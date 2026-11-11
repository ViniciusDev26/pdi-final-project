"""Página Triagem: fluxo do profissional da Atenção Primária."""

import httpx
import streamlit as st

from web.common import api_error_message, get_client

LABELS = {
    "malignant": ("Suspeita de melanoma", "Recomenda-se encaminhar ao dermatologista."),
    "benign": ("Sem sinais de malignidade", "Manter acompanhamento de rotina."),
}

client = get_client()

st.title("Triagem de lesões de pele")
st.caption(
    "Protótipo acadêmico de apoio à triagem. **Não substitui o diagnóstico médico nem a biópsia.**"
)

uploaded = st.file_uploader("Imagem dermatoscópica da lesão", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

col_img, col_result = st.columns(2)
col_img.image(uploaded, caption=uploaded.name, width="stretch")

with col_result, st.spinner("Analisando..."):
    try:
        result = client.predict(uploaded.getvalue(), uploaded.name, uploaded.type)
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
        st.stop()

    title, advice = LABELS[result["label"]]
    prob = result["probability_malignant"]
    (st.error if result["label"] == "malignant" else st.success)(f"**{title}**")
    st.metric("Probabilidade de malignidade", f"{prob:.1%}")
    st.progress(prob)
    st.write(advice)
    st.caption(
        f"Modelo {result['model']} v{result['model_version']} · limiar {result['threshold']:.2f}"
    )
    if result["model"] == "dummy":
        st.warning("Modelo de demonstração: o resultado não tem valor diagnóstico.")
