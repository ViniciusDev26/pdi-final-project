"""Interface do protótipo. Rodar com: just web"""

import httpx
import streamlit as st

from web.api_client import ApiClient

LABELS = {
    "malignant": ("Suspeita de melanoma", "Recomenda-se encaminhar ao dermatologista."),
    "benign": ("Sem sinais de malignidade", "Manter acompanhamento de rotina."),
}

st.set_page_config(page_title="Triagem de melanoma", page_icon="🔬", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


client = get_client()

with st.sidebar:
    st.header("Status")
    status = client.health()
    if status is None:
        st.error("API indisponível")
    else:
        st.success(f"API online · modelo: {status['model']}")

st.title("Triagem de lesões de pele")
st.caption(
    "Protótipo acadêmico de apoio à triagem. **Não substitui o diagnóstico médico nem a biópsia.**"
)

uploaded = st.file_uploader("Imagem dermatoscópica da lesão", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

col_img, col_result = st.columns(2)
col_img.image(uploaded, caption=uploaded.name, use_container_width=True)

with col_result, st.spinner("Analisando..."):
    try:
        result = client.predict(uploaded.getvalue(), uploaded.name, uploaded.type)
    except httpx.HTTPStatusError as exc:
        st.error(f"A API recusou a imagem: {exc.response.json().get('detail', exc)}")
        st.stop()
    except httpx.HTTPError:
        st.error("Não foi possível conectar à API.")
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
