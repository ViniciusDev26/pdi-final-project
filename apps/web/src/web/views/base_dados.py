"""Página Base de dados (plano experimental, seção 7.1).

Distribuição das classes e divisões, duplicatas removidas, prevalência da vinheta por
classe e exemplos por classe; envia uma imagem da base para o Simulador.
"""

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from web.common import api_error_message, get_client

SPLITS = {"train": "Treino", "val": "Validação", "test": "Teste"}
LABELS = {"benign": "Benigna", "malignant": "Maligna"}
PAGE_SIZE = 12

client = get_client()

st.title("Base de dados")
st.caption(
    "Melanoma Skin Cancer Dataset of 10000 Images (Kaggle, CC0). Duplicatas removidas (mantida "
    "a cópia do teste) e 15% do treino separados para validação, estratificado, semente 42."
)

try:
    summary = client.dataset_summary()
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Imagens usadas", f"{summary['total']:,}".replace(",", "."))
c2.metric("Imagens originais", f"{summary['imagens_originais']:,}".replace(",", "."))
c3.metric("Duplicatas removidas", summary["duplicatas_removidas"])

counts = pd.DataFrame(summary["contagem"]).T.rename(index=SPLITS, columns=LABELS)
counts["Total"] = counts.sum(axis=1)
vig = (pd.DataFrame(summary["vinheta"]).T * 100).rename(index=SPLITS, columns=LABELS)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Classes por divisão")
    st.dataframe(counts, width="stretch")
    long = (
        counts.drop(columns="Total")
        .reset_index(names="Divisão")
        .melt("Divisão", var_name="Classe", value_name="Imagens")
    )
    st.altair_chart(
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("Divisão:N", sort=list(SPLITS.values())),
            y="Imagens:Q",
            color="Classe:N",
            xOffset="Classe:N",
        )
        .properties(height=260),
        width="stretch",
    )
with col_b:
    st.subheader("Prevalência da vinheta (%)")
    st.dataframe(vig.style.format("{:.1f}"), width="stretch")
    geral = summary["vinheta_geral"]
    st.write(
        f"Na base inteira, {geral['malignant']:.1%} das malignas têm vinheta (borda escura "
        f"do dermatoscópio), contra {geral['benign']:.1%} das benignas. Por isso a vinheta é um "
        "atalho: um modelo pode acertar a classe olhando só para ela (cenários V1 a V5)."
    )

st.subheader("Exemplos")
if not summary["dataset_disponivel"]:
    st.info(
        "As imagens não estão nesta máquina (modo leitura). Baixe o dataset com "
        "`just data-download` para ver os exemplos e enviá-los ao Simulador."
    )
    st.stop()

f1, f2, f3, f4 = st.columns(4)
split = f1.selectbox("Divisão", ["todas", *SPLITS], format_func=lambda k: SPLITS.get(k, "Todas"))
label = f2.selectbox("Classe", ["todas", *LABELS], format_func=lambda k: LABELS.get(k, "Todas"))
VIGNETTE = {"todas": "Todas", "sim": "Com", "nao": "Sem"}
vinheta = f3.selectbox("Vinheta", list(VIGNETTE), format_func=VIGNETTE.get)
seed = f4.number_input("Sorteio", min_value=0, value=0, step=1, help="Muda a amostra exibida")

try:
    listing = client.dataset_images(
        split=None if split == "todas" else split,
        label=None if label == "todas" else label,
        vinheta=None if vinheta == "todas" else vinheta == "sim",
        limit=PAGE_SIZE,
        seed=int(seed),
    )
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

st.caption(f"{listing['total']} imagens com estes filtros; mostrando {len(listing['items'])}.")
cols = st.columns(6)
for i, item in enumerate(listing["items"]):
    with cols[i % 6]:
        try:
            data = client.dataset_image(item["path"])
        except httpx.HTTPError as exc:
            st.error(api_error_message(exc))
            continue
        name = item["path"].rsplit("/", 1)[-1]
        st.image(data, width="stretch")
        st.caption(
            f"{LABELS[item['label']]} · {SPLITS[item['split']]}"
            f"{' · vinheta' if item['vinheta'] else ''}"
        )
        if st.button("Simulador", key=f"base_sim_{item['path']}", icon=":material/compare:"):
            st.session_state["sim_base_image"] = {
                "name": name,
                "path": item["path"],
                "data": data,
            }
            st.session_state.pop("sim_result", None)
            st.switch_page("views/simulador.py")
