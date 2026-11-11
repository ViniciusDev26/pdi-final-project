"""Página Experimentos: cenários congelados, execução pela fila da API e progresso.

Plano experimental, seção 7.1: nenhum cenário depende do terminal. A API executa um
cenário por vez num processo separado; esta página consulta o estado periodicamente.
"""

import httpx
import pandas as pd
import streamlit as st

from web.common import api_error_message, get_client

REFRESH_SECONDS = 3
STATE_LABELS = {
    "nao_executado": "⚪ não executado",
    "na_fila": "🕓 na fila",
    "executando": "🔵 executando",
    "concluido": "🟢 concluído",
    "erro": "🔴 erro",
    "interrompido": "🟠 interrompido",
}
GROUP_LABELS = {
    "A": "A · acumulativos (seção 5.1)",
    "L": "L · remoção de etapa (5.2)",
    "T": "T · técnica alternativa (6.3)",
    "S": "S · segmentação (5.5)",
    "C": "C · comparação de modelos (5.3)",
    "V": "V · viés e artefatos (5.4)",
}

client = get_client()

st.title("Experimentos")
st.caption(
    "Cenários congelados do plano experimental. Cada execução treina e avalia o cenário no "
    "servidor e grava as métricas com IC 95%, o modelo e o log. Os cenários rodam um por "
    "vez, na ordem da fila; os de CNN usam a GPU."
)


@st.fragment(run_every=REFRESH_SECONDS)
def board() -> None:
    try:
        scenarios = client.scenarios()
        jobs = client.jobs()
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
        return

    running = jobs["executando"]
    c1, c2, c3 = st.columns(3)
    done = sum(s["status"]["state"] == "concluido" for s in scenarios)
    c1.metric("Concluídos", f"{done} de {len(scenarios)}")
    c2.metric("Executando", running or "—")
    c3.metric("Na fila", len(jobs["fila"]))

    if running:
        st_run = next(s for s in scenarios if s["id"] == running)["status"]
        st.progress(
            float(st_run.get("progress") or 0), text=f"{running}: {st_run.get('message', '')}"
        )

    rows = []
    for s in scenarios:
        stt = s["status"]
        rows.append(
            {
                "ID": s["id"],
                "Grupo": s["group"],
                "Descrição": s["description"],
                "Estado": STATE_LABELS.get(stt["state"], stt["state"]),
                "Progresso": float(stt.get("progress") or 0),
                "Mensagem": stt.get("message", ""),
                "Tempo estimado": s["estimate"],
                "GPU": "sim" if s["uses_gpu"] else "",
                "Depende de": ", ".join(s["depends"]),
            }
        )
    df = pd.DataFrame(rows)
    group = st.segmented_control(
        "Grupo",
        ["Todos", *GROUP_LABELS],
        default="Todos",
        key="exp_group",
        format_func=lambda g: g if g == "Todos" else GROUP_LABELS[g],
    )
    if group and group != "Todos":
        df = df[df["Grupo"] == group]
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Progresso": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="percent"
            ),
        },
    )


board()

st.subheader("Executar")
try:
    scenario_list = client.scenarios()
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

ids = [s["id"] for s in scenario_list]
by_id = {s["id"]: s for s in scenario_list}
selected = st.multiselect(
    "Cenários",
    ids,
    key="exp_selected",
    format_func=lambda i: f"{i} · {by_id[i]['description']}",
)
if selected:
    st.caption(
        "Tempo estimado: "
        + "; ".join(f"{i}: {by_id[i]['estimate']}" for i in selected)
        + ". Cenários que dependem de outros (V2, V4, V5) usam os modelos já treinados."
    )
c1, c2, c3 = st.columns(3)
if c1.button(
    "Executar selecionados",
    type="primary",
    disabled=not selected,
    width="stretch",
    icon=":material/play_arrow:",
):
    try:
        res = client.enqueue(selected)
        st.toast(f"Adicionados à fila: {', '.join(res['adicionados']) or 'nenhum (já na fila)'}")
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
if c2.button("Executar todos", width="stretch", icon=":material/playlist_play:"):
    try:
        res = client.enqueue(["todos"])
        st.toast(f"{len(res['adicionados'])} cenários adicionados à fila")
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
cancel_id = c3.selectbox("Cancelar", ["—", *ids], key="exp_cancel", label_visibility="collapsed")
if cancel_id != "—" and c3.button(f"Cancelar {cancel_id}", width="stretch"):
    try:
        client.cancel(cancel_id)
        st.toast(f"{cancel_id} cancelado")
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))

st.subheader("Log")
log_id = st.selectbox(
    "Cenário", ids, key="exp_log", format_func=lambda i: f"{i} · {by_id[i]['description']}"
)


@st.fragment(run_every=REFRESH_SECONDS)
def show_log() -> None:
    try:
        text = client.scenario_log(log_id)
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
        return
    st.code(text or "(sem log)", language=None, height=300)
    err = by_id[log_id]["status"].get("erro")
    if err:
        with st.expander("Erro"):
            st.code(err, language=None)


show_log()
