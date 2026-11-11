"""Página Simulador: compara combinações de técnicas sobre a mesma imagem.

Plano experimental, seção 6.2. Cada combinação escolhe a técnica de cada etapa (ou a
desliga), partindo de um cenário congelado como preset ou montada livremente.
"""

import httpx
import pandas as pd
import streamlit as st

from web.common import api_error_message, b64_image, features_table, get_client

# presets sugeridos para as colunas: sem × com pré-processamento, técnica X × Y
DEFAULT_PRESETS = ("A0", "A5", "T4", "L1")
CUSTOM = "custom"
# o selectbox do Streamlit trata None como "nada selecionado"; na interface, etapa
# desligada é OFF, convertido para None só ao montar a configuração
OFF = "off"

client = get_client()

st.title("Simulador")
st.caption(
    "Compare a mesma imagem processada por diferentes combinações de técnicas: com ou sem "
    "pré-processamento, técnica X ou Y em cada etapa. Os presets são os cenários congelados "
    "do plano experimental."
)


@st.cache_data(ttl=300, show_spinner=False)
def load_options() -> dict:
    return client.pipeline_options()


try:
    options = load_options()
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

stages = options["stages"]
presets = {p["id"]: p for p in options["presets"]}
seg_labels = {s["key"]: s["label"] for s in options["segmentation"]}
technique_labels = {s["id"]: {t["key"]: t["label"] for t in s["techniques"]} for s in stages}


# ---------------------------------------------------------------- estado das combinações


def _key(i: int, name: str) -> str:
    return f"sim_c{i}_{name}"


def _apply_config(i: int, config: dict, label: str) -> None:
    for stage in stages:
        st.session_state[_key(i, stage["id"])] = config[stage["field"]] or OFF
    st.session_state[_key(i, "seg")] = config["segmentation"] or "otsu"
    st.session_state[_key(i, "post")] = config["seg_postprocess"]
    st.session_state[_key(i, "label")] = label


def _on_preset(i: int) -> None:
    pid = st.session_state[_key(i, "preset")]
    if pid != CUSTOM:
        p = presets[pid]
        _apply_config(i, p["config"], f"{pid} · {p['description']}")


def _on_manual_change(i: int) -> None:
    st.session_state[_key(i, "preset")] = CUSTOM


def _init(i: int) -> None:
    if _key(i, "preset") in st.session_state:
        return
    pid = DEFAULT_PRESETS[i % len(DEFAULT_PRESETS)]
    st.session_state[_key(i, "preset")] = pid
    _apply_config(i, presets[pid]["config"], f"{pid} · {presets[pid]['description']}")


def _config_of(i: int) -> dict:
    config = {}
    for stage in stages:
        value = st.session_state[_key(i, stage["id"])]
        config[stage["field"]] = None if value == OFF else value
    config["segmentation"] = st.session_state[_key(i, "seg")]
    config["seg_postprocess"] = st.session_state[_key(i, "post")]
    return config


def _preset_label(pid: str) -> str:
    return "Personalizada" if pid == CUSTOM else f"{pid} · {presets[pid]['description']}"


# ---------------------------------------------------------------- entrada

uploaded = st.file_uploader("Imagem dermatoscópica", type=["jpg", "jpeg", "png"])

n = st.segmented_control(
    "Número de combinações",
    options=list(range(1, options["max_combinations"] + 1)),
    default=2,
    key="sim_n",
)
n = n or 2

cols = st.columns(n)
for i, col in enumerate(cols):
    _init(i)
    with col, st.container(border=True):
        st.selectbox(
            "Preset (cenário)",
            [CUSTOM, *presets],
            key=_key(i, "preset"),
            format_func=_preset_label,
            on_change=_on_preset,
            args=(i,),
        )
        st.text_input("Nome", key=_key(i, "label"))
        for stage in stages:
            labels = technique_labels[stage["id"]]
            st.selectbox(
                f"{stage['id']} · {stage['name']}",
                [OFF, *labels],
                key=_key(i, stage["id"]),
                format_func=lambda k, labels=labels: "Desligada" if k == OFF else labels[k],
                on_change=_on_manual_change,
                args=(i,),
            )
        st.selectbox(
            "S · Segmentação",
            list(seg_labels),
            key=_key(i, "seg"),
            format_func=seg_labels.get,
            on_change=_on_manual_change,
            args=(i,),
        )
        st.checkbox(
            "Pós-processamento (morfologia + componente central)",
            key=_key(i, "post"),
            on_change=_on_manual_change,
            args=(i,),
        )

if uploaded is None:
    st.info("Envie uma imagem para comparar as combinações.")
    st.stop()

if st.button("Comparar", type="primary", icon=":material/play_arrow:"):
    combos = [
        {
            "label": st.session_state[_key(i, "label")] or f"Combinação {i + 1}",
            "config": _config_of(i),
        }
        for i in range(n)
    ]
    with st.spinner("Processando..."):
        try:
            st.session_state["sim_result"] = {
                "file": uploaded.file_id,
                "data": client.compare(uploaded.getvalue(), uploaded.name, uploaded.type, combos),
            }
        except httpx.HTTPError as exc:
            st.error(api_error_message(exc))
            st.stop()

saved = st.session_state.get("sim_result")
if saved is None or saved["file"] != uploaded.file_id:
    st.stop()

result = saved["data"]
combos = result["combinations"]

# ---------------------------------------------------------------- resultados

tab_final, tab_steps, tab_dice, tab_feats, tab_fig = st.tabs(
    [
        "Resultado final",
        "Etapa a etapa",
        "Concordância das máscaras",
        "Características",
        "Figura para slides",
    ]
)

with tab_final:
    for col, combo in zip(st.columns(len(combos)), combos, strict=True):
        r = combo["result"]
        with col:
            st.markdown(f"**{combo['label']}**")
            st.image(b64_image(r["overlay"] or r["final"]), width="content")
            area = r["lesion_area_fraction"]
            st.metric("Área da lesão", "—" if area is None else f"{area:.1%}")
            st.metric("Tempo de processamento", f"{r['total_seconds'] * 1000:.0f} ms")
            st.caption("Vinheta detectada" if r["vignette_detected"] else "Sem vinheta detectada")
            h, w = r["original_shape"][:2]
            if (h, w) != (300, 300):
                st.caption(f"Entrada {w} × {h} normalizada para 300 × 300 (P0)")

with tab_steps:
    # uma linha por combinação, uma coluna por etapa (mesmo formato da figura para slides)
    show_masks = st.toggle("Mostrar as máscaras produzidas pelas etapas (vinheta, pelos)")
    headers = ["Entrada", *(f"{s['id']} · {s['name']}" for s in stages), "S · Segmentação"]
    for col, header in zip(st.columns(len(headers)), headers, strict=True):
        col.markdown(f"**{header}**")
    for combo in combos:
        r = combo["result"]
        st.markdown(f"##### {combo['label']}")
        cols = st.columns(len(headers))
        cols[0].image(b64_image(r["input"]), width="stretch")
        for col, out in zip(cols[1:-1], r["stages"], strict=True):
            if out["technique"]:
                caption = f"{out['technique_label']} · {out['seconds'] * 1000:.0f} ms"
            else:
                caption = "Desligada"
            image = out["mask"] if show_masks and out["mask"] else out["image"]
            col.image(b64_image(image), caption=caption, width="stretch")
        seg = r["config"]["segmentation"]
        cols[-1].image(
            b64_image(r["overlay"] or r["final"]),
            caption=seg_labels.get(seg, "Desligada"),
            width="stretch",
        )

with tab_dice:
    st.write(
        "Índice de Dice entre as máscaras de cada par de combinações: 1 = máscaras idênticas, "
        "0 = nenhuma sobreposição. Mostra o quanto a escolha das técnicas muda a região "
        "segmentada, da qual dependem os indicadores ABCD."
    )
    labels = [c["label"] for c in combos]
    st.dataframe(
        pd.DataFrame(result["mask_dice"], index=labels, columns=labels).style.format(
            "{:.2f}", na_rep="—"
        ),
        width="stretch",
    )

with tab_feats:
    st.write(
        "Características extraídas dentro da máscara da lesão, sobre a imagem pré-processada "
        "(lista fechada da seção 3.1 do plano): forma e borda (critérios A, B e D), cor (C) e "
        "textura (GLCM e LBP). Comparar as colunas mostra o quanto cada combinação de técnicas "
        "altera o que o classificador vê. Valores vazios: nenhuma lesão segmentada."
    )
    table = features_table(combos, options["feature_groups"])
    group = st.segmented_control(
        "Grupo", ["Todos", *options["feature_groups"]], default="Todos", key="sim_feat_group"
    )
    if group and group != "Todos":
        table = table.xs(group, level="Grupo", drop_level=False)
    st.dataframe(table.style.format("{:.3f}", na_rep="—"), width="stretch")

with tab_fig:
    figure = b64_image(result["figure"])
    st.image(figure, width="stretch")
    st.download_button(
        "Baixar figura (PNG)",
        figure,
        file_name=f"simulador_{uploaded.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
        icon=":material/download:",
    )
