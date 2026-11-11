"""Página Resultados: simulador no modo "base inteira" (plano experimental, seções 5.6 e 6.3).

Consolida os cenários executados: tabela comparativa com IC 95% (com e sem vinheta),
impacto por etapa (A/L), técnicas alternativas (T), modelos (C), segmentação (S), viés (V),
métricas locais, tempo por etapa e figuras exportáveis para os slides.
"""

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from web.common import api_error_message, get_client

client = get_client()

st.title("Resultados")
st.caption(
    "Métricas no conjunto de teste, com intervalo de confiança de 95% por bootstrap (1.000 "
    "reamostragens). AUC independe do limiar; sensibilidade e especificidade usam o limiar "
    "escolhido na validação (sensibilidade ≥ 0,90). Diferenças entre etapas e técnicas são "
    "exploratórias (seção 4 do plano)."
)

try:
    report = client.report()
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()


def fmt_ci(value, ci) -> str:
    if value is None:
        return "—"
    if not ci or ci[0] is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]"


def fmt_diff(d: dict | None, metric: str = "auc") -> str:
    if not d or metric not in d:
        return "—"
    v = d[metric]
    return f"{v['diferenca']:+.3f} [{v['ic95'][0]:+.3f}; {v['ic95'][1]:+.3f}]"


def download(df: pd.DataFrame, name: str) -> None:
    st.download_button(
        "Baixar tabela (CSV)",
        df.to_csv(index=False).encode(),
        file_name=f"{name}.csv",
        mime="text/csv",
        key=f"dl_{name}",
        icon=":material/download:",
    )


executed = [r for r in report["cenarios"] if r.get("auc") is not None or r.get("dice_teste")]
if not executed:
    st.info("Nenhum cenário executado ainda. Execute os cenários na página Experimentos.")

tabs = st.tabs(
    [
        "Visão geral",
        "Impacto por etapa",
        "Técnicas",
        "Modelos",
        "Segmentação",
        "Viés",
        "Métricas locais",
        "Figuras",
    ]
)

# ---------------------------------------------------------------- visão geral
with tabs[0]:
    subset = st.radio(
        "Subconjunto do teste", ["completo", "sem vinheta"], horizontal=True, key="res_subset"
    )
    prefix = "" if subset == "completo" else "sem_vinheta_"
    rows = []
    for r in report["cenarios"]:
        rows.append(
            {
                "ID": r["id"],
                "Descrição": r["descricao"],
                "Estado": r["estado"],
                "AUC": fmt_ci(r.get(f"{prefix}auc"), r.get(f"{prefix}auc_ic95")),
                "Sensibilidade": fmt_ci(
                    r.get(f"{prefix}sensibilidade"), r.get(f"{prefix}sensibilidade_ic95")
                ),
                "Especificidade": fmt_ci(
                    r.get(f"{prefix}especificidade"), r.get(f"{prefix}especificidade_ic95")
                ),
                "Dice (teste)": "—" if r.get("dice_teste") is None else f"{r['dice_teste']:.3f}",
                "ms/imagem": r.get("ms_por_imagem"),
                "Commit": r.get("commit"),
            }
        )
    overview = pd.DataFrame(rows)
    only_done = st.toggle("Só cenários executados", value=True, key="res_only_done")
    if only_done:
        overview = overview[overview["AUC"] != "—"]
    st.dataframe(overview, hide_index=True, width="stretch")
    download(overview, f"cenarios_{prefix or 'completo'}")
    if report["triagem"]:
        t = report["triagem"]
        st.success(
            f"Modelo da Triagem pela regra da seção 2 (maior AUC na validação entre os "
            f"cenários C): **{t['id']}** (AUC validação {t['auc_validacao']:.3f})."
        )

# ---------------------------------------------------------------- impacto por etapa
with tabs[1]:
    st.write(
        "Impacto de cada etapa no Random Forest de referência: **incremental** (A_k − A_{k−1}) "
        "e **remoção** (A5 − L_k), com IC 95% por bootstrap pareado no teste. Uma etapa é "
        "**útil** só se as duas leituras forem positivas e os IC não contiverem zero."
    )
    rows = [
        {
            "Etapa": r["etapa"],
            "ΔAUC incremental": fmt_diff(r["incremental"]),
            "ΔAUC remoção": fmt_diff(r["remocao"]),
            "ΔSens. incremental": fmt_diff(r["incremental"], "sensibilidade"),
            "ΔSens. remoção": fmt_diff(r["remocao"], "sensibilidade"),
            "Conclusão": (
                "útil"
                if r["util"]
                else ("sem efeito mensurável" if r["completo"] else "cenários pendentes")
            ),
        }
        for r in report["impacto_etapas"]
    ]
    impact = pd.DataFrame(rows)
    st.dataframe(impact, hide_index=True, width="stretch")
    download(impact, "impacto_etapas")
    chart_rows = []
    for r in report["impacto_etapas"]:
        for reading, key in (("incremental (A)", "incremental"), ("remoção (L)", "remocao")):
            d = r[key].get("auc")
            if d:
                chart_rows.append(
                    {
                        "etapa": r["etapa"],
                        "leitura": reading,
                        "ΔAUC": d["diferenca"],
                        "inf": d["ic95"][0],
                        "sup": d["ic95"][1],
                    }
                )
    if chart_rows:
        cd = pd.DataFrame(chart_rows)
        base = alt.Chart(cd).encode(x=alt.X("etapa:N"), xOffset="leitura:N")
        st.altair_chart(
            base.mark_bar().encode(y="ΔAUC:Q", color="leitura:N")
            + base.mark_errorbar().encode(y="inf:Q", y2="sup:Q"),
            use_container_width=True,
        )

# ---------------------------------------------------------------- técnicas
with tabs[2]:
    st.write(
        "Cada cenário T troca a técnica de uma única etapa em relação ao A5. Diferença = "
        "A5 (técnica padrão) − T (alternativa): positiva favorece a técnica padrão."
    )
    tech = pd.DataFrame(
        [
            {
                "Cenário": r["cenario"],
                "Etapa": r["etapa"],
                "Troca": r["descricao"],
                "ΔAUC (A5 − T)": fmt_diff(r["diferenca"]),
                "ΔSens. (A5 − T)": fmt_diff(r["diferenca"], "sensibilidade"),
                "Conclusão": (
                    "padrão melhor"
                    if r["padrao_melhor"]
                    else "alternativa melhor"
                    if r["alternativa_melhor"]
                    else ("sem diferença mensurável" if r["diferenca"] else "pendente")
                ),
            }
            for r in report["tecnicas"]
        ]
    )
    st.dataframe(tech, hide_index=True, width="stretch")
    download(tech, "tecnicas")

# ---------------------------------------------------------------- modelos
with tabs[3]:
    models = report["modelos"]
    if not models:
        st.info("Nenhum cenário C executado ainda.")
    else:
        table = pd.DataFrame(
            [
                {
                    "ID": m["id"],
                    "Descrição": m["descricao"],
                    "AUC validação": round(m["auc_validacao"], 3),
                    **{
                        k.capitalize(): fmt_ci(m["teste"][k]["valor"], m["teste"][k]["ic95"])
                        for k in ("auc", "sensibilidade", "especificidade", "f1")
                    },
                    "Limiar": round(m["limiar"], 3),
                }
                for m in models
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        download(table, "modelos")
        roc = pd.DataFrame(
            [
                {"modelo": m["id"], "1 − especificidade": x, "sensibilidade": y}
                for m in models
                for x, y in zip(m["roc"]["fpr"], m["roc"]["tpr"], strict=True)
            ]
        )
        st.altair_chart(
            alt.Chart(roc)
            .mark_line()
            .encode(x="1 − especificidade:Q", y="sensibilidade:Q", color="modelo:N")
            .properties(height=420, title="Curvas ROC (teste)"),
            use_container_width=True,
        )
        cols = st.columns(min(len(models), 4))
        for i, m in enumerate(models):
            (tn, fp), (fn, tp) = m["matriz_confusao"]
            cols[i % len(cols)].dataframe(
                pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["real benigna", "real maligna"],
                    columns=["pred. benigna", "pred. maligna"],
                ),
                width="stretch",
            )
            cols[i % len(cols)].caption(f"{m['id']}: matriz de confusão no limiar")

# ---------------------------------------------------------------- segmentação
with tabs[4]:
    seg = report["segmentacao"]
    if not seg:
        st.info(
            "Cenários S ainda não executados. Eles precisam das máscaras anotadas do teste "
            "(página Anotação)."
        )
    else:

        def ci(s: dict, key: str) -> str:
            v = s[key] or {}
            return fmt_ci(v.get("valor"), v.get("ic95"))

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": s["id"],
                        "Descrição": s["descricao"],
                        "Dice teste": ci(s, "teste_dice"),
                        "IoU teste": ci(s, "teste_iou"),
                        "n teste": s["n_teste"],
                        "Dice validação": ci(s, "val_dice"),
                    }
                    for s in seg
                ]
            ),
            hide_index=True,
            width="stretch",
        )

# ---------------------------------------------------------------- viés
with tabs[5]:
    vies = report["vies"]
    if "V1" in vies:
        v1 = vies["V1"]
        st.markdown("**V1 · atalho da vinheta**")
        st.write(
            f"Um classificador que só sabe se a imagem tem vinheta chega a AUC "
            f"{fmt_ci(v1['teste']['auc']['valor'], v1['teste']['auc']['ic95'])} no teste."
        )
        st.dataframe(pd.DataFrame(v1["prevalencia_vinheta"]).T, width="stretch")
    if "V2" in vies:
        st.markdown("**V2 · desempenho com e sem vinheta**")
        rows = []
        for sid, d in vies["V2"]["modelos"].items():
            if d.get("pendente"):
                continue
            for sub, key in (("com vinheta", "com_vinheta"), ("sem vinheta", "sem_vinheta")):
                rows.append(
                    {
                        "Modelo": sid,
                        "Subconjunto": sub,
                        "n": d[key]["n"],
                        "AUC": fmt_ci(d[key]["auc"]["valor"], d[key]["auc"]["ic95"]),
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if "V4" in vies:
        st.markdown("**V4 · onde a CNN olha (Grad-CAM)**")
        st.json(vies["V4"].get("modelos", {}), expanded=False)
    if "V5" in vies:
        st.markdown("**V5 · vinheta artificial em benignas**")
        rows = [
            {
                "Modelo": sid,
                "P1 ligada": d.get("p1_ligada"),
                "Variação média da probabilidade": fmt_ci(d.get("variacao_media"), d.get("ic95")),
                "Viraram malignas": d.get("viraram_malignas"),
            }
            for sid, d in vies["V5"]["modelos"].items()
            if not d.get("pendente")
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not vies:
        st.info("Nenhum cenário V executado ainda.")

# ---------------------------------------------------------------- métricas locais
with tabs[6]:
    local = report["local"]
    if not local:
        st.info("Execute cenários A, L ou T para ver as métricas locais das etapas.")
    else:
        rows = []
        for sid, m in local.items():
            row = {"ID": sid}
            for stage in ("P1", "P2", "P3", "P4", "P5"):
                for k, v in m.get(stage, {}).items():
                    if isinstance(v, int | float):
                        row[f"{stage} {k}"] = round(v, 4)
            rows.append(row)
        local_df = pd.DataFrame(rows)
        st.dataframe(local_df, hide_index=True, width="stretch")
        download(local_df, "metricas_locais")
        st.caption(next(iter(local.values())).get("amostra", ""))
    tempo = report["tempo"]
    if tempo:
        st.markdown("**Tempo médio por etapa (ms por imagem)**")
        st.dataframe(
            pd.DataFrame({sid: t["ms_por_etapa"] for sid, t in tempo.items()}).T.round(2),
            width="stretch",
        )

# ---------------------------------------------------------------- figuras
with tabs[7]:
    try:
        figures = client.figures()
    except httpx.HTTPError as exc:
        st.error(api_error_message(exc))
        figures = {}
    name = st.selectbox("Figura", list(figures), format_func=figures.get, key="res_fig")
    if name:
        png = client.figure(name)
        st.image(png, width="stretch")
        st.download_button(
            "Baixar figura (PNG)",
            png,
            file_name=f"{name}.png",
            mime="image/png",
            icon=":material/download:",
        )
