"""Simulador, modo "base inteira": consolida `results/` em tabelas e figuras (seção 6.3).

- Tabela geral de todos os cenários executados (AUC e métricas no limiar, com IC 95%).
- Impacto por etapa (seção 4): ganho incremental (A_k − A_{k−1}) e perda na remoção
  (A5 − L_k), em AUC e sensibilidade, com IC 95% por bootstrap pareado no teste; uma etapa
  é "útil" só se as duas leituras forem positivas com IC que não contém zero.
- Técnicas alternativas: A5 − T_k (mesmo critério).
- Comparação de modelos (C), segmentação (S) e viés (V).
- Figuras para os slides (PNG), geradas a partir das mesmas tabelas.
"""

import io
import json

import numpy as np
import pandas as pd

from training import evaluation, status
from training.scenarios import SCENARIOS

STAGE_PAIRS = {  # etapa → (cenário com, cenário sem) nas duas leituras
    "P1": (("A1", "A0"), ("A5", "L1")),
    "P2": (("A2", "A1"), ("A5", "L2")),
    "P3": (("A3", "A2"), ("A5", "L3")),
    "P4": (("A4", "A3"), ("A5", "L4")),
    "P5": (("A5", "A4"), ("A5", "L5")),
}
TECHNIQUE_PAIRS = {"T1": "P2", "T2": "P3", "T3": "P3", "T4": "P4", "T5": "P5"}


def load_metrics() -> dict[str, dict]:
    out = {}
    for sid in SCENARIOS:
        path = status.scenario_dir(sid) / "metrics.json"
        if path.exists():
            out[sid] = json.loads(path.read_text())
    return out


def _predictions(sid: str) -> pd.DataFrame | None:
    path = status.scenario_dir(sid) / "predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["split"] == "test"].set_index("path") if "split" in df else None


def _metric(m: dict, block: str, key: str):
    v = m.get(block, {}).get(key)
    return (v["valor"], *v["ic95"]) if v else (None, None, None)


def overview(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for sid, sc in SCENARIOS.items():
        m = metrics.get(sid)
        st = status.read(sid) or {}
        row = {
            "id": sid,
            "grupo": sc.group,
            "descricao": sc.description,
            "estado": st.get("state", "nao_executado"),
        }
        if m:
            for block, prefix in (("teste", ""), ("teste_sem_vinheta", "sem_vinheta_")):
                for key in ("auc", "sensibilidade", "especificidade"):
                    v, lo, hi = _metric(m, block, key)
                    row[f"{prefix}{key}"] = v
                    row[f"{prefix}{key}_ic95"] = [lo, hi] if v is not None else None
            seg = m.get("segmentacao", {}).get("teste", {})
            row["dice_teste"] = seg.get("dice", {}).get("valor")
            row["ms_por_imagem"] = m.get("tempo", {}).get("ms_total")
            row["commit"] = m.get("commit")
        rows.append(row)
    return rows


def _paired(a: str, b: str, metrics: dict[str, dict]) -> dict | None:
    pa, pb = _predictions(a), _predictions(b)
    if pa is None or pb is None or a not in metrics or b not in metrics:
        return None
    common = pa.index.intersection(pb.index)
    y = pa.loc[common, "y"].to_numpy()
    out = {}
    for metric in ("auc", "sensibilidade"):
        out[metric] = evaluation.paired_difference(
            y,
            pa.loc[common, "p"].to_numpy(),
            pb.loc[common, "p"].to_numpy(),
            metric,
            threshold_a=metrics[a]["limiar"],
            threshold_b=metrics[b]["limiar"],
        )
    return out


def _significant_positive(d: dict | None) -> bool:
    return d is not None and d["auc"]["ic95"][0] > 0


def stage_impact(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for stage, ((a_with, a_without), (l_with, l_without)) in STAGE_PAIRS.items():
        inc = _paired(a_with, a_without, metrics)
        rem = _paired(l_with, l_without, metrics)
        both = inc is not None and rem is not None
        rows.append(
            {
                "etapa": stage,
                "incremental": {"cenarios": f"{a_with} − {a_without}", **(inc or {})},
                "remocao": {"cenarios": f"{l_with} − {l_without}", **(rem or {})},
                "completo": both,
                "util": both and _significant_positive(inc) and _significant_positive(rem),
            }
        )
    return rows


def technique_impact(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for tid, stage in TECHNIQUE_PAIRS.items():
        d = _paired("A5", tid, metrics)
        rows.append(
            {
                "cenario": tid,
                "etapa": stage,
                "descricao": SCENARIOS[tid].description,
                "diferenca": d,  # A5 (técnica padrão) − T (alternativa)
                "padrao_melhor": _significant_positive(d),
                "alternativa_melhor": d is not None and d["auc"]["ic95"][1] < 0,
            }
        )
    return rows


def model_comparison(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for sid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "V3"):
        m = metrics.get(sid)
        if not m:
            continue
        rows.append(
            {
                "id": sid,
                "descricao": SCENARIOS[sid].description,
                "auc_validacao": m["validacao"]["auc"],
                "teste": {k: m["teste"][k] for k in evaluation.METRICS},
                "matriz_confusao": m["teste"]["matriz_confusao"],
                "limiar": m["limiar"],
                "roc": m["roc_teste"],
            }
        )
    return rows


def triage_choice(metrics: dict[str, dict]) -> dict | None:
    """Modelo da Triagem: o cenário C com maior AUC na validação (seção 2)."""
    candidates = {sid: m["validacao"]["auc"] for sid, m in metrics.items() if sid.startswith("C")}
    if not candidates:
        return None
    best = max(candidates, key=candidates.get)
    return {"id": best, "auc_validacao": candidates[best], "candidatos": candidates}


def segmentation(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for sid in ("S1", "S2", "S3", "S4"):
        m = metrics.get(sid)
        if m:
            seg = m["segmentacao"]
            rows.append(
                {
                    "id": sid,
                    "descricao": SCENARIOS[sid].description,
                    **{
                        f"{c}_{k}": seg[c].get(k) for c in ("teste", "val") for k in ("dice", "iou")
                    },
                    "n_teste": seg["teste"]["n"],
                    "n_val": seg["val"]["n"],
                }
            )
    return rows


def build() -> dict:
    metrics = load_metrics()
    return {
        "cenarios": overview(metrics),
        "impacto_etapas": stage_impact(metrics),
        "tecnicas": technique_impact(metrics),
        "modelos": model_comparison(metrics),
        "triagem": triage_choice(metrics),
        "segmentacao": segmentation(metrics),
        "vies": {sid: metrics[sid] for sid in ("V1", "V2", "V4", "V5") if sid in metrics},
        "local": {sid: metrics[sid]["local"] for sid in metrics if "local" in metrics[sid]},
        "tempo": {sid: metrics[sid]["tempo"] for sid in metrics if "tempo" in metrics[sid]},
    }


# ---------------------------------------------------------------- figuras (PNG)

FIGURES = {
    "roc_modelos": "Curvas ROC dos cenários C (teste)",
    "confusao_modelos": "Matrizes de confusão dos cenários C no limiar fixado",
    "impacto_etapas": "Impacto por etapa (ΔAUC com IC 95%): incremental e remoção",
    "tecnicas": "Técnica padrão × alternativa (ΔAUC com IC 95%)",
    "tempo_etapas": "Tempo médio de processamento por etapa (A5)",
}


def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def figure(name: str, report: dict | None = None) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if name not in FIGURES:
        raise KeyError(name)
    report = report or build()
    if name == "roc_modelos":
        fig, ax = plt.subplots(figsize=(6, 6))
        for row in report["modelos"]:
            auc = row["teste"]["auc"]["valor"]
            ax.plot(row["roc"]["fpr"], row["roc"]["tpr"], label=f"{row['id']} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set(xlabel="1 − especificidade", ylabel="Sensibilidade", title=FIGURES[name])
        ax.legend(loc="lower right")
    elif name == "confusao_modelos":
        rows = report["modelos"] or [{"id": "—", "matriz_confusao": [[0, 0], [0, 0]]}]
        fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 3), squeeze=False)
        for ax, row in zip(axes[0], rows, strict=True):
            cm = np.asarray(row["matriz_confusao"])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set(title=row["id"], xticks=[0, 1], yticks=[0, 1])
            ax.set_xticklabels(["benigna", "maligna"])
            ax.set_yticklabels(["benigna", "maligna"])
            ax.set_xlabel("predito")
        axes[0][0].set_ylabel("real")
    elif name in ("impacto_etapas", "tecnicas"):
        fig, ax = plt.subplots(figsize=(7, 4))
        if name == "impacto_etapas":
            labels, series = (
                [r["etapa"] for r in report["impacto_etapas"]],
                {
                    "incremental (A)": [r["incremental"] for r in report["impacto_etapas"]],
                    "remoção (L)": [r["remocao"] for r in report["impacto_etapas"]],
                },
            )
        else:
            labels = [r["cenario"] for r in report["tecnicas"]]
            series = {"A5 − T": [r["diferenca"] or {} for r in report["tecnicas"]]}
        width = 0.8 / len(series)
        x = np.arange(len(labels))
        for k, (label, values) in enumerate(series.items()):
            vals = [v.get("auc", {}).get("diferenca", np.nan) for v in values]
            lo = [v.get("auc", {}).get("ic95", [np.nan] * 2)[0] for v in values]
            hi = [v.get("auc", {}).get("ic95", [np.nan] * 2)[1] for v in values]
            err = [np.subtract(vals, lo), np.subtract(hi, vals)]
            ax.bar(x + k * width - 0.4 + width / 2, vals, width, yerr=err, capsize=3, label=label)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xticks=x, ylabel="ΔAUC (teste)", title=FIGURES[name])
        ax.set_xticklabels(labels)
        ax.legend()
    else:  # tempo_etapas
        fig, ax = plt.subplots(figsize=(6, 3.5))
        t = report["tempo"].get("A5", {}).get("ms_por_etapa", {})
        ax.bar(list(t), list(t.values()))
        ax.set(ylabel="ms por imagem", title=FIGURES[name])
    return _png(fig)
