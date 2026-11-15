"""Cenário C8: classificador 100% PDI pela regra ABCD da dermatoscopia (TDS, `pdi.tds`).

1. Pipeline completo (A5) na base inteira, com imagens e máscaras em cache.
2. Medidas de B (borda por setor) e D (estruturas) em cada lesão; A e C vêm das
   características F já extraídas.
3. Cortes de cada medida = mediana nas lesões do **treino**, sem usar os rótulos.
4. TDS de cada imagem e decisão pelo corte da literatura (≥ 4,75 = encaminhar).

Nenhum parâmetro é ajustado com os rótulos: nem os cortes das medidas, nem os pesos, nem o
limiar. O "modelo" salvo (`results/C8/model.json`) são só esses cortes e constantes.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pdi import tds
from training import dataset, evaluation

_state: dict = {}


def _init(cache_dir: str) -> None:
    import cv2

    cv2.setNumThreads(1)
    _state["images"] = np.load(f"{cache_dir}/images.npy", mmap_mode="r")
    _state["masks"] = np.load(f"{cache_dir}/masks.npy", mmap_mode="r")


def _measure(i: int) -> dict | None:
    return tds.measures(np.asarray(_state["images"][i]), np.asarray(_state["masks"][i]))


def all_measures(cache: dataset.PipelineCache, progress) -> list[dict | None]:
    n = len(cache.features)
    out: list[dict | None] = []
    with ProcessPoolExecutor(initializer=_init, initargs=(str(cache.dir),)) as pool:
        for m in pool.map(_measure, range(n), chunksize=64):
            out.append(m)
            if len(out) % 1000 == 0:
                progress(len(out) / n, f"medidas TDS: {len(out)}/{n}")
    return out


def label_free_cutoffs(feats: pd.DataFrame, meas: list[dict | None], train: np.ndarray):
    """Medianas no treino (sem rótulos) de cada medida, só nas lesões segmentadas."""
    rows = [m for m, t in zip(meas, train, strict=True) if t and m is not None]
    f = feats[train]
    asym = pd.concat([f["assimetria_eixo_maior"], f["assimetria_eixo_menor"]]).dropna()
    return tds.Cutoffs(
        assimetria=float(asym.median()),
        borda_setor=float(np.median([v for m in rows for v in m["borda_setor"]])),
        **{s: float(np.median([m[s] for m in rows])) for s in tds.STRUCTURES},
    )


def score_all(feats: pd.DataFrame, meas: list[dict | None], cut: tds.Cutoffs) -> pd.DataFrame:
    rows = []
    for (_, f), m in zip(feats.iterrows(), meas, strict=True):
        if m is None:  # sem lesão segmentada: nenhum critério pontua
            rows.append({"A": 0, "B": 0, "C": 1, "D": 1, "tds": tds.TDS_MIN, "estruturas": ""})
            continue
        c = tds.criteria(f.to_dict(), m, cut)
        rows.append(
            {
                **{k: c[k] for k in "ABCD"},
                "tds": tds.score(c),
                "estruturas": ",".join(c["estruturas"]),
            }
        )
    return pd.DataFrame(rows)


def run(sc, out, progress, classification_block) -> dict:
    progress(0.02, "pipeline completo (A5) com imagens e máscaras")
    cache = dataset.pipeline_cache(
        sc.pipeline, with_images=True, progress=lambda f, m: progress(0.02 + 0.4 * f, m)
    )
    splits = dataset.load_splits()
    feats = cache.features.reset_index(drop=True)
    meas = all_measures(cache, lambda f, m: progress(0.45 + 0.4 * f, m))
    train = (splits["split"] == "train").to_numpy()
    cut = label_free_cutoffs(feats, meas, train)
    progress(0.88, "TDS de cada imagem")
    scored = score_all(feats, meas, cut)
    model = {
        "regra": "TDS (Stolz et al., 1994)",
        "pesos": tds.WEIGHTS,
        "corte_suspeita": tds.CUTOFF_SUSPICIOUS,
        "corte_melanoma": tds.CUTOFF_MELANOMA,
        "cortes_medidas": cut.to_dict(),
        "origem_cortes": "mediana de cada medida nas lesões do treino, sem usar os rótulos",
    }
    (out / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
    pred = pd.concat([splits[["path", "split", "y", "vinheta"]], scored], axis=1)
    pred = pred.assign(p=pred["tds"])
    pred = pred[pred["split"] != "train"]
    pred.to_csv(out / "predictions.csv", index=False)
    block = classification_block(pred, tds.CUTOFF_SUSPICIOUS)
    val = pred[pred["split"] == "val"]
    test = pred[pred["split"] == "test"]
    block["modelo"] = model | {
        "sem_lesao": float(np.mean([m is None for m in meas])),
        "media_criterios_teste": {
            lab: test[test["y"] == y][list("ABCD") + ["tds"]].mean().round(3).to_dict()
            for lab, y in (("benigna", 0), ("maligna", 1))
        },
        # comparação com o protocolo dos outros modelos (limiar escolhido na validação);
        # a decisão do C8 continua sendo o corte da literatura
        "limiar_validacao": evaluation.choose_threshold(val["y"], val["p"]),
    }
    t = block["modelo"]["limiar_validacao"]
    block["modelo"]["teste_com_limiar_validacao"] = evaluation.point_metrics(
        test["y"], test["p"], t
    )
    if not math.isfinite(block["teste"]["auc"]["valor"]):
        raise RuntimeError("AUC indefinida")
    return block
