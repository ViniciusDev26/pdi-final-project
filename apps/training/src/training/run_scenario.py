"""Executa um cenário congelado e grava `results/<ID>/` (plano experimental, seção 7).

Saídas: `metrics.json` (métricas com IC 95%, configuração, commit e ambiente),
`predictions.csv` (probabilidades na validação e no teste), o modelo treinado e
`status.json` com o progresso. A API chama este módulo num processo separado
(`python -m training.run_scenario <ID>`); o atalho `just experimento <ID>` também.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd

from pdi import PipelineConfig, load_rgb, preprocess, run
from pdi.metrics import dice, iou
from training import annotations, dataset, evaluation, local_metrics, status, traditional
from training import config as cfg
from training.scenarios import SCENARIOS, Scenario

Progress = Callable[[float, str], None]


class MissingDependency(RuntimeError):
    """Um cenário do qual este depende ainda não foi executado."""


# ---------------------------------------------------------------- utilidades


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cfg.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cfg.REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-modificado" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        import os

        return os.environ.get("PDI_GIT_COMMIT", "desconhecido")


def environment() -> dict:
    import cv2
    import sklearn

    env = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            env["driver"] = smi.stdout.strip() or None
    except ImportError:
        pass
    return env


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def no_vignette_paths() -> set[str]:
    return set(pd.read_csv(cfg.EXPERIMENTS_DIR / "teste_sem_vinheta.csv")["path"])


def read_predictions(sid: str) -> pd.DataFrame:
    path = status.scenario_dir(sid) / "predictions.csv"
    if not path.exists():
        raise MissingDependency(f"execute antes o cenário {sid} (predições não encontradas)")
    return pd.read_csv(path)


def read_metrics(sid: str) -> dict:
    path = status.scenario_dir(sid) / "metrics.json"
    if not path.exists():
        raise MissingDependency(f"execute antes o cenário {sid}")
    return json.loads(path.read_text())


def classification_block(pred: pd.DataFrame, threshold: float) -> dict:
    """Métricas de validação e teste (completo e sem vinheta) a partir das predições."""
    val, test = pred[pred["split"] == "val"], pred[pred["split"] == "test"]
    clean = test[test["path"].isin(no_vignette_paths())]
    return {
        "limiar": threshold,
        "validacao": evaluation.point_metrics(val["y"], val["p"], threshold),
        "teste": evaluation.evaluate(test["y"], test["p"], threshold),
        "teste_sem_vinheta": evaluation.evaluate(clean["y"], clean["p"], threshold),
        "roc_teste": evaluation.roc_points(test["y"], test["p"]),
    }


def segmentation_block(config: PipelineConfig) -> dict:
    """Dice/IoU da segmentação nas máscaras de referência (validação = ajuste; teste)."""
    out = {}
    for conjunto in annotations.SETS:
        refs = annotations.reference_masks(conjunto, cfg.EXPERIMENTS_DIR, cfg.MASKS_DIR)
        rows = []
        for path, ref in refs.items():
            res = run(load_rgb(cfg.DATASET_DIR / path), config)
            pred = res.lesion_mask if res.lesion_mask is not None else np.zeros_like(ref)
            rows.append({"path": path, "dice": dice(pred, ref), "iou": iou(pred, ref)})
        df = pd.DataFrame(rows, columns=["path", "dice", "iou"])
        block: dict = {"n": len(df)}
        for m in ("dice", "iou"):
            if len(df):
                boots = [df[m].to_numpy()[i].mean() for i in evaluation.bootstrap_indices(len(df))]
                block[m] = {
                    "valor": float(df[m].mean()),
                    "ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                }
        block["por_imagem"] = df.to_dict("records")
        out[conjunto] = block
    return out


def timing_block(features: pd.DataFrame) -> dict:
    cols = [c for c in features.columns if c.startswith("t_")]
    per_stage = {c.removeprefix("t_"): float(features[c].mean() * 1000) for c in cols}
    return {"ms_por_etapa": per_stage, "ms_total": float(sum(per_stage.values()))}


# ---------------------------------------------------------------- tipos de cenário


def _fit_traditional(
    sc: Scenario, cache: dataset.PipelineCache, out, progress: Progress
) -> tuple[dict, pd.DataFrame]:
    splits = dataset.load_splits()
    X = dataset.feature_matrix(cache, sc.feature_groups).to_numpy()
    train = (splits["split"] == "train").to_numpy()
    progress(0.4, f"busca em grade 5-fold no treino ({sc.model}, {X.shape[1]} características)")
    t0 = time.perf_counter()
    fit = traditional.fit(sc.model, X[train], splits["y"].to_numpy()[train])
    progress(0.8, f"modelo ajustado em {time.perf_counter() - t0:.0f} s; avaliando")
    joblib.dump(fit.model, out / "model.joblib")
    pred = splits[["path", "split", "y", "vinheta"]].assign(
        p=traditional.predict_proba(fit.model, X)
    )
    pred = pred[pred["split"] != "train"]
    val = pred[pred["split"] == "val"]
    threshold = evaluation.choose_threshold(val["y"], val["p"])
    block = classification_block(pred, threshold)
    block["modelo"] = {
        "tipo": sc.model,
        "melhores_parametros": fit.best_params,
        "auc_cv_treino": {"media": fit.cv_auc, "dp": fit.cv_auc_std},
        "caracteristicas": X.shape[1],
    }
    if sc.model == "rf":
        names = dataset.feature_matrix(cache, sc.feature_groups).columns
        imp = fit.model.named_steps["clf"].feature_importances_
        block["modelo"]["importancia"] = dict(
            sorted(zip(names, imp.round(4).tolist(), strict=True), key=lambda kv: -kv[1])
        )
    return block, pred


def run_ablation(sc: Scenario, out, progress: Progress) -> dict:
    progress(0.02, "pipeline + características na base inteira")
    cache = dataset.pipeline_cache(sc.pipeline, progress=lambda f, m: progress(0.02 + 0.3 * f, m))
    block, pred = _fit_traditional(sc, cache, out, progress)
    pred.to_csv(out / "predictions.csv", index=False)
    progress(0.85, "métricas locais das etapas e segmentação")
    splits = dataset.load_splits()
    feats = cache.features.assign(label=splits["label"])
    local = local_metrics.compute(sc.pipeline, local_metrics.sample_paths(splits), cache.features)
    local["P1"]["taxa_deteccao_por_classe"] = (
        feats.groupby("label")["vinheta_p1"].mean().round(4).to_dict()
    )
    seg_fail = float((cache.features["area_lesao"] == 0).mean())
    return block | {
        "segmentacao": segmentation_block(sc.pipeline) | {"fracao_sem_lesao": seg_fail},
        "local": local,
        "tempo": timing_block(cache.features),
    }


def run_traditional(sc: Scenario, out, progress: Progress) -> dict:
    progress(0.02, "pipeline + características (A5) na base inteira")
    cache = dataset.pipeline_cache(sc.pipeline, progress=lambda f, m: progress(0.02 + 0.3 * f, m))
    block, pred = _fit_traditional(sc, cache, out, progress)
    pred.to_csv(out / "predictions.csv", index=False)
    return block | {"tempo": timing_block(cache.features)}


def run_segmentation(sc: Scenario, out, progress: Progress) -> dict:
    progress(0.1, "segmentando as imagens anotadas")
    seg = segmentation_block(sc.pipeline)
    if seg["teste"]["n"] == 0:
        raise MissingDependency(
            "nenhuma máscara de referência do teste anotada ainda (página Anotação)"
        )
    return {"segmentacao": seg}


def run_vignette_shortcut(sc: Scenario, out, progress: Progress) -> dict:
    """V1: o indicador de vinheta (P1 na imagem bruta) como único preditor."""
    from sklearn.linear_model import LogisticRegression

    splits = dataset.load_splits()
    if splits["vinheta"].isna().any():
        raise MissingDependency("experiments/vinheta.csv ausente (training.experiment_lists)")
    X = splits[["vinheta"]].astype(float).to_numpy()
    train = (splits["split"] == "train").to_numpy()
    model = LogisticRegression().fit(X[train], splits["y"][train])
    joblib.dump(model, out / "model.joblib")
    pred = splits[["path", "split", "y", "vinheta"]].assign(p=model.predict_proba(X)[:, 1])
    pred = pred[pred["split"] != "train"]
    pred.to_csv(out / "predictions.csv", index=False)
    val = pred[pred["split"] == "val"]
    block = classification_block(pred, evaluation.choose_threshold(val["y"], val["p"]))
    prevalence = splits.groupby(["split", "label"])["vinheta"].mean().unstack().round(4)
    block["prevalencia_vinheta"] = prevalence.to_dict("index")
    return block


def run_vignette_split(sc: Scenario, out, progress: Progress) -> dict:
    """V2: AUC de cada modelo nas imagens do teste com e sem vinheta."""
    out_block = {}
    for dep in sc.depends:
        try:
            pred = read_predictions(dep)
        except MissingDependency:
            out_block[dep] = {"pendente": True}
            continue
        threshold = read_metrics(dep)["limiar"]
        test = pred[pred["split"] == "test"]
        with_v, without = test[test["vinheta"].astype(bool)], test[~test["vinheta"].astype(bool)]
        out_block[dep] = {
            "com_vinheta": evaluation.evaluate(with_v["y"], with_v["p"], threshold),
            "sem_vinheta": evaluation.evaluate(without["y"], without["p"], threshold),
        }
    if all(v.get("pendente") for v in out_block.values()):
        raise MissingDependency(f"execute antes ao menos um de {sc.depends}")
    return {"modelos": out_block}


def _traditional_proba(sid: str, images: list[np.ndarray]) -> np.ndarray:
    from pdi import features

    sc = SCENARIOS[sid]
    model = joblib.load(status.scenario_dir(sid) / "model.joblib")
    rows = []
    for img in images:
        res = run(img, sc.pipeline)
        rows.append(features.extract(res.image, res.lesion_mask, res.roi))
    X = pd.DataFrame(rows)[list(features.FEATURE_NAMES)].to_numpy()
    return traditional.predict_proba(model, X)


def run_synthetic_vignette(sc: Scenario, out, progress: Progress) -> dict:
    """V5: probabilidade de melanoma antes e depois de inserir vinheta artificial."""
    from pdi import artifacts

    paths = list(pd.read_csv(cfg.EXPERIMENTS_DIR / "v5_ids.csv")["path"])
    originals = [preprocess.normalize_input(load_rgb(cfg.DATASET_DIR / p)) for p in paths]
    vignetted = [artifacts.add_vignette(img)[0] for img in originals]
    results, rows = {}, {"path": paths}
    for dep in sc.depends:
        if not (status.scenario_dir(dep) / "metrics.json").exists():
            results[dep] = {"pendente": True}
            continue
        progress(0.2, f"probabilidades do modelo de {dep}")
        if SCENARIOS[dep].model == "cnn":
            from training import cnn

            before, after = cnn.predict_images(dep, originals), cnn.predict_images(dep, vignetted)
        else:
            before, after = _traditional_proba(dep, originals), _traditional_proba(dep, vignetted)
        delta = after - before
        threshold = read_metrics(dep)["limiar"]
        rows[f"{dep}_antes"], rows[f"{dep}_depois"] = before, after
        idx = evaluation.bootstrap_indices(len(delta))
        boots = [delta[i].mean() for i in idx]
        results[dep] = {
            "p1_ligada": bool(SCENARIOS[dep].pipeline.enabled("P1")),
            "variacao_media": float(delta.mean()),
            "ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "viraram_malignas": int(((before < threshold) & (after >= threshold)).sum()),
        }
    if all(v.get("pendente") for v in results.values()):
        raise MissingDependency(f"execute antes ao menos um de {sc.depends}")
    pd.DataFrame({k: v for k, v in rows.items()}).to_csv(out / "predictions.csv", index=False)
    return {"n": len(paths), "modelos": results}


def _run_cnn(sc: Scenario, out, progress: Progress) -> dict:
    from training import cnn

    return cnn.run(sc, out, progress)


def _run_gradcam(sc: Scenario, out, progress: Progress) -> dict:
    from training import cnn

    return cnn.run_gradcam_scenario(sc, out, progress)


HANDLERS = {
    "ablation": run_ablation,
    "traditional": run_traditional,
    "segmentation": run_segmentation,
    "vignette_shortcut": run_vignette_shortcut,
    "vignette_split": run_vignette_split,
    "synthetic_vignette": run_synthetic_vignette,
    "cnn": _run_cnn,
    "gradcam": _run_gradcam,
}


# ---------------------------------------------------------------- execução


def run_scenario(sid: str) -> dict:
    if sid not in SCENARIOS:
        raise SystemExit(f"cenário desconhecido: {sid}; válidos: {list(SCENARIOS)}")
    sc = SCENARIOS[sid]
    out = status.scenario_dir(sid)
    out.mkdir(parents=True, exist_ok=True)
    started = status.now()
    t0 = time.perf_counter()

    def progress(fraction: float, message: str) -> None:
        print(f"[{sid}] {fraction:5.1%} {message}", flush=True)
        status.write(sid, state="executando", progress=round(fraction, 3), message=message)

    status.write(
        sid,
        state="executando",
        progress=0.0,
        message="iniciando",
        iniciado_em=started,
        finalizado_em=None,
        erro=None,
    )
    try:
        body = HANDLERS[sc.kind](sc, out, progress)
    except Exception as exc:
        traceback.print_exc()
        status.write(
            sid,
            state="erro",
            message=str(exc),
            erro=traceback.format_exc(limit=5),
            finalizado_em=status.now(),
        )
        raise
    metrics = {
        "id": sid,
        "grupo": sc.group,
        "descricao": sc.description,
        "tipo": sc.kind,
        "configuracao": sc.pipeline.to_dict() if sc.pipeline else None,
        "grupos_caracteristicas": sc.feature_groups,
        "commit": git_commit(),
        "ambiente": environment(),
        "iniciado_em": started,
        "finalizado_em": status.now(),
        "duracao_s": round(time.perf_counter() - t0, 1),
        **body,
    }
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2))
    status.write(
        sid, state="concluido", progress=1.0, message="concluído", finalizado_em=status.now()
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", help="IDs dos cenários (ou 'todos')")
    args = parser.parse_args()
    ids = list(SCENARIOS) if args.ids == ["todos"] else args.ids
    failed = []
    for sid in ids:
        try:
            m = run_scenario(sid)
            auc = m.get("teste", {}).get("auc", {}).get("valor")
            print(f"[{sid}] concluído" + (f": AUC teste = {auc:.3f}" if auc else ""))
        except Exception as exc:  # um cenário com erro não para os outros
            print(f"[{sid}] erro: {exc}", file=sys.stderr)
            failed.append(sid)
    if failed:
        raise SystemExit(f"cenários com erro: {failed}")


if __name__ == "__main__":
    main()
