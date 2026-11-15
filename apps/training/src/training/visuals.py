"""Saídas visuais obrigatórias da seção 5.6 que dependem das imagens da base.

Gera em `results/visuais/` (versionado, para o modo leitura funcionar sem o dataset):

- `segmentacao_melhores.png` / `segmentacao_piores.png`: máscara automática (S3, vermelho)
  × manual (verde) nos casos de maior e menor Dice do teste;
- `erros_fn.png` / `erros_fp.png` e `erros/<imagem>.png`: os 10 falsos negativos e os 10
  falsos positivos de maior confiança do modelo da Triagem (regra da seção 2), com imagem,
  segmentação do pipeline tradicional e Grad-CAM;
- `exemplo_<n>.png`: grades do simulador (A0 × A5 × T4 × L1) para as 6 imagens de exemplo;
- `index.json`: lista das figuras, casos de erro e Dice por imagem.

As figuras de ROC, matrizes de confusão, impacto por etapa e tempo vêm de `report.figure`.
A causa provável de cada erro é anotada pela equipe na página Resultados e fica em
`experiments/analise_erros.csv` (`save_error_note`).

Uso: python -m training.visuals
"""

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pdi import load_rgb, preprocess
from pdi import run as run_pipeline
from pdi.metrics import dice
from pdi.visualize import captioned, grid, overlay_contour, overlay_heatmap, overlay_two_contours
from training import annotations, report, status
from training import config as cfg
from training.compare import Combination, compare, figure
from training.scenarios import SCENARIOS

N_SEGMENTATION = 4
N_ERRORS = 10
EXAMPLE_PRESETS = ("A0", "A5", "T4", "L1")
SEGMENTATION_SCENARIO = "S3"  # segmentação padrão do pipeline
ERROR_CAUSES = (
    "pelos",
    "vinheta",
    "régua ou marcação",
    "baixo contraste",
    "lesão atípica",
    "reflexo ou bolha",
    "iluminação ou cor",
    "outra",
)
NOTE_COLUMNS = ("path", "causa", "nota", "atualizado_em")


def out_dir() -> Path:
    return cfg.RESULTS_DIR / "visuais"


def notes_path() -> Path:
    return cfg.EXPERIMENTS_DIR / "analise_erros.csv"


def _save(image: np.ndarray, name: str) -> str:
    path = out_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, optimize=True)
    return name


def _load(path: str) -> np.ndarray:
    return preprocess.normalize_input(load_rgb(cfg.DATASET_DIR / path))


def _commit() -> str:
    from training.run_scenario import git_commit

    return git_commit()


# ---------------------------------------------------------------- segmentação


def segmentation_cases(progress=lambda f, m: None) -> dict:
    refs = annotations.reference_masks("teste", cfg.EXPERIMENTS_DIR, cfg.MASKS_DIR)
    if not refs:
        return {"pendente": "nenhuma máscara de referência do teste"}
    config = SCENARIOS[SEGMENTATION_SCENARIO].pipeline
    rows = []
    for i, (path, manual) in enumerate(refs.items()):
        progress(i / len(refs), "segmentação")
        image = _load(path)
        auto = run_pipeline(image, config).lesion_mask
        d = dice(auto, manual) if auto is not None else 0.0
        rows.append({"path": path, "dice": float(d), "_img": image, "_a": auto, "_m": manual})
    rows.sort(key=lambda r: r["dice"], reverse=True)
    figures = {}
    for name, subset in (
        ("segmentacao_melhores", rows[:N_SEGMENTATION]),
        ("segmentacao_piores", rows[-N_SEGMENTATION:][::-1]),
    ):
        tiles = [
            captioned(
                overlay_two_contours(r["_img"], r["_a"], r["_m"]),
                [Path(r["path"]).stem, f"Dice {r['dice']:.2f}"],
            )
            for r in subset
        ]
        figures[name] = _save(grid([tiles]), f"{name}.png")
    return {
        "cenario": SEGMENTATION_SCENARIO,
        "figuras": figures,
        "dice_por_imagem": [{"path": r["path"], "dice": r["dice"]} for r in rows],
    }


# ---------------------------------------------------------------- análise de erros


def error_candidates(sid: str, n: int = N_ERRORS) -> pd.DataFrame:
    """FN e FP de maior confiança no teste (FN: maligna com menor p; FP: benigna com maior p)."""
    pred = pd.read_csv(status.scenario_dir(sid) / "predictions.csv")
    threshold = json.loads((status.scenario_dir(sid) / "metrics.json").read_text())["limiar"]
    test = pred[pred["split"] == "test"]
    fn = test[(test["y"] == 1) & (test["p"] < threshold)].nsmallest(n, "p").assign(tipo="FN")
    fp = test[(test["y"] == 0) & (test["p"] >= threshold)].nlargest(n, "p").assign(tipo="FP")
    return pd.concat([fn, fp])[["path", "y", "p", "tipo"]].reset_index(drop=True)


def error_cases(progress=lambda f, m: None) -> dict:
    choice = report.triage_choice(report.load_metrics())
    if choice is None:
        return {"pendente": "nenhum cenário C executado"}
    sid = choice["id"]
    cands = error_candidates(sid)
    is_cnn = SCENARIOS[sid].model == "cnn"
    seg_config = SCENARIOS["A5"].pipeline
    items, tiles = [], {"FN": [], "FP": []}
    for i, r in cands.iterrows():
        progress(i / max(len(cands), 1), "análise de erros")
        image = _load(r["path"])
        seg = run_pipeline(image, seg_config).lesion_mask
        panels = [captioned(image, [Path(r["path"]).stem, f"{r['tipo']} · p = {r['p']:.3f}"])]
        panels.append(captioned(overlay_contour(image, seg), ["Segmentação (A5)", ""]))
        if is_cnn:
            from training import cnn

            _, cam, _ = cnn.gradcam(sid, image)
            panels.append(captioned(overlay_heatmap(image, cam), [f"Grad-CAM ({sid})", ""]))
        name = f"erros/{Path(r['path']).stem}.png"
        _save(grid([panels]), name)
        tiles[r["tipo"]].append(panels)
        items.append(
            {
                "path": r["path"],
                "imagem": Path(r["path"]).stem,
                "tipo": r["tipo"],
                "y": int(r["y"]),
                "p": float(r["p"]),
                "figura": name,
            }
        )
    figures = {}
    for kind, row in tiles.items():
        if row:  # dois casos por linha: imagem, segmentação e Grad-CAM de cada um
            rows = [[t for case in row[j : j + 2] for t in case] for j in range(0, len(row), 2)]
            figures[f"erros_{kind.lower()}"] = _save(grid(rows), f"erros_{kind.lower()}.png")
    return {"modelo": sid, "figuras": figures, "casos": items}


def load_notes() -> dict[str, dict]:
    if not notes_path().exists():
        return {}
    df = pd.read_csv(notes_path(), keep_default_na=False)
    return {r["path"]: r for r in df.to_dict("records")}


def save_error_note(path: str, causa: str, nota: str = "") -> dict:
    if causa not in ERROR_CAUSES:
        raise ValueError(f"causa inválida; opções: {ERROR_CAUSES}")
    notes = load_notes()
    notes[path] = {
        "path": path,
        "causa": causa,
        "nota": nota.strip(),
        "atualizado_em": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    notes_path().parent.mkdir(parents=True, exist_ok=True)
    with notes_path().open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NOTE_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(notes.values(), key=lambda n: n["path"]))
    return notes[path]


# ---------------------------------------------------------------- grades do simulador


def example_grids(progress=lambda f, m: None) -> dict:
    examples = pd.read_csv(cfg.EXPERIMENTS_DIR / "exemplos_ids.csv")
    combos = [
        Combination(f"{pid} · {SCENARIOS[pid].description}", SCENARIOS[pid].pipeline)
        for pid in EXAMPLE_PRESETS
    ]
    figures = {}
    for i, r in enumerate(examples.itertuples(), start=1):
        progress(i / len(examples), "grades do simulador")
        figures[f"exemplo_{i}"] = _save(figure(compare(_load(r.path), combos)), f"exemplo_{i}.png")
    return {"presets": list(EXAMPLE_PRESETS), "figuras": figures, "imagens": list(examples.path)}


# ---------------------------------------------------------------- índice


def generate(progress=lambda f, m: None) -> dict:
    if not cfg.DATASET_DIR.is_dir():
        raise FileNotFoundError("as saídas visuais precisam do dataset (just data-download)")
    out_dir().mkdir(parents=True, exist_ok=True)
    index = {
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "segmentacao": segmentation_cases(lambda f, m: progress(0.3 * f, m)),
        "erros": error_cases(lambda f, m: progress(0.3 + 0.4 * f, m)),
        "exemplos": example_grids(lambda f, m: progress(0.7 + 0.3 * f, m)),
    }
    (out_dir() / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return index


def load_index() -> dict | None:
    path = out_dir() / "index.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    index = generate(lambda f, m: print(f"{f:5.0%} {m}", flush=True))
    print(json.dumps({k: v.get("figuras") for k, v in index.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
