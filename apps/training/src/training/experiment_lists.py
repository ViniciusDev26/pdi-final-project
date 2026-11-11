"""Gera as listas fixas de `experiments/` (plano experimental, seções 2, 3, 5.4 e 5.8).

As listas são versionadas no git e geradas uma única vez, com a semente do projeto. A
seleção usa só o rótulo e o detector de vinheta da etapa P1 (não há ajuste de nada no teste):

- `vinheta.csv`: vinheta detectada pela P1 em cada imagem da base;
- `exemplos_ids.csv`: 6 imagens da validação para as figuras "antes → depois";
- `anotacao_val_ids.csv`: 20 da validação (10 benignas, 10 malignas; metade com vinheta);
- `anotacao_teste_ids.csv`: 60 do teste (30 + 30; metade com vinheta quando houver), 10
  delas anotadas pelos três integrantes;
- `teste_sem_vinheta.csv`: imagens do teste sem vinheta;
- `v5_ids.csv`: 50 benignas do teste sem vinheta (cenário V5).

Uso: python -m training.experiment_lists [--force]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from pdi import load_rgb, preprocess
from training.config import ANNOTATORS, DATASET_DIR, EXPERIMENTS_DIR, SEED, SPLITS_CSV

N_SHARED = 10


def _has_vignette(path: str) -> bool:
    return preprocess.remove_vignette(load_rgb(DATASET_DIR / path)).detected


def vignette_table(splits: pd.DataFrame) -> pd.DataFrame:
    with ProcessPoolExecutor() as pool:
        flags = list(pool.map(_has_vignette, splits["path"], chunksize=32))
    return splits[["path", "label", "split"]].assign(vinheta=flags)


def _balanced(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """`per_class` por classe, metade com vinheta (completa com sem vinheta se faltar)."""
    parts = []
    for label in ("benign", "malignant"):
        cls = df[df["label"] == label]
        with_v = cls[cls["vinheta"]]
        n_v = min(per_class // 2, len(with_v))
        take_v = with_v.sample(n_v, random_state=seed)
        take_n = cls[~cls["vinheta"]].sample(per_class - n_v, random_state=seed)
        parts += [take_v, take_n]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _assign(df: pd.DataFrame, shared: int = 0) -> pd.DataFrame:
    """Divide as imagens entre os anotadores; as `shared` primeiras são de todos."""
    df = df.copy()
    df["compartilhada"] = [i < shared for i in range(len(df))]
    df["responsavel"] = [ANNOTATORS[i % len(ANNOTATORS)] for i in range(len(df))]
    return df


def build(vig: pd.DataFrame) -> dict[str, pd.DataFrame]:
    val, test = vig[vig["split"] == "val"], vig[vig["split"] == "test"]
    cols = ["path", "label", "vinheta"]
    examples = _balanced(val, 3, SEED + 1)[cols]
    ann_val = _assign(_balanced(val, 10, SEED))
    ann_test = _assign(_balanced(test, 30, SEED), shared=N_SHARED)
    no_vig = test[~test["vinheta"]][["path", "label"]]
    v5 = no_vig[no_vig["label"] == "benign"].sample(50, random_state=SEED)[["path"]]
    return {
        "vinheta.csv": vig,
        "exemplos_ids.csv": examples,
        "anotacao_val_ids.csv": ann_val[[*cols, "responsavel", "compartilhada"]],
        "anotacao_teste_ids.csv": ann_test[[*cols, "responsavel", "compartilhada"]],
        "teste_sem_vinheta.csv": no_vig.sort_values("path"),
        "v5_ids.csv": v5.sort_values("path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="sobrescreve listas existentes")
    args = parser.parse_args()
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    existing = [p.name for p in EXPERIMENTS_DIR.glob("*.csv")]
    if existing and not args.force:
        raise SystemExit(f"Listas já existem ({existing}); são fixas. Use --force para refazer.")
    vig = vignette_table(pd.read_csv(SPLITS_CSV))
    for name, df in build(vig).items():
        df.to_csv(EXPERIMENTS_DIR / name, index=False)
        print(f"{name}: {len(df)} linhas")
    print(vig.groupby(["split", "label"])["vinheta"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
