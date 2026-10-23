"""Verifica o dataset, remove duplicatas e gera data/splits.csv.

O CSV tem uma linha por imagem (path, label, split), com split em
train/val/test. A validação sai do treino, estratificada e com semente fixa,
para que todos os modelos sejam comparados nas mesmas divisões.

Uso: python -m training.prepare_data [--val-size 0.15]
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from training.config import CLASSES, DATASET_DIR, SEED, SPLITS_CSV


def scan(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for label in CLASSES:
            folder = dataset_dir / split / label
            if not folder.is_dir():
                raise SystemExit(f"Pasta não encontrada: {folder} (rode `just data-download`)")
            for path in sorted(folder.glob("*.jpg")):
                rows.append(
                    {
                        "path": str(path.relative_to(dataset_dir)),
                        "label": label,
                        "split": split,
                        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                    }
                )
    return pd.DataFrame(rows)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma cópia por conteúdo, preferindo a do teste para não vazar treino no teste."""
    labels_per_hash = df.groupby("md5")["label"].nunique()
    conflicting = list(labels_per_hash[labels_per_hash > 1].index)
    if conflicting:
        raise SystemExit(f"Imagens idênticas com rótulos diferentes: {conflicting}")
    order = df["split"].map({"test": 0, "train": 1})
    return (
        df.assign(_order=order)
        .sort_values(["_order", "path"])
        .drop_duplicates("md5")
        .drop(columns="_order")
    )


def add_validation(df: pd.DataFrame, val_size: float) -> pd.DataFrame:
    train_idx = df.index[df["split"] == "train"]
    _, val_idx = train_test_split(
        train_idx, test_size=val_size, stratify=df.loc[train_idx, "label"], random_state=SEED
    )
    df = df.copy()
    df.loc[val_idx, "split"] = "val"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    df = scan(DATASET_DIR)
    deduped = drop_duplicates(df)
    splits = add_validation(deduped, args.val_size)
    splits[["path", "label", "split"]].sort_values("path").to_csv(SPLITS_CSV, index=False)

    print(f"Imagens encontradas: {len(df)} | duplicatas removidas: {len(df) - len(deduped)}")
    print(pd.crosstab(splits["split"], splits["label"], margins=True))
    print(f"Gerado: {SPLITS_CSV}")


if __name__ == "__main__":
    main()
