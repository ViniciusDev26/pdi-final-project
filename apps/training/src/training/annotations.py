"""Máscaras de referência das listas de anotação (plano experimental, seção 5.8).

Listas: `experiments/anotacao_{val,teste}_ids.csv` (20 da validação, 60 do teste).

Armazenamento (PNG binário 300 × 300, versionado): `experiments/mascaras/<conjunto>/<imagem>.png`,
uma máscara por imagem, com cada salvamento registrado em `experiments/mascaras/log.csv`.

As máscaras do teste não podem ser usadas no ajuste do pipeline: só a avaliação final
(cenários S e V4) chama `reference_masks("teste")`.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from training.config import EXPERIMENTS_DIR, MASKS_DIR

SETS = ("val", "teste")
LOG_COLUMNS = ("conjunto", "imagem", "salvo_em")


@dataclass(frozen=True)
class Item:
    conjunto: str
    path: str  # caminho relativo ao dataset (coluna `path` do splits.csv)
    label: str
    vinheta: bool

    @property
    def name(self) -> str:
        return Path(self.path).stem


def _check_set(conjunto: str) -> None:
    if conjunto not in SETS:
        raise ValueError(f"conjunto {conjunto!r} inválido; válidos: {SETS}")


def items(conjunto: str, experiments_dir: Path = EXPERIMENTS_DIR) -> list[Item]:
    _check_set(conjunto)
    df = pd.read_csv(experiments_dir / f"anotacao_{conjunto}_ids.csv")
    return [
        Item(conjunto=conjunto, path=r.path, label=r.label, vinheta=bool(r.vinheta))
        for r in df.itertuples()
    ]


def find(conjunto: str, name: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Item:
    for item in items(conjunto, experiments_dir):
        if item.name == name:
            return item
    raise KeyError(f"imagem {name!r} não está na lista de anotação de {conjunto}")


def mask_path(conjunto: str, name: str, masks_dir: Path = MASKS_DIR) -> Path:
    return masks_dir / conjunto / f"{name}.png"


def save_mask(item: Item, mask: np.ndarray, masks_dir: Path = MASKS_DIR) -> Path:
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (300, 300) or not mask.any():
        raise ValueError("a máscara deve ser 300 × 300 e não vazia")
    path = mask_path(item.conjunto, item.name, masks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    log = masks_dir / "log.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(LOG_COLUMNS)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        writer.writerow((item.conjunto, item.name, stamp))
    return path


def load_mask(conjunto: str, name: str, masks_dir: Path = MASKS_DIR) -> np.ndarray | None:
    path = mask_path(conjunto, name, masks_dir)
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 127


def reference_masks(
    conjunto: str, experiments_dir: Path = EXPERIMENTS_DIR, masks_dir: Path = MASKS_DIR
) -> dict[str, np.ndarray]:
    """Máscara de referência de cada imagem já anotada, por `path`."""
    out = {}
    for item in items(conjunto, experiments_dir):
        mask = load_mask(conjunto, item.name, masks_dir)
        if mask is not None:
            out[item.path] = mask
    return out


def progress(experiments_dir: Path = EXPERIMENTS_DIR, masks_dir: Path = MASKS_DIR) -> pd.DataFrame:
    """Uma linha por conjunto: imagens da lista e já anotadas."""
    rows = []
    for conjunto in SETS:
        all_items = items(conjunto, experiments_dir)
        done = sum(mask_path(conjunto, i.name, masks_dir).exists() for i in all_items)
        rows.append({"conjunto": conjunto, "total": len(all_items), "feitas": done})
    return pd.DataFrame(rows)
