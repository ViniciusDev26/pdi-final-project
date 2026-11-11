"""Máscaras de referência anotadas pela equipe (plano experimental, seção 5.8).

Listas: `experiments/anotacao_{val,teste}_ids.csv` (coluna `responsavel` = quem anota;
`compartilhada` = anotada pelos três, para o Dice entre anotadores).

Armazenamento (PNG binário 300 × 300, versionado):
`experiments/mascaras/<conjunto>/<anotador>/<imagem>.png`, com cada salvamento registrado
em `experiments/mascaras/anotadores.csv`. A máscara de **referência** de uma imagem é a do
responsável; as dos outros dois anotadores (nas compartilhadas) servem só ao Dice entre
anotadores.

As máscaras do teste não podem ser usadas no ajuste do pipeline: só a avaliação final
(cenários S e V4) chama `reference_masks("teste")`.
"""

import csv
import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pdi.metrics import dice
from training.config import ANNOTATORS, EXPERIMENTS_DIR, MASKS_DIR

SETS = ("val", "teste")
LOG_COLUMNS = ("conjunto", "imagem", "anotador", "salvo_em")


@dataclass(frozen=True)
class Item:
    conjunto: str
    path: str  # caminho relativo ao dataset (coluna `path` do splits.csv)
    label: str
    vinheta: bool
    responsavel: str
    compartilhada: bool

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def annotators(self) -> tuple[str, ...]:
        """Quem deve anotar esta imagem."""
        return ANNOTATORS if self.compartilhada else (self.responsavel,)


def _check_set(conjunto: str) -> None:
    if conjunto not in SETS:
        raise ValueError(f"conjunto {conjunto!r} inválido; válidos: {SETS}")


def items(conjunto: str, experiments_dir: Path = EXPERIMENTS_DIR) -> list[Item]:
    _check_set(conjunto)
    df = pd.read_csv(experiments_dir / f"anotacao_{conjunto}_ids.csv")
    return [
        Item(
            conjunto=conjunto,
            path=r.path,
            label=r.label,
            vinheta=bool(r.vinheta),
            responsavel=r.responsavel,
            compartilhada=bool(r.compartilhada),
        )
        for r in df.itertuples()
    ]


def find(conjunto: str, name: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Item:
    for item in items(conjunto, experiments_dir):
        if item.name == name:
            return item
    raise KeyError(f"imagem {name!r} não está na lista de anotação de {conjunto}")


def mask_path(conjunto: str, annotator: str, name: str, masks_dir: Path = MASKS_DIR) -> Path:
    return masks_dir / conjunto / annotator / f"{name}.png"


def save_mask(item: Item, annotator: str, mask: np.ndarray, masks_dir: Path = MASKS_DIR) -> Path:
    if annotator not in item.annotators:
        raise ValueError(f"{annotator} não anota {item.name} (anotadores: {item.annotators})")
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (300, 300) or not mask.any():
        raise ValueError("a máscara deve ser 300 × 300 e não vazia")
    path = mask_path(item.conjunto, annotator, item.name, masks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    log = masks_dir / "anotadores.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(LOG_COLUMNS)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        writer.writerow((item.conjunto, item.name, annotator, stamp))
    return path


def load_mask(
    conjunto: str, annotator: str, name: str, masks_dir: Path = MASKS_DIR
) -> np.ndarray | None:
    path = mask_path(conjunto, annotator, name, masks_dir)
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 127


def reference_masks(
    conjunto: str, experiments_dir: Path = EXPERIMENTS_DIR, masks_dir: Path = MASKS_DIR
) -> dict[str, np.ndarray]:
    """Máscara de referência (do responsável) de cada imagem já anotada, por `path`."""
    out = {}
    for item in items(conjunto, experiments_dir):
        mask = load_mask(conjunto, item.responsavel, item.name, masks_dir)
        if mask is not None:
            out[item.path] = mask
    return out


def progress(experiments_dir: Path = EXPERIMENTS_DIR, masks_dir: Path = MASKS_DIR) -> pd.DataFrame:
    """Uma linha por (conjunto, anotador): imagens atribuídas e já anotadas."""
    rows = []
    for conjunto in SETS:
        all_items = items(conjunto, experiments_dir)
        for annotator in ANNOTATORS:
            mine = [i for i in all_items if annotator in i.annotators]
            done = sum(mask_path(conjunto, annotator, i.name, masks_dir).exists() for i in mine)
            rows.append(
                {"conjunto": conjunto, "anotador": annotator, "total": len(mine), "feitas": done}
            )
    return pd.DataFrame(rows)


def inter_annotator_dice(
    experiments_dir: Path = EXPERIMENTS_DIR, masks_dir: Path = MASKS_DIR
) -> pd.DataFrame:
    """Dice entre cada par de anotadores nas imagens compartilhadas do teste."""
    rows = []
    for item in items("teste", experiments_dir):
        if not item.compartilhada:
            continue
        masks = {a: load_mask("teste", a, item.name, masks_dir) for a in ANNOTATORS}
        for a, b in itertools.combinations(ANNOTATORS, 2):
            if masks[a] is not None and masks[b] is not None:
                rows.append(
                    {"imagem": item.name, "par": f"{a} × {b}", "dice": dice(masks[a], masks[b])}
                )
    return pd.DataFrame(rows, columns=["imagem", "par", "dice"])
