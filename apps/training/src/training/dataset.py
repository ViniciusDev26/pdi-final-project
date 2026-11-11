"""Base de dados dos experimentos e cache do pipeline por configuração.

`load_splits` junta o `data/splits.csv` com a vinheta detectada (`experiments/vinheta.csv`).
`pipeline_cache` executa o pipeline + características F sobre a base inteira, em paralelo,
e guarda o resultado em `results/_cache/<chave>/`: `features.pkl` (uma linha por imagem,
na ordem de `load_splits`) e, se pedido, `images.npy` e `masks.npy` (memmap uint8/bool,
entrada da CNN). A chave combina a configuração e o código-fonte de `packages/pdi`: mudar
uma constante de PDI invalida o cache.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import pdi
from pdi import PipelineConfig, features, load_rgb, run
from training import config as cfg

SIZE = 300
Progress = Callable[[float, str], None]


def load_splits() -> pd.DataFrame:
    """Uma linha por imagem: path, label, split, y (1 = maligna), vinheta (P1 na imagem bruta)."""
    df = pd.read_csv(cfg.SPLITS_CSV)[["path", "label", "split"]]
    vig_csv = cfg.EXPERIMENTS_DIR / "vinheta.csv"
    if vig_csv.exists():
        vig = pd.read_csv(vig_csv)[["path", "vinheta"]]
        df = df.merge(vig, on="path", how="left")
    else:
        df["vinheta"] = np.nan
    df["y"] = (df["label"] == "malignant").astype(int)
    return df.reset_index(drop=True)


def _pdi_source_hash() -> str:
    h = hashlib.sha1()
    for path in sorted(Path(pdi.__file__).parent.glob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def cache_key(config: PipelineConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True) + _pdi_source_hash()
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PipelineCache:
    key: str
    dir: Path
    features: pd.DataFrame  # path + características + tempos (s) por etapa + vinheta_p1

    def images(self) -> np.ndarray:
        return np.load(self.dir / "images.npy", mmap_mode="r")

    def masks(self) -> np.ndarray:
        return np.load(self.dir / "masks.npy", mmap_mode="r")

    @property
    def has_images(self) -> bool:
        return (self.dir / "images.npy").exists() and (self.dir / "masks.npy").exists()


# ---------------------------------------------------------------- execução paralela

_worker_state: dict = {}


def _init_worker(config_dict: dict, out_dir: str | None, n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)
    _worker_state["config"] = PipelineConfig.from_dict(config_dict)
    if out_dir is not None:
        d = Path(out_dir)
        _worker_state["images"] = np.lib.format.open_memmap(d / "images.tmp.npy", mode="r+")
        _worker_state["masks"] = np.lib.format.open_memmap(d / "masks.tmp.npy", mode="r+")


def _process(args: tuple[int, str]) -> dict:
    i, path = args
    result = run(load_rgb(cfg.DATASET_DIR / path), _worker_state["config"])
    row = {"path": path, **features.extract(result.image, result.lesion_mask, result.roi)}
    row |= {f"t_{k}": v for k, v in result.timings.items()}
    row["vinheta_p1"] = result.vignette_detected
    row["area_lesao"] = float(result.lesion_mask.mean()) if result.lesion_mask is not None else 0.0
    if "images" in _worker_state:
        _worker_state["images"][i] = result.image
        mask = result.lesion_mask if result.lesion_mask is not None else np.zeros((SIZE, SIZE))
        _worker_state["masks"][i] = mask.astype(bool)
    return row


def pipeline_cache(
    config: PipelineConfig,
    with_images: bool = False,
    progress: Progress | None = None,
    workers: int | None = None,
) -> PipelineCache:
    key = cache_key(config)
    out = cfg.RESULTS_DIR / "_cache" / key
    feats_path = out / "features.pkl"
    cache = PipelineCache(key, out, pd.DataFrame())
    if feats_path.exists() and (cache.has_images or not with_images):
        return PipelineCache(key, out, pd.read_pickle(feats_path))

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    paths = list(load_splits()["path"])
    n = len(paths)
    if with_images:
        np.lib.format.open_memmap(out / "images.tmp.npy", "w+", np.uint8, (n, SIZE, SIZE, 3))
        np.lib.format.open_memmap(out / "masks.tmp.npy", "w+", bool, (n, SIZE, SIZE))

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(config.to_dict(), str(out) if with_images else None, n),
    ) as pool:
        for row in pool.map(_process, enumerate(paths), chunksize=16):
            rows.append(row)
            if progress and len(rows) % 500 == 0:
                elapsed = time.perf_counter() - t0
                progress(len(rows) / n, f"pipeline: {len(rows)}/{n} imagens ({elapsed:.0f} s)")
    df = pd.DataFrame(rows)
    if with_images:
        (out / "images.tmp.npy").replace(out / "images.npy")
        (out / "masks.tmp.npy").replace(out / "masks.npy")
    df.to_pickle(feats_path)
    return PipelineCache(key, out, df)


def feature_matrix(cache: PipelineCache, groups: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Colunas de características (todas ou só as dos grupos indicados)."""
    names = (
        features.FEATURE_NAMES
        if groups is None
        else tuple(n for g in groups for n in features.FEATURE_GROUPS[g])
    )
    return cache.features[list(names)]
