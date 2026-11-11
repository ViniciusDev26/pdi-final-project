"""Calibração do alongamento mínimo do pelo (P2) com pelos sintéticos — só treino/validação.

Para cada imagem amostrada: aplica P1, insere pelos sintéticos (`pdi.artifacts.add_hair`),
remove os pelos com cada candidato de `HAIR_MIN_ELONGATION` e mede:

- PSNR da restauração em relação à imagem antes dos pelos sintéticos (métrica local de P2,
  seção 3 do plano): penaliza tanto pelo não removido quanto estrutura da lesão apagada;
- cobertura dos pelos sintéticos pela máscara detectada;
- fração da lesão (segmentada na imagem sem pelos sintéticos) marcada como pelo: o defeito
  que motivou a correção (glóbulos escuros apagados).

O candidato escolhido é o de maior PSNR médio no treino; a validação confirma.

Uso: python -m training.calibrate_p2 [--n 150]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from pdi import artifacts, load_rgb, preprocess
from pdi.metrics import psnr
from pdi.segment import segment
from training.config import DATASET_DIR, SEED, SPLITS_CSV

THRESHOLDS = (12, 20, 30, 40, 50)
ELONGATIONS = (0.0, 4.0, 8.0, 12.0, 20.0)
CANDIDATES = tuple((t, e) for t in THRESHOLDS for e in ELONGATIONS)


def _hair_mask(
    image: np.ndarray, roi: np.ndarray, threshold: int, min_elongation: float
) -> np.ndarray:
    k = preprocess.HAIR_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raw = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) > threshold) & roi
    mask = preprocess.keep_elongated(raw, min_elongation)
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _evaluate(args: tuple[str, int]) -> list[dict]:
    path, seed = args
    vig = preprocess.remove_vignette(load_rgb(DATASET_DIR / path))
    base, roi = vig.image, ~vig.mask
    hairy, hair = artifacts.add_hair(base, np.random.default_rng(seed))
    hair &= roi
    lesion = segment(base, roi)
    rows = []
    for thr, elong in CANDIDATES:
        mask = _hair_mask(hairy, roi, thr, elong)
        restored = cv2.inpaint(
            hairy, mask.astype(np.uint8), preprocess.HAIR_INPAINT_RADIUS, cv2.INPAINT_TELEA
        )
        clean_mask = _hair_mask(base, roi, thr, elong)  # o que seria apagado sem pelo sintético
        rows.append(
            {
                "path": path,
                "threshold": thr,
                "min_elongation": elong,
                "psnr": psnr(restored, base),
                "cobertura_pelo": float(mask[hair].mean()) if hair.any() else np.nan,
                "lesao_apagada": float(clean_mask[lesion].mean()) if lesion.any() else np.nan,
            }
        )
    return rows


def calibrate(n: int) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_CSV)
    parts = []
    for split in ("train", "val"):
        sample = df[df["split"] == split].sample(n, random_state=SEED)
        jobs = [(p, SEED + i) for i, p in enumerate(sample["path"])]
        with ProcessPoolExecutor() as pool:
            rows = [r for chunk in pool.map(_evaluate, jobs, chunksize=4) for r in chunk]
        parts.append(pd.DataFrame(rows).assign(split=split))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150, help="imagens por divisão")
    args = parser.parse_args()
    res = calibrate(args.n)
    table = (
        res.groupby(["split", "threshold", "min_elongation"])[["psnr", "cobertura_pelo", "lesao_apagada"]]
        .mean()
        .unstack("split")
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table.round(4))
    best = table[("psnr", "train")].idxmax()
    print(f"\nMelhor no treino (PSNR médio): (HAIR_THRESHOLD, HAIR_MIN_ELONGATION) = {best}")


if __name__ == "__main__":
    main()
