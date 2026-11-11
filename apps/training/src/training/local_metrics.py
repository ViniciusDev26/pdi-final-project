"""Métricas locais das etapas (plano experimental, tabela da seção 3).

Calculadas para a configuração de um cenário, numa amostra fixa do teste (avaliação, nada
é escolhido com elas), para comparar cenários A, L e T etapa a etapa:

- P1: taxa de detecção da vinheta por classe e área recortada (na base inteira, do cache);
- P2: PSNR/SSIM da restauração de pelos sintéticos (técnica de P2 da configuração sobre a
  saída de P1; sem P2, a imagem com pelos é comparada direto);
- P3: dispersão da cor média da pele entre imagens (desvio-padrão de L, a, b) e ΔE entre as
  saídas da imagem original e da mesma imagem sob dominante de cor sintética;
- P4: contraste RMS (canal L) e gradiente médio de Sobel na borda da lesão;
- P5: estabilidade da máscara sob ruído gaussiano sintético (Dice entre as máscaras).
"""

import cv2
import numpy as np
import pandas as pd

from pdi import STAGES, PipelineConfig, artifacts, load_rgb, preprocess, run
from pdi.metrics import delta_e, dice, lab, psnr, rms_contrast, ssim
from training import config as cfg

SAMPLE_SIZE = 100
COLOR_CAST = (1.15, 1.0, 0.8)  # dominante amarelada (lâmpada quente)
NOISE_SIGMA = 10.0


def sample_paths(splits: pd.DataFrame, n: int = SAMPLE_SIZE) -> list[str]:
    test = splits[splits["split"] == "test"]
    return list(test.sample(n, random_state=cfg.SEED)["path"])


def _p2_restoration(image: np.ndarray, config: PipelineConfig, rng) -> tuple[float, float]:
    vig = preprocess.remove_vignette(image)
    base, roi = vig.image, ~vig.mask
    hairy, _ = artifacts.add_hair(base, rng)
    key = config.technique("P2")
    if key is None:
        restored = hairy
    else:
        stage = next(s for s in STAGES if s.id == "P2")
        restored, _ = stage.technique(key).fn(hairy, roi)
    return psnr(restored, base), ssim(restored, base)


def _border_gradient(image: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None or not mask.any():
        return np.nan
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate(contours)[:, 0, :]
    return float(mag[pts[:, 1], pts[:, 0]].mean())


def compute(config: PipelineConfig, paths: list[str], cache_features: pd.DataFrame | None) -> dict:
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for path in paths:
        image = preprocess.normalize_input(load_rgb(cfg.DATASET_DIR / path))
        res = run(image, config)
        cast = run(artifacts.add_color_cast(image, COLOR_CAST), config)
        noisy_in = np.clip(image + rng.normal(0, NOISE_SIGMA, image.shape), 0, 255)
        noisy = run(noisy_in.astype(np.uint8), config)
        skin = res.roi & ~(res.lesion_mask if res.lesion_mask is not None else False)
        p2_psnr, p2_ssim = _p2_restoration(image, config, rng)
        a, b = res.lesion_mask, noisy.lesion_mask
        rows.append(
            {
                "vinheta_area": float((~res.roi).mean()),
                "p2_psnr": p2_psnr,
                "p2_ssim": p2_ssim,
                **dict(
                    zip(
                        ("pele_L", "pele_a", "pele_b"),
                        lab(res.image)[skin].mean(axis=0),
                        strict=True,
                    )
                ),
                "p3_delta_e_dominante": delta_e(res.image, cast.image, res.roi & cast.roi),
                "p4_contraste_rms": rms_contrast(res.image, res.roi),
                "p4_gradiente_borda": _border_gradient(res.image, res.lesion_mask),
                "p5_dice_ruido": dice(a, b) if a is not None and b is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    p2 = df["p2_psnr"].replace(np.inf, np.nan)
    out = {
        "amostra": f"{len(paths)} imagens do teste (semente {cfg.SEED})",
        "P1": {"area_recortada_media": float(df["vinheta_area"].mean())},
        "P2": {"psnr_pelos_sinteticos": float(p2.mean()), "ssim": float(df["p2_ssim"].mean())},
        "P3": {
            "dispersao_pele_L": float(df["pele_L"].std()),
            "dispersao_pele_a": float(df["pele_a"].std()),
            "dispersao_pele_b": float(df["pele_b"].std()),
            "delta_e_dominante": float(df["p3_delta_e_dominante"].mean()),
        },
        "P4": {
            "contraste_rms": float(df["p4_contraste_rms"].mean()),
            "gradiente_borda": float(df["p4_gradiente_borda"].mean()),
        },
        "P5": {"dice_sob_ruido": float(df["p5_dice_ruido"].mean())},
    }
    if cache_features is not None and "vinheta_p1" in cache_features:
        out["P1"]["taxa_deteccao"] = float(cache_features["vinheta_p1"].mean())
    return out
