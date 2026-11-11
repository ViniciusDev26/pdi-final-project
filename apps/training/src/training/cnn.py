"""CNN dos cenários C6, C7 e V3 e Grad-CAM (V4); protocolo congelado da seção 5.7.

- EfficientNet-B0 pré-treinada na ImageNet (torchvision), cabeça trocada por uma camada
  linear de 1 saída.
- Entrada 224 × 224 (redimensionada de 300 × 300), normalização da ImageNet.
- Aumento só geométrico: espelhamentos, rotações múltiplas de 90°, recorte aleatório com
  escala 0,8–1,0. Sem aumento de cor (não confundir com P3).
- AdamW (lr 3e-4, wd 1e-4), agenda cosseno, batch 32, precisão mista, até 30 épocas, early
  stopping com paciência 5 pela AUC na validação.
- Semente 42 e `cudnn.deterministic = True`.

Entradas por cenário (`Scenario.cnn_input`): `raw` = imagem após P0 (C6); `preprocessed` =
saída de P1–P5 (C7); `lesion` = saída de P1–P5 com a pele fora da máscara de S zerada (V3;
se a segmentação vier vazia, a imagem fica inteira).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinismo do cuBLAS

import json
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2

from pdi import PipelineConfig
from pdi import run as run_pipeline
from training import annotations, dataset, evaluation, status
from training import config as cfg
from training.device import get_device
from training.scenarios import SCENARIOS, Scenario

INPUT_SIZE = 224
BATCH_SIZE = 32
# 30 é o protocolo; PDI_CNN_MAX_EPOCHS só serve para testes rápidos (fica no metrics.json)
MAX_EPOCHS = int(os.environ.get("PDI_CNN_MAX_EPOCHS", 30))
PATIENCE = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
CROP_SCALE = (0.8, 1.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_WORKERS = min(8, os.cpu_count() or 1)


def seed_everything(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


# ---------------------------------------------------------------- dados


def _pipeline_for(sc: Scenario) -> PipelineConfig:
    return PipelineConfig.raw() if sc.cnn_input == "raw" else sc.pipeline


def apply_input(image: np.ndarray, mask: np.ndarray | None, cnn_input: str) -> np.ndarray:
    """Imagem (após o pipeline do cenário) como a CNN a recebe."""
    if cnn_input == "lesion" and mask is not None and mask.any():
        return image * mask[..., None].astype(image.dtype)
    return image


def _normalize() -> v2.Normalize:
    return v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(INPUT_SIZE, scale=CROP_SCALE, ratio=(1.0, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            _normalize(),
        ]
    )


def eval_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(INPUT_SIZE, antialias=True),
            _normalize(),
        ]
    )


class CachedImages(Dataset):
    """Imagens do cache do pipeline (memmap), na ordem de `load_splits`."""

    def __init__(self, cache: dataset.PipelineCache, indices, labels, cnn_input, train: bool):
        self.dir = cache.dir
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.cnn_input = cnn_input
        self.train = train
        self.transform = train_transform() if train else eval_transform()
        self._images = self._masks = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._images is None:  # abre o memmap em cada processo do DataLoader
            self._images = np.load(self.dir / "images.npy", mmap_mode="r")
            self._masks = np.load(self.dir / "masks.npy", mmap_mode="r")
        idx = self.indices[i]
        img = apply_input(np.array(self._images[idx]), np.array(self._masks[idx]), self.cnn_input)
        x = torch.from_numpy(img).permute(2, 0, 1)
        if self.train:
            x = torch.rot90(x, k=random.randint(0, 3), dims=(1, 2))
        return self.transform(x), self.labels[i]


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.SEED)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


# ---------------------------------------------------------------- treino


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True)).squeeze(1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_auc: float
    seconds: float


def train(
    train_ds: Dataset, val_ds: Dataset, y_val: np.ndarray, progress, device: torch.device
) -> TrainResult:
    seed_everything()
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader, val_loader = _loader(train_ds, True), _loader(val_ds, False)

    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        te, total, n = time.perf_counter(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        val_auc = float(roc_auc_score(y_val, predict(model, val_loader, device)))
        history.append(
            {
                "epoca": epoch,
                "perda_treino": total / n,
                "auc_validacao": val_auc,
                "lr": opt.param_groups[0]["lr"],
                "segundos": time.perf_counter() - te,
            }
        )
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        progress(
            0.3 + 0.6 * epoch / MAX_EPOCHS,
            f"época {epoch}: perda {total / n:.4f}, AUC val {val_auc:.4f} "
            f"(melhor {best_auc:.4f} na época {best_epoch})",
        )
        if epoch - best_epoch >= PATIENCE:
            progress(0.9, f"early stopping na época {epoch} (melhor: {best_epoch})")
            break
    model.load_state_dict(best_state)
    return TrainResult(model, pd.DataFrame(history), best_epoch, best_auc, time.perf_counter() - t0)


def run(sc: Scenario, out, progress) -> dict:
    """Executa um cenário de CNN (C6, C7, V3)."""
    from training.run_scenario import classification_block

    device = get_device()
    seed_everything()
    progress(0.02, f"pipeline da entrada ({sc.cnn_input}) na base inteira; dispositivo {device}")
    cache = dataset.pipeline_cache(
        _pipeline_for(sc), with_images=True, progress=lambda f, m: progress(0.02 + 0.25 * f, m)
    )
    splits = dataset.load_splits()
    y = splits["y"].to_numpy()
    idx = {s: np.flatnonzero(splits["split"].to_numpy() == s) for s in ("train", "val", "test")}
    make = lambda s, train: CachedImages(cache, idx[s], y[idx[s]], sc.cnn_input, train)  # noqa: E731
    progress(0.3, f"treinando EfficientNet-B0 ({len(idx['train'])} imagens de treino)")
    result = train(make("train", True), make("val", False), y[idx["val"]], progress, device)

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "scenario": sc.id,
            "cnn_input": sc.cnn_input,
            "pipeline": _pipeline_for(sc).to_dict(),
            "best_epoch": result.best_epoch,
        },
        out / "model.pt",
    )
    result.history.to_csv(out / "history.csv", index=False)
    progress(0.92, "predições na validação e no teste")
    preds = []
    for s in ("val", "test"):
        p = predict(result.model, _loader(make(s, False), False), device)
        preds.append(splits.iloc[idx[s]][["path", "split", "y", "vinheta"]].assign(p=p))
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(out / "predictions.csv", index=False)
    val = pred[pred["split"] == "val"]
    block = classification_block(pred, evaluation.choose_threshold(val["y"], val["p"]))
    epochs = len(result.history)
    block["treino"] = {
        "dispositivo": str(device),
        "epocas": epochs,
        "melhor_epoca": result.best_epoch,
        "auc_validacao_melhor": result.best_val_auc,
        "segundos": round(result.seconds, 1),
        "segundos_por_epoca": round(result.seconds / epochs, 1),
        "protocolo": {
            "arquitetura": "EfficientNet-B0 (ImageNet)",
            "entrada": INPUT_SIZE,
            "batch": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epocas": MAX_EPOCHS,
            "paciencia": PATIENCE,
        },
    }
    block["historico"] = result.history.round(5).to_dict("records")
    return block


# ---------------------------------------------------------------- inferência e Grad-CAM

_loaded: dict[str, tuple[nn.Module, dict]] = {}


def load(sid: str, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    if sid not in _loaded:
        path = status.scenario_dir(sid) / "model.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device or get_device()).eval()
        _loaded[sid] = (model, ckpt)
    return _loaded[sid]


def prepare(sid: str, image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Aplica o pipeline do cenário e devolve (imagem vista pela CNN, tensor 1×3×224×224)."""
    sc = SCENARIOS[sid]
    res = run_pipeline(image, _pipeline_for(sc))
    img = apply_input(res.image, res.lesion_mask, sc.cnn_input)
    x = eval_transform()(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1))
    return img, x.unsqueeze(0)


@torch.no_grad()
def predict_images(sid: str, images: list[np.ndarray]) -> np.ndarray:
    model, _ = load(sid)
    device = next(model.parameters()).device
    out = []
    for img in images:
        _, x = prepare(sid, img)
        out.append(torch.sigmoid(model(x.to(device)).squeeze()).item())
    return np.asarray(out)


def gradcam(sid: str, image: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Grad-CAM na última camada convolucional. Devolve (prob., mapa 300×300 em [0,1], img)."""
    model, _ = load(sid)
    device = next(model.parameters()).device
    seen, x = prepare(sid, image)
    acts, grads = {}, {}

    def keep(module, inputs, output) -> None:
        acts["a"] = output
        output.register_hook(lambda g: grads.__setitem__("g", g))

    handle = model.features[-1].register_forward_hook(keep)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = model(x.to(device)).squeeze()
            logit.backward()
    finally:
        handle.remove()
    weights = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=seen.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return float(torch.sigmoid(logit).item()), cam, seen


def run_gradcam_scenario(sc: Scenario, out, progress) -> dict:
    """V4: fração da ativação do Grad-CAM dentro da máscara manual (teste anotado)."""
    from pdi import load_rgb, preprocess

    refs = annotations.reference_masks("teste", cfg.EXPERIMENTS_DIR, cfg.MASKS_DIR)
    if not refs:
        from training.run_scenario import MissingDependency

        raise MissingDependency("nenhuma máscara de referência do teste anotada ainda")
    results, rows = {}, []
    for dep in sc.depends:
        if not (status.scenario_dir(dep) / "model.pt").exists():
            results[dep] = {"pendente": True}
            continue
        progress(0.1, f"Grad-CAM de {dep} em {len(refs)} imagens anotadas")
        fractions = []
        for path, mask in refs.items():
            image = preprocess.normalize_input(load_rgb(cfg.DATASET_DIR / path))
            _, cam, _ = gradcam(dep, image)
            frac = float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else np.nan
            fractions.append(frac)
            rows.append(
                {
                    "modelo": dep,
                    "path": path,
                    "fracao_na_lesao": frac,
                    "area_lesao": float(mask.mean()),
                }
            )
        f = np.asarray(fractions)
        boots = [np.nanmean(f[i]) for i in evaluation.bootstrap_indices(len(f))]
        results[dep] = {
            "fracao_ativacao_na_lesao": float(np.nanmean(f)),
            "ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "area_media_lesao": float(np.mean([m.mean() for m in refs.values()])),
        }
    if all(v.get("pendente") for v in results.values()):
        from training.run_scenario import MissingDependency

        raise MissingDependency(f"execute antes ao menos um de {sc.depends}")
    pd.DataFrame(rows).to_csv(out / "gradcam.csv", index=False)
    (out / "resumo.json").write_text(json.dumps(results, indent=2))
    return {"n": len(refs), "modelos": results}
