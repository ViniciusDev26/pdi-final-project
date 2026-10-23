import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(os.environ.get("PDI_DATA_DIR", REPO_ROOT / "data"))
DATASET_DIR = DATA_DIR / "melanoma_cancer_dataset"
SPLITS_CSV = DATA_DIR / "splits.csv"
MODELS_DIR = Path(os.environ.get("PDI_MODELS_DIR", REPO_ROOT / "models"))

CLASSES = ("benign", "malignant")
SEED = 42
