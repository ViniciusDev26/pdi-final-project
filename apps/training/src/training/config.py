import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(os.environ.get("PDI_DATA_DIR", REPO_ROOT / "data"))
DATASET_DIR = DATA_DIR / "melanoma_cancer_dataset"
SPLITS_CSV = DATA_DIR / "splits.csv"
MODELS_DIR = Path(os.environ.get("PDI_MODELS_DIR", REPO_ROOT / "models"))

CLASSES = ("benign", "malignant")
SEED = 42

EXPERIMENTS_DIR = Path(os.environ.get("PDI_EXPERIMENTS_DIR", REPO_ROOT / "experiments"))
RESULTS_DIR = Path(os.environ.get("PDI_RESULTS_DIR", REPO_ROOT / "results"))
MASKS_DIR = EXPERIMENTS_DIR / "mascaras"
# integrantes que anotam as máscaras de referência (seção 5.8 do plano)
ANNOTATORS = ("Renan", "Guilherme", "Carlos")
