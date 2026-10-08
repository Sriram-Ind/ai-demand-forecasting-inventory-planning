"""Central place for file paths and settings, so every script agrees on them."""
from pathlib import Path

# Project root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

TRAIN_CSV = DATA_RAW / "train.csv"

# Columns we expect in Kaggle's Store Item Demand train.csv
EXPECTED_COLUMNS = ["date", "store", "item", "sales"]

RANDOM_SEED = 42
