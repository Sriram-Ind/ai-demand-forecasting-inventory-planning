"""Load the Kaggle train.csv and run basic sanity checks.

Run from the project root:
    python -m src.data_loader
"""
import sys

import pandas as pd

from src.config import EXPECTED_COLUMNS, TRAIN_CSV


def load_sales(path=TRAIN_CSV) -> pd.DataFrame:
    """Read the sales file, parse dates, and sort by store, item, date."""
    if not path.exists():
        sys.exit(
            f"Could not find {path}.\n"
            "Download train.csv from Kaggle's 'Store Item Demand Forecasting Challenge' "
            "and put it in data/raw/."
        )

    df = pd.read_csv(path, parse_dates=["date"])

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        sys.exit(f"train.csv is missing columns: {sorted(missing)}")

    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    """Print a short health check of the data."""
    print(f"Rows:            {len(df):,}")
    print(f"Date range:      {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Stores:          {df['store'].nunique()}")
    print(f"Items:           {df['item'].nunique()}")
    print(f"Store-item pairs:{df.groupby(['store', 'item']).ngroups:>6}")
    print(f"Missing values:  {int(df.isna().sum().sum())}")
    print(f"Duplicate rows:  {int(df.duplicated(['date', 'store', 'item']).sum())}")
    print(f"Daily sales:     min {df['sales'].min()}, mean {df['sales'].mean():.1f}, max {df['sales'].max()}")


if __name__ == "__main__":
    summarize(load_sales())
