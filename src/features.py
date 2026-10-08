"""Stage 3: Feature engineering.

Turns the raw (date, store, item, sales) table into a table of "clues"
(features) a machine-learning model can learn from.

DESIGN RULE: we forecast up to HORIZON = 7 days ahead. So every clue built
from past sales must be at least 7 days old. That is exactly what you'd know
on the day you place a weekly order, and it prevents "data leakage"
(accidentally letting the model peek at the answer).

Run from the project root:
    python -m src.features
"""
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.data_loader import load_sales

HORIZON = 7                      # how many days ahead we forecast
LAGS = [7, 14, 21, 28, 364]      # "sales N days ago" (364 = same weekday last year)
ROLL_WINDOWS = [7, 28]           # rolling average lengths in days
FEATURES_PATH = DATA_PROCESSED / "features.parquet"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar clues. These need no sales history, so they never leak."""
    d = df["date"].dt
    df["dayofweek"] = d.dayofweek            # 0 = Monday ... 6 = Sunday
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["day"] = d.day
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["year"] = d.year
    df["dayofyear"] = d.dayofyear
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"] = d.is_month_end.astype(int)
    # Cyclical encoding: makes December (12) "close" to January (1) for the model
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag = the value from N days earlier, within the same store-item."""
    g = df.groupby(["store", "item"])["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling average of the last N days, ending HORIZON days before the target day.

    shift(HORIZON) happens BEFORE rolling, so the window never touches
    anything newer than HORIZON days ago.
    """
    g = df.groupby(["store", "item"])["sales"]
    for w in ROLL_WINDOWS:
        df[f"rollmean_{w}"] = g.transform(lambda s: s.shift(HORIZON).rolling(w).mean())
    df["rollstd_28"] = g.transform(lambda s: s.shift(HORIZON).rolling(28).std())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def check_no_leakage(df: pd.DataFrame, n_checks: int = 200, seed: int = 42) -> None:
    """Prove features never use data newer than HORIZON days before the row.

    For random rows: rebuild the features from sales that STOP at (date - HORIZON)
    only, and confirm the lag/rolling values are identical to the full-data ones.
    """
    rng = np.random.default_rng(seed)
    raw_cols = ["date", "store", "item", "sales"]
    feat_cols = [c for c in df.columns if c.startswith(("lag_", "rollmean_", "rollstd_"))]
    rows = df.dropna(subset=["lag_364"]).sample(n_checks, random_state=seed)
    for _, r in rows.iterrows():
        cutoff = r["date"] - pd.Timedelta(days=HORIZON)
        hist = df[(df.store == r["store"]) & (df.item == r["item"]) & (df.date <= cutoff)][raw_cols]
        # add the target day with its sales hidden, then rebuild features
        target = pd.DataFrame({"date": [r["date"]], "store": [r["store"]],
                               "item": [r["item"]], "sales": [np.nan]})
        # fill the gap days with NaN so shifts line up on the calendar
        span = pd.date_range(hist["date"].min(), r["date"])
        full = (pd.concat([hist, target]).set_index("date").reindex(span)
                .rename_axis("date").reset_index())
        full["store"], full["item"] = r["store"], r["item"]
        rebuilt = build_features(full).iloc[-1]
        for c in feat_cols:
            assert np.isclose(rebuilt[c], r[c], equal_nan=True), f"LEAK in {c} at {r['date']}"
    print(f"Leakage check passed: {n_checks} random rows rebuilt from history "
          f"ending {HORIZON} days earlier gave identical features.")


if __name__ == "__main__":
    raw = load_sales()
    feats = build_features(raw)
    check_no_leakage(feats)

    # Drop the first rows of each series where the short lags/rollings don't exist yet.
    # lag_364 stays missing for the first year; XGBoost can handle that.
    core = ["lag_7", "lag_14", "lag_21", "lag_28", "rollmean_7", "rollmean_28", "rollstd_28"]
    before = len(feats)
    feats = feats.dropna(subset=core).reset_index(drop=True)
    print(f"\nRows before dropping warm-up days: {before:,}; after: {len(feats):,} "
          f"({before - len(feats):,} dropped, {(before - len(feats)) // 500} days per store-item)")
    print(f"Rows still missing lag_364 (first year only): {feats['lag_364'].isna().sum():,}")
    print(f"Columns ({len(feats.columns)}): {list(feats.columns)}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PATH, index=False)
    print(f"\nSaved {FEATURES_PATH.relative_to(FEATURES_PATH.parents[2])}")
    num = feats.drop(columns=["date"]).select_dtypes("number")
    corr = num.corr()["sales"].drop("sales").sort_values(key=abs, ascending=False)
    print("\nCorrelation of each feature with sales (closer to +1 or -1 = more related):")
    print(corr.round(3).to_string())
    print("\nSample rows:")
    print(feats[(feats.store == 1) & (feats.item == 1)].iloc[:3, :14].to_string())
