"""Stage 4: Baseline model, "same as last week".

Prediction for a day = actual sales on the same weekday one week earlier.
Any smarter model must beat this to be worth the effort.

Run from the project root:
    python -m src.baseline
"""
import pandas as pd

from src.config import DATA_PROCESSED
from src.data_loader import load_sales
from src.evaluation import TEST_START, mae, mape, rmse, score, time_split
from src.features import FEATURES_PATH

PRED_PATH = DATA_PROCESSED / "pred_baseline.parquet"


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    """'Same as last week' = the lag_7 column (sales 7 days earlier)."""
    return df["lag_7"]


if __name__ == "__main__":
    feats = pd.read_parquet(FEATURES_PATH)
    train, test = time_split(feats)
    print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows)")
    print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test):,} rows, "
          f"{test['date'].nunique()} days)")

    # Independent check: rebuild "last week" straight from the raw file, not from features
    raw = load_sales().set_index(["store", "item", "date"])["sales"]
    idx = pd.MultiIndex.from_arrays([test["store"], test["item"], test["date"] - pd.Timedelta(days=7)])
    assert (raw.reindex(idx).to_numpy() == test["lag_7"].to_numpy()).all(), "lag_7 mismatch!"
    print("Check passed: baseline equals actual sales 7 days earlier from the raw file.")

    test["pred"] = baseline_predict(test)
    test[["date", "store", "item", "sales", "pred"]].to_parquet(PRED_PATH, index=False)

    s = score(test["sales"], test["pred"])
    print("\n== BASELINE RESULTS (all 500 store-items, test period) ==")
    print(f"MAE   : {s['MAE']:.2f} units")
    print(f"RMSE  : {s['RMSE']:.2f} units")
    print(f"MAPE  : {s['MAPE_%']:.1f} %")
    print(f"(average actual sales in test period: {test['sales'].mean():.1f} units/day)")

    print("\nMAE by weekday of the day being predicted:")
    test["dow"] = test["date"].dt.day_name().str[:3]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(test.groupby("dow").apply(lambda g: mae(g["sales"], g["pred"]), include_groups=False)
          .reindex(order).round(2).to_string())

    # Is the error bigger for high-volume items? (percentage error says no, unit error says yes)
    pair = test.groupby(["store", "item"]).apply(
        lambda g: pd.Series({"avg_sales": g["sales"].mean(), "MAE": mae(g["sales"], g["pred"]),
                             "MAPE": mape(g["sales"], g["pred"])}), include_groups=False)
    print("\nAcross the 500 store-items:")
    print(f"  MAE  range {pair['MAE'].min():.2f} to {pair['MAE'].max():.2f} (median {pair['MAE'].median():.2f})")
    print(f"  MAPE range {pair['MAPE'].min():.1f}% to {pair['MAPE'].max():.1f}% (median {pair['MAPE'].median():.1f}%)")
    print(f"  Correlation of avg sales with MAE:  {pair['avg_sales'].corr(pair['MAE']):.2f}")
    print(f"  Correlation of avg sales with MAPE: {pair['avg_sales'].corr(pair['MAPE']):.2f}")
    print("\nWorst-forecast days (by units missed, summed over all store-items):")
    by_day = test.groupby("date").apply(lambda g: (g["sales"] - g["pred"]).abs().sum(), include_groups=False)
    print(by_day.sort_values(ascending=False).head(3).astype(int).to_string())
