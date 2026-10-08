"""Stage 5b: Prophet model.

Prophet (by Meta) fits one model PER store-item: trend + weekly pattern +
yearly pattern + holidays. It does not use lag features.

IMPORTANT difference from XGBoost: Prophet is trained up to 2017-09-30 and
then forecasts all 92 test days in one go. It never sees the recent actuals
as the test period unfolds, so its job is harder (1 to 92 days ahead) than
XGBoost's (7 days ahead, with fresh lags). Comparing them is therefore not
perfectly apples-to-apples; we report that honestly.

Run from the project root:
    python -m src.train_prophet
"""
import logging
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

from src.config import DATA_PROCESSED
from src.data_loader import load_sales
from src.evaluation import TEST_START, score

PRED_PATH = DATA_PROCESSED / "pred_prophet.parquet"


def fit_one(store: int, item: int, g: pd.DataFrame, horizon_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Fit Prophet on one store-item's history and forecast the test dates."""
    from prophet import Prophet  # imported inside so each parallel worker loads it
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    hist = g.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode="multiplicative",  # seasonal swings grow as the level grows
    )
    m.add_country_holidays(country_name="US")  # the data is from a US Kaggle competition (assumed)
    m.fit(hist)
    fc = m.predict(pd.DataFrame({"ds": horizon_dates}))
    return pd.DataFrame({"date": fc["ds"], "store": store, "item": item,
                         "pred": fc["yhat"].clip(lower=0)})


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # optional: only first N pairs (for timing)
    raw = load_sales()
    train = raw[raw["date"] < TEST_START]
    test = raw[raw["date"] >= TEST_START]
    dates = pd.DatetimeIndex(sorted(test["date"].unique()))

    pairs = list(train.groupby(["store", "item"]))
    if limit:
        pairs = pairs[:limit]
    print(f"Fitting Prophet for {len(pairs)} store-items...")
    t0 = time.time()
    out = Parallel(n_jobs=-1)(delayed(fit_one)(s, i, g, dates) for (s, i), g in pairs)
    print(f"Done in {time.time() - t0:.0f}s")

    pred = pd.concat(out).merge(test, on=["date", "store", "item"])[["date", "store", "item", "sales", "pred"]]
    if not limit:
        pred.to_parquet(PRED_PATH, index=False)
    s = score(pred["sales"], pred["pred"])
    print(f"\n== PROPHET RESULTS ({pred.groupby(['store', 'item']).ngroups} store-items, test period) ==")
    print(f"MAE {s['MAE']:.2f} | RMSE {s['RMSE']:.2f} | MAPE {s['MAPE_%']:.1f}%")
