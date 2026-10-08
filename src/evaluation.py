"""Shared train/test split and error metrics, used by every model.

Keeping these in one place guarantees all models are judged on the SAME test
period with the SAME formulas.
"""
import numpy as np
import pandas as pd

# Time-based split: train on the past, test on the final ~3 months.
# (A random split would let the model train on days AFTER the ones it is tested on.)
TEST_START = pd.Timestamp("2017-10-01")


def time_split(df: pd.DataFrame):
    """Return (train, test): everything before TEST_START, and everything from it on."""
    return df[df["date"] < TEST_START].copy(), df[df["date"] >= TEST_START].copy()


def mae(y, p):
    """Mean Absolute Error: average miss, in units."""
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    """Root Mean Squared Error: like MAE but big misses count extra."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    """Mean Absolute Percentage Error: average miss as a % of actual sales.
    Days with zero actual sales are skipped (can't divide by zero)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def score(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE_%": mape(y, p)}
