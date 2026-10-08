"""Stage 7: Turn forecasts into inventory decisions.

Setup (defaults, change at the top):
  * Orders are placed weekly. LEAD_TIME_DAYS = 7: an order placed at the end of
    one week arrives for the start of the next week.
  * Each week we need enough stock to cover that week's demand:
        order-up-to level = forecast for the week + safety stock
  * Safety stock = z * (typical forecast error over the week), where z comes
    from the target service level (95% -> z = 1.645).
  * The typical error is measured on a VALIDATION period (Jul-Sep 2017) that comes
    before the test period, so we never use the test answers to set stock levels.

Run from the project root:
    python -m src.inventory
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from src.config import DATA_PROCESSED, MODELS, ROOT
from src.evaluation import TEST_START, time_split
from src.features import FEATURES_PATH
from src.train_xgboost import NOT_FEATURES, PARAMS, VALID_DAYS

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
Z = float(norm.ppf(SERVICE_LEVEL))
MODEL_NAMES = {"Same as last week": "baseline", "XGBoost": "xgboost"}


def validation_predictions() -> pd.DataFrame:
    """Out-of-sample predictions for Jul-Sep 2017 (model trained only on data before July)."""
    import json
    feats = pd.read_parquet(FEATURES_PATH)
    train, _ = time_split(feats)
    cut = train["date"].max() - pd.Timedelta(days=VALID_DAYS - 1)
    fit, valid = train[train["date"] < cut].copy(), train[train["date"] >= cut].copy()
    cols = [c for c in feats.columns if c not in NOT_FEATURES]
    n_trees = json.load(open(MODELS / "xgboost_meta.json"))["n_trees"]
    m = xgb.XGBRegressor(**{**PARAMS, "n_estimators": n_trees})
    m.fit(fit[cols], fit["sales"])
    valid["XGBoost"] = m.predict(valid[cols]).clip(min=0)
    valid["Same as last week"] = valid["lag_7"]
    return valid[["date", "store", "item", "sales", "XGBoost", "Same as last week"]]


def weekly_sigma(valid: pd.DataFrame, model: str) -> pd.Series:
    """Typical error of a WEEKLY forecast, per store-item, measured on validation.

    Uses every 7-day window (overlapping) of summed forecast error, then takes the
    root-mean-square. RMS (not plain std) so a consistent over/under-forecast counts too.
    """
    v = valid.sort_values(["store", "item", "date"]).copy()
    v["err"] = v["sales"] - v[model]
    wk = v.groupby(["store", "item"])["err"].transform(lambda s: s.rolling(LEAD_TIME_DAYS).sum())
    v["wk_err_sq"] = wk ** 2
    return np.sqrt(v.groupby(["store", "item"])["wk_err_sq"].mean())


def simulate(test: pd.DataFrame, valid: pd.DataFrame, z: float = Z) -> pd.DataFrame:
    """Weekly order-up-to simulation. Returns one row per (model, store, item, week)."""
    out = []
    for model in MODEL_NAMES:
        sigma = weekly_sigma(valid, model).rename("sigma")
        wk = (test.assign(week=(test["date"] - TEST_START).dt.days // 7)
              .groupby(["store", "item", "week"])
              .agg(actual=("sales", "sum"), forecast=(model, "sum"), days=("sales", "size")).reset_index())
        wk = wk[wk["days"] == 7].merge(sigma.reset_index(), on=["store", "item"])
        wk["model"] = model
        wk["safety_stock"] = z * wk["sigma"]
        out.append(wk)
    return pd.concat(out, ignore_index=True)


def summarize(sim: pd.DataFrame, safety_col: str = "safety_stock") -> pd.DataFrame:
    sim = sim.copy()
    sim["order_up_to"] = sim["forecast"] + sim[safety_col]
    sim["short"] = (sim["actual"] - sim["order_up_to"]).clip(lower=0)
    sim["left_over"] = (sim["order_up_to"] - sim["actual"]).clip(lower=0)
    g = sim.groupby("model")
    return pd.DataFrame({
        "weeks_with_stockout_%": g["short"].apply(lambda s: (s > 0).mean() * 100),
        "service_level_%": 100 - g["short"].apply(lambda s: (s > 0).mean() * 100),
        "fill_rate_%": (1 - g["short"].sum() / g["actual"].sum()) * 100,
        "units_short_per_week": g["short"].mean(),
        "avg_safety_stock": g[safety_col].mean(),
        "avg_left_over_stock": g["left_over"].mean(),
        "avg_order_up_to": g["order_up_to"].mean(),
    }).round(2)


def make_chart(a: pd.DataFrame, c: pd.DataFrame) -> None:
    """Bar chart: stock held (A) and stockout weeks at equal safety stock (C)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.config import FIGURES

    cols = {"Same as last week": "#8d8c85", "XGBoost": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [
        (axes[0], a["avg_safety_stock"], "Safety stock needed for ~95% service\n(units per store-item per week, lower = cheaper)", "{:.1f}"),
        (axes[1], c["weeks_with_stockout_%"], "Same safety stock for both:\nweeks with a stockout (%, lower = better)", "{:.2f}%"),
    ]
    for ax, vals, title, fmt in panels:
        ax.bar(vals.index, vals.values, color=[cols[m] for m in vals.index], width=0.55)
        for i, v in enumerate(vals.values):
            ax.text(i, v * 1.02, fmt.format(v), ha="center", fontsize=10, color="#52514e")
        ax.set_ylim(0, vals.max() * 1.18)
        ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "10_inventory_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    test = pd.read_parquet(DATA_PROCESSED / "pred_baseline.parquet").rename(columns={"pred": "Same as last week"})
    test = test.merge(pd.read_parquet(DATA_PROCESSED / "pred_xgboost.parquet").rename(columns={"pred": "XGBoost"}),
                      on=["date", "store", "item", "sales"])
    valid = validation_predictions()
    print(f"Target service level {SERVICE_LEVEL:.0%} (z = {Z:.3f}); lead time {LEAD_TIME_DAYS} days")

    sim = simulate(test, valid)
    n_weeks = sim["week"].nunique()
    print(f"Simulated {n_weeks} full weeks x 500 store-items per model = {len(sim) // 2:,} order decisions each\n")

    # A) each model sizes its OWN safety stock for a 95% service level
    a = summarize(sim)
    print("A) Each model uses its own error to size safety stock (target 95% of weeks without stockout)")
    print(a.T.to_string())

    # B) no safety stock at all: pure forecast
    b = summarize(sim.assign(zero=0.0), "zero")
    print("\nB) No safety stock (order exactly the forecast)")
    print(b[["weeks_with_stockout_%", "fill_rate_%", "units_short_per_week"]].T.to_string())

    # C) same safety stock units for both (the baseline's, since it is the larger) -> pure forecast-quality effect
    base_ss = sim[sim["model"] == "Same as last week"][["store", "item", "week", "safety_stock"]].rename(
        columns={"safety_stock": "ss_base"})
    c_in = sim.merge(base_ss, on=["store", "item", "week"])
    c = summarize(c_in, "ss_base")
    print("\nC) Same safety stock for both (the baseline's amount): which forecast leaves fewer stockouts?")
    print(c[["weeks_with_stockout_%", "fill_rate_%", "units_short_per_week", "avg_left_over_stock"]].T.to_string())

    # Sample table: order decision for the first test week, 10 store-items
    first = sim[(sim["week"] == 0) & (sim["model"] == "XGBoost")].copy()
    sample = first.sort_values(["store", "item"]).iloc[::50].head(10)
    sample["reorder_point"] = (sample["forecast"] + sample["safety_stock"]).round(0)
    sample = sample[["store", "item", "forecast", "safety_stock", "reorder_point", "actual"]].round(0).astype(int)
    print("\nSample order decisions for the week starting", TEST_START.date(), "(XGBoost):")
    print(sample.to_string(index=False))

    make_chart(a, c)
    sim.to_parquet(DATA_PROCESSED / "inventory_simulation.parquet", index=False)
    a.to_csv(ROOT / "reports" / "inventory_results.csv")
