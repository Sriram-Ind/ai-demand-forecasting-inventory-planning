"""Stage 6: Compare all models on the same test rows; make charts and a results table.

Run from the project root:
    python -m src.evaluate
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, FIGURES, ROOT
from src.evaluation import score

MODELS = {"Same as last week": "baseline", "XGBoost": "xgboost", "Prophet": "prophet"}
COLORS = {"Same as last week": "#8d8c85", "XGBoost": "#2a78d6", "Prophet": "#eb6834", "Actual": "#0b0b0b"}
INK2, GRID = "#52514e", "#ececea"

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "font.size": 10,
})
KEYS = ["date", "store", "item"]


def load_all() -> pd.DataFrame:
    """One table: actual sales + one prediction column per model, same rows for all."""
    out = None
    for name, f in MODELS.items():
        d = pd.read_parquet(DATA_PROCESSED / f"pred_{f}.parquet").rename(columns={"pred": name})
        out = d if out is None else out.merge(d[KEYS + [name]], on=KEYS, how="inner", validate="1:1")
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    df = load_all()
    n = len(df)
    print(f"Rows compared (identical for all models): {n:,}; store-items: {df.groupby(['store','item']).ngroups}; "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    # ---- 1. Results table ----
    rows = {m: score(df["sales"], df[m]) for m in MODELS}
    table = pd.DataFrame(rows).T
    base = table.loc["Same as last week"]
    table["MAE improvement vs baseline"] = ((base["MAE"] - table["MAE"]) / base["MAE"] * 100).round(1)
    table["Bias (avg pred - actual)"] = [float((df[m] - df["sales"]).mean()) for m in MODELS]
    table = table.round(2)
    print("\n== RESULTS TABLE ==")
    print(table.to_string())
    table.to_csv(ROOT / "reports" / "results_table.csv")

    # ---- 2. Does the winner win broadly, or only on average? ----
    pair_mae = {m: df.assign(e=(df["sales"] - df[m]).abs()).groupby(["store", "item"])["e"].mean() for m in MODELS}
    pm = pd.DataFrame(pair_mae)
    print("\n== HOW BROADLY DOES EACH MODEL WIN? (per store-item MAE) ==")
    print(f"XGBoost beats baseline on {(pm['XGBoost'] < pm['Same as last week']).mean():.1%} of the 500 store-items")
    print(f"Prophet beats baseline on {(pm['Prophet'] < pm['Same as last week']).mean():.1%} of the 500 store-items")
    print(f"XGBoost beats Prophet on  {(pm['XGBoost'] < pm['Prophet']).mean():.1%} of the 500 store-items")
    print(f"Best model per store-item: {pm.idxmin(axis=1).value_counts().to_dict()}")

    # ---- 3. By month ----
    df["month"] = df["date"].dt.month_name().str[:3]
    bym = pd.DataFrame({m: df.groupby("month").apply(lambda g, m=m: (g["sales"] - g[m]).abs().mean(), include_groups=False)
                        for m in MODELS}).reindex(["Oct", "Nov", "Dec"]).round(2)
    print("\nMAE by month of test period:\n", bym.to_string())
    dec1 = df[df["date"] <= "2017-12-03"]
    dec1 = dec1[dec1["date"] >= "2017-12-01"]
    print("\nMAE on 1-3 Dec (the baseline's worst days):",
          {m: round(float((dec1["sales"] - dec1[m]).abs().mean()), 2) for m in MODELS})

    # ---- Chart 06: metric comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE_%"]):
        vals = table[metric]
        ax.bar(range(3), vals.values, color=[COLORS[m] for m in vals.index], width=0.65)
        ax.set_xticks(range(3), ["Last week", "XGBoost", "Prophet"])
        for i, v in enumerate(vals.values):
            ax.text(i, v * 1.015, f"{v:.1f}" if metric == "MAPE_%" else f"{v:.2f}", ha="center", fontsize=9, color=INK2)
        ax.set_ylim(0, vals.max() * 1.15)
        ax.set_title(metric.replace("_%", " (%)") + " (lower is better)", fontsize=11)
        ax.grid(axis="x", visible=False)
    save(fig, "06_model_comparison.png")

    # ---- Chart 07: total daily demand, actual vs predicted ----
    daily = df.groupby("date")[["sales"] + list(MODELS)].sum()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily["sales"], color=COLORS["Actual"], lw=2.2, label="Actual")
    for m in MODELS:
        ax.plot(daily.index, daily[m], color=COLORS[m], lw=1.4, label=m)
    ax.set_title("Total daily demand across all 500 store-items: predicted vs actual")
    ax.set_ylabel("Units per day")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(top=daily[["sales"] + list(MODELS)].max().max() * 1.12)
    save(fig, "07_total_predicted_vs_actual.png")

    # ---- Chart 08: one store-item ----
    s, i = 2, 15  # highest-selling store and item from EDA
    one = df[(df.store == s) & (df.item == i)].set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(one.index, one["sales"], color=COLORS["Actual"], lw=2.2, label="Actual")
    for m in MODELS:
        ax.plot(one.index, one[m], color=COLORS[m], lw=1.4, label=m)
    ax.set_title(f"One store-item up close: store {s}, item {i}")
    ax.set_ylabel("Units per day")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(top=one[["sales"] + list(MODELS)].max().max() * 1.15)
    save(fig, "08_single_store_item_forecast.png")
    print(f"\nStore {s}, item {i} MAE:", {m: round(float((one['sales'] - one[m]).abs().mean()), 2) for m in MODELS})

    # ---- Chart 09: scatter predicted vs actual for XGBoost ----
    samp = df.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(samp["sales"], samp["XGBoost"], s=5, alpha=0.25, color=COLORS["XGBoost"], linewidths=0)
    lim = max(samp["sales"].max(), samp["XGBoost"].max())
    ax.plot([0, lim], [0, lim], color=COLORS["Actual"], lw=1.2, ls="--")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("XGBoost prediction")
    ax.set_title("XGBoost: predicted vs actual")
    ax.text(lim * 0.05, lim * 0.93, "dashed line = perfect forecast", fontsize=8.5, color=INK2)
    save(fig, "09_xgboost_scatter.png")
    print("Saved charts 06-09 and reports/results_table.csv")
