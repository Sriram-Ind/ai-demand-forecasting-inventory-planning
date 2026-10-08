"""Stage 2: Exploratory data analysis (EDA).

Prints key numbers and saves charts to reports/figures/.
Run from the project root:
    python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES
from src.data_loader import load_sales

# Colours from the dataviz reference palette (light mode)
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#c9c8c2", "#0b0b0b", "#52514e"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved reports/figures/{name}")


def data_quality(df):
    print("\n== DATA QUALITY ==")
    print("Missing values per column:\n", df.isna().sum().to_string())
    per_pair = df.groupby(["store", "item"]).size()
    print(f"Days per store-item: min {per_pair.min()}, max {per_pair.max()} "
          f"(full calendar = {df['date'].nunique()})")
    print(f"Days with zero sales: {(df['sales'] == 0).sum()} rows "
          f"({(df['sales'] == 0).mean():.3%})")
    print(f"Negative sales: {(df['sales'] < 0).sum()}")
    # Outliers: anything more than 3 standard deviations above that pair's own mean
    g = df.groupby(["store", "item"])["sales"]
    z = (df["sales"] - g.transform("mean")) / g.transform("std")
    print(f"Spikes > 3 std above own mean: {(z > 3).sum()} rows ({(z > 3).mean():.3%})")


def trend(df):
    daily = df.groupby("date")["sales"].sum()
    monthly = daily.resample("MS").mean()
    yearly = df.groupby(df["date"].dt.year)["sales"].sum()
    print("\n== OVERALL TREND (all stores and items) ==")
    print("Total units sold per year:\n", yearly.to_string())
    print("Year-on-year growth:\n", (yearly.pct_change() * 100).round(1).dropna().to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, color=GREY, lw=0.8, label="Daily total")
    ax.plot(monthly.index, monthly.values, color=BLUE, lw=2, label="Monthly average")
    ax.set_title("Total daily sales across all stores and items")
    ax.set_ylabel("Units sold per day")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "01_total_sales_trend.png")


def weekly(df):
    dow = df.groupby(df["date"].dt.dayofweek)["sales"].mean()
    print("\n== WEEKLY PATTERN (avg units per store-item per day) ==")
    print(pd.Series(dow.values, index=DOW).round(1).to_string())
    print(f"Busiest day vs quietest day: {dow.max() / dow.min() - 1:.1%} higher")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ORANGE if v == dow.max() else BLUE for v in dow.values]
    ax.bar(DOW, dow.values, color=colors, width=0.6)
    ax.set_ylim(0, dow.max() * 1.12)
    for i, v in enumerate(dow.values):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_title("Average sales by day of week")
    ax.set_ylabel("Units per store-item per day")
    ax.grid(axis="x", visible=False)
    save(fig, "02_weekly_seasonality.png")


def yearly(df):
    month = df.groupby(df["date"].dt.month)["sales"].mean()
    print("\n== YEARLY PATTERN (avg units per store-item per day) ==")
    print(pd.Series(month.values, index=MONTHS).round(1).to_string())
    print(f"Peak month {MONTHS[month.idxmax() - 1]} vs lowest {MONTHS[month.idxmin() - 1]}: "
          f"{month.max() / month.min() - 1:.1%} higher")

    # One line per year so we can see the same shape repeats while the level climbs
    ym = df.groupby([df["date"].dt.year, df["date"].dt.month])["sales"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    shades = ["#c6d8f0", "#9fbfe8", "#6fa0dd", "#3f84d9", BLUE]
    for col, shade in zip(ym.columns, shades):
        ax.plot(MONTHS, ym[col].values, color=shade, lw=2)
        ax.text(11.15, ym[col].values[-1], str(col), color=INK2, fontsize=9, va="center")
    ax.set_xlim(-0.2, 12)
    ax.set_title("Yearly seasonality: same shape every year, higher each year")
    ax.set_ylabel("Units per store-item per day")
    save(fig, "03_yearly_seasonality.png")


def stores_items(df):
    store = df.groupby("store")["sales"].mean().sort_values()
    item = df.groupby("item")["sales"].mean().sort_values()
    print("\n== STORE DIFFERENCES (avg units per item per day) ==")
    print(store.round(1).to_string())
    print(f"Best store vs worst store: {store.max() / store.min() - 1:.1%} higher")
    print("\n== ITEM DIFFERENCES ==")
    print(f"Top 3 items:\n{item.tail(3).round(1)[::-1].to_string()}")
    print(f"Bottom 3 items:\n{item.head(3).round(1).to_string()}")
    print(f"Best item vs worst item: {item.max() / item.min():.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].barh(store.index.astype(str), store.values, color=BLUE, height=0.6)
    axes[0].set_yticks(range(len(store)), [f"Store {s}" for s in store.index])
    axes[0].set_title("Stores")
    axes[0].set_xlabel("Avg units per item per day")
    axes[1].bar(range(len(item)), item.values, color=BLUE, width=0.7)
    axes[1].set_xticks(range(len(item)), item.index, fontsize=7, rotation=90)
    axes[1].set_title("Items (sorted)")
    axes[1].set_xlabel("Item id")
    axes[1].set_ylabel("Avg units per store per day")
    for a in axes:
        a.grid(visible=False, axis="y" if a is axes[0] else "x")
    save(fig, "04_store_item_differences.png")


def sample_pair(df, store=1, item=1):
    s = df[(df.store == store) & (df.item == item)].set_index("date")["sales"]
    print(f"\n== ONE STORE-ITEM EXAMPLE (store {store}, item {item}) ==")
    print(s.describe().round(1).to_string())
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s.index, s.values, color=GREY, lw=0.8, label="Daily")
    ax.plot(s.index, s.rolling(30, center=True).mean(), color=BLUE, lw=2, label="30-day average")
    ax.set_title(f"What one forecast target looks like: store {store}, item {item}")
    ax.set_ylabel("Units sold per day")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "05_single_store_item.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_sales()
    data_quality(data)
    trend(data)
    weekly(data)
    yearly(data)
    stores_items(data)
    sample_pair(data)
