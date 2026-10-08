"""Stage 8: Streamlit dashboard.

Pick a store and item, see the forecast vs actual, the suggested order for a
chosen week, and how accurate each model was.

Run from the project root (after the pipeline scripts have created the data files):
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src` works

from src.config import DATA_PROCESSED, TRAIN_CSV  # noqa: E402
from src.evaluation import TEST_START, score  # noqa: E402

MODELS = ["XGBoost", "Prophet", "Same as last week"]
COLORS = {"Actual": "#0b0b0b", "XGBoost": "#2a78d6", "Prophet": "#eb6834", "Same as last week": "#8d8c85"}
FILES = {"XGBoost": "xgboost", "Prophet": "prophet", "Same as last week": "baseline"}

st.set_page_config(page_title="Demand forecasting & inventory planning", layout="wide")


@st.cache_data
def load_data():
    """Actual sales, test-period predictions for all models, and the weekly inventory simulation."""
    for p in [TRAIN_CSV, DATA_PROCESSED / "pred_xgboost.parquet", DATA_PROCESSED / "inventory_simulation.parquet"]:
        if not p.exists():
            st.error(f"Missing {p.name}. Run the pipeline scripts first (see README).")
            st.stop()
    hist = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    preds = None
    for name, f in FILES.items():
        d = pd.read_parquet(DATA_PROCESSED / f"pred_{f}.parquet").rename(columns={"pred": name})
        preds = d if preds is None else preds.merge(d[["date", "store", "item", name]], on=["date", "store", "item"])
    sim = pd.read_parquet(DATA_PROCESSED / "inventory_simulation.parquet")
    return hist, preds, sim


hist, preds, sim = load_data()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Choose what to look at")
store = st.sidebar.selectbox("Store", sorted(hist["store"].unique()))
item = st.sidebar.selectbox("Item", sorted(hist["item"].unique()))
model = st.sidebar.radio("Forecast model", ["XGBoost", "Same as last week"], help="Safety stock is available for these two models.")
service = st.sidebar.slider("Target service level", 0.90, 0.99, 0.95, 0.01, format="%.2f",
                            help="Share of weeks you want to get through without running out of stock.")
n_weeks = sim["week"].nunique()
week = st.sidebar.selectbox(
    "Ordering week", list(range(n_weeks)), index=0,
    format_func=lambda w: f"Week of {(TEST_START + pd.Timedelta(days=7 * w)).date()}")
on_hand = st.sidebar.number_input("Stock on hand at order time (units)", min_value=0, value=0, step=10)

z = float(norm.ppf(service))
st.title("AI-based demand forecasting & inventory planning")
st.caption(f"Store {store} · Item {item} · Backtest on the last 3 months of a public Kaggle dataset "
           f"(not real company data). Forecast horizon 7 days; lead time 7 days.")

# ---------------- Suggested order ----------------
row = sim[(sim.model == model) & (sim.store == store) & (sim.item == item) & (sim.week == week)].iloc[0]
safety = z * row["sigma"]
order_up_to = row["forecast"] + safety
order_qty = max(0.0, order_up_to - on_hand)
week_start = TEST_START + pd.Timedelta(days=7 * week)

st.subheader(f"Order decision for the week of {week_start.date()}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast demand (7 days)", f"{row['forecast']:.0f} units")
c2.metric(f"Safety stock ({service:.0%})", f"{safety:.0f} units")
c3.metric("Reorder point / order-up-to", f"{order_up_to:.0f} units")
c4.metric("Suggested order", f"{order_qty:.0f} units", help="Order-up-to level minus stock on hand.")
actual_week = row["actual"]
verdict = "enough stock" if order_up_to >= actual_week else "a stockout would have happened"
st.caption(f"What actually happened that week: {actual_week:.0f} units sold, so this plan had {verdict} "
           f"(short by {max(0, actual_week - order_up_to):.0f}, left over {max(0, order_up_to - actual_week):.0f}) "
           f"if the week started with the order-up-to level.")

# ---------------- Forecast chart ----------------
st.subheader("Forecast vs actual")
pair_hist = hist[(hist.store == store) & (hist.item == item)]
pair_pred = preds[(preds.store == store) & (preds.item == item)].sort_values("date")
context = pair_hist[(pair_hist["date"] < TEST_START) & (pair_hist["date"] >= TEST_START - pd.Timedelta(days=60))]

fig = go.Figure()
fig.add_trace(go.Scatter(x=context["date"], y=context["sales"], name="Actual (history)",
                         line=dict(color=COLORS["Actual"], width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=pair_pred["date"], y=pair_pred["sales"], name="Actual (test period)",
                         line=dict(color=COLORS["Actual"], width=2.5)))
for m in MODELS:
    fig.add_trace(go.Scatter(x=pair_pred["date"], y=pair_pred[m], name=m,
                             line=dict(color=COLORS[m], width=2 if m == model else 1.2),
                             opacity=1 if m == model else 0.6))
fig.add_vrect(x0=week_start, x1=week_start + pd.Timedelta(days=7), fillcolor="#2a78d6", opacity=0.08,
              line_width=0, annotation_text="selected week", annotation_position="top left")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  yaxis_title="Units per day", legend=dict(orientation="h", y=-0.15),
                  plot_bgcolor="white")
fig.update_yaxes(gridcolor="#ececea")
st.plotly_chart(fig, width="stretch")

# ---------------- Accuracy ----------------
st.subheader("Model accuracy on the test period (Oct 1 - Dec 31, 2017)")
left, right = st.columns(2)
this_pair = pd.DataFrame({m: score(pair_pred["sales"], pair_pred[m]) for m in MODELS}).T
all_pairs = pd.DataFrame({m: score(preds["sales"], preds[m]) for m in MODELS}).T
left.markdown(f"**This store-item** (store {store}, item {item})")
left.dataframe(this_pair.round(2).rename(columns={"MAPE_%": "MAPE %"}), width="stretch")
right.markdown("**All 500 store-items**")
right.dataframe(all_pairs.round(2).rename(columns={"MAPE_%": "MAPE %"}), width="stretch")
st.caption("MAE = average miss in units. RMSE = like MAE but big misses count extra. MAPE = average miss as % of actual sales. "
           "Prophet forecast all 92 days blind, so it is not a like-for-like comparison with the 7-day-ahead models.")

# ---------------- Week-by-week table ----------------
with st.expander("Week-by-week plan for this store-item"):
    t = sim[(sim.model == model) & (sim.store == store) & (sim.item == item)].copy()
    t["week_start"] = [(TEST_START + pd.Timedelta(days=7 * w)).date() for w in t["week"]]
    t["safety_stock"] = z * t["sigma"]
    t["order_up_to"] = t["forecast"] + t["safety_stock"]
    t["stockout?"] = t["actual"] > t["order_up_to"]
    st.dataframe(t[["week_start", "forecast", "safety_stock", "order_up_to", "actual", "stockout?"]].round(0),
                 width="stretch", hide_index=True)
