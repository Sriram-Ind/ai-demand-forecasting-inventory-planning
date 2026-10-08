# Stage 10: Resume bullets and 60-second pitch

Every number below comes from the real runs (see README and reports/results_table.csv).
Say "backtest on a public Kaggle dataset" if asked; never present these as savings at a real company.

## Resume bullets (pick 2-3)

**Project: AI-Based Demand Forecasting & Inventory Planning** | Python, pandas, XGBoost, Prophet, Streamlit | GitHub

- Built an end-to-end demand forecasting pipeline on 913K rows of daily retail sales (10 stores x 50 items, 500 series): engineered calendar, lag and rolling-average features, verified no data leakage, and validated on a time-based hold-out of the final 92 days.
- Reduced forecast error 34.7% versus a "same as last week" baseline (MAE 9.08 to 5.92 units; MAPE 19.9% to 13.0%) using XGBoost, which beat the baseline on all 500 store-items; benchmarked against Prophet (MAE 6.36).
- Converted forecasts into inventory decisions (safety stock and reorder level at a 95% service target); in a 13-week backtest the XGBoost forecast needed 32% less safety stock and left 42% less end-of-week stock, or cut stockout weeks from 3.0% to 1.0% at equal safety stock. Delivered an interactive Streamlit dashboard for store-item planning.

Shorter one-line version: *Built a demand-forecasting and inventory-planning tool (XGBoost, Prophet, Streamlit) that cut forecast error 35% vs a naive baseline and translated it into safety-stock and reorder recommendations in a 13-week backtest.*

## 60-second interview explanation (about 150 words)

"I wanted a project that connects data science to a real operations decision, so I built a demand forecasting and inventory planning tool. I used a public Kaggle dataset: five years of daily sales for 50 items across 10 stores, so 500 separate demand series.

First I set a baseline, just repeating last week's sales, so I'd know what a model has to beat. Then I built features like sales a week ago, rolling averages and calendar effects, and trained XGBoost, testing on the final three months using a time-based split so the model never saw the future. It cut forecast error by about 35% against the baseline.

The part I care about most is the business layer. I turned forecast error into safety stock and reorder levels. In a 13-week simulation the better forecast needed about a third less safety stock for the same service level. It's a public dataset with no costs, so the next step is adding real holding and stockout costs."

## Likely questions, with honest answers

- **Why a time-based split?** Random splitting lets the model train on days after the ones it's tested on, which inflates accuracy. Real forecasting only has the past.
- **Why the baseline?** Without it, "MAPE 13%" means nothing. The baseline showed 13% is a real improvement over 19.9%.
- **What is safety stock?** An extra cushion sized from forecast error: z x typical weekly error, where z comes from the service level (1.645 for 95%).
- **What surprised you?** With no safety stock, XGBoost had *more* stockout weeks than the baseline (49% vs 42.5%), because the baseline over-forecast by ~1.3 units/day, acting as an accidental cushion. XGBoost was unbiased, so it needs safety stock to protect service.
- **Is Prophet worse?** Not exactly: it forecast 92 days blind while XGBoost forecast 7 days ahead, so the comparison isn't like-for-like. It even beat XGBoost on some store-items (about 11%).
- **Limitations?** Public, apparently synthetic data; sales is not true demand; one test quarter; simplified inventory model with no costs or supplier delays.
- **What next?** Probabilistic (quantile) forecasts, cost-based optimisation of service level, time-series cross-validation, real data with promotions and prices.
