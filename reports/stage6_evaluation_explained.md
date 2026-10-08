# Stage 6: Which model won, and why (plain-English version)

All three models were scored on the identical 46,000 test rows
(500 store-items x 92 days, 2017-10-01 to 2017-12-31). Run: `python -m src.evaluate`.

| Model | MAE | RMSE | MAPE | MAE better than baseline | Bias |
|---|---|---|---|---|---|
| Same as last week | 9.08 | 12.05 | 19.9% | 0% | +1.33 |
| XGBoost | 5.92 | 7.67 | 13.0% | 34.7% | -0.06 |
| Prophet | 6.36 | 8.30 | 13.8% | 30.0% | -0.16 |

**Bias** = average (prediction - actual). Positive = over-forecasting (excess stock),
negative = under-forecasting (stockouts). The baseline over-predicts by 1.33 units
on average; both ML models are nearly neutral.

## Winner: XGBoost
- Lowest error on all three metrics: 34.7% lower MAE than the baseline.
- It wins broadly, not just on average: it beats the baseline on all 500 store-items,
  and beats Prophet on 88.6% of them (443 store-items best with XGBoost, 57 with Prophet).
- By month (MAE) XGBoost / Prophet / baseline: Oct 6.07 / 6.17 / 8.52,
  Nov 6.30 / 6.92 / 8.83, Dec 5.42 / 5.99 / 9.87.
- On 1-3 Dec (the baseline's worst days): baseline 20.41, Prophet 9.09, XGBoost 5.68.

## Why XGBoost won
- It sees recent sales (lag_7 and the 7-day average did ~67% of the work) AND the
  calendar, so it adapts when the level shifts, like the late-November to December drop.
- The baseline only copies one past value, including that day's random noise.
- Prophet only knows long-run trend and seasonality; it can't react to recent changes.

## The fairness caveat (important)
XGBoost predicts 7 days ahead with fresh recent sales. Prophet forecast all 92 days
from a single cut-off date with no new sales. A fair claim is: "Prophet, which looks
far ahead without recent data, got within 0.44 units of XGBoost's MAE", not
"XGBoost is better than Prophet at the same task". The XGBoost vs baseline comparison
is fair (same horizon, same information).

## Charts (reports/figures/)
- 06 metrics side by side. 07 total daily demand, actual vs predicted: XGBoost
  hugs the actual line; the baseline lags a week (see the early-December bump).
  08 one store-item (store 2, item 15, the biggest seller): MAE 8.00 XGBoost,
  9.19 Prophet, 12.76 baseline. 09 predicted vs actual scatter; closer to the dashed line = better.

## Limits
- One test quarter (Q4 2017). Different seasons could rank models differently.
- Errors are not zero: XGBoost still misses by about 6 units a day, or 13%. That
  leftover uncertainty is what safety stock covers in Stage 7.
- Public dataset, no promotions/prices/stockouts, so real-world accuracy would differ.
