# Stage 4: Baseline model, "same as last week" (plain-English version)

## What is a baseline and why bother?
A **baseline** is the simplest sensible forecast. It's the bar every fancy model
must clear. If XGBoost can't beat "just repeat last week", it isn't worth
deploying. Ours: predict a day's sales = sales on the same weekday 7 days earlier.
(The weekly pattern from EDA is why this isn't a silly guess.)

## The test setup (used for every model from here on)
- **Train** = 2013-02-04 to 2017-09-30 (850,000 rows). **Test** = 2017-10-01 to
  2017-12-31 (46,000 rows, 92 days, all 500 store-items).
- **Time-based split**: train on the past, test on the future. A random split
  would shuffle days, so the model could learn from days AFTER the ones it's
  tested on, like studying with tomorrow's exam answers. Real forecasting never
  has that luxury, so a random split makes scores look better than reality.
- The baseline doesn't learn anything, but it is scored on the exact same test
  period, so comparisons in Stage 6 are fair.

## The three error scores
- **MAE (Mean Absolute Error)**: on average, how many units are we off? Easy to
  read; in the same units as sales.
- **RMSE (Root Mean Squared Error)**: like MAE but squares each miss first, so
  a few big misses hurt a lot. RMSE much larger than MAE means some big misses.
- **MAPE (Mean Absolute Percentage Error)**: average miss as a % of that day's
  real sales. Good for comparing small and big items. Weakness: it blows up when
  actual sales are tiny (and can't handle zero; we skip zero-sales days, only 1 in
  the whole dataset).

## Results (real, from `python -m src.baseline`)
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Same as last week | 9.08 | 12.05 | 19.9% |

The average store-item sells 54.7 units/day in the test period, so we're off
by roughly 9 units, about a sixth of typical sales. Check passed: I rebuilt
"last week" straight from the raw file and it matched exactly.

## What else we learned
- Error by weekday: Monday 7.95 (best) up to Sunday 9.97 (worst). Busy days are
  harder.
- Across the 500 store-items, MAE ranges 4.05 to 15.67 units, and MAPE 10.9% to
  40.6%. Big sellers miss by more units (correlation 0.94) but by a smaller
  percentage (-0.85), because noise grows slower than volume.
- Worst days are 1-3 December, when the baseline misses by 9,300-10,700 units in
  total. Yearly seasonality showed sales drop sharply from November to December,
  and "last week" was still a November week, so it overshoots. A model that knows
  the calendar and last year's pattern should fix part of this.

## Limits
Single test period (one quarter) is a small sample: Q4 includes the December drop,
so the baseline may look better or worse than on another quarter.
