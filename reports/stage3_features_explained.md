# Stage 3: Feature engineering (plain-English version)

**Feature** = one clue column the model uses to guess tomorrow's sales.
**Feature engineering** = creating good clues from the raw table (which only has
date, store, item, sales). Output: `data/processed/features.parquet`,
896,000 rows x 26 columns (built by `python -m src.features`).

## The rule everything follows: forecast horizon = 7 days
We forecast up to 7 days ahead, like a weekly ordering cycle. So every clue built
from past sales is at least 7 days old. Using yesterday's sales would be
**data leakage**: the model would get a clue it won't have in real life, scores
look great, and the project falls apart in real use. (This is a decision I made
as a default; it can be changed, and Stage 7 will tie it to ordering lead time.)

## The clues
**Date features** (no sales history needed)
- dayofweek, is_weekend, day, month, quarter, year, dayofyear, weekofyear,
  is_month_start / is_month_end. Why: EDA showed Sunday is 50% above Monday and
  July is 89% above January, so the calendar is a strong clue.
- Sine/cosine of month and weekday: a trick so the model knows December is next
  to January (as plain numbers 12 and 1 look far apart).

**Lag features** = sales N days ago for the same store-item.
- lag_7, lag_14, lag_21, lag_28: same weekday 1-4 weeks back. Why: recent
  similar days predict well.
- lag_364: same weekday one year ago (364 = exactly 52 weeks). Why: captures
  yearly seasonality. Missing for the first year of data (165,000 rows); XGBoost
  can handle gaps.

**Rolling averages** = average of a window of recent days.
- rollmean_7 and rollmean_28: average over the 7 / 28 days ending 7 days before
  the target. Why: smooths out day-to-day noise to show the current level.
- rollstd_28: how much sales have been swinging lately. Why: busy, unstable
  items are harder to forecast (this will help safety stock later).

**store, item**: lets one model learn that store 2 sells more than store 7.

## How I proved there's no leakage
For 200 random rows I rebuilt every lag/rolling feature using only sales up to
7 days before that row, and confirmed the values were identical to the full-data
ones. The check passed (it's in `check_no_leakage`).

## Real numbers
- 913,000 rows -> 896,000 after dropping the first 34 days of each store-item
  (the rolling windows need history first).
- Correlation with sales (+1/-1 = strongly related, 0 = unrelated):
  lag_364 0.932, lag_7 0.929, rollmean_7 0.918, rollmean_28 0.913, dayofweek 0.225.
- Honest note: the big correlations are partly because stores and items sell
  very different amounts (item 15 vs item 5), so any past-sales column lines up
  with today's. Real accuracy comes from Stage 5-6 on held-out data, not from this table.
