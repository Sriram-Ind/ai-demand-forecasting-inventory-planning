# Stage 5: XGBoost and Prophet (plain-English version)

Both models use the same time-based split as the baseline: train on data up to
2017-09-30, test on 2017-10-01..2017-12-31 (92 days, 46,000 rows, 500 store-items).

## XGBoost (`python -m src.train_xgboost`)
- **How it works:** XGBoost builds hundreds of small "decision trees" (think of
  yes/no flowcharts: "is it a weekend? was last week above 60? ..."). Each new
  tree is trained to fix the mistakes of the trees before it. This is called
  **gradient boosting**. One model learns all 500 store-items; store and item
  are just input columns, alongside the Stage 3 features.
- **Overfitting** = memorizing the training data instead of learning the pattern
  (great on old data, bad on new). Guards used: a validation period, early
  stopping, and each tree seeing only a random 80% of rows and columns.
- **Early stopping:** I held out Jul-Sep 2017 (the end of the training period)
  and kept adding trees until that error stopped improving. It stopped at 700
  trees (validation MAE 6.57). Then I refit on all training data with 700 trees.
  The test period was never used for any of these choices.
- **Result:** MAE 5.92, RMSE 7.67, MAPE 13.0% (baseline: 9.08 / 12.05 / 19.9%).
- **Feature importance** (how much each clue helped): lag_7 42.9%, rollmean_7
  23.7%, lag_14 9.7%, rollmean_28 6.5%, lag_364 3.1%. Recent sales do most of
  the work; calendar clues add the rest.

## Prophet (`python -m src.train_prophet`)
- **How it works:** Prophet (from Meta) fits one model per store-item, splitting
  sales into a trend + weekly pattern + yearly pattern + holidays. No lag features.
  I used 500 separate models, multiplicative seasonality (swings scale with
  the sales level), and US holidays (assumption: it's a US dataset; I don't know that).
- **Result (all 500):** MAE 6.36, RMSE 8.30, MAPE 13.8%. Took 45 seconds.

## Important: the comparison isn't perfectly fair
XGBoost forecasts 7 days ahead using fresh recent sales. Prophet was trained
once up to Sep 30 and forecasts all 92 days blind, never seeing newer sales.
So Prophet had the harder job. Stage 6 compares them in one table and I'll
repeat this caveat there.

## Limits
- One test quarter only. No hyperparameter search beyond sensible defaults.
- Prophet's holiday setting is a guess. XGBoost wasn't tuned beyond early stopping.
