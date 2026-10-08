"""Stage 5a: XGBoost model.

XGBoost builds many small decision trees, each one fixing the mistakes of the
ones before it ("gradient boosting"). One model is trained for all 500
store-items; store and item are just input columns.

Run from the project root:
    python -m src.train_xgboost
"""
import json

import pandas as pd
import xgboost as xgb

from src.config import DATA_PROCESSED, MODELS, RANDOM_SEED
from src.evaluation import score, time_split
from src.features import FEATURES_PATH

PRED_PATH = DATA_PROCESSED / "pred_xgboost.parquet"
NOT_FEATURES = ["date", "sales"]
VALID_DAYS = 92  # last 92 days of the training period are used to decide when to stop

PARAMS = dict(
    n_estimators=2000,        # upper limit on number of trees; early stopping picks the real number
    learning_rate=0.05,       # how big a correction each new tree makes (smaller = more careful)
    max_depth=7,              # how complex each tree can be
    subsample=0.8,            # each tree sees a random 80% of rows (reduces overfitting)
    colsample_bytree=0.8,     # ...and a random 80% of columns
    min_child_weight=5,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


if __name__ == "__main__":
    feats = pd.read_parquet(FEATURES_PATH)
    train, test = time_split(feats)
    cols = [c for c in feats.columns if c not in NOT_FEATURES]

    # Validation = the final days of training data (still the past relative to test).
    cut = train["date"].max() - pd.Timedelta(days=VALID_DAYS - 1)
    fit, valid = train[train["date"] < cut], train[train["date"] >= cut]
    print(f"Fit on {fit['date'].min().date()}..{fit['date'].max().date()} ({len(fit):,} rows); "
          f"validate on {valid['date'].min().date()}..{valid['date'].max().date()} ({len(valid):,} rows)")

    m = xgb.XGBRegressor(early_stopping_rounds=50, eval_metric="mae", **PARAMS)
    m.fit(fit[cols], fit["sales"], eval_set=[(valid[cols], valid["sales"])], verbose=False)
    best = m.best_iteration + 1
    print(f"Early stopping chose {best} trees (validation MAE {m.best_score:.3f})")

    # Refit on ALL training data with that number of trees. The test period is still untouched.
    final_params = {**PARAMS, "n_estimators": best}
    final = xgb.XGBRegressor(**final_params)
    final.fit(train[cols], train["sales"], verbose=False)

    test["pred"] = final.predict(test[cols]).clip(min=0)
    test[["date", "store", "item", "sales", "pred"]].to_parquet(PRED_PATH, index=False)

    s = score(test["sales"], test["pred"])
    print("\n== XGBOOST RESULTS (test period, all 500 store-items) ==")
    print(f"MAE {s['MAE']:.2f} | RMSE {s['RMSE']:.2f} | MAPE {s['MAPE_%']:.1f}%")

    imp = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop 10 features by importance (share of the model's splitting 'gain'):")
    print((imp.head(10) * 100).round(1).astype(str).add("%").to_string())

    MODELS.mkdir(exist_ok=True)
    final.save_model(MODELS / "xgboost.json")
    json.dump({"n_trees": best, "features": cols, "test_start": str(test["date"].min().date())},
              open(MODELS / "xgboost_meta.json", "w"), indent=2)
