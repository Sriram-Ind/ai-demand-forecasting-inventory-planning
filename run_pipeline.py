"""Run every stage in order, from raw data to the files the dashboard needs.

    python run_pipeline.py

Takes a few minutes. Needs data/raw/train.csv first (see README).
"""
import subprocess
import sys
import time

STEPS = [
    ("Check the data", "src.data_loader"),
    ("Explore the data (charts)", "src.eda"),
    ("Build features", "src.features"),
    ("Baseline: same as last week", "src.baseline"),
    ("Train XGBoost", "src.train_xgboost"),
    ("Train Prophet", "src.train_prophet"),
    ("Compare models (table + charts)", "src.evaluate"),
    ("Safety stock and reorder simulation", "src.inventory"),
]

if __name__ == "__main__":
    start = time.time()
    for i, (label, module) in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] {label} ===", flush=True)
        if subprocess.run([sys.executable, "-m", module]).returncode != 0:
            sys.exit(f"Step failed: {module}")
    print(f"\nAll done in {time.time() - start:.0f}s. Now run:  streamlit run app/dashboard.py")
