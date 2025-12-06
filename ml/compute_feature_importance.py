import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from load_dataset import load_cmaps  # your existing loader

# 1. Load data
train_df, test_df, rul_df = load_cmaps()

print("Test DF head:")
print(test_df.head())
print("\nTest DF columns:", list(test_df.columns))

# 2. Build RUL + binary label for TEST set (same logic as evaluate_cmaps_model.py)
test_df = test_df.copy()
max_cycles_per_unit = test_df.groupby("unit_nr")["time_cycles"].transform("max")
test_df["RUL"] = max_cycles_per_unit - test_df["time_cycles"]

N = 30  # failure within next 30 cycles
test_df["label"] = (test_df["RUL"] <= N).astype(int)

# 3. Load model + feature columns
feature_cols = joblib.load("feature_cols.pkl")
model = joblib.load("engine_model.pkl")

X_test = test_df[feature_cols]
y_test = test_df["label"]

print("\nUsing feature columns:")
print(feature_cols)

# 4. Baseline ROC-AUC
y_proba = model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_proba)
print(f"\nBaseline ROC-AUC: {baseline_auc:.3f}")

# 5. Permutation importance
print("\nComputing permutation importance (this can take a bit)...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
    scoring="roc_auc",
)

importances_mean = result.importances_mean
importances_mean = importances_mean / importances_mean.sum()  # normalize to 1

pairs = list(zip(feature_cols, importances_mean))
pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

print("\n=== Global feature importance (permutation, normalized) ===")
for name, w in pairs_sorted:
    print(f"{name}: {w:.4f}  ({w*100:.1f}%)")
