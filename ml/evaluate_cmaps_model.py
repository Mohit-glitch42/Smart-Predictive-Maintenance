import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import joblib

from load_dataset import load_cmaps  # your existing loader

# 1. Load full C-MAPSS dataset (pre-split by your loader)
train_df, test_df, rul_df = load_cmaps()

print("Test DF head:")
print(test_df.head())
print("\nTest DF columns:", list(test_df.columns))

# 2. Create RUL and binary label for TEST set
# RUL = (max time_cycles for this engine) - current time_cycles
test_df = test_df.copy()
max_cycles_per_unit = test_df.groupby("unit_nr")["time_cycles"].transform("max")
test_df["RUL"] = max_cycles_per_unit - test_df["time_cycles"]

# Define failure within next N cycles as positive class
N = 30
test_df["label"] = (test_df["RUL"] <= N).astype(int)

print("\nLabel distribution in TEST set (RUL <=", N, "):")
print(test_df["label"].value_counts(normalize=True).rename("proportion"))
print(test_df["label"].value_counts().rename("count"))

# 3. Load model + feature columns
feature_cols = joblib.load("feature_cols.pkl")   # from your training stage
model = joblib.load("engine_model.pkl")          # main maintenance model

X_test = test_df[feature_cols]
y_test = test_df["label"]

# 4. Predict probabilities + default predictions
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

# 5. Metrics at default threshold = 0.5
acc = accuracy_score(y_test, y_pred_default)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_default, average="binary"
)
roc_auc = roc_auc_score(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

print("\n=== C-MAPSS model – Metrics at threshold = 0.50 ===")
print(f"Accuracy     : {acc:.3f}")
print(f"Precision    : {precision:.3f}")
print(f"Recall       : {recall:.3f}")
print(f"F1-score     : {f1:.3f}")
print(f"ROC-AUC      : {roc_auc:.3f}")
print(f"PR-AUC       : {avg_prec:.3f}\n")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_default))

print("\nClassification report:")
print(classification_report(y_test, y_pred_default))

# 6. Threshold sweep
print("\n=== Threshold sweep (C-MAPSS model) ===")
thresholds = np.linspace(0.2, 0.8, 7)  # 0.20, 0.30, ..., 0.80

for th in thresholds:
    y_pred_th = (y_proba >= th).astype(int)
    prec_th, rec_th, f1_th, _ = precision_recall_fscore_support(
        y_test, y_pred_th, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred_th)
    print(f"\n--- Threshold = {th:.2f} ---")
    print(f"Precision: {prec_th:.3f} | Recall: {rec_th:.3f} | F1: {f1_th:.3f}")
    print("Confusion matrix:")
    print(cm)

# 7. ROC Curve
fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 0, 1], [0, 1, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve – C-MAPSS model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 8. Precision–Recall Curve
precisions, recalls, pr_thresh = precision_recall_curve(y_test, y_proba)

plt.figure()
plt.plot(recalls, precisions, label=f"PR curve (AP = {avg_prec:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – C-MAPSS model")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
