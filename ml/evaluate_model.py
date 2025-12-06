import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
import joblib

# 1. Load data
df = pd.read_csv("jet_engine_sample_data.csv")

# 2. Define features & target (same as training)
feature_cols = [
    "cycles_since_maintenance",
    "avg_turbine_temp",
    "compressor_pressure_ratio",
    "vibration_level",
    "fuel_flow_variation",
    "previous_failures",
]
X = df[feature_cols]
y = df["failed_within_30_cycles"]

# 3. Train/test split (same random_state as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Load your trained model
model = joblib.load("jet_engine_model.pkl")

# 5. Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6. Metrics at default threshold = 0.5
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
roc_auc = roc_auc_score(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

print("=== Metrics at threshold = 0.50 ===")
print(f"Accuracy     : {acc:.3f}")
print(f"Precision    : {precision:.3f}")
print(f"Recall       : {recall:.3f}")
print(f"F1-score     : {f1:.3f}")
print(f"ROC-AUC      : {roc_auc:.3f}")
print(f"PR-AUC       : {avg_prec:.3f}\n")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 7. Threshold sweep
print("\n=== Threshold sweep ===")
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

# 8. ROC Curve
fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 0, 1], [0, 1, 1], linestyle="--")  # optional "perfect" guide
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 9. Precision–Recall Curve
precisions, recalls, pr_thresh = precision_recall_curve(y_test, y_proba)

plt.figure()
plt.plot(recalls, precisions, label=f"PR curve (AP = {avg_prec:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
