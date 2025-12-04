import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
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

# 6. Metrics
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy     : {acc:.3f}")
print(f"Precision    : {precision:.3f}")
print(f"Recall       : {recall:.3f}")
print(f"F1-score     : {f1:.3f}")
print(f"ROC-AUC      : {roc_auc:.3f}\n")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))
