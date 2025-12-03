import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Load data
df = pd.read_csv("jet_engine_sample_data.csv")

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

# 2. Train-test split (to check generalization on unseen data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline: scaling + RandomForest
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

# 4. Cross-validation to reduce overfitting risk
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"Cross-val ROC AUC scores: {cv_scores}")
print(f"Mean CV ROC AUC: {cv_scores.mean():.3f}")

# 5. Train on full training set
pipeline.fit(X_train, y_train)

# 6. Evaluate on untouched test set
y_proba_test = pipeline.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

print("\n=== Test Set Performance ===")
print(classification_report(y_test, y_pred_test))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba_test))

# 7. Save model & feature list
joblib.dump(pipeline, "jet_engine_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\nModel saved to jet_engine_model.pkl")
