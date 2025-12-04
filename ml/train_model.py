import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import add_rul
from load_dataset import load_cmaps

# Load and preprocess C-MAPSS dataset
train_df, test_df, rul_df = load_cmaps()
train_df = add_rul(train_df)

# Create classification labels based on RUL threshold
train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x <= 30 else 0)

# Select feature columns (operational settings + sensor values)
feature_cols = [col for col in train_df.columns if 'sensor' in col or 'op_setting' in col]

# SAVE feature columns list here
joblib.dump(feature_cols, "feature_cols.pkl")

X = train_df[feature_cols]
y = train_df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Save trained model
joblib.dump(model, "engine_model.pkl")
print("Model saved successfully as engine_model.pkl")
