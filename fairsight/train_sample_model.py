#!/usr/bin/env python3
"""
Train a sample model on the biased sample_data.csv and save it for ModelSight testing.
Creates:
  - sample_model.pkl  (trained GradientBoostingClassifier)
  - sample_test.csv   (held-out test set with protected attributes preserved)
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading sample_data.csv...")
df = pd.read_csv("sample_data.csv")

target = "loan_approved"
protected = ["age", "gender", "race"]

# Split into train/test — preserving protected attrs in test set
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])

# Prepare features (encode categoricals, but KEEP protected attrs in the feature set
# to simulate a naively trained biased model)
feature_cols = [c for c in df.columns if c != target]

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = train_df[target].values
y_test = test_df[target].values

# Encode string columns
encoders = {}
for c in X_train.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))
    encoders[c] = le

# Train a model (deliberately including protected attributes to create bias)
print("Training GradientBoostingClassifier (including protected attributes)...")
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.1%}")
print(f"Test accuracy:  {test_acc:.1%}")

# Save model
model_path = "sample_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved → {model_path}")

# Save the test set with ORIGINAL (non-encoded) protected attributes
# so ModelSight can analyze them
test_output = test_df.copy()
test_output.to_csv("sample_test.csv", index=False)
print(f"Test set saved → sample_test.csv ({len(test_output)} rows)")

print("\nNow run:")
print(f"  python modelsight.py --model {model_path} --csv sample_test.csv --target {target}")
