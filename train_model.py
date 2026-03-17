# ============================================================
#  train_model.py  –  Train & save Iris Logistic Regression
#  Run this ONCE:  python train_model.py
# ============================================================

import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── 1. Load dataset ──────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names   # ['sepal length (cm)', ...]
target_names  = iris.target_names    # ['setosa', 'versicolor', 'virginica']

print("Dataset shape :", X.shape)
print("Classes       :", target_names)

# ── 2. Train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Feature scaling ───────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train Logistic Regression ─────────────────────────────
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_sc, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy : {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 6. Save model + scaler + metadata ────────────────────────
bundle = {
    "model"        : model,
    "scaler"       : scaler,
    "feature_names": list(feature_names),
    "target_names" : list(target_names),
    "X_train"      : X_train,
    "X_test"       : X_test,
    "y_train"      : y_train,
    "y_test"       : y_test,
    "accuracy"     : acc,
}

with open("iris_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nModel saved to iris_model.pkl")