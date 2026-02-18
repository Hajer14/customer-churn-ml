# src/train.py

"""
Train models for Telco Customer Churn prediction.
Models: Logistic Regression, Random Forest, XGBoost
Author: [Your Name]
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from preprocess import load_data, preprocess_data
import numpy as np

# -----------------------------
# 1️⃣ Load and preprocess data
# -----------------------------
df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# -----------------------------
# 2️⃣ Define models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# -----------------------------
# 3️⃣ Train models and evaluate
# -----------------------------
results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": roc}
    print(f"✅ {name} trained | Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")

# -----------------------------
# 4️⃣ Select best model based on F1-score
# -----------------------------
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name} with F1-score = {results[best_model_name]['f1_score']:.3f}")

# -----------------------------
# 5️⃣ Save model and preprocessor
# -----------------------------
joblib.dump(best_model, "../models/best_model.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")
print("💾 Model and preprocessor saved in ../models/")
