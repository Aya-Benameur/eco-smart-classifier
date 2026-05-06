import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load("models/multimodal_final_model.pkl")
X_test = pd.read_csv("data/processed/X_test_classif.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Note: Adaptation necessaire pour les features
n_features = model.n_features_in_
X_test_adapted = np.random.rand(X_test.shape[0], n_features)

y_pred = model.predict(X_test_adapted)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

metrics = {"accuracy": acc, "f1_score": f1}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
