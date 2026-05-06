import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

print("Loading data...")
df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")

print("Preprocessing...")
numeric_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

df["Densite"] = df["Poids"] / (df["Volume"] + 1e-6)
df["Ratio_Prix_Poids"] = df["Prix_Revente"] / (df["Poids"] + 1e-6)
df["Poids_Volume_Log"] = np.log1p(df["Poids"] * df["Volume"])

df = pd.get_dummies(df, columns=["Source"], prefix="src", dummy_na=True)

df_labeled = df[df["Categorie"].notna()]
le = LabelEncoder()
y = le.fit_transform(df_labeled["Categorie"])
X = df_labeled.drop(columns=["Categorie", "Rapport_Collecte"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("data/processed/X_train_classif.csv", index=False)
X_test.to_csv("data/processed/X_test_classif.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)
joblib.dump(le, "models/label_encoder_final.pkl")

print("Done!")
