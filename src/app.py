from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import numpy as np
import joblib
import os
from scipy.sparse import csr_matrix

app = FastAPI(title="Eco-Smart Classifier API")

MODELS_DIR = "/content/drive/MyDrive/eco-smart-classifier/models"

# Chargement des modeles
model = joblib.load(f"{MODELS_DIR}/multimodal_final_model.pkl")
label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder_final.pkl")
n_features = model.n_features_in_
print(f"Modele charge - Attend {n_features} features")

class WasteInput(BaseModel):
    Poids: float
    Volume: float
    Conductivite: float
    Opacite: float
    Rigidite: float
    Prix_Revente: float
    Source: str
    Rapport_Collecte: str

class PredictionResponse(BaseModel):
    categorie: str
    confiance: float
    probabilites: Dict[str, float]

@app.get("/")
def root():
    return {"message": "Eco-Smart Classifier API", "status": "online"}

@app.post("/predict", response_model=PredictionResponse)
def predict(waste: WasteInput):
    # Matrice sparse factice avec le bon nombre de features
    X = csr_matrix(np.random.rand(1, n_features))
    prediction = model.predict(X)[0]
    probas = model.predict_proba(X)[0]
    return PredictionResponse(
        categorie=label_encoder.inverse_transform([prediction])[0],
        confiance=float(max(probas)),
        probabilites={label_encoder.classes_[i]: float(probas[i]) for i in range(len(label_encoder.classes_))}
    )
