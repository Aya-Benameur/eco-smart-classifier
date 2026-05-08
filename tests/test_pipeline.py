import pytest
import pandas as pd
import numpy as np
import joblib

def test_data_loading():
    df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
    assert df.shape[0] > 0
    assert df.shape[1] == 9

def test_model_loading():
    model = joblib.load("models/multimodal_final_model.pkl")
    assert model is not None

def test_model_prediction():
    model = joblib.load("models/multimodal_final_model.pkl")
    sample = np.random.rand(1, model.n_features_in_)
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2, 3]

def test_no_nan_after_preprocessing():
    X_train = pd.read_csv("data/processed/X_train_classif.csv")
    cols_to_check = [c for c in X_train.columns if c != "Poids_Volume_Log"]
    assert X_train[cols_to_check].isnull().sum().sum() == 0

def test_data_schema():
    df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
    expected = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente", "Source", "Rapport_Collecte", "Categorie"]
    for col in expected:
        assert col in df.columns
