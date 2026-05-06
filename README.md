# Eco-Smart Classifier

Classification automatique des dechets et estimation de leur valeur de revente.

## Description

Ce projet developpe un pipeline complet allant de la donnee brute a une application web capable de :
- Classifier les dechets (Metal, Papier, Plastique, Verre)
- Estimer leur valeur de revente
- Analyser des descriptions textuelles (NLP)
- Segmenter les dechets par clustering

## 3 commandes pour rejouer le pipeline

```bash
# 1. Installer les dependances
pip install -r requirements.txt

# 2. Executer le pipeline de preprocessing et training
python src/data_preprocessing.py && python src/train_model.py

# 3. Lancer l API ou l application web
streamlit run src/app_streamlit.py
# OU
uvicorn src.app:app --reload
```

## Performances

| Metrique | Score |
|----------|-------|
| Classification Accuracy | 99.93% |
| Regression R² | 0.998 |
| NLP Accuracy | 100% |
| Clustering | 3 clusters |

## Docker

```bash
# Construire l image
docker build -t eco-smart-classifier .

# Lancer le conteneur
docker run -p 8000:8000 eco-smart-classifier
```

## MLflow - Tracking des experiences

```bash
# Lancer l interface MLflow
mlflow ui --backend-store-uri file://./mlruns
```

## Structure du projet

```
eco-smart-classifier/
├── data/               # Donnees brutes et traitees
│   ├── raw/           # Dataset original
│   └── processed/     # Donnees preparees
├── models/            # 20 modeles sauvegardes (.pkl)
├── src/               # Code source
│   ├── app.py         # API FastAPI
│   ├── app_streamlit.py # Application web Streamlit
│   ├── data_preprocessing.py
│   └── train_model.py
├── tests/             # Tests pytest
├── figures/           # 11 visualisations PNG
├── mlruns/            # Tracking MLflow
├── requirements.txt   # Dependances Python
├── Dockerfile         # Containerisation
└── README.md         # Documentation
```

## Technologies utilisees

- Data Engineering: Pandas, NumPy, Scikit-learn
- Modelisation: Random Forest, XGBoost, SVM
- NLP: NLTK, TF-IDF, Word2Vec
- MLOps: MLflow, DVC, Docker, GitHub Actions
- Deploiement: FastAPI, Streamlit

## Auteur

Projet realise dans le cadre du cours ML Engineering.

## Licence

MIT
