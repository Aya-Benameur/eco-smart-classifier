import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Eco-Smart Classifier", layout="wide")
st.title("Eco-Smart Classifier Demo")

# Version de démonstration - modèles simplifiés
@st.cache_resource
def load_demo_models():
    # Modèle simple pour la démo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    label_encoder = LabelEncoder()
    label_encoder.fit(['Metal', 'Papier', 'Plastique', 'Verre'])
    # Entraîner un modèle factice
    X_demo = np.random.rand(100, 14)
    y_demo = np.random.randint(0, 4, 100)
    model.fit(X_demo, y_demo)
    return model, label_encoder

model, label_encoder = load_demo_models()

# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/dataset_ProjetML_2026.csv")

df = load_data()

# ============================================
# ONGLET 1: DASHBOARD DATA
# ============================================
tab1, tab2, tab3 = st.tabs(["Dashboard Data", "Prediction Manuelle", "Assistant Intelligent"])

with tab1:
    st.header("Visualisation du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des categories")
        st.bar_chart(df["Categorie"].value_counts())
    
    with col2:
        st.subheader("Apercu des donnees")
        st.dataframe(df.head(10))
    
    st.subheader("Visualisation PCA")
    numeric_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
    df_num = df[numeric_cols].dropna()
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(df_num)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], title="Projection PCA")
    st.plotly_chart(fig)

# ============================================
# ONGLET 2: PREDICTION MANUELLE
# ============================================
with tab2:
    st.header("Prediction en temps reel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        poids = st.slider("Poids (kg)", 0.0, 50.0, 1.0)
        volume = st.slider("Volume (L)", 0.0, 100.0, 1.0)
        conductivite = st.slider("Conductivite", 0.0, 100.0, 10.0)
    
    with col2:
        opacite = st.slider("Opacite", 0.0, 1.0, 0.5)
        rigidite = st.slider("Rigidite", 0.0, 100.0, 50.0)
        prix = st.slider("Prix (€)", 0.0, 100.0, 10.0)
    
    features = np.array([[
        poids, volume, conductivite, opacite, rigidite, prix,
        0, 0, 0, 0, 0, 0, 0, 0
    ]])
    
    prediction = model.predict(features)[0]
    categorie = label_encoder.inverse_transform([prediction])[0]
    probas = model.predict_proba(features)[0]
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Categorie predite", categorie)
    with col_b:
        st.metric("Confiance", f"{max(probas):.1%}")
    st.progress(max(probas))

# ============================================
# ONGLET 3: ASSISTANT INTELLIGENT
# ============================================
with tab3:
    st.header("Description textuelle du dechet")
    
    description = st.text_area("Decrivez votre dechet:", 
                                "Bouteille en plastique transparent, 50cl")
    
    if st.button("Analyser", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Simulation NLP simple
            mots = description.lower().split()
            if "plastique" in mots or "bouteille" in mots:
                categorie = "Plastique"
                confiance = 0.95
            elif "verre" in mots or "bouteille" in mots:
                categorie = "Verre"
                confiance = 0.92
            elif "metal" in mots or "canette" in mots:
                categorie = "Metal"
                confiance = 0.90
            elif "papier" in mots or "carton" in mots:
                categorie = "Papier"
                confiance = 0.88
            else:
                categorie = "Plastique"
                confiance = 0.75
            
            st.success(f"Categorie identifiee : {categorie}")
            st.metric("Confiance", f"{confiance:.1%}")
            
            st.subheader("Mots cles detectes")
            st.write(list(set(mots) & {"plastique", "verre", "metal", "papier", "bouteille", "canette", "carton"}))
