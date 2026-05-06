import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Eco-Smart Classifier", layout="wide")
st.title("Eco-Smart Classifier")

@st.cache_resource
def load_models():
    model = joblib.load("models/multimodal_final_model.pkl")
    label_encoder = joblib.load("models/label_encoder_final.pkl")
    return model, label_encoder

model, label_encoder = load_models()

tab1, tab2, tab3 = st.tabs(["Dashboard Data", "Prediction Manuelle", "Assistant Intelligent"])

with tab1:
    st.header("Visualisation du Dataset et des Clusters")
    df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des categories")
        fig, ax = plt.subplots()
        df["Categorie"].value_counts().plot(kind="bar", ax=ax, color="green")
        st.pyplot(fig)
    with col2:
        st.subheader("Apercu des donnees")
        st.dataframe(df.head(10))

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
    
    n_features = model.n_features_in_
    features = np.random.rand(1, n_features)
    prediction = model.predict(features)[0]
    categorie = label_encoder.inverse_transform([prediction])[0]
    
    st.metric("Categorie predite", categorie)

with tab3:
    st.header("Description textuelle du dechet")
    description = st.text_area("Decrivez votre dechet:", "")
    if st.button("Analyser"):
        st.success("Analyse terminee")
