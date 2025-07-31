# app.py
import streamlit as st
import pandas as pd
import re
import spacy
import joblib
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Health Claim Verifier", layout="centered")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    bert = joblib.load("bert_model.pkl")
    return model, bert

nlp = load_nlp()
model, bert_model = load_model()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

st.title("🔬 Health Claim Verifier")
st.markdown("Enter a health-related claim. The model will predict if it's **True**, **False**, or **Misleading**.")

user_input = st.text_area("📝 Enter claim:")

if st.button("Verify"):
    if not user_input.strip():
        st.warning("Please enter a claim.")
    else:
        processed = preprocess(user_input)
        emb = bert_model.encode([processed])
        prediction = model.predict(emb)[0]

        label_map = {0: "True ✅", 1: "False ❌", 2: "Misleading ⚠️"}
        st.success(f"Prediction: **{label_map.get(prediction, 'Unknown')}**")
