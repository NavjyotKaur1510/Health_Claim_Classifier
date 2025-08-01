import streamlit as st
import re
import spacy
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_nlp():
    import spacy.cli
    try:
        return spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_bert():
    return SentenceTransformer("bert-base-nli-mean-tokens")

@st.cache_resource
def load_model():
    return joblib.load("trained_model.pkl")

nlp = load_nlp()
bert = load_bert()
model = load_model()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

st.title("🧠 Health Claim Classifier")
user_input = st.text_input("Enter a health claim:")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a claim.")
    else:
        clean = preprocess(user_input)
        embed = bert.encode([clean])
        prediction = model.predict(embed)[0]
        label_map = {0: "True ✅", 1: "False ❌", 2: "Misleading ⚠️"}
        st.success(f"**Prediction:** {label_map.get(prediction)}")
