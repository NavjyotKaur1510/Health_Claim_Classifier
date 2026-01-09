import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Caching and Model Loading ---
# Use st.cache_resource to load models and data only once
@st.cache_resource
def load_resources():
    """
    Loads all the necessary resources for the app, including the dataset,
    spaCy model, SentenceTransformer model, and trains the classification model.
    This function is cached so it only runs once.
    """
    # Load dataset
    try:
        df = pd.read_csv("Datensatz.csv")
    except FileNotFoundError:
        st.error("Error: 'Datensatz.csv' not found. Please make sure the dataset is in the same directory as your Streamlit app.")
        return None, None, None, None

    df = df.dropna(subset=['en_claim', 'label'])

    # Load spaCy model
    # You might need to run: python -m spacy download en_core_web_sm
    import spacy
    import subprocess

    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")



    # Text preprocessing and lemmatization function
    def preprocess_and_lemmatize(text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    df['cleaned'] = df['en_claim'].apply(preprocess_and_lemmatize)

    # Load BERT Embeddings model
    bert_model = SentenceTransformer("bert-base-nli-mean-tokens")
    
    with st.spinner("Generating text embeddings... This might take a few minutes."):
        X = bert_model.encode(df['cleaned'].tolist(), show_progress_bar=True)

    # Labels
    y = df['label']

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model to display metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, bert_model, nlp, (accuracy, report)

# --- Streamlit App UI ---

st.set_page_config(page_title="Fact-Checking Claim Classifier", layout="wide")

st.title("🔎 Fact-Checking Health Claim Classifier")
st.markdown("""
This application uses a trained Machine Learning model (Random Forest) to classify whether a given claim is likely true or false.
Enter a claim in the text box below and click the 'Classify' button to see the result.
""")

# Load all resources
classifier_model, sentence_bert_model, nlp_model, metrics = load_resources()

if classifier_model:
    st.header("Enter a Claim to Classify")

    # User input
    user_claim = st.text_area("Enter the claim text here:", height=150)

    if st.button("Classify Claim"):
        if user_claim:
            with st.spinner("Analyzing the claim..."):
                # Preprocess the user's input
                def preprocess_user_input(text, nlp):
                    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
                    doc = nlp(text)
                    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

                cleaned_claim = preprocess_user_input(user_claim, nlp_model)

                # Generate embedding for the user's claim
                claim_embedding = sentence_bert_model.encode([cleaned_claim])

                # Predict using the trained model
                prediction = classifier_model.predict(claim_embedding)
                prediction_proba = classifier_model.predict_proba(claim_embedding)

            # Display the result
            st.subheader("Classification Result")
            if prediction[0] == 1 or prediction[0].lower() == 'true':
                st.success("✅ The claim is classified as: **True**")
            else:
                st.error("❌ The claim is classified as: **False**")

            # Display probabilities
            st.write("Confidence Scores:")
            st.write(f"**False:** {prediction_proba[0][0]:.2%}")
            st.write(f"**True:** {prediction_proba[0][1]:.2%}")

        else:
            st.warning("Please enter a claim to classify.")

    # Display model performance metrics
    with st.expander("View Model Performance Metrics"):
        accuracy, report_dict = metrics
        st.subheader("Model Evaluation")
        st.write(f"**Overall Accuracy:** {accuracy:.2%}")
        
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format('{:.2f}'))
else:
    st.info("The application could not be loaded due to the errors mentioned above.")

