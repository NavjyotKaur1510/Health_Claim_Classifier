# import streamlit as st
# import pandas as pd
# import re
# import spacy
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split

# @st.cache_resource
# def load_nlp():
#     return spacy.load("en_core_web_sm")

# @st.cache_resource
# def load_bert_model():
#     return SentenceTransformer("bert-base-nli-mean-tokens")

# nlp = load_nlp()
# bert_model = load_bert_model()

# def preprocess_and_lemmatize(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# @st.cache_data
# def train_model():
#     df = pd.read_csv("Datensatz.csv")
#     df = df.dropna(subset=['en_claim', 'label'])
#     df['cleaned'] = df['en_claim'].apply(preprocess_and_lemmatize)

#     X = bert_model.encode(df['cleaned'].tolist(), show_progress_bar=True)
#     y = df['label']

#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)

#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#     model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     conf_mat = confusion_matrix(y_test, y_pred)

#     return model, report, conf_mat

# model, report, conf_mat = train_model()

# st.title("🔬 Health Claim Verifier")
# st.markdown("Enter a health claim to classify it as **True**, **False**, or **Misleading** using a BERT + Random Forest model.")

# user_input = st.text_area("📝 Enter a health claim:", height=100)

# if st.button("Verify Claim"):
#     if not user_input.strip():
#         st.warning("Please enter a claim.")
#     else:
#         processed = preprocess_and_lemmatize(user_input)
#         embedded = bert_model.encode([processed])
#         prediction = model.predict(embedded)[0]

#         label_map = {
#             0: "True ✅",
#             1: "False ❌",
#             2: "Misleading ⚠️"
#         }

#         st.subheader("📌 Prediction Result:")
#         st.success(f"**This claim is predicted as: {label_map.get(prediction, 'Unknown')}**")

# with st.expander("📊 Show Model Performance"):
#     st.subheader("Classification Report")
#     st.dataframe(pd.DataFrame(report).transpose())

#     st.subheader("Confusion Matrix Heatmap")
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["True", "False", "Misleading"], yticklabels=["True", "False", "Misleading"])
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     st.pyplot(plt)
# app.py (Updated to load saved model instead of retraining)
import streamlit as st
import pandas as pd
import re
import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_bert_model():
    return SentenceTransformer("bert-base-nli-mean-tokens")

@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl"), pd.read_json("report.json"), np.load("confusion_matrix.npy")

nlp = load_nlp()
bert_model = load_bert_model()
model, report_df, conf_mat = load_model()

def preprocess_and_lemmatize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

st.title("🔬 Health Claim Verifier")
st.markdown("Enter a health claim below to classify it as **True**, **False**, or **Misleading** using BERT + Random Forest model.")

user_input = st.text_area("📝 Enter a health claim:", height=100)

if st.button("Verify Claim"):
    if not user_input.strip():
        st.warning("Please enter a claim.")
    else:
        processed = preprocess_and_lemmatize(user_input)
        embedded = bert_model.encode([processed])
        prediction = model.predict(embedded)[0]

        label_map = {
            0: "True ✅",
            1: "False ❌",
            2: "Misleading ⚠️"
        }

        st.subheader("📌 Prediction Result:")
        st.success(f"**This claim is predicted as: {label_map.get(prediction, 'Unknown')}**")

with st.expander("📊 Show Model Performance"):
    st.subheader("Classification Report")
    st.dataframe(report_df)

    st.subheader("Confusion Matrix Heatmap")
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["True", "False", "Misleading"],
                yticklabels=["True", "False", "Misleading"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
