# train_model.py
import pandas as pd
import re
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import joblib

# Load models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("bert-base-nli-mean-tokens")

# Preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load and process dataset
df = pd.read_csv("Datensatz.csv")
df = df.dropna(subset=['en_claim', 'label'])
df['cleaned'] = df['en_claim'].apply(preprocess)

X = bert_model.encode(df['cleaned'].tolist(), show_progress_bar=True)
y = df['label']

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Save model and BERT
joblib.dump(model, "rf_model.pkl")
joblib.dump(bert_model, "bert_model.pkl")
