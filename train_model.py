import pandas as pd
import re
import spacy
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load NLP
import spacy.cli
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("Datensatz.csv")
df = df.dropna(subset=['en_claim', 'label'])
df['cleaned'] = df['en_claim'].apply(preprocess)

bert = SentenceTransformer("bert-base-nli-mean-tokens")
X = bert.encode(df['cleaned'].tolist(), show_progress_bar=True)
y = df['label']

X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "trained_model.pkl")
