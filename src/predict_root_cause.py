# predict_root_cause.py
import joblib

model = joblib.load("models/root_cause_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_root_cause(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]                  # predicted class
    conf = model.predict_proba(X).max() 
    return pred,conf