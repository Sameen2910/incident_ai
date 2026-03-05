import os
import joblib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess  # your preprocessing function
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Load and preprocess data ---
df = pd.read_csv("data/incidents_10000.csv")
df = preprocess(df)

def add_text_noise(text, p=0.1):
    words = text.split()
    n_swap = max(1, int(len(words) * p))
    for _ in range(n_swap):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

df["text"] = df["text"].apply(lambda x: add_text_noise(x, p=0.2))

np.random.seed(42)
mask = np.random.rand(len(df)) < 0.05  # 5% labels flipped
y_noisy = df["root_cause"].copy()
y_noisy[mask] = np.random.choice(df["root_cause"].unique(), mask.sum())

X = df["text"].values  # raw text
y = y_noisy.values

# --- Convert text to numeric features ---
vectorizer = TfidfVectorizer(max_features=500)
X_numeric = vectorizer.fit_transform(X)

# --- Split the data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42
)

# --- Train the model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# --- Confusion matrix ---
# y_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
# disp.plot(xticks_rotation=45, cmap='Blues')
# plt.show()

# --- Save the model safely ---
project_root = os.path.dirname(os.path.dirname(__file__))
model_folder = os.path.join(project_root, "models")
model_path = os.path.join(model_folder, "root_cause_model.pkl")

os.makedirs(model_folder, exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# --- Optional: save vectorizer as well ---
vectorizer_path = os.path.join(model_folder, "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)
print(f"Vectorizer saved to {vectorizer_path}")