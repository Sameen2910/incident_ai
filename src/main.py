# main.py
from embed import build_and_save_index
from rag import IncidentRAG
import pandas as pd
from preprocess import preprocess

# -----------------------------
# Load dataset and preprocess
# -----------------------------
df = pd.read_csv("data/incidents_10000.csv")
df = preprocess(df)

# -----------------------------
# Build FAISS index
# -----------------------------
build_and_save_index(df["text"].tolist())

# -----------------------------
# Initialize RAG
# -----------------------------
rag = IncidentRAG(model_name="llama3:8b")

# -----------------------------
# Interactive loop
# -----------------------------
while True:
    query = input("\nEnter incident description (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Directly generate AI response (root cause + recommendations)
    answer, refs = rag.generate(query, predicted_root="")  # <-- predicted_root not used
    print("\nAI Recommendation:\n", answer)

    print("\nTop Similar Incidents:")
    for i, r in enumerate(refs, start=1):
        print(f"{i}. {r}")