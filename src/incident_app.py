# streamlit.py
import streamlit as st
import pandas as pd
from preprocess import preprocess
from embed import build_and_save_index
from rag import IncidentRAG

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("AI Incident Intelligence Assistant")

# -----------------------------
# Setup: Load Data and RAG
# -----------------------------
@st.cache_resource
def setup():
    df = pd.read_csv("data/incidents.csv")
    df = preprocess(df)

    # Build FAISS index
    build_and_save_index(df["text"].tolist())

    # Initialize RAG with smaller 7b model
    rag = IncidentRAG(model_name="llama3:8b")
    return rag

rag = setup()

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Describe the incident")

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze") and user_input:
    with st.spinner("Generating AI recommendation..."):
        # Directly generate AI response (no ML classifier)
        answer, refs = rag.generate(user_input, predicted_root="")  # predicted_root unused
        refs = [r[:512] for r in refs]  # safety truncation

    st.subheader("AI Recommendation")
    st.write(answer)

    st.subheader("Similar Incidents Used as Reference")
    for i, r in enumerate(refs, start=1):
        st.write(f"{i}. {r}")