import streamlit as st
import pandas as pd
import os
import numpy as np
from fpdf import FPDF

from preprocess import preprocess
from embed import build_and_save_index
from rag import IncidentRAG
from predict_root_cause import predict_root_cause
from incident_memory import IncidentMemory
from incident_graph import IncidentGraph
from pattern_detector import FailurePatternDetector
from utils import extract_timeline

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="AI Incident Copilot", layout="wide")
st.title("🚑 AI Incident Intelligence Copilot")

# -----------------------------
# Session State Initialization
# -----------------------------
if "analyzed_incidents" not in st.session_state:
    st.session_state.analyzed_incidents = []

if "memory" not in st.session_state:
    st.session_state.memory = IncidentMemory()

if "graph" not in st.session_state:
    st.session_state.graph = IncidentGraph()

# -----------------------------
# Setup RAG + Embeddings
# -----------------------------
@st.cache_resource
def setup_rag():
    df = pd.read_csv("data/incidents_10000.csv")
    df = preprocess(df)
    if not os.path.exists("embeddings/faiss.index"):
        build_and_save_index(df["text"].tolist())
    return IncidentRAG()

rag = setup_rag()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Settings")
show_similar = st.sidebar.checkbox("Show Similar Incidents")
show_confidence = st.sidebar.checkbox("Show Confidence Score")
show_clusters = st.sidebar.checkbox("Show Incident Clusters")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Incident Name")
incident_name = st.text_input(
    "Incident Name",
    placeholder="Enter a short descriptive name...",
    label_visibility="collapsed"
)

st.subheader("Describe the Incident")
user_input = st.text_area(
    "Incident Description",
    placeholder="Example: Production API returning 500 errors...",
    label_visibility="collapsed"
)

# -----------------------------
# Analyze Incident
# -----------------------------
if st.button("Analyze Incident") and user_input and incident_name:

    with st.spinner("AI analyzing incident..."):

        # 1️⃣ Predict Root Cause
        predicted_root, confidence = predict_root_cause(user_input)

        # 2️⃣ Generate RAG-based analysis
        answer, refs = rag.generate(user_input, predicted_root, confidence)

        # 3️⃣ Save in session
        incident_record = {
            "name": incident_name,
            "predicted_root": predicted_root,
            "confidence": confidence,
            "analysis": answer,
            "refs": refs
        }

        st.session_state.analyzed_incidents.append(incident_record)

        # 4️⃣ Add to memory
        st.session_state.memory.add_incident(
            name=incident_name,
            description=user_input,
            root_cause=predicted_root
        )

        # 5️⃣ Add to graph
        st.session_state.graph.add_incident(incident_name, predicted_root)

    st.success(f"Incident '{incident_name}' added to session!")

# -----------------------------
# Display Last Incident
# -----------------------------
if st.session_state.analyzed_incidents:
    last = st.session_state.analyzed_incidents[-1]

    st.divider()
    st.subheader(f"Predicted Root Cause for '{last['name']}'")
    st.write(last["predicted_root"])

    if show_confidence:
        st.subheader("AI Confidence")
        st.write(f"{last['confidence']*100:.1f}%")

    st.subheader("🤖 AI Incident Analysis")
    st.write(last["analysis"])

    # Timeline extraction
    timeline = extract_timeline(last["analysis"])
    if timeline:
        st.subheader("📅 Incident Timeline")
        for event in timeline:
            st.write(event)

# -----------------------------
# Similar Incidents (RAG + Memory)
# -----------------------------
if show_similar and st.session_state.analyzed_incidents:
    refs = st.session_state.analyzed_incidents[-1]["refs"]
    memory_results = st.session_state.memory.search(user_input)

    with st.expander("🔎 Similar Historical Incidents (RAG + Memory)"):
        for i, r in enumerate(refs, start=1):
            st.write(f"**RAG Incident {i}**")
            st.write(r)
        for j, r in enumerate(memory_results, start=1):
            st.write(f"**Memory Match {j}**")
            st.write(r)

# -----------------------------
# Incident Analytics / Clustering
# -----------------------------
if show_clusters and len(st.session_state.analyzed_incidents) > 1:
    st.subheader("📊 Incident Clusters (Failure Patterns)")

    # Example: using simple embeddings placeholder (replace with real embeddings)
    # Here we simulate embeddings for demonstration
    num_incidents = len(st.session_state.analyzed_incidents)
    embedding_dim = 384
    embeddings = np.random.rand(num_incidents, embedding_dim)

    detector = FailurePatternDetector(embeddings)
    clusters = detector.detect_patterns(n_clusters=min(5, num_incidents))

    for cluster_id, incident_ids in clusters.items():
        st.write(f"**Cluster {cluster_id}**")
        for i in incident_ids:
            st.write(st.session_state.analyzed_incidents[i]["name"])

# -----------------------------
# Graph-based Related Incidents
# -----------------------------
st.subheader("🔗 Graph-based Related Incidents")
if st.session_state.analyzed_incidents:
    last_root = st.session_state.analyzed_incidents[-1]["predicted_root"]
    related = st.session_state.graph.find_by_root(last_root)
    if related:
        st.write(f"Incidents with same root cause ({last_root}):")
        for r in related:
            st.write(r)

# -----------------------------
# PDF Generation
# -----------------------------
def generate_multi_incident_pdf(incidents):
    pdf = FPDF()
    for inc in incidents:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Incident: {inc['name']}", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(
            0, 8,
            f"Predicted Root Cause: {inc['predicted_root']} "
            f"(Confidence: {inc['confidence']*100:.1f}%)"
        )
        pdf.ln(3)
        pdf.multi_cell(0, 8, inc["analysis"])
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin1")

if st.session_state.analyzed_incidents:
    pdf_bytes = generate_multi_incident_pdf(st.session_state.analyzed_incidents)
    st.download_button(
        label="📄 Download Multi-Incident Report",
        data=pdf_bytes,
        file_name="incident_report.pdf",
        mime="application/pdf"
    )