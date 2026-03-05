import streamlit as st
import pandas as pd
import os
from fpdf import FPDF
from preprocess import preprocess
from embed import build_and_save_index
from rag import IncidentRAG
from predict_root_cause import predict_root_cause
from executive_summary import generate_executive_summary

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Incident Copilot",
    layout="wide"
)

st.title("🚑 AI Incident Intelligence Copilot")

# -----------------------------
# Session State Initialization
# -----------------------------
if "analyzed_incidents" not in st.session_state:
    st.session_state.analyzed_incidents = []

# -----------------------------
# Setup
# -----------------------------
@st.cache_resource
def setup():
    df = pd.read_csv("data/incidents_10000.csv")
    df = preprocess(df)

    if not os.path.exists("embeddings/faiss.index"):
        build_and_save_index(df["text"].tolist())

    return IncidentRAG()

rag = setup()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Settings")
debug_mode = st.sidebar.checkbox("Show Similar Incidents")
show_confidence = st.sidebar.checkbox("Show Confidence Score")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Incident Name")
incident_name = st.text_input(
    "",
    placeholder="Enter a short descriptive name for the incident..."
)

st.subheader("Describe the Incident")
user_input = st.text_area(
    "",
    placeholder="Example: Production API returning 500 errors due to database connection failures..."
)

# -----------------------------
# Analyze Incident Button
# -----------------------------
if st.button("Analyze Incident") and user_input and incident_name:
    with st.spinner("AI analyzing incident..."):

        # 1️⃣ ML Prediction
        predicted_root, confidence = predict_root_cause(user_input)

        # 2️⃣ RAG Recommendation
        answer, refs = rag.generate(user_input)

        # 3️⃣ Save incident to session (without executive summary)
        st.session_state.analyzed_incidents.append({
            "name": incident_name,
            "predicted_root": predicted_root,
            "confidence": confidence,
            "answer": answer,   # display on Streamlit
            "refs": refs        # for similar incidents & PDF generation
        })

    st.success(f"Incident '{incident_name}' added to session!")

# -----------------------------
# Display last incident
# -----------------------------
if st.session_state.analyzed_incidents:
    last_incident = st.session_state.analyzed_incidents[-1]

    st.divider()
    st.subheader(f"Predicted Root Cause for '{last_incident['name']}'")
    st.markdown(
        f"<span title='{last_incident['predicted_root']}'>{last_incident['predicted_root']}</span>",
        unsafe_allow_html=True
    )

    if show_confidence:
        st.subheader("AI Confidence")
        st.write(f"{last_incident['confidence']*100:.1f}%")

    st.subheader("🤖 AI Recommended Resolution")
    st.write(last_incident['answer'])  # RAG answer, not exec summary

# -----------------------------
# PDF Generation
# -----------------------------
def generate_multi_incident_pdf(incidents):
    pdf = FPDF()
    for inc in incidents:
        # Generate executive summary dynamically for PDF
        exec_summary = generate_executive_summary(
            inc['predicted_root'],
            inc['confidence'],
            inc['answer'],
            inc['refs']
        )

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Executive Summary: {inc['name']}", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, f"Predicted Root Cause: {inc['predicted_root']} (Confidence: {inc['confidence']*100:.1f}%)")
        pdf.ln(3)
        pdf.multi_cell(0, 8, exec_summary)
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin1")

if st.session_state.analyzed_incidents:
    pdf_bytes = generate_multi_incident_pdf(st.session_state.analyzed_incidents)
    st.download_button(
        label="📄 Download Multi-Incident Executive Summary PDF",
        data=pdf_bytes,
        file_name="executive_summary.pdf",
        mime="application/pdf"
    )

# -----------------------------
# Similar Incidents (optional)
# -----------------------------
if debug_mode and st.session_state.analyzed_incidents:
    last_refs = st.session_state.analyzed_incidents[-1]["refs"]
    with st.expander("🔎 Similar Historical Incidents"):
        for i, r in enumerate(last_refs, start=1):
            st.write(f"**Incident {i}**")
            st.write(r)
            st.divider()