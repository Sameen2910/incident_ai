# -----------------------------
# Executive Summary using AI
# -----------------------------
# executive_summary.py
from ollama import Client

def generate_executive_summary(predicted_root, confidence, answer, refs, model_name="llama2:7b"):
    """
    Generates a concise 5-6 line executive summary for management.

    Args:
        predicted_root (str): ML predicted root cause
        confidence (float): Confidence score of ML prediction (0-1)
        answer (str): RAG-generated recommendation text
        refs (list): List of similar historical incidents
        model_name (str): LLM model name to use (default: llama2:7b)

    Returns:
        str: Executive-friendly summary
    """
    client = Client()
    # Format similar incidents nicely
    refs_text = "\n".join([f"- {r}" for r in refs]) if refs else "None"

    prompt = f"""
You are an executive summary generator. Create a concise 5-6 line summary for management.

Predicted Root Cause:
{predicted_root} (confidence: {confidence*100:.1f}%)

AI Recommendation:
{answer}

Similar Incidents:
{refs_text}

Provide a short executive-friendly summary.
"""
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return response["message"]["content"]