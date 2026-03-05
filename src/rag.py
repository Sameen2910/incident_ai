from ollama import Client
from hybrid_retriever import HybridRetriever
from predict_root_cause import predict_root_cause
import subprocess

class IncidentRAG:

    def __init__(self, model_name="llama3:8b"):

        self.client = Client()
        self.model_name = model_name
        self.retriever = HybridRetriever()

        if not self._is_model_available():
            subprocess.run(["ollama", "pull", model_name], check=True)

    def _is_model_available(self):

        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )

        return self.model_name in result.stdout

    def generate(self, query):

        predicted_root,confidence = predict_root_cause(query)

        retrieved = self.retriever.search(query)

        context = "\n".join([f"- {r}" for r in retrieved])

        prompt = f"""
You are an AI Incident Resolution Assistant.

Incident:
{query}

Predicted Root Cause:
{predicted_root}

AI Confidence: {confidence*100:.1f}%

Relevant historical incidents:
{context}

Respond in this format:

Root Cause Analysis:
Explain the likely root cause.

Recommended Actions:
1.
2.
3.

Prevention Steps:
List steps to avoid recurrence.
"""

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )

        return response["message"]["content"], retrieved