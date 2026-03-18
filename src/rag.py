from ollama import Client
from hybrid_retriever import HybridRetriever

class IncidentRAG:

    def __init__(self, model_name="phi3:mini", top_k=3):

        self.client = Client()
        self.model_name = model_name
        self.retriever = HybridRetriever()
        self.top_k = top_k

    def generate(self, query, predicted_root, confidence):

        retrieved = self.retriever.search(query)[:self.top_k]

        MAX_CONTEXT = 400
        clean_refs = [r[:MAX_CONTEXT] for r in retrieved]

        context = "\n".join(
            [f"{i+1}. {r}" for i, r in enumerate(clean_refs)]
        )

        prompt = f"""
You are an AI Incident Copilot.

Incident:
{query}

Predicted Root Cause:
{predicted_root}

Confidence:
{confidence*100:.1f}%

Similar historical incidents:
{context}

Respond with:

Executive Summary:
(2-3 lines)

Root Cause Analysis:
Explain briefly.

Recommended Actions:
1.
2.
3.

Prevention Steps:
1.
2.

Automation Opportunity:
Suggest automation commands if possible.
"""

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 150
            }
        )

        return response["message"]["content"], clean_refs