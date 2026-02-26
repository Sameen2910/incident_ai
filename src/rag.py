# rag.py
from embed import embed_texts, load_index
from ollama import Client
import subprocess

class IncidentRAG:

    def __init__(self, model_name="llama3:8b"):
        """
        Initialize the RAG pipeline.
        Automatically downloads the model if not available locally.
        """
        self.model_name = model_name
        self.client = Client()
        self.index, self.texts = load_index()

        # -------------------------------
        # Auto-download model if missing
        # -------------------------------
        if not self._is_model_available():
            print(f"Model {self.model_name} not found locally. Downloading...")
            subprocess.run(["ollama", "pull", self.model_name], check=True)
            print(f"Model {self.model_name} downloaded successfully.")

    def _is_model_available(self):
        """
        Check if Ollama model exists locally.
        """
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return self.model_name in result.stdout

    # -------------------------------
    # Retrieve Top-k Similar Incidents
    # -------------------------------
    def retrieve(self, query, k=3):
        """
        Retrieve top-k most similar incidents.
        Truncates each retrieved incident to 512 chars for faster AI response.
        """
        query_vec = embed_texts([query])
        D, I = self.index.search(query_vec, k)
        retrieved = [self.texts[i][:512] for i in I[0]]  # truncate
        return retrieved

    # -------------------------------
    # Generate AI Response
    # -------------------------------
    def generate(self, query, predicted_root=""):
        """
        Generate root cause reasoning + recommended actions.
        predicted_root is optional (can be empty).
        """
        retrieved = self.retrieve(query)
        context = "\n".join([f"{i+1}. {t}" for i, t in enumerate(retrieved)])  # numbered

        prompt = f"""
You are an AI Incident Resolution Assistant.

New Incident:
{query}

Suggested Root Cause (optional):
{predicted_root}

Similar Historical Incidents:
{context}

Provide:
1. Root cause reasoning
2. Recommended next best actions
3. Reference which similar incidents support your suggestion
"""
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"], retrieved