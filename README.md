A # AI Incident Intelligence Copilot
An AI-powered Streamlit app to analyze incidents, predict root causes, provide recommendations, and generate executive summary PDFs.

B ## **Prerequisites**
- Python 3.10+
- Ollama CLI (for LLM inference)
- Required Python packages:

```bash
pip install -r requirements.txtsee `requirements.txt`):

C Installation / Setup**

This is where you guide the user step by step:

1. **Install Ollama CLI**
2. **Pull required Ollama models** (`llama3:8b` for RAG, `llama2:7b` for executive summary)
3. **Verify Ollama installation** (`ollama list`)
4. **Create and activate a virtual environment**
5. **Install Python packages from `requirements.txt`**

Example:

```markdown

Installation / Setup

1. Install Ollama CLI:

- macOS: `brew install ollama`
- Windows: Download installer from https://ollama.com/
- Linux: see https://ollama.com/docs

2. Pull required models:

```bash
ollama pull llama3:8b   # RAG model
ollama pull llama2:7b   # Executive summary

D Verify Models:

```bash
ollama list

E Create and activate a virtual environment:
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

F Install Python dependencies
pip install -r requirements.txt

G Running the App**

After setup, include instructions on **how to start Streamlit**:

```markdown
## Running the App

1. Make sure your virtual environment is active.
2. Run the app:

```bash
streamlit run src/incident_app.py

H In the app, provide:
* Incident description
* Incident name
* Click Analyze Incident to see:
* App will show:
-----------> ML Root Cause
-----------> AI Recommendations (RAG)
-----------> Similar Historical Incidents
-----------> Executive Summary (PDF export Optional)