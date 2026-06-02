# FactCheck Agent

Automated PDF fact-checking app built with Streamlit.

## What it does

1. Extracts key verifiable claims from a PDF
2. Searches the web for each claim
3. Verifies claims with an LLM (Groq by default, Gemini optional)
4. Returns verdicts: Verified / Inaccurate / False / Unverifiable

## Tech stack

- Streamlit (python library)
- Groq SDK (`llama-3.1-8b-instant`) as default LLM
- Google GenAI SDK as optional fallback
- pdfplumber
- duckduckgo-search

## Local setup

```bash
git clone https://github.com/Saumya-bhardwaj04/factcheck-agent.git
cd factcheck-agent
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud secrets

Add at least one key:

```toml
GROQ_API_KEY = "gsk_..."
GEMINI_API_KEY = "AIza..." # optional fallback
```

## Get API keys

- Groq (recommended): https://console.groq.com/keys
- Gemini: https://aistudio.google.com/app/apikey
