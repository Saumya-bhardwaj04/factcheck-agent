# 🔍 FactCheck Agent

**Automated PDF Fact-Checking Web App** — CogCulture Management Trainee Assessment (Part 2)

---

## What it does

Upload any PDF and the app will:
1. **Extract** all verifiable claims — stats, dates, financial figures, technical facts
2. **Search** the live web for each claim using DuckDuckGo (no API key needed)
3. **Verify** each claim using Google Gemini AI (free)
4. **Report** every claim as ✅ Verified / ⚠️ Inaccurate / ❌ False / 🔵 Unverifiable

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| AI / LLM | Google Gemini 2.0 Flash-Lite via `google-genai` (free tier) |
| PDF Parsing | pdfplumber |
| Web Search | DuckDuckGo Search (free, no key needed) |
| Deployment | Streamlit Cloud (free) |

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/factcheck-agent.git
cd factcheck-agent
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → main file: `app.py`
4. **Advanced settings → Secrets** → add:
   ```
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy** → live in ~2 minutes

---

## Get a Free Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API Key** → **Create API key**
4. Paste it in the app sidebar — done!
