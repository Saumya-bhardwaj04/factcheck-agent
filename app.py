import io
import json
import os
import textwrap
import time

import pdfplumber
import streamlit as st
from duckduckgo_search import DDGS
from fpdf import FPDF
from groq import Groq


st.set_page_config(
    page_title="FactCheck Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CLAIMS = 5
MAX_SOURCES_PER_CLAIM = 2

try:
    DEFAULT_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, .stApp {
        background-color: #04080f !important;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
    }

    /* ── Animated background grid ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(14, 165, 233, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 165, 233, 0.03) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
    }
    .stApp::after {
        content: '';
        position: fixed;
        top: -40%;
        left: 50%;
        transform: translateX(-50%);
        width: 900px;
        height: 500px;
        background: radial-gradient(ellipse, rgba(99, 102, 241, 0.08) 0%, rgba(14, 165, 233, 0.05) 40%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060c18 0%, #0a1120 100%) !important;
        border-right: 1px solid rgba(30, 45, 69, 0.8) !important;
        padding-top: 1.5rem;
        backdrop-filter: blur(10px);
    }
    section[data-testid="stSidebar"] > div { padding: 0 1.2rem; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] h2 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
        letter-spacing: 0.2px;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 13px !important;
        color: #64748b !important;
        line-height: 1.8;
    }

    /* ── Hero ── */
    .hero-wrap {
        text-align: center;
        padding: 3rem 1rem 2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.07);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 100px;
        padding: 6px 18px;
        font-size: 11.5px;
        color: #38bdf8 !important;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 1.4rem;
        animation: fadeSlideDown 0.6s ease both;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.6rem, 6vw, 3.6rem);
        font-weight: 800;
        letter-spacing: -2px;
        margin: 0 0 1rem;
        line-height: 1.05;
        background: linear-gradient(135deg, #f8fafc 0%, #93c5fd 45%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeSlideDown 0.7s 0.1s ease both;
    }
    .hero-sub {
        font-size: 1rem;
        color: #475569 !important;
        max-width: 500px;
        width: 100%;
        margin: 0 auto 0.5rem;
        line-height: 1.7;
        font-weight: 400;
        animation: fadeSlideDown 0.7s 0.2s ease both;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0ea5e915;
        border: 1px solid #0ea5e930;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px;
        color: #38bdf8 !important;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin-bottom: 0.3rem;
    }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }

    /* ── File uploader ── */
    @keyframes scanLine {
        0%   { top: -100%; opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    div[data-testid="stFileUploader"] {
        position: relative;
        background: linear-gradient(135deg, #070d1a, #0d1729) !important;
        border-radius: 16px;
        padding: 8px 10px;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow:
            0 0 0 1px rgba(125, 211, 252, 0.25),
            0 0 0 3px rgba(99, 102, 241, 0.1),
            0 20px 60px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }
    div[data-testid="stFileUploader"]::after {
        content: '';
        position: absolute;
        left: 0; right: 0; height: 100%;
        background: linear-gradient(to bottom, transparent, rgba(2, 132, 199, 0.06) 80%, rgba(125, 211, 252, 0.3) 100%);
        pointer-events: none;
        animation: scanLine 4s linear infinite;
    }
    div[data-testid="stFileUploader"]:hover {
        box-shadow:
            0 0 0 1px rgba(125, 211, 252, 0.45),
            0 0 0 3px rgba(99, 102, 241, 0.2),
            0 24px 70px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.06);
        transform: translateY(-2px);
    }
    div[data-testid="stFileUploader"] * { color: #94a3b8 !important; }

    .upload-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(1.3rem, 3vw, 1.65rem);
        font-weight: 800;
        letter-spacing: -0.8px;
        text-align: center;
        margin: 0.5rem 0 1.2rem;
        background: linear-gradient(135deg, #e2e8f0 0%, #7dd3fc 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
        color: #fff !important;
        border: none;
        border-radius: 12px;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700;
        font-size: 15px;
        padding: 14px 28px;
        width: 100%;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35), 0 0 0 1px rgba(255,255,255,0.05) inset;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        transition: left 0.4s ease;
    }
    .stButton > button:hover::before { left: 100%; }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #4f46e5 100%);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.5), 0 0 0 1px rgba(255,255,255,0.08) inset;
        transform: translateY(-2px);
    }
    .stButton > button:active { transform: translateY(0); }
    .stButton > button * { color: #fff !important; }

    /* ── Stat boxes ── */
    .stat-box {
        background: linear-gradient(145deg, #080e1c, #0d1525);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        border: 1px solid rgba(30, 45, 69, 0.9);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent, #60a5fa), transparent);
        opacity: 0.9;
    }
    .stat-box::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 50% 0%, var(--accent, #60a5fa) 0%, transparent 70%);
        opacity: 0.04;
        pointer-events: none;
    }
    .stat-box:hover { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
    .stat-number {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1.5px;
    }
    .stat-label { font-size: 10.5px; color: #334155; margin-top: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

    /* ── Claim cards ── */
    .claim-card {
        background: linear-gradient(135deg, #080e1c 0%, #0d1525 100%);
        border-radius: 16px;
        padding: 24px 26px;
        margin-bottom: 16px;
        border: 1px solid rgba(30, 45, 69, 0.9);
        border-left: 4px solid #1e2d45;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.025);
        transition: transform 0.2s, box-shadow 0.2s;
        animation: fadeIn 0.4s ease both;
    }
    .claim-card:hover { transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0,0,0,0.5); }
    .verified   { border-left-color: #10b981 !important; }
    .inaccurate { border-left-color: #f59e0b !important; }
    .false      { border-left-color: #ef4444 !important; }
    .unknown    { border-left-color: #6366f1 !important; }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 13px;
        border-radius: 100px;
        font-size: 10.5px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
    }
    .badge-verified   { background:#011f16; color:#34d399; border:1px solid rgba(16,185,129,0.25); }
    .badge-inaccurate { background:#1f1100; color:#fbbf24; border:1px solid rgba(245,158,11,0.25); }
    .badge-false      { background:#1f0505; color:#f87171; border:1px solid rgba(239,68,68,0.25); }
    .badge-unknown    { background:#0c0b36; color:#a5b4fc; border:1px solid rgba(99,102,241,0.25); }

    .claim-text {
        font-family: 'Syne', sans-serif;
        font-size: 14.5px;
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 9px;
        line-height: 1.55;
    }
    .explanation { font-size: 13px; color: #64748b !important; line-height: 1.75; }
    .correct-val {
        font-size: 12.5px; color: #6ee7b7; margin-top: 12px; font-weight: 500;
        background: rgba(2,44,34,0.4); padding: 8px 14px; border-radius: 8px;
        border: 1px solid rgba(6,95,70,0.3); display: inline-block;
    }
    .source-link { font-size: 12px; color: #60a5fa; margin-top: 10px; display: inline-flex; align-items: center; gap: 5px; }
    .source-link a { color: #60a5fa; text-decoration: none; }
    .source-link a:hover { color: #93c5fd; text-decoration: underline; }

    .pill-row { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    .cat-pill {
        background: rgba(30, 45, 69, 0.8);
        color: #94a3b8;
        padding: 3px 11px;
        border-radius: 100px;
        font-size: 10.5px;
        font-weight: 700;
        border: 1px solid rgba(45,62,85,0.8);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .conf-pill { padding: 3px 11px; border-radius: 100px; font-size: 10.5px; font-weight: 700; background: rgba(8,14,28,0.8); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(8, 14, 28, 0.8);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(30, 45, 69, 0.8);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        color: #475569 !important;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 18px;
        font-family: 'DM Sans', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f2744, #1a1f4e) !important;
        color: #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    div[data-testid="stStatusWidget"] {
        background: rgba(8,14,28,0.9) !important;
        border: 1px solid rgba(30,45,69,0.8) !important;
        border-radius: 12px !important;
    }
    hr { border-color: rgba(30, 45, 69, 0.6) !important; margin: 1.5rem 0 !important; }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    p { color: #64748b !important; }
    div[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid rgba(30,45,69,0.8) !important; }

    @media (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .hero-title { font-size: 1.9rem !important; letter-spacing: -1px; }
        .stat-number { font-size: 1.6rem !important; }
        .claim-card { padding: 16px 18px; }
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #04080f; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, #0284c7, #1e3a5f); 
    border-radius: 10px; 
    border: 1px solid rgba(14, 165, 233, 0.15);
    }
    ::-webkit-scrollbar-thumb:hover { 
    background: linear-gradient(180deg, #38bdf8, #0369a1); 
    }
    ::-webkit-scrollbar-corner { background: #04080f; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def clean_json(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 FactCheck Agent")
    st.markdown("Verify claims in a PDF using live web search + fast LLM checks.")
    st.divider()
    groq_client = None
    groq_error = ""
    if DEFAULT_GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=DEFAULT_GROQ_API_KEY)
        except Exception as e:
            groq_error = str(e)

    api_ready = groq_client is not None
    model_name = GROQ_MODEL
    if api_ready:
        st.success("✅ Groq API Key Activated")
    else:
        st.error(f"❌ Groq API Key Missing/Invalid in secrets. {groq_error[:120]}")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
<div style="font-size: 13px; line-height: 2; color: #94a3b8;">
    <b style="color:#e2e8f0">1.</b> 📄 Upload a PDF<br>
    <b style="color:#e2e8f0">2.</b> 🤖 LLM extracts key claims<br>
    <b style="color:#e2e8f0">3.</b> 🌐 Live web search verification<br>
    <b style="color:#e2e8f0">4.</b> 🎯 Rapid batched verdicts
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""
<div style="font-size:12px; line-height:2.2; color:#64748b;">
    ✅ <b style="color:#34d399">Verified</b> — Confirmed by sources<br>
    ⚠️ <b style="color:#fbbf24">Inaccurate</b> — Wrong numbers/dates<br>
    ❌ <b style="color:#f87171">False</b> — Contradicted by evidence<br>
    🔵 <b style="color:#a5b4fc">Unverifiable</b> — No evidence found
</div>
""", unsafe_allow_html=True)


# ── Core logic ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str, retries: int = 3) -> str:
    if groq_client is None:
        raise RuntimeError("Groq client is not initialized.")
    last_error = ""
    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return exactly the requested output format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2400,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            raise RuntimeError("Groq returned an empty response.")
        except Exception as e:
            err = str(e)
            last_error = err
            if "429" in err or "rate limit" in err.lower():
                wait = 4 * (attempt + 1)
                st.warning(f"Rate limit hit — waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Groq API failed after retries. Last error: {last_error[:180]}")


def extract_claims(text: str) -> list:
    prompt = f"""You are a fact-checking assistant.
Extract the most important verifiable claims from the text below.

Rules:
- Return ONLY a JSON array.
- No markdown. No extra text.
- Each item must contain:
  - "claim" (string)
  - "category" (one of Statistic|Date|Financial|Technical|Attribution|Ranking)
  - "context" (max 80 chars)
- Extract up to {MAX_CLAIMS} claims.

TEXT:
{text[:5000]}
"""
    raw = call_llm(prompt)
    claims = json.loads(clean_json(raw))
    return claims[:MAX_CLAIMS]


def web_search(query: str, n: int = 4) -> list:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query[:120], max_results=n):
                results.append({
                    "title": r.get("title", ""),
                    "body":  r.get("body", ""),
                    "href":  r.get("href", ""),
                })
    except Exception:
        pass
    return results


def verify_claims_batch(claims: list, search_map: dict) -> list:
    payload = []
    for i, claim in enumerate(claims):
        hits = search_map.get(i, [])[:MAX_SOURCES_PER_CLAIM]
        payload.append({
            "id": i,
            "claim":    claim.get("claim", ""),
            "category": claim.get("category", ""),
            "context":  claim.get("context", ""),
            "sources": [
                {
                    "title":   h.get("title", "")[:140],
                    "url":     h.get("href", ""),
                    "snippet": h.get("body", "")[:240],
                }
                for h in hits
            ],
        })

    prompt = f"""You are a professional fact-checker.
Verify each claim using the provided sources.

Input JSON:
{json.dumps(payload, ensure_ascii=True)}

Return ONLY a JSON array with one object per input item.
Each object must include:
- "id" (matching input id, integer)
- "verdict" ("Verified" | "Inaccurate" | "False" | "Unverifiable")
- "explanation" (1-2 short sentences)
- "correct_value" (string or null)
- "source_url" (string or null)
- "confidence" ("High" | "Medium" | "Low")
"""
    raw   = call_llm(prompt)
    items = json.loads(clean_json(raw))
    by_id = {
        int(item.get("id")): item
        for item in items
        if isinstance(item, dict) and str(item.get("id", "")).isdigit()
    }

    normalized    = []
    valid_verdicts = {"Verified", "Inaccurate", "False", "Unverifiable"}
    valid_conf     = {"High", "Medium", "Low"}

    for i, claim in enumerate(claims):
        item       = by_id.get(i, {})
        verdict    = item.get("verdict", "Unverifiable")
        confidence = item.get("confidence", "Low")
        if verdict not in valid_verdicts:
            verdict = "Unverifiable"
        if confidence not in valid_conf:
            confidence = "Low"
        normalized.append({
            "claim":         claim.get("claim", "Unknown"),
            "category":      claim.get("category", ""),
            "context":       claim.get("context", ""),
            "verdict":       verdict,
            "explanation":   item.get("explanation", "Insufficient evidence from available sources."),
            "correct_value": item.get("correct_value"),
            "source_url":    item.get("source_url"),
            "confidence":    confidence,
        })
    return normalized


# ── UI helpers ─────────────────────────────────────────────────────────────────

def verdict_badge(verdict: str) -> str:
    icons = {
        "Verified":     ("✅", "badge-verified"),
        "Inaccurate":   ("⚠️", "badge-inaccurate"),
        "False":        ("❌", "badge-false"),
        "Unverifiable": ("🔵", "badge-unknown"),
    }
    icon, cls = icons.get(verdict, ("❓", "badge-unknown"))
    return f'<span class="verdict-badge {cls}">{icon} {verdict.upper()}</span>'


def card_class(verdict: str) -> str:
    return {
        "Verified":     "verified",
        "Inaccurate":   "inaccurate",
        "False":        "false",
        "Unverifiable": "unknown",
    }.get(verdict, "unknown")


def render_cards(items):
    if not items:
        st.info("No claims in this category.")
        return
    for r in items:
        v          = r.get("verdict", "Unverifiable")
        badge      = verdict_badge(v)
        cc         = card_class(v)
        conf_color = {"High": "#34d399", "Medium": "#fbbf24", "Low": "#f87171"}.get(r.get("confidence", ""), "#94a3b8")
        correct    = f'<div class="correct-val">📌 Correct: {r["correct_value"]}</div>' if r.get("correct_value") else ""
        source     = f'<div class="source-link">🔗 <a href="{r["source_url"]}" target="_blank">{r["source_url"][:70]}</a></div>' if r.get("source_url") else ""
        cat_pill   = f'<span class="cat-pill">{r.get("category","")}</span>'
        conf_pill  = f'<span class="conf-pill" style="color:{conf_color}; border:1px solid {conf_color}30;">Confidence: {r.get("confidence","")}</span>'
        st.markdown(f"""
        <div class="claim-card {cc}">
            <div class="pill-row">{badge} {cat_pill} {conf_pill}</div>
            <div class="claim-text">"{r['claim']}"</div>
            <div class="explanation">{r.get('explanation', '')}</div>
            {correct}{source}
        </div>""", unsafe_allow_html=True)


def build_pdf_report(results: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(0, 10, "FactCheck Agent Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    summary = {
        "Verified": sum(1 for r in results if r.get("verdict") == "Verified"),
        "Inaccurate": sum(1 for r in results if r.get("verdict") == "Inaccurate"),
        "False": sum(1 for r in results if r.get("verdict") == "False"),
        "Unverifiable": sum(1 for r in results if r.get("verdict") == "Unverifiable"),
    }
    pdf.cell(0, 8, f"Claims checked: {len(results)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Verified: {summary['Verified']} | Inaccurate: {summary['Inaccurate']} | False: {summary['False']} | Unverifiable: {summary['Unverifiable']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    content_width = pdf.w - pdf.l_margin - pdf.r_margin

    def safe_text(value: str) -> str:
        return str(value).encode("latin-1", "replace").decode("latin-1")

    def normalize_text(text: str) -> str:
        text = safe_text(text)
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        return " ".join(text.split())

    def split_long_tokens(text: str, max_len: int = 30) -> str:
        tokens = []
        for token in normalize_text(text).split():
            if len(token) > max_len:
                chunks = [token[i : i + max_len] for i in range(0, len(token), max_len)]
                token = " ".join(chunks)
            tokens.append(token)
        return " ".join(tokens)

    def write_wrapped(label: str, text: str) -> None:
        if not text:
            return
        pdf.set_font("Helvetica", size=11, style="B")
        pdf.cell(0, 6, safe_text(f"{label}:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        normalized = split_long_tokens(text)
        pdf.set_x(pdf.l_margin)
        for line in textwrap.wrap(normalized, width=80, break_long_words=True, break_on_hyphens=False):
            pdf.multi_cell(content_width, 6, safe_text(line), new_x="LMARGIN", new_y="NEXT")

    for idx, r in enumerate(results, start=1):
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(0, 8, safe_text(f"{idx}. {r.get('verdict', 'Unverifiable')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)

        write_wrapped("Claim", r.get("claim", ""))
        write_wrapped("Reason", r.get("explanation", ""))
        write_wrapped("Correct", r.get("correct_value"))
        write_wrapped("Source", r.get("source_url"))

        pdf.ln(2)

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", "replace")


# ── Main page ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <span class="hero-eyebrow">⚡ Powered by Groq · DuckDuckGo · LLaMA 3</span>
    <h1 class="hero-title">FactCheck Agent</h1>
    <p class="hero-sub">Upload any document and get key claims instantly verified against live web data using LLMs.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

if not api_ready:
    st.error("Groq API key not configured. Add `GROQ_API_KEY` in Streamlit secrets.")
    st.stop()

st.markdown('<div class="upload-title">Get your file uploaded today</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Drop your PDF here", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    st.write("")
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        run = st.button("🚀 Run Fact Check", use_container_width=True)

    if run:
        with st.status("Running fact-check pipeline...", expanded=True) as status:

            st.write("📄 Reading PDF...")
            try:
                uploaded_file.seek(0)
                pdf_text = read_pdf(uploaded_file)
                if not pdf_text:
                    st.error("Could not extract text. Ensure it is not an image-only PDF.")
                    st.stop()
                st.write(f"✅ Got **{len(pdf_text.split()):,} words** from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"PDF error: {e}")
                st.stop()

            st.write("🤖 Extracting claims with Groq...")
            try:
                claims = extract_claims(pdf_text)
                st.write(f"✅ Found **{len(claims)} claims** to verify")
            except Exception as e:
                st.error(f"Claim extraction failed: {e}")
                st.stop()

            st.write("🌐 Searching web sources...")
            progress   = st.progress(0)
            search_map = {}
            for i, claim in enumerate(claims):
                search_map[i] = web_search(claim.get("claim", ""), n=MAX_SOURCES_PER_CLAIM)
                progress.progress((i + 1) / max(1, len(claims)))
                time.sleep(0.25)

            st.write("🤖 Verifying claims in one Groq batch call...")
            try:
                results = verify_claims_batch(claims, search_map)
            except Exception as ex:
                results = [{
                    "claim":         claim.get("claim", "Unknown"),
                    "category":      claim.get("category", ""),
                    "context":       claim.get("context", ""),
                    "verdict":       "Unverifiable",
                    "explanation":   f"Batch verification error: {str(ex)[:160]}",
                    "correct_value": None,
                    "source_url":    None,
                    "confidence":    "Low",
                } for claim in claims]

            status.update(label="✅ Fact Check Complete!", state="complete", expanded=False)

        # Stats
        st.markdown("---")
        counts = {
            v: sum(1 for r in results if r["verdict"] == v)
            for v in ["Verified", "Inaccurate", "False", "Unverifiable"]
        }
        accuracy = round(counts["Verified"] / len(results) * 100) if results else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, num, label, color in [
            (c1, len(results),         "Claims Checked", "#60a5fa"),
            (c2, counts["Verified"],   "✅ Verified",    "#34d399"),
            (c3, counts["Inaccurate"], "⚠️ Inaccurate", "#fbbf24"),
            (c4, counts["False"],      "❌ False",       "#ef4444"),
            (c5, f"{accuracy}%",       "Accuracy Rate",  "#a78bfa"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat-box" style="--accent:{color}">
                    <div class="stat-number" style="color:{color}">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        # Results
        st.markdown("---")
        st.markdown("### 📋 Detailed Results")
        tab_all, tab_false, tab_inaccurate, tab_verified = st.tabs([
            f"All ({len(results)})",
            f"❌ False ({counts['False']})",
            f"⚠️ Inaccurate ({counts['Inaccurate']})",
            f"✅ Verified ({counts['Verified']})",
        ])
        with tab_all:        render_cards(results)
        with tab_false:      render_cards([r for r in results if r["verdict"] == "False"])
        with tab_inaccurate: render_cards([r for r in results if r["verdict"] == "Inaccurate"])
        with tab_verified:   render_cards([r for r in results if r["verdict"] == "Verified"])

        st.markdown("---")
        pdf_bytes = build_pdf_report(results)
        st.download_button(
            "⬇️ Download Full Report (PDF)",
            data=pdf_bytes,
            file_name="factcheck_report.pdf",
            mime="application/pdf",
        )
