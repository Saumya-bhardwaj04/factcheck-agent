import io
import json
import re
import time

import pdfplumber
import streamlit as st
from duckduckgo_search import DDGS
from google import genai
from groq import Groq


st.set_page_config(
    page_title="FactCheck Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

GEMINI_MODEL = "gemini-2.0-flash-lite"
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CLAIMS = 5
MAX_SOURCES_PER_CLAIM = 2

DEFAULT_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
DEFAULT_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")


st.markdown(
    """
<style>
    .stApp { background-color: #0f172a; }

    .claim-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-left: 5px solid #64748b;
    }
    .verified   { border-left-color: #10b981 !important; }
    .inaccurate { border-left-color: #f59e0b !important; }
    .false      { border-left-color: #ef4444 !important; }
    .unknown    { border-left-color: #6366f1 !important; }

    .verdict-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .badge-verified   { background:#064e3b; color:#34d399; }
    .badge-inaccurate { background:#451a03; color:#fbbf24; }
    .badge-false      { background:#450a0a; color:#f87171; }
    .badge-unknown    { background:#1e1b4b; color:#a5b4fc; }

    .claim-text  { font-size:15px; color:#e2e8f0; font-weight:500; margin-bottom:6px; }
    .explanation { font-size:13px; color:#94a3b8; line-height:1.6; }
    .correct-val { font-size:13px; color:#6ee7b7; margin-top:6px; }
    .source-link { font-size:11px; color:#60a5fa; margin-top:4px; }

    .stat-box {
        background:#1e293b; border-radius:10px;
        padding:16px; text-align:center;
    }
    .stat-number { font-size:28px; font-weight:700; }
    .stat-label  { font-size:12px; color:#94a3b8; margin-top:2px; }

    h1, h2, h3 { color:#f1f5f9 !important; }
    p { color:#94a3b8 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #0d9488, #0891b2);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 28px;
        font-size: 15px; width: 100%;
    }
    section[data-testid="stSidebar"] { background:#1e293b !important; }
    section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
</style>
""",
    unsafe_allow_html=True,
)


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


with st.sidebar:
    st.markdown("## 🔍 FactCheck Agent")
    st.markdown("Verify claims in a PDF using live web search + fast LLM checks.")
    st.divider()

    provider = st.selectbox("LLM Provider", ["Groq", "Gemini"], index=0)

    groq_key = st.text_input(
        "Groq API key",
        value="",
        type="password",
        help="Recommended for this app if Gemini free-tier is rate-limited.",
    ).strip() or DEFAULT_GROQ_API_KEY
    groq_model = st.text_input(
        "Groq model",
        value=GROQ_MODEL,
        help="Change this if your Groq project restricts model permissions.",
    ).strip() or GROQ_MODEL

    gemini_key = st.text_input(
        "Gemini API key",
        value="",
        type="password",
        help="Optional fallback.",
    ).strip() or DEFAULT_GEMINI_API_KEY

    groq_client = None
    gemini_client = None
    groq_error = ""
    gemini_error = ""

    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
        except Exception as e:
            groq_error = str(e)
    if gemini_key:
        try:
            gemini_client = genai.Client(api_key=gemini_key)
        except Exception as e:
            gemini_error = str(e)

    if provider == "Groq":
        api_ready = groq_client is not None
        model_name = groq_model
        if api_ready:
            st.success("✅ Groq API Connected")
        else:
            st.error(f"❌ Groq key missing/invalid. {groq_error[:120]}")
    else:
        api_ready = gemini_client is not None
        model_name = GEMINI_MODEL
        if api_ready:
            st.success("✅ Gemini API Connected")
        else:
            st.error(f"❌ Gemini key missing/invalid. {gemini_error[:120]}")

    if api_ready and st.button("Test API key now", use_container_width=True):
        try:
            if provider == "Groq":
                probe = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    temperature=0,
                    max_tokens=12,
                )
                text = (probe.choices[0].message.content or "").strip()
            else:
                probe = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents="Reply with exactly: OK",
                )
                text = (getattr(probe, "text", None) or "").strip()
            st.success(f"Provider test passed: {text or 'OK'}")
        except Exception as probe_err:
            st.error(f"Provider test failed: {probe_err}")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. LLM extracts key verifiable claims")
    st.markdown("3. Each claim is web searched")
    st.markdown("4. LLM returns verdicts in one batch")


def call_llm(prompt: str, retries: int = 3) -> str:
    if provider == "Groq":
        if groq_client is None:
            raise RuntimeError("Groq client is not initialized.")
        last_error = ""
        for attempt in range(retries):
            try:
                resp = groq_client.chat.completions.create(
                    model=groq_model,
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
                    st.warning(f"Rate limit hit - waiting {wait}s before retry {attempt+1}/{retries}...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Groq API failed after retries. Last error: {last_error[:180]}")

    if gemini_client is None:
        raise RuntimeError("Gemini client is not initialized.")

    last_error = ""
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text
            raise RuntimeError("Gemini returned an empty response.")
        except Exception as e:
            err = str(e)
            last_error = err
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                retry_hint = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", err, flags=re.IGNORECASE)
                if retry_hint:
                    wait = max(5, int(float(retry_hint.group(1)) + 2))
                else:
                    wait = 20 * (attempt + 1)
                st.warning(f"Rate limit hit - waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini API failed after retries. Last error: {last_error[:180]}")


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
                results.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    }
                )
    except Exception:
        pass
    return results


def verify_claims_batch(claims: list, search_map: dict) -> list:
    payload = []
    for i, claim in enumerate(claims):
        hits = search_map.get(i, [])[:MAX_SOURCES_PER_CLAIM]
        payload.append(
            {
                "id": i,
                "claim": claim.get("claim", ""),
                "category": claim.get("category", ""),
                "context": claim.get("context", ""),
                "sources": [
                    {
                        "title": h.get("title", "")[:140],
                        "url": h.get("href", ""),
                        "snippet": h.get("body", "")[:240],
                    }
                    for h in hits
                ],
            }
        )

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

    raw = call_llm(prompt)
    items = json.loads(clean_json(raw))

    by_id = {
        int(item.get("id")): item
        for item in items
        if isinstance(item, dict) and str(item.get("id", "")).isdigit()
    }

    normalized = []
    valid_verdicts = {"Verified", "Inaccurate", "False", "Unverifiable"}
    valid_conf = {"High", "Medium", "Low"}

    for i, claim in enumerate(claims):
        item = by_id.get(i, {})
        verdict = item.get("verdict", "Unverifiable")
        confidence = item.get("confidence", "Low")
        if verdict not in valid_verdicts:
            verdict = "Unverifiable"
        if confidence not in valid_conf:
            confidence = "Low"

        normalized.append(
            {
                "claim": claim.get("claim", "Unknown"),
                "category": claim.get("category", ""),
                "context": claim.get("context", ""),
                "verdict": verdict,
                "explanation": item.get("explanation", "Insufficient evidence from available sources."),
                "correct_value": item.get("correct_value"),
                "source_url": item.get("source_url"),
                "confidence": confidence,
            }
        )

    return normalized


def verdict_badge(verdict: str) -> str:
    icons = {
        "Verified": ("✅", "badge-verified"),
        "Inaccurate": ("⚠️", "badge-inaccurate"),
        "False": ("❌", "badge-false"),
        "Unverifiable": ("🔵", "badge-unknown"),
    }
    icon, cls = icons.get(verdict, ("❓", "badge-unknown"))
    return f'<span class="verdict-badge {cls}">{icon} {verdict.upper()}</span>'


def card_class(verdict: str) -> str:
    return {
        "Verified": "verified",
        "Inaccurate": "inaccurate",
        "False": "false",
        "Unverifiable": "unknown",
    }.get(verdict, "unknown")


def render_cards(items):
    if not items:
        st.info("No claims in this category.")
        return
    for r in items:
        v = r.get("verdict", "Unverifiable")
        badge = verdict_badge(v)
        cc = card_class(v)
        conf_color = {"High": "#34d399", "Medium": "#fbbf24", "Low": "#f87171"}.get(
            r.get("confidence", ""),
            "#94a3b8",
        )
        correct = (
            f'<div class="correct-val">📌 Correct: {r["correct_value"]}</div>'
            if r.get("correct_value")
            else ""
        )
        source = (
            f'<div class="source-link">🔗 <a href="{r["source_url"]}" target="_blank">{r["source_url"][:70]}</a></div>'
            if r.get("source_url")
            else ""
        )
        cat_pill = f'<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:10px;font-size:11px;">{r.get("category","")}</span>'
        conf_pill = f'<span style="background:#1e293b;color:{conf_color};padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid {conf_color}40;">Confidence: {r.get("confidence","")}</span>'
        st.markdown(
            f"""
        <div class="claim-card {cc}">
            {badge} &nbsp; {cat_pill} &nbsp; {conf_pill}
            <div class="claim-text">"{r['claim']}"</div>
            <div class="explanation">{r.get('explanation', '')}</div>
            {correct}{source}
        </div>""",
            unsafe_allow_html=True,
        )


st.markdown("# 🔍 FactCheck Agent")
st.markdown("Upload any PDF and get key claims verified against live web data.")
st.divider()

if not api_ready:
    st.error("API key not configured for selected provider. Add keys in sidebar or Streamlit secrets.")
    st.stop()

uploaded_file = st.file_uploader("📂 Drop your PDF here", type=["pdf"])

if uploaded_file:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run = st.button("🚀 Run Fact Check", use_container_width=True)

    if run:
        with st.status("Running fact-check pipeline...", expanded=True) as status:
            st.write("📄 Reading PDF...")
            try:
                uploaded_file.seek(0)
                pdf_text = read_pdf(uploaded_file)
                if not pdf_text:
                    st.error("Could not extract text. Ensure it is not image-only PDF.")
                    st.stop()
                st.write(f"✅ Got **{len(pdf_text.split()):,} words** from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"PDF error: {e}")
                st.stop()

            st.write(f"🤖 Extracting claims with {provider}...")
            try:
                claims = extract_claims(pdf_text)
                st.write(f"✅ Found **{len(claims)} claims** to verify")
            except Exception as e:
                st.error(f"Claim extraction failed: {e}")
                st.stop()

            st.write("🌐 Searching web sources...")
            progress = st.progress(0)
            search_map = {}
            for i, claim in enumerate(claims):
                search_map[i] = web_search(claim.get("claim", ""), n=MAX_SOURCES_PER_CLAIM)
                progress.progress((i + 1) / max(1, len(claims)))
                time.sleep(0.25)

            st.write(f"🤖 Verifying claims in one {provider} batch call...")
            try:
                results = verify_claims_batch(claims, search_map)
            except Exception as ex:
                results = [
                    {
                        "claim": claim.get("claim", "Unknown"),
                        "category": claim.get("category", ""),
                        "context": claim.get("context", ""),
                        "verdict": "Unverifiable",
                        "explanation": f"Batch verification error: {str(ex)[:160]}",
                        "correct_value": None,
                        "source_url": None,
                        "confidence": "Low",
                    }
                    for claim in claims
                ]

            status.update(label="✅ Done!", state="complete", expanded=False)

        st.markdown("---")
        counts = {
            v: sum(1 for r in results if r["verdict"] == v)
            for v in ["Verified", "Inaccurate", "False", "Unverifiable"]
        }
        accuracy = round(counts["Verified"] / len(results) * 100) if results else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, num, label, color in [
            (c1, len(results), "Claims Checked", "#60a5fa"),
            (c2, counts["Verified"], "✅ Verified", "#34d399"),
            (c3, counts["Inaccurate"], "⚠️ Inaccurate", "#fbbf24"),
            (c4, counts["False"], "❌ False", "#f87171"),
            (c5, f"{accuracy}%", "Accuracy Rate", "#a78bfa"),
        ]:
            with col:
                st.markdown(
                    f"""<div class="stat-box">
                    <div class="stat-number" style="color:{color}">{num}</div>
                    <div class="stat-label">{label}</div></div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### Detailed Results")
        tab_all, tab_false, tab_inaccurate, tab_verified = st.tabs(
            [
                f"All ({len(results)})",
                f"❌ False ({counts['False']})",
                f"⚠️ Inaccurate ({counts['Inaccurate']})",
                f"✅ Verified ({counts['Verified']})",
            ]
        )
        with tab_all:
            render_cards(results)
        with tab_false:
            render_cards([r for r in results if r["verdict"] == "False"])
        with tab_inaccurate:
            render_cards([r for r in results if r["verdict"] == "Inaccurate"])
        with tab_verified:
            render_cards([r for r in results if r["verdict"] == "Verified"])

        st.markdown("---")
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="factcheck_report.json",
            mime="application/json",
        )
