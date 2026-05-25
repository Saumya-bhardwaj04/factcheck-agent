import streamlit as st
from google import genai
import pdfplumber
import json
import time
import io
import re
from duckduckgo_search import DDGS

# ── Setup ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from Streamlit secrets (set in Streamlit Cloud dashboard)
MODEL_NAME = "gemini-2.0-flash-lite"
DEFAULT_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FactCheck Agent")
    st.markdown("Verifies claims in any PDF using **Gemini 2.0 Flash-Lite** + live web search.")
    st.divider()

    user_api_key = st.text_input(
        "Gemini API key (optional override)",
        type="password",
        help="Use your own key if the shared app key is rate-limited.",
    ).strip()
    active_api_key = user_api_key or DEFAULT_API_KEY

    try:
        client = genai.Client(api_key=active_api_key) if active_api_key else None
        api_ready = client is not None
    except Exception:
        client = None
        api_ready = False

    if api_ready:
        if user_api_key:
            st.success("✅ Gemini API Connected (personal key)")
        else:
            st.success("✅ Gemini API Connected")

        if st.button("Test API key now", use_container_width=True):
            try:
                probe = client.models.generate_content(
                    model=MODEL_NAME,
                    contents="Reply with exactly: OK",
                )
                probe_text = (getattr(probe, "text", None) or "").strip()
                st.success(f"Gemini test passed: {probe_text or 'OK'}")
            except Exception as probe_err:
                st.error(f"Gemini test failed: {probe_err}")
    else:
        st.error("❌ API key missing/invalid — add `GEMINI_API_KEY` in Streamlit secrets or paste a personal key above.")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. 📄 Upload a PDF")
    st.markdown("2. 🤖 Gemini extracts all verifiable claims")
    st.markdown("3. 🌐 Each claim gets searched on the web")
    st.markdown("4. ✅ Every claim is flagged as Verified / Inaccurate / False")
    st.divider()
    st.markdown("✅ **Verified** — Confirmed by sources")
    st.markdown("⚠️ **Inaccurate** — Wrong numbers or dates")
    st.markdown("❌ **False** — Contradicted by evidence")
    st.markdown("🔵 **Unverifiable** — Not enough info found")


# ── Core functions ─────────────────────────────────────────────────────────────

def read_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def call_gemini(prompt: str, retries: int = 3) -> str:
    if client is None:
        raise RuntimeError("Gemini client is not initialized.")

    last_error = ""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text

            parts = []
            for candidate in (getattr(response, "candidates", None) or []):
                content = getattr(candidate, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            if parts:
                return "\n".join(parts).strip()

            raise RuntimeError("Gemini returned an empty response.")
        except Exception as e:
            err = str(e)
            last_error = err
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                retry_hint = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", err, flags=re.IGNORECASE)
                if retry_hint:
                    wait = max(5, int(float(retry_hint.group(1)) + 2))
                else:
                    wait = 20 * (attempt + 1)  # 20s, 40s, 60s fallback
                st.warning(f"Rate limit hit — waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
        raise RuntimeError(
            "Gemini API quota/rate limit reached (HTTP 429). "
            "Wait for reset in AI Studio Usage, or use another API key."
        )
    raise RuntimeError(f"Gemini API failed after all retries. Last error: {last_error[:180]}")


def clean_json(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_claims(text: str) -> list:
    prompt = f"""You are a fact-checking assistant. Read the text below and extract the most important, verifiable claims.

Look for: statistics, percentages, dates, financial figures, named attributions, technical specs, rankings.

Return ONLY a JSON array — no markdown, no explanation, no code fences.
Each item must have:
- "claim": the exact claim (string)
- "category": one of Statistic | Date | Financial | Technical | Attribution | Ranking
- "context": brief surrounding context (max 80 chars)

Extract up to 12 claims.

TEXT:
{text[:5000]}

JSON:"""

    raw = call_gemini(prompt)
    return json.loads(clean_json(raw))


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


def verify_claim(claim: dict, search_results: list) -> dict:
    sources = "\n\n".join([
        f"Source: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in search_results
    ]) if search_results else "No results found."

    prompt = f"""You are a professional fact-checker. Use the search results below to verify the claim.

CLAIM: "{claim['claim']}"
CATEGORY: {claim['category']}

SEARCH RESULTS:
{sources}

Return ONLY a JSON object — no markdown, no code fences.
Fields:
- "verdict": one of "Verified" | "Inaccurate" | "False" | "Unverifiable"
- "explanation": 1-2 sentence summary (string)
- "correct_value": corrected fact if wrong, else null
- "source_url": best source URL (string or null)
- "confidence": "High" | "Medium" | "Low"

JSON:"""

    raw = call_gemini(prompt)
    result = json.loads(clean_json(raw))
    result["claim"]    = claim["claim"]
    result["category"] = claim["category"]
    result["context"]  = claim.get("context", "")
    return result


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
    return {"Verified": "verified", "Inaccurate": "inaccurate",
            "False": "false", "Unverifiable": "unknown"}.get(verdict, "unknown")


def render_cards(items):
    if not items:
        st.info("No claims in this category.")
        return
    for r in items:
        v     = r.get("verdict", "Unverifiable")
        badge = verdict_badge(v)
        cc    = card_class(v)
        conf_color = {"High": "#34d399", "Medium": "#fbbf24", "Low": "#f87171"}.get(r.get("confidence", ""), "#94a3b8")
        correct = f'<div class="correct-val">📌 Correct: {r["correct_value"]}</div>' if r.get("correct_value") else ""
        source  = f'<div class="source-link">🔗 <a href="{r["source_url"]}" target="_blank">{r["source_url"][:70]}</a></div>' if r.get("source_url") else ""
        cat_pill  = f'<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:10px;font-size:11px;">{r.get("category","")}</span>'
        conf_pill = f'<span style="background:#1e293b;color:{conf_color};padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid {conf_color}40;">Confidence: {r.get("confidence","")}</span>'
        st.markdown(f"""
        <div class="claim-card {cc}">
            {badge} &nbsp; {cat_pill} &nbsp; {conf_pill}
            <div class="claim-text">"{r['claim']}"</div>
            <div class="explanation">{r.get('explanation', '')}</div>
            {correct}{source}
        </div>""", unsafe_allow_html=True)


# ── Main page ──────────────────────────────────────────────────────────────────

st.markdown("# 🔍 FactCheck Agent")
st.markdown("Upload any PDF — report, press release, marketing doc — and get every claim verified against live web data.")
st.divider()

if not api_ready:
    st.error("Gemini API key not configured. Add `GEMINI_API_KEY` in your Streamlit Cloud app secrets.")
    st.stop()

uploaded_file = st.file_uploader("📂 Drop your PDF here", type=["pdf"])

if uploaded_file:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run = st.button("🚀 Run Fact Check", use_container_width=True)

    if run:
        with st.status("Running fact-check pipeline...", expanded=True) as status:

            # Step 1 — Read PDF
            st.write("📄 Reading PDF...")
            try:
                uploaded_file.seek(0)
                pdf_text = read_pdf(uploaded_file)
                if not pdf_text:
                    st.error("Couldn't extract text. Make sure it's not a scanned/image PDF.")
                    st.stop()
                st.write(f"✅ Got **{len(pdf_text.split()):,} words** from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"PDF error: {e}")
                st.stop()

            # Step 2 — Extract claims
            st.write("🤖 Extracting claims with Gemini...")
            try:
                claims = extract_claims(pdf_text)
                st.write(f"✅ Found **{len(claims)} claims** to verify")
            except Exception as e:
                st.error(f"Claim extraction failed: {e}")
                st.stop()

            # Step 3 — Verify claims
            st.write("🌐 Searching the web and verifying...")
            results  = []
            progress = st.progress(0)

            for i, claim in enumerate(claims):
                try:
                    hits     = web_search(claim["claim"])
                    verified = verify_claim(claim, hits)
                    results.append(verified)
                except Exception as ex:
                    results.append({
                        "claim":         claim.get("claim", "Unknown"),
                        "category":      claim.get("category", ""),
                        "context":       claim.get("context", ""),
                        "verdict":       "Unverifiable",
                        "explanation":   f"Verification error: {str(ex)[:80]}",
                        "correct_value": None,
                        "source_url":    None,
                        "confidence":    "Low",
                    })
                progress.progress((i + 1) / len(claims))
                time.sleep(1.5)  # stay within free-tier rate limits

            status.update(label="✅ Done!", state="complete", expanded=False)

        # Summary stats
        st.markdown("---")
        counts   = {v: sum(1 for r in results if r["verdict"] == v) for v in ["Verified", "Inaccurate", "False", "Unverifiable"]}
        accuracy = round(counts["Verified"] / len(results) * 100) if results else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, num, label, color in [
            (c1, len(results),         "Claims Checked", "#60a5fa"),
            (c2, counts["Verified"],   "✅ Verified",    "#34d399"),
            (c3, counts["Inaccurate"], "⚠️ Inaccurate", "#fbbf24"),
            (c4, counts["False"],      "❌ False",       "#f87171"),
            (c5, f"{accuracy}%",       "Accuracy Rate",  "#a78bfa"),
        ]:
            with col:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number" style="color:{color}">{num}</div>
                    <div class="stat-label">{label}</div></div>""",
                    unsafe_allow_html=True)

        # Result tabs
        st.markdown("---")
        st.markdown("### 📋 Detailed Results")
        tab_all, tab_false, tab_inaccurate, tab_verified = st.tabs([
            f"All ({len(results)})",
            f"❌ False ({counts['False']})",
            f"⚠️ Inaccurate ({counts['Inaccurate']})",
            f"✅ Verified ({counts['Verified']})",
        ])
        with tab_all:         render_cards(results)
        with tab_false:       render_cards([r for r in results if r["verdict"] == "False"])
        with tab_inaccurate:  render_cards([r for r in results if r["verdict"] == "Inaccurate"])
        with tab_verified:    render_cards([r for r in results if r["verdict"] == "Verified"])

        # Export
        st.markdown("---")
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="factcheck_report.json",
            mime="application/json",
        )
