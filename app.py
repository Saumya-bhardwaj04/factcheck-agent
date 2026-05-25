import streamlit as st
from google import genai
import pdfplumber
import json
import time
import io
from duckduckgo_search import DDGS

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
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

    h1 { color:#f1f5f9 !important; }
    h2, h3 { color:#cbd5e1 !important; }
    p  { color:#94a3b8 !important; }

    .stButton > button {
        background: linear-gradient(135deg,#0d9488,#0891b2);
        color:white; border:none; border-radius:8px;
        font-weight:600; padding:10px 28px; font-size:15px; width:100%;
    }
    section[data-testid="stSidebar"] { background:#1e293b !important; }
    section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def call_gemini(client, prompt: str) -> str:
    """Call Gemini Flash and return the text response."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text.strip()


def extract_claims(text: str, client) -> list[dict]:
    prompt = f"""You are a fact-checking assistant. Analyze the following text and extract ALL specific, verifiable claims.

Focus on:
- Statistics and percentages (e.g., "revenue grew 45%", "3 billion users")
- Dates and timelines (e.g., "launched in 2019", "founded in 1998")
- Financial figures (e.g., "valued at $10B", "earned $500M in Q3")
- Named facts and attributed claims (e.g., "Einstein said...", "WHO reports...")
- Technical claims (e.g., "processes 1M requests/sec", "99.9% uptime")
- Rankings and comparisons (e.g., "largest company", "first to market")

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each object must have:
- "claim": exact claim as stated in the text (string)
- "category": one of: Statistic | Date | Financial | Technical | Attribution | Ranking
- "context": brief surrounding context (string, max 80 chars)

Extract maximum 15 most important/verifiable claims.

TEXT:
{text[:6000]}

JSON array:"""

    raw = call_gemini(client, prompt)
    # Strip markdown fences if Gemini adds them
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]
    return json.loads(raw.strip())


def search_web(query: str, max_results: int = 4) -> list[dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body":  r.get("body", ""),
                    "href":  r.get("href", ""),
                })
    except Exception:
        pass
    return results


def verify_claim(claim: dict, search_results: list[dict], client) -> dict:
    search_text = "\n\n".join([
        f"Source: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in search_results
    ]) if search_results else "No search results found."

    prompt = f"""You are a professional fact-checker. Verify the following claim using the web search results provided.

CLAIM: "{claim['claim']}"
CATEGORY: {claim['category']}

WEB SEARCH RESULTS:
{search_text}

Respond with ONLY a valid JSON object. No markdown, no code fences, no explanation outside the JSON.
Fields:
- "verdict": exactly one of: "Verified" | "Inaccurate" | "False" | "Unverifiable"
  * Verified = search results confirm the claim
  * Inaccurate = claim has wrong numbers/dates/facts but is partially true
  * False = claim is clearly contradicted by evidence
  * Unverifiable = not enough evidence to confirm or deny
- "explanation": 1-2 sentence explanation (string)
- "correct_value": if Inaccurate or False, the correct fact/figure from sources (string or null)
- "source_url": most relevant source URL (string or null)
- "confidence": "High" | "Medium" | "Low"

JSON:"""

    raw = call_gemini(client, prompt)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]
    result = json.loads(raw.strip())
    result["claim"]    = claim["claim"]
    result["category"] = claim["category"]
    result["context"]  = claim.get("context", "")
    return result


# ─── UI helpers ──────────────────────────────────────────────────────────────

def verdict_badge(verdict: str) -> str:
    icons = {
        "Verified":     ("✅", "badge-verified"),
        "Inaccurate":   ("⚠️", "badge-inaccurate"),
        "False":        ("❌", "badge-false"),
        "Unverifiable": ("🔵", "badge-unknown"),
    }
    icon, cls = icons.get(verdict, ("❓", "badge-unknown"))
    return f'<span class="verdict-badge {cls}">{icon} {verdict.upper()}</span>'

def card_cls(verdict: str) -> str:
    return {"Verified":"verified","Inaccurate":"inaccurate",
            "False":"false","Unverifiable":"unknown"}.get(verdict,"unknown")

def render_results(items):
    if not items:
        st.info("No claims in this category.")
        return
    for r in items:
        v   = r.get("verdict","Unverifiable")
        cc  = card_cls(v)
        badge = verdict_badge(v)
        correct = f'<div class="correct-val">📌 Correct value: {r["correct_value"]}</div>' if r.get("correct_value") else ""
        source  = f'<div class="source-link">🔗 <a href="{r["source_url"]}" target="_blank">{r["source_url"][:70]}</a></div>' if r.get("source_url") else ""
        conf_color = {"High":"#34d399","Medium":"#fbbf24","Low":"#f87171"}.get(r.get("confidence",""),"#94a3b8")
        cat_pill  = f'<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:10px;font-size:11px;">{r.get("category","")}</span>'
        conf_pill = f'<span style="background:#1e293b;color:{conf_color};padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid {conf_color}40;">Confidence: {r.get("confidence","")}</span>'
        st.markdown(f"""
        <div class="claim-card {cc}">
            {badge} &nbsp; {cat_pill} &nbsp; {conf_pill}
            <div class="claim-text">"{r['claim']}"</div>
            <div class="explanation">{r.get('explanation','')}</div>
            {correct}{source}
        </div>""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FactCheck Agent")
    st.markdown("AI-powered claim verification using **Gemini Flash** (free) + live web search.")
    st.divider()

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free at aistudio.google.com — no credit card needed"
    )

    st.markdown("""
**Get your free key:**
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API key**
3. Paste it above ☝️
""")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. 📄 Upload a PDF")
    st.markdown("2. 🤖 Gemini extracts all claims")
    st.markdown("3. 🌐 Each claim is web-searched")
    st.markdown("4. ✅ Claims flagged Verified / Inaccurate / False")
    st.divider()
    st.markdown("✅ **Verified** — Confirmed by web sources")
    st.markdown("⚠️ **Inaccurate** — Wrong numbers / dates")
    st.markdown("❌ **False** — Contradicted by evidence")
    st.markdown("🔵 **Unverifiable** — No evidence found")


# ─── Main ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 FactCheck Agent")
st.markdown("Upload any PDF — marketing doc, report, press release — and get every claim verified against live web data instantly.")
st.divider()

uploaded_file = st.file_uploader("📂 Drop your PDF here", type=["pdf"])

if uploaded_file and not api_key:
    st.warning("⚠️ Please enter your **free** Google Gemini API Key in the sidebar to continue.")

if uploaded_file and api_key:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        run_btn = st.button("🚀 Run Fact Check", use_container_width=True)

    if run_btn:
        client = genai.Client(api_key=api_key)

        with st.status("⚙️ Running fact-check pipeline...", expanded=True) as status:

            # Step 1 – Extract PDF text
            st.write("📄 Reading PDF...")
            try:
                uploaded_file.seek(0)
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text:
                    st.error("Could not extract text. Please use a text-based (non-scanned) PDF.")
                    st.stop()
                st.write(f"✅ Extracted **{len(pdf_text.split()):,} words** from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"PDF read error: {e}")
                st.stop()

            # Step 2 – Extract claims
            st.write("🤖 Identifying verifiable claims with Gemini...")
            try:
                claims = extract_claims(pdf_text, client)
                st.write(f"✅ Found **{len(claims)} verifiable claims** to check")
            except Exception as e:
                st.error(f"Claim extraction failed: {e}")
                st.stop()

            # Step 3 – Verify each claim
            st.write("🌐 Searching the web and verifying each claim...")
            results  = []
            progress = st.progress(0)

            for i, claim in enumerate(claims):
                try:
                    search_results = search_web(claim["claim"][:120], max_results=4)
                    time.sleep(0.4)
                    verified = verify_claim(claim, search_results, client)
                    results.append(verified)
                except Exception as ex:
                    results.append({
                        "claim": claim.get("claim","Unknown"),
                        "category": claim.get("category",""),
                        "context": claim.get("context",""),
                        "verdict": "Unverifiable",
                        "explanation": f"Verification error: {str(ex)[:80]}",
                        "correct_value": None,
                        "source_url": None,
                        "confidence": "Low",
                    })
                progress.progress((i+1)/len(claims))
                time.sleep(0.2)

            status.update(label="✅ Fact-check complete!", state="complete", expanded=False)

        # ── Summary stats
        st.markdown("---")
        counts = {v: sum(1 for r in results if r["verdict"]==v)
                  for v in ["Verified","Inaccurate","False","Unverifiable"]}
        accuracy = round(counts["Verified"]/len(results)*100) if results else 0

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, num, label, color in [
            (c1, len(results),        "Claims Checked",  "#60a5fa"),
            (c2, counts["Verified"],  "✅ Verified",     "#34d399"),
            (c3, counts["Inaccurate"],"⚠️ Inaccurate",  "#fbbf24"),
            (c4, counts["False"],     "❌ False",        "#f87171"),
            (c5, f"{accuracy}%",      "Accuracy Rate",   "#a78bfa"),
        ]:
            with col:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number" style="color:{color}">{num}</div>
                    <div class="stat-label">{label}</div></div>""",
                    unsafe_allow_html=True)

        # ── Result cards
        st.markdown("---")
        st.markdown("### 📋 Detailed Results")

        tab_all, tab_false, tab_inaccurate, tab_verified = st.tabs([
            f"All ({len(results)})",
            f"❌ False ({counts['False']})",
            f"⚠️ Inaccurate ({counts['Inaccurate']})",
            f"✅ Verified ({counts['Verified']})",
        ])
        with tab_all:        render_results(results)
        with tab_false:      render_results([r for r in results if r["verdict"]=="False"])
        with tab_inaccurate: render_results([r for r in results if r["verdict"]=="Inaccurate"])
        with tab_verified:   render_results([r for r in results if r["verdict"]=="Verified"])

        # ── Export
        st.markdown("---")
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="factcheck_report.json",
            mime="application/json",
        )