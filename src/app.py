"""
app.py — Phase 4 (v2): ChargeClarity Streamlit UI
---------------------------------------------------
Run: streamlit run src/app.py
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ChargeClarity",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background:#0d0d14; color:#e2e2ee; }
.stApp { background:#0d0d14; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2.5rem; padding-bottom:3rem; max-width:760px; }

/* hero */
.hero { text-align:center; padding:2.5rem 1rem 0.5rem; }
.hero-eyebrow {
    display:inline-block; background:rgba(139,92,246,0.12);
    border:1px solid rgba(139,92,246,0.25); color:#a78bfa;
    font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase;
    padding:0.28rem 0.9rem; border-radius:999px; margin-bottom:1.1rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.9rem; font-weight:800;
    background:linear-gradient(130deg,#fff 40%,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.08; letter-spacing:-0.03em; margin-bottom:0.7rem;
}
.hero-sub { color:#52526e; font-size:0.95rem; font-weight:300; max-width:440px; margin:0 auto 1.8rem; line-height:1.65; }

/* chips */
.stButton>button {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important;
    color:#7b7b9a !important; font-size:0.78rem !important; border-radius:999px !important;
    padding:0.3rem 0.9rem !important; transition:all 0.18s !important; font-family:'DM Sans' !important;
}
.stButton>button:hover { background:rgba(139,92,246,0.12) !important; color:#c4b5fd !important; border-color:rgba(139,92,246,0.35) !important; }

/* ask button override */
div[data-testid="column"]:last-child .stButton>button {
    background:linear-gradient(135deg,#6d28d9,#7c3aed) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:500 !important; font-size:0.95rem !important;
    padding:0.65rem 1.2rem !important;
}

/* input */
.stTextInput input {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:#e2e2ee !important; font-size:0.97rem !important;
    padding:0.8rem 1.1rem !important; font-family:'DM Sans' !important;
}
.stTextInput input:focus { border-color:rgba(139,92,246,0.55) !important; box-shadow:0 0 0 3px rgba(139,92,246,0.1) !important; }
.stTextInput input::placeholder { color:#3a3a52 !important; }
.stTextInput label { display:none !important; }

/* answer card */
.ans-card {
    background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.6rem 1.8rem; margin-top:1.4rem;
}
.ans-card p { margin:0 0 0.75rem; line-height:1.75; color:#c8c8e0; font-size:0.96rem; }
.ans-card p:last-child { margin-bottom:0; }
.ans-card ul { padding-left:1.2rem; margin:0.3rem 0 0.75rem; }
.ans-card li { color:#c8c8e0; font-size:0.96rem; line-height:1.75; margin-bottom:0.25rem; }
.ans-card strong { color:#e8e8f8; font-weight:500; }

/* badges */
.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:1.1rem; align-items:center; }
.badge {
    font-size:0.72rem; font-weight:500; letter-spacing:0.04em;
    padding:0.22rem 0.7rem; border-radius:999px;
}
.b-high   { background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.2); }
.b-medium { background:rgba(245,158,11,0.12);  color:#fcd34d; border:1px solid rgba(245,158,11,0.2); }
.b-low    { background:rgba(239,68,68,0.12);   color:#fca5a5; border:1px solid rgba(239,68,68,0.2); }
.b-none   { background:rgba(100,100,120,0.12); color:#9ca3af; border:1px solid rgba(100,100,120,0.2); }
.b-web    { background:rgba(59,130,246,0.12);  color:#93c5fd; border:1px solid rgba(59,130,246,0.2); }
.b-src    { background:rgba(139,92,246,0.1);   color:#c4b5fd; border:1px solid rgba(139,92,246,0.18); }
.b-time   { background:rgba(255,255,255,0.04); color:#52526e; border:1px solid rgba(255,255,255,0.07); }

/* links section */
.links-section { margin-top:1.1rem; }
.links-label { font-size:0.72rem; color:#3a3a52; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem; }
.links-row { display:flex; flex-wrap:wrap; gap:0.45rem; }
.ext-link {
    font-size:0.78rem; color:#7c6faa; text-decoration:none;
    background:rgba(139,92,246,0.07); border:1px solid rgba(139,92,246,0.15);
    padding:0.22rem 0.75rem; border-radius:6px; transition:all 0.15s;
}
.ext-link:hover { background:rgba(139,92,246,0.15); color:#c4b5fd; }

/* blocked / warn */
.blocked { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.18); border-radius:10px; padding:1rem 1.2rem; color:#fca5a5; font-size:0.9rem; margin-top:1rem; }
.warn    { background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.18); border-radius:10px; padding:0.7rem 1rem; color:#fcd34d; font-size:0.82rem; margin-top:0.7rem; }

/* history */
.hist-wrap { margin-top:2.5rem; }
.hist-label { font-size:0.7rem; color:#2e2e42; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.9rem; }
.hist-item { border-left:2px solid rgba(139,92,246,0.2); padding:0.5rem 0.9rem; margin-bottom:0.7rem; }
.hist-q { font-size:0.86rem; color:#6b6b8a; font-weight:500; margin-bottom:0.2rem; }
.hist-a { font-size:0.82rem; color:#3a3a52; line-height:1.55; }

hr { border-color:rgba(255,255,255,0.05) !important; margin:2rem 0 !important; }
.stSpinner>div { border-top-color:#7c3aed !important; }
/* Hide "Press Enter to submit form" text */
.stForm small { display: none !important; }
.stForm [data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_chain():
    from pathlib import Path
    import subprocess, sys

    # Build index if it doesn't exist (first run on HuggingFace)
    index_path = Path(__file__).parent.parent / "data" / "processed" / "faiss_index.bin"
    if not index_path.exists():
        with st.spinner("First run — building knowledge base index (2-3 mins)..."):
            subprocess.run([sys.executable, str(Path(__file__).parent / "chunker.py")], check=True)
            subprocess.run([sys.executable, str(Path(__file__).parent / "embedder.py")], check=True)

    from chain import ChargeChain
    return ChargeChain()

@st.cache_resource(show_spinner=False)
def get_guardrail_fn():
    from guardrails import run_with_guardrails
    return run_with_guardrails


# ── Session state ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_q" not in st.session_state:
    st.session_state.last_q = ""


# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · RAG · Fintech · India</div>
    <div class="hero-title">ChargeClarity</div>
    <div class="hero-sub">Ask anything about payment fees, currency charges,
    UPI, taxes, or fintech — get a plain-English answer with sources.</div>
</div>
""", unsafe_allow_html=True)

# ── Example chips ─────────────────────────────────────────
EXAMPLES = [
    "Why did PayPal charge me so much?",
    "How does currency conversion work?",
    "Is UPI free for merchants?",
    "What is a chargeback?",
    "Do I pay GST on foreign income?",
]

chip_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if chip_cols[i].button(ex, key=f"chip_{i}"):
        st.session_state["prefill"] = ex
        st.rerun()

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")

with st.form(key="query_form", clear_on_submit=False):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_query = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="e.g. Why did PayPal deduct 7% from my payment?",
            value=prefill, key="main_input"
        )
    with col_btn:
        ask_clicked = st.form_submit_button("Ask →", use_container_width=True)

# Also trigger on Enter key
if user_query and user_query != st.session_state.get("last_submitted", ""):
    if ask_clicked or (user_query.endswith("\n") or st.session_state.get("enter_pressed")):
        ask_clicked = True


# ── Helper: render answer text as clean HTML ──────────────
def render_answer(text: str) -> str:
    """
    Converts LLM markdown-ish output → clean HTML for the answer card.
    Handles: **bold**, bullet lists, numbered lists, paragraphs.
    """
    # Strip confidence tag
    text = re.sub(r'\s*\[(High|Medium|Low|None)\]\s*$', '', text, flags=re.IGNORECASE).strip()

    lines   = text.split("\n")
    html    = ""
    in_ul   = False
    in_ol   = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_ul:  html += "</ul>"; in_ul = False
            if in_ol:  html += "</ol>"; in_ol = False
            continue

        # Bold
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)

        # Bullet list
        if re.match(r'^[-•*] ', line):
            if in_ol:  html += "</ol>"; in_ol = False
            if not in_ul: html += "<ul>"; in_ul = True
            html += f"<li>{line[2:].strip()}</li>"
            continue

        # Numbered list
        if re.match(r'^\d+\. ', line):
            if in_ul:  html += "</ul>"; in_ul = False
            if not in_ol: html += "<ol>"; in_ol = True
            html += f"<li>{re.sub(r'^\d+\. ', '', line)}</li>"
            continue

        # Close open lists
        if in_ul:  html += "</ul>"; in_ul = False
        if in_ol:  html += "</ol>"; in_ol = False

        html += f"<p>{line}</p>"

    if in_ul: html += "</ul>"
    if in_ol: html += "</ol>"
    return html


# ── Process query ─────────────────────────────────────────
if ask_clicked and user_query.strip():
    with st.spinner("Thinking..."):
        chain        = load_chain()
        guardrail_fn = get_guardrail_fn()
        result       = guardrail_fn(user_query.strip(), chain.ask)
        st.session_state.result = result
        st.session_state.last_q = user_query.strip()

        if not result["blocked"]:
            clean_a = re.sub(
                r'\s*\[(High|Medium|Low|None)\]\s*$', '',
                result["answer"], flags=re.IGNORECASE
            ).strip()
            st.session_state.history.insert(0, {
                "q": user_query.strip(),
                "a": clean_a[:160] + "..." if len(clean_a) > 160 else clean_a,
                "confidence": result["confidence"]
            })


# ── Render result ─────────────────────────────────────────
result = st.session_state.result

if result:
    if result["blocked"]:
        reason = result["guardrail_warnings"][0] if result["guardrail_warnings"] else "Request not supported."
        st.markdown(f'<div class="blocked">🚫 &nbsp;{reason}</div>', unsafe_allow_html=True)

    else:
        answer     = result["answer"]
        confidence = result.get("confidence", "medium")
        sources    = result.get("sources", [])
        latency    = result.get("latency_ms", 0)
        warnings   = result.get("guardrail_warnings", [])
        web_used   = result.get("web_search_used", False)
        links      = result.get("official_links", [])

        # ── Answer card ───────────────────────────────────
        rendered = render_answer(answer)
        st.markdown(f'<div class="ans-card">{rendered}</div>', unsafe_allow_html=True)

        # ── Badges ────────────────────────────────────────
        conf_map = {
            "high":   ("b-high",   "High Confidence"),
            "medium": ("b-medium", "Medium Confidence"),
            "low":    ("b-low",    "Low Confidence"),
            "none":   ("b-none",   "No Data"),
        }
        cls, label = conf_map.get(confidence.lower(), ("b-none", confidence))

        badges = f'<span class="badge {cls}">{label}</span>'

        if web_used:
            badges += ' <span class="badge b-web">🌐 Live Search</span>'

        clean_sources = [
            s for s in sources
            if s not in ("live_web_search",)
        ]
        for s in clean_sources:
            readable = s.replace("_", " ").replace("manual", "").strip().title()
            badges += f' <span class="badge b-src">📄 {readable}</span>'

        badges += f' <span class="badge b-time">⚡ {latency:.0f}ms</span>'

        st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

        # ── Warnings ──────────────────────────────────────
        for w in warnings:
            st.markdown(f'<div class="warn">⚠️ {w}</div>', unsafe_allow_html=True)

        # ── Official links ────────────────────────────────
        if links:
            link_items = ""
            for url in links[:5]:
                domain = url.replace("https://","").replace("http://","").split("/")[0]
                link_items += f'<a class="ext-link" href="{url}" target="_blank">↗ {domain}</a>'
            st.markdown(f"""
            <div class="links-section">
                <div class="links-label">Verify at official sources</div>
                <div class="links-row">{link_items}</div>
            </div>
            """, unsafe_allow_html=True)


# ── History ───────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="hist-label">Recent Questions</div>', unsafe_allow_html=True)
    for item in st.session_state.history[:5]:
        dot_color = {"high":"#6ee7b7","medium":"#fcd34d","low":"#fca5a5"}.get(item["confidence"],"#52526e")
        st.markdown(f"""
        <div class="hist-item">
            <div class="hist-q"><span style="color:{dot_color};margin-right:6px">●</span>{item['q']}</div>
            <div class="hist-a">{item['a']}</div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;border-top:1px solid rgba(255,255,255,0.04);padding-top:1.5rem;">
    <p style="color:#2a2a3a;font-size:0.72rem;letter-spacing:0.06em;">
        CHARGECLARITY &nbsp;·&nbsp; RAG + LIVE SEARCH &nbsp;·&nbsp;
        LLAMA 3.1 via GROQ &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; LANGCHAIN
    </p>
</div>
""", unsafe_allow_html=True)