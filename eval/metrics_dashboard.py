"""
metrics_dashboard.py — Phase 5 of PayLens
-------------------------------------------
Streamlit page showing RAGAS evaluation scores over time.
Embedded as a tab inside the main PayLens app.

HOW TO VIEW:
  Accessible via the "Eval" tab in the main app.
  Or run standalone: streamlit run eval/metrics_dashboard.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd

SCORES_PATH = Path(__file__).parent / "scores_history.json"

# ── Threshold for passing ─────────────────────────────────
PASS_THRESHOLD = 0.70


def load_scores():
    if not SCORES_PATH.exists():
        return []
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render_dashboard():
    """Main dashboard render function — called from app.py."""

    st.markdown("""
    <div style="padding:1.5rem 0 0.5rem;">
        <p style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
        color:#e2e2ee;margin:0;">Evaluation Dashboard</p>
        <p style="color:#52526e;font-size:0.88rem;margin:0.3rem 0 0;">
        Tracking RAG pipeline quality over time — PayLens</p>
    </div>
    """, unsafe_allow_html=True)

    history = load_scores()

    # ── No data state ─────────────────────────────────────
    if not history:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
        border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;">
            <p style="color:#52526e;font-size:0.95rem;">
                No evaluation runs yet.<br>
                Run <code>python eval/ragas_eval.py</code> to generate scores.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Latest run scores ─────────────────────────────────
    latest = history[-1]

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#3a3a52;letter-spacing:0.1em;'
        'text-transform:uppercase;margin-bottom:0.8rem;">Latest Run</p>',
        unsafe_allow_html=True
    )

    metrics = [
        ("Faithfulness",       latest["faithfulness"],      "Hallucination check"),
        ("Answer Relevancy",   latest["answer_relevancy"],  "On-topic check"),
        ("Context Coverage",  latest["context_coverage"], "Coverage check"),
    ]

    cols = st.columns(3)
    for col, (name, score, desc) in zip(cols, metrics):
        passed     = score >= PASS_THRESHOLD
        color      = "#6ee7b7" if passed else "#fca5a5"
        bg         = "rgba(16,185,129,0.08)" if passed else "rgba(239,68,68,0.08)"
        border     = "rgba(16,185,129,0.2)"  if passed else "rgba(239,68,68,0.2)"
        status     = "PASS" if passed else "FAIL"

        col.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:12px;
        padding:1.1rem;text-align:center;">
            <p style="font-size:0.7rem;color:#52526e;letter-spacing:0.08em;
            text-transform:uppercase;margin:0 0 0.4rem;">{name}</p>
            <p style="font-family:'Syne',sans-serif;font-size:1.8rem;
            font-weight:700;color:{color};margin:0;line-height:1;">{score:.0%}</p>
            <p style="font-size:0.68rem;color:#3a3a52;margin:0.3rem 0 0;">{desc}</p>
            <p style="font-size:0.65rem;color:{color};margin:0.2rem 0 0;
            font-weight:600;">{status}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Overall + system stats ────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    stat_cols = st.columns(4)
    stats = [
        ("Overall Score",    f"{latest['overall_score']:.0%}"),
        ("Avg Latency",      f"{latest['avg_latency_ms']:.0f}ms"),
        ("Web Search Rate",  f"{latest['web_search_rate']:.0%}"),
        ("Questions Tested", str(latest["num_questions"])),
    ]
    for col, (label, val) in zip(stat_cols, stats):
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid
        rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem;text-align:center;">
            <p style="font-size:0.68rem;color:#3a3a52;text-transform:uppercase;
            letter-spacing:0.08em;margin:0 0 0.3rem;">{label}</p>
            <p style="font-size:1.3rem;font-weight:600;color:#a78bfa;margin:0;">{val}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Score history chart ───────────────────────────────
    if len(history) > 1:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.72rem;color:#3a3a52;letter-spacing:0.1em;'
            'text-transform:uppercase;margin-bottom:0.8rem;">Score History</p>',
            unsafe_allow_html=True
        )

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m/%d %H:%M")
        df = df.set_index("timestamp")

        chart_df = df[["faithfulness","answer_relevancy","context_coverage"]]
        chart_df.columns = ["Faithfulness","Answer Relevancy","Context Coverage"]

        st.line_chart(chart_df, use_container_width=True, height=220)

    # ── Run history table ─────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#3a3a52;letter-spacing:0.1em;'
        'text-transform:uppercase;margin-bottom:0.8rem;">All Runs</p>',
        unsafe_allow_html=True
    )

    table_df = pd.DataFrame(history[::-1])  # newest first
    table_df["timestamp"] = pd.to_datetime(table_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    table_df = table_df[[
        "timestamp", "overall_score", "faithfulness",
        "answer_relevancy", "context_coverage",
        "avg_latency_ms"
    ]]
    table_df.columns = [
       "Run Time", "Overall", "Faithfulness",
       "Relevancy", "Coverage", "Avg Latency(ms)"
    ]
    for col in ["Overall","Faithfulness","Relevancy","Coverage"]:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.0%}")

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ── What scores mean ──────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.expander("What do these scores mean?"):
        st.markdown("""
        | Metric | What it checks | Target |
        |---|---|---|
        | **Faithfulness** | Does the answer only use information from retrieved context? Low = hallucinating | > 70% |
        | **Answer Relevancy** | Does the answer actually address the question asked? Low = off-topic | > 70% |
        | **Context Coverage** | Were the chunks retrieved actually useful for answering? Low = wrong docs retrieved | > 70% |

        **How to improve scores:**
        - Low Faithfulness → tighten the system prompt constraints
        - Low Relevancy → improve query reformulation in retriever
        - Low Coverage → increase RAG confidence threshold
        """)


# ── Standalone run ────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="PayLens Eval",
        page_icon="📊",
        layout="centered"
    )
    render_dashboard()