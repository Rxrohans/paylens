"""
chain.py — Phase 3 of PayLens
----------------------------------------
WHAT THIS FILE DOES:
  Wires the retriever (Phase 2) to an LLM (Groq/Llama3) using LangChain.
  This is the core RAG chain — the brain of PayLens.

INDUSTRY PATTERNS USED HERE:
  1. Prompt Templates     — structured, versioned, reusable prompts
  2. Context Injection    — retrieved chunks fed into prompt safely
  3. Source Citation      — LLM forced to cite which document it used
  4. Hallucination Guard  — LLM told to say "I don't know" vs making things up
  5. Structured Output    — consistent JSON-like response every time
  6. Latency + Token Tracking — cost & performance monitoring
  7. Chain of Thought     — LLM reasons step by step before answering

WHY THESE MATTER:
  In production fintech AI (like Pine Labs), an answer without a source
  is a liability. A hallucinated fee amount could cause real financial harm.
  Every one of these patterns exists to prevent that.
"""
"""
chain.py — Phase 3 (v2) of ChargeClarity
------------------------------------------
WHAT'S NEW IN THIS VERSION:
  Hybrid RAG + Live Web Search
  - If FAISS retrieval confidence >= 0.60: answer from KB only (fast, grounded)
  - If FAISS retrieval confidence < 0.60: trigger DuckDuckGo live search
  - Both contexts are combined and fed to LLM together
  - LLM synthesizes KB knowledge + live results into one answer

FREE TOOLS USED:
  - DuckDuckGoSearchRun (langchain_community): no API key, no cost, no restrictions
  - Groq llama-3.1-8b-instant: 14,400 free requests/day
  - all-MiniLM-L6-v2: local embeddings, no API needed
"""
"""
chain.py — Phase 3 (v2) of ChargeClarity
------------------------------------------
WHAT'S NEW IN THIS VERSION:
  Hybrid RAG + Live Web Search
  - If FAISS retrieval confidence >= 0.60: answer from KB only (fast, grounded)
  - If FAISS retrieval confidence < 0.60: trigger DuckDuckGo live search
  - Both contexts are combined and fed to LLM together
  - LLM synthesizes KB knowledge + live results into one answer

FREE TOOLS USED:
  - DuckDuckGoSearchRun (langchain_community): no API key, no cost, no restrictions
  - Groq llama-3.1-8b-instant: 14,400 free requests/day
  - all-MiniLM-L6-v2: local embeddings, no API needed
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "chain.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PayLens.chain")

# ─────────────────────────────────────────────────────────
# OFFICIAL LINKS REGISTRY
# Every answer includes the relevant official link
# so users always have somewhere to verify.
# ─────────────────────────────────────────────────────────
OFFICIAL_LINKS = {
    "paypal":   "https://www.paypal.com/in/webapps/mpp/paypal-fees",
    "stripe":   "https://stripe.com/in/pricing",
    "razorpay": "https://razorpay.com/pricing/",
    "upi":      "https://www.npci.org.in/what-we-do/upi/product-overview",
    "rbi":      "https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx",
    "gst":      "https://www.gst.gov.in",
    "tax":      "https://incometaxindia.gov.in",
    "fema":     "https://fema.rbi.org.in",
    "wise":     "https://wise.com/in",
    "neft":     "https://www.rbi.org.in/Scripts/neft.aspx",
}

# ─────────────────────────────────────────────────────────
# RESPONSE DATACLASS
# ─────────────────────────────────────────────────────────
@dataclass
class ChargeAnswer:
    question:         str
    answer:           str
    sources:          List[str]
    confidence:       str
    retrieved_chunks: int
    latency_ms:       float
    tokens_used:      Optional[int]
    fallback_used:    bool
    web_search_used:  bool          # NEW: did we trigger live search?
    official_links:   List[str]     # NEW: relevant official URLs


# ─────────────────────────────────────────────────────────
# CONFIDENCE THRESHOLD
# Below this score → trigger live web search
# ─────────────────────────────────────────────────────────
RAG_CONFIDENCE_THRESHOLD = 0.60

# ─────────────────────────────────────────────────────────
# PROMPTS — two versions depending on context source
# ─────────────────────────────────────────────────────────

# Used when RAG confidence is high (KB only)
SYSTEM_PROMPT_RAG_ONLY = """You are PayLens, a friendly expert AI that helps people \
understand payment fees, currency charges, taxes, and fintech concepts in plain English.

## Your Role
Explain things simply and practically — like a knowledgeable friend, not a legal document. \
Accuracy is non-negotiable. Never guess numbers.

## Strict Output Rules
1. Answer ONLY from the CONTEXT below. Never invent fees or percentages.
2. Do NOT mention document names, scores, or internal labels like [DOC 1] in your answer.
3. Write in clean, plain English with short paragraphs.
4. Use bullet points (- item) for lists of 3 or more items.
5. Use **bold** for important numbers, percentages, and key terms.
6. If context is missing info, say what you DO know, then say what to check.
7. Keep answers under 200 words unless the question genuinely needs more.
8. End your answer on a new line with exactly one of: [High] [Medium] [Low]

## Context
{context}
"""

# Used when web search is triggered (combined context)
SYSTEM_PROMPT_HYBRID = """You are PayLens , a friendly expert AI that helps people \
understand payment fees, currency charges, taxes, and fintech in plain English.

You have access to a curated knowledge base AND fresh live web search results.

## Strict Output Rules
1. Synthesize BOTH sources into one clear, practical answer.
2. Do NOT mention document names, scores, or labels like [DOC 1] in your answer.
3. Write in clean, plain English with short paragraphs.
4. Use bullet points (- item) for lists of 3 or more items.
5. Use **bold** for important numbers, percentages, and key terms.
6. If the web results have more recent info than the KB, use the web result and say "as of [date]".
7. Keep answers under 200 words unless the question genuinely needs more.
8. Add a brief disclaimer for tax/legal questions: "This is general information, not professional advice."
9. End your answer on a new line with exactly one of: [High] [Medium] [Low]

## Knowledge Base
{kb_context}

## Live Web Search Results
{web_context}
"""

HUMAN_PROMPT = "Question: {question}"


# ─────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file!")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        api_key=api_key,
    )


# ─────────────────────────────────────────────────────────
# WEB SEARCH HELPER
# DuckDuckGo — free, no API key, no restrictions
# ─────────────────────────────────────────────────────────
def run_web_search(query: str) -> str:
    """
    Runs a DuckDuckGo search and returns results as a string.

    INDUSTRY PRACTICE — Query reformulation:
        We don't just pass the raw user question to the search engine.
        We reformulate it to be more specific for better results.
        e.g. "why am I charged?" → "PayPal India international payment fees 2024"
    """
    try:
        search = DuckDuckGoSearchRun()
        # Reformulate query to be search-engine friendly
        search_query = f"{query} India 2026"
        result = search.invoke(search_query)
        logger.info(f"Web search completed for: {search_query[:60]}")
        return result if result else "No live results found."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Live search unavailable: {e}"


# ─────────────────────────────────────────────────────────
# OFFICIAL LINKS DETECTOR
# Looks for platform mentions in question → returns relevant links
# ─────────────────────────────────────────────────────────
def detect_relevant_links(question: str, answer: str) -> List[str]:
    """Returns official links for platforms mentioned in the question/answer."""
    combined = (question + " " + answer).lower()
    links = []
    for platform, url in OFFICIAL_LINKS.items():
        if platform in combined:
            links.append(url)
    # Always include RBI if India context detected
    if "india" in combined or "inr" in combined or "rupee" in combined:
        if OFFICIAL_LINKS["rbi"] not in links:
            links.append(OFFICIAL_LINKS["rbi"])
    return list(dict.fromkeys(links))  # deduplicate while preserving order


# ─────────────────────────────────────────────────────────
# MAIN CHAIN CLASS
# ─────────────────────────────────────────────────────────
class ChargeChain:
    """
    Hybrid RAG + Web Search chain for PayLens.

    Decision logic:
        top RAG score >= 0.60  →  RAG only (fast, grounded, no web call)
        top RAG score < 0.60   →  RAG + DuckDuckGo live search (slower, broader)
    """

    def __init__(self):
        logger.info("Initialising ChargeChain v2 (Hybrid)...")
        from retriever import ChargeRetriever
        self.retriever    = ChargeRetriever()
        self.llm          = get_llm()
        self.rag_prompt   = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_RAG_ONLY),
            ("human",  HUMAN_PROMPT),
        ])
        self.hybrid_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_HYBRID),
            ("human",  HUMAN_PROMPT),
        ])
        self.parser = StrOutputParser()
        logger.info("ChargeChain v2 ready [OK]")

    def ask(self, question: str, top_k: int = 7) -> ChargeAnswer:
        """
        Full hybrid pipeline:
        question → FAISS retrieval → confidence check
            → if high: RAG only
            → if low:  RAG + DuckDuckGo web search
        → LLM synthesizes → structured answer
        """
        start = time.time()
        logger.info(f"Query: {question[:100]}")

        # ── Step 1: FAISS Retrieval ─────────────────────────
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved:
            return self._fallback_answer(question, time.time() - start)

        top_score = retrieved[0]["score"]
        logger.info(f"Top RAG score: {top_score:.3f} | Threshold: {RAG_CONFIDENCE_THRESHOLD}")

        # ── Step 2: Decide — RAG only or Hybrid ────────────
        web_search_used = False
        web_context     = ""

        if top_score < RAG_CONFIDENCE_THRESHOLD:
            logger.info("Low RAG confidence — triggering web search")
            web_context     = run_web_search(question)
            web_search_used = True

        # ── Step 3: Build KB context ────────────────────────
        kb_context = self._format_context(retrieved)

        # ── Step 4: Invoke correct prompt ──────────────────
        if web_search_used:
            chain = self.hybrid_prompt | self.llm | self.parser
            raw   = chain.invoke({
                "kb_context":  kb_context,
                "web_context": web_context,
                "question":    question
            })
        else:
            chain = self.rag_prompt | self.llm | self.parser
            raw   = chain.invoke({
                "context":  kb_context,
                "question": question
            })

        # ── Step 5: Parse + enrich answer ──────────────────
        latency_ms     = (time.time() - start) * 1000
        official_links = detect_relevant_links(question, raw)
        answer         = self._parse_answer(
            question, raw, retrieved, latency_ms,
            web_search_used, official_links
        )
        self._log_answer(answer)
        return answer

    def _format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[DOC {i} | Source: {chunk['source']} | Score: {chunk['score']:.2f}]\n"
                f"{chunk['text']}"
            )
        return "\n\n".join(parts)

    def _parse_answer(
        self,
        question:        str,
        raw:             str,
        chunks:          List[Dict],
        latency_ms:      float,
        web_search_used: bool,
        official_links:  List[str],
    ) -> ChargeAnswer:
        fallback_phrases = [
            "don't have enough information",
            "not enough information",
            "please check the platform"
        ]
        fallback_used = any(p in raw.lower() for p in fallback_phrases)

        # Extract confidence
        confidence = "medium"
        last_line  = raw.strip().split("\n")[-1].lower()
        for level in ["high", "medium", "low", "none"]:
            if level in last_line:
                confidence = level
                break

        # Extract cited sources
        cited = list(dict.fromkeys(c["source"] for c in chunks[:3]))
        if web_search_used:
            cited.append("live_web_search")

        return ChargeAnswer(
            question         = question,
            answer           = raw.strip(),
            sources          = cited,
            confidence       = confidence,
            retrieved_chunks = len(chunks),
            latency_ms       = round(latency_ms, 2),
            tokens_used      = None,
            fallback_used    = fallback_used,
            web_search_used  = web_search_used,
            official_links   = official_links,
        )

    def _fallback_answer(self, question: str, elapsed: float) -> ChargeAnswer:
        return ChargeAnswer(
            question         = question,
            answer           = (
                "I couldn't find relevant information in my knowledge base. "
                "Please check the official fee page of the platform you're asking about."
            ),
            sources          = [],
            confidence       = "none",
            retrieved_chunks = 0,
            latency_ms       = round(elapsed * 1000, 2),
            tokens_used      = None,
            fallback_used    = True,
            web_search_used  = False,
            official_links   = list(OFFICIAL_LINKS.values())[:3],
        )

    def _log_answer(self, answer: ChargeAnswer):
        log_path = LOG_DIR / "answers.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(answer), ensure_ascii=False) + "\n")
        logger.info(
            f"Logged | confidence={answer.confidence} | "
            f"latency={answer.latency_ms}ms | "
            f"web_search={answer.web_search_used} | "
            f"fallback={answer.fallback_used}"
        )


# ─────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    chain = ChargeChain()

    questions = [
        "Why does PayPal charge so much when I receive money from Outlier?",
        "What is Razorpay's fee for UPI payments?",
        "How does currency conversion work and why do I lose money?",
        "Do I need to pay GST on my freelance income from abroad?",
    ]

    for q in questions:
        print(f"\n{'='*65}")
        print(f"Q: {q}")
        print(f"{'='*65}")
        r = chain.ask(q)
        # Clean confidence tag
        clean = re.sub(r'\s*\[(High|Medium|Low|None)\]\s*$', '', r.answer, flags=re.IGNORECASE)
        print(f"\nA: {clean}")
        print(f"\n  Sources     : {r.sources}")
        print(f"  Confidence  : {r.confidence}")
        print(f"  Latency     : {r.latency_ms}ms")
        print(f"  Web search  : {r.web_search_used}")
        print(f"  Links       : {r.official_links}")