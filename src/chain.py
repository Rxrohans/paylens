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
from datetime_parser import parse_datetime_query, DateTimeParser
from exchange_rate_fetcher import get_exchange_rate

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

# DATETIME PARSER — for temporal query awareness
datetime_parser = DateTimeParser()

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
DISABLE_WEB_SEARCH = os.getenv("DISABLE_WEB_SEARCH", "0") == "1"
RAG_CONFIDENCE_THRESHOLD = 0.99 if DISABLE_WEB_SEARCH else 0.60

# ─────────────────────────────────────────────────────────
# PROMPTS — two versions depending on context source
# ─────────────────────────────────────────────────────────

# Used when RAG confidence is high (KB only)
SYSTEM_PROMPT_RAG_ONLY = """You are PayLens, a friendly expert AI that helps people \
understand payment fees, currency charges, taxes, and fintech concepts in plain English.
{date_context}

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
SYSTEM_PROMPT_HYBRID = """You are PayLens, a friendly expert AI that helps people \
understand payment fees, currency charges, taxes, and fintech in plain English.

You have access to a curated knowledge base AND fresh live web search results.

{date_context}

**FORBIDDEN PHRASES** - NEVER use these unless the exact data is in the web results:
❌ "As of [date], the exchange rate is..."
❌ "The current rate is approximately..."
❌ "Based on today's rate of..."
❌ "At the current exchange rate of..."

**REQUIRED BEHAVIOR** for exchange rate queries:
1. Check if web results contain a specific exchange rate number.
2. If YES → Quote it exactly: "According to [source], the rate is X"
3. If NO → Say: "I don't have today's exchange rate. Please check Google Finance, XE.com, or your bank for current rates."
4. NEVER estimate, approximate, or use general knowledge for current rates

**WHY THIS MATTERS:** 
Making up exchange rates could cause users financial harm. Better to admit uncertainty 
than provide incorrect numbers.

## CRITICAL RULES FOR CURRENT DATA
**NEVER invent, estimate, or guess current numbers.** For ANY time-sensitive data (exchange rates, \
current prices, today's fees, recent news, etc.), you MUST:
1. Check the Live Web Search Results section below first
2. Use ONLY information from those web results for current data
3. If web results don't have the data, say "I found [what you did find], but I don't have \
current [what's missing]. Please check [official source]."
4. NEVER say "approximately" or "as of [date], the rate is X" unless X comes directly from the web results

## Strict Output Rules
1. For historical/general info: Use knowledge base
2. For current data: Use ONLY web search results - never make up numbers
3. Do NOT mention document names, scores, or labels like [DOC 1] in your answer
4. Write in clean, plain English with short paragraphs
5. Use bullet points (- item) for lists of 3 or more items
6. Use **bold** for important numbers, percentages, and key terms
7. If uncertain about current data, explicitly say what you don't know and where to verify
8. Keep answers under 200 words unless the question genuinely needs more
9. Add a brief disclaimer for tax/legal questions: "This is general information, not professional advice."
10. End your answer on a new line with exactly one of: [High] [Medium] [Low]

## Knowledge Base (Historical/General Info)
{kb_context}

## Live Web Search Results (Current Data - USE THIS FOR TIME-SENSITIVE INFO)
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
    Runs web search with smart handling for exchange rate queries.
    For currency conversions, directly fetches rates instead of searching.
    """
    try:
        # Parse datetime info from query
        dt_info = parse_datetime_query(query)
        
        # DETECT CURRENCY CONVERSION QUERIES
        query_lower = query.lower()
        
        # Detect which currency the user is asking about
        currency_patterns = {
            'USD': r'\b(usd|dollar|dollars|\$|stripe|paypal)\b',
            'EUR': r'\b(eur|euro|euros|€)\b',
            'GBP': r'\b(gbp|pound|pounds|£)\b',
        }
        
        detected_currency = None
        for code, pattern in currency_patterns.items():
            if re.search(pattern, query_lower):
                detected_currency = code
                break
        
        # Check if this is a currency conversion query
        is_currency_query = (
            detected_currency and 
            any(kw in query_lower for kw in [
                'inr', 'rupee', 'convert', 'exchange', 'cash out', 
                'get', 'rate', 'conversion', 'how much'
            ])
        )
        
        # ──────────────────────────────────────────────────────
        # DIRECT RATE FETCH for currency queries
        # ──────────────────────────────────────────────────────
        if is_currency_query and detected_currency:
            logger.info(f"💱 Currency conversion detected: {detected_currency} to INR")
            logger.info(f"🌐 Fetching live exchange rate (not searching)...")
            
            rate_info = get_exchange_rate(detected_currency, "INR")
            
            if rate_info:
                # Successfully fetched rate - format it beautifully for the LLM
                current_date = dt_info['date_context'].split(':')[1].strip()
                
                rate_str = f"""LIVE EXCHANGE RATE
Retrieved: {current_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Current Rate:** 1 {detected_currency} = {rate_info['rate']:.4f} INR

**Source:** {rate_info['source']}
**Verification URL:** {rate_info['url']}

This is the ACTUAL current exchange rate fetched directly from {rate_info['source']}.
Use this exact rate for all calculations in your response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
                logger.info(f"✓ Rate fetched successfully: 1 {detected_currency} = {rate_info['rate']:.4f} INR from {rate_info['source']}")
                return rate_str
            else:
                # Failed to fetch rate - inform LLM explicitly
                logger.warning(f"Failed to fetch exchange rate from all sources")
                return f"""EXCHANGE RATE FETCH FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Unable to retrieve the current {detected_currency} to INR exchange rate.

Tell the user to check:
- Google Finance (search '{detected_currency} to INR')
- XE.com currency converter
- Their bank's current rates

Do NOT estimate or guess the exchange rate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # ──────────────────────────────────────────────────────
        # REGULAR WEB SEARCH for non-currency queries
        # ──────────────────────────────────────────────────────
        search_query = dt_info['augmented_query'] if dt_info['has_temporal'] else query
        
        if dt_info['has_temporal']:
            logger.info(f"🕐 Temporal refs detected: {dt_info['temporal_refs']}")
            logger.info(f"📝 Search query: {search_query}")
        
        search = DuckDuckGoSearchRun()
        raw_results = search.run(search_query)
        
        formatted_results = f"""WEB SEARCH RESULTS:

{raw_results}
"""
        
        logger.info(f"Web search completed | query_len={len(search_query)} | result_len={len(formatted_results)}")
        return formatted_results
    
    except Exception as e:
        logger.error(f"Web search/fetch failed: {e}")
        return ""

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
        
        # RAG-only prompt (includes date_context placeholder)
        self.rag_prompt   = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_RAG_ONLY),
            ("human",  HUMAN_PROMPT),
        ])
        
        # Hybrid prompt (includes date_context placeholder)
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

        # CRITICAL: Force web search for currency/rate queries regardless of RAG score
        # These queries ALWAYS need current data, even if KB has good context
        question_lower = question.lower()
        is_currency_query = any(kw in question_lower for kw in [
            'exchange rate', 'current rate', 'today', 'conversion rate',
            'usd to inr', 'eur to inr', 'gbp to inr', 'dollar to rupee',
            'euro to rupee', 'pound to rupee', 'convert', 'cash out',
            'how much inr', 'how much rupee'
        ])
    
        if is_currency_query:
            logger.info("🔥 Currency query detected — FORCING web search (overriding RAG score)")
            web_context     = run_web_search(question)
            web_search_used = True
        elif top_score < RAG_CONFIDENCE_THRESHOLD:
            logger.info("Low RAG confidence — triggering web search")
            web_context     = run_web_search(question)
            web_search_used = True


        # ── Step 3: Build KB context ────────────────────────
        kb_context = self._format_context(retrieved)

        # ── Step 4: Inject date context and invoke correct prompt ──
        date_context = datetime_parser.get_current_date_context()
        
        if web_search_used:
            chain = self.hybrid_prompt | self.llm | self.parser
            raw   = chain.invoke({
                "date_context": date_context,  # ADD THIS
                "kb_context":   kb_context,
                "web_context":  web_context,
                "question":     question
            })
        else:
            chain = self.rag_prompt | self.llm | self.parser
            raw   = chain.invoke({
                "date_context": date_context,  # ADD THIS
                "context":      kb_context,
                "question":     question
            })

        # ── Step 5: Parse + enrich answer ──────────────────
        latency_ms     = (time.time() - start) * 1000
        official_links = detect_relevant_links(question, raw)
        answer         = self._parse_answer(
            question, raw, retrieved, latency_ms,
            web_search_used, official_links, web_context
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
        web_context:     str = "",  

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
        
        # Lower confidence for queries asking about current rates/prices
        # if we used web search (these are time-sensitive)
        if web_search_used:
            current_data_keywords = ["exchange rate", "current", "today", "price", "rate", "how much"]
            if any(kw in question.lower() for kw in current_data_keywords):
                # Cap confidence at medium for time-sensitive data
                if confidence == "high":
                    confidence = "medium"
                    logger.info("Capped confidence to medium for time-sensitive query")

        # Extract cited sources
        cited = list(dict.fromkeys(c["source"] for c in chunks[:3]))
        if web_search_used:
            cited.append("live_web_search")

        # HALLUCINATION DETECTION for exchange rates
        # If query asks about rates but web results don't contain them,
        # and LLM gives specific numbers → likely hallucination
        hallucination_phrases = [
            r"exchange rate is (approximately )?[\d.]+",
            r"current rate (is|of) (approximately )?[\d.]+",
            r"as of .+, the rate is [\d.]+",
        ]
        
        is_rate_query = any(kw in question.lower() for kw in [
            "exchange rate", "conversion rate", "convert", "euro to", "usd to", "rate"
        ])
        
        if is_rate_query and web_search_used:
            # Check if web context actually has numbers
            import re as regex
            web_has_numbers = bool(regex.search(r'\d+\.\d+', web_context)) if web_context else False
            llm_gives_numbers = any(regex.search(pattern, raw.lower()) for pattern in hallucination_phrases)
            
            if llm_gives_numbers and not web_has_numbers:
                # Likely hallucination - override response
                logger.warning("⚠️  Hallucination detected - LLM gave exchange rate but web results don't have it")
                raw = (
                    "I found information about payment platforms and fees, but I don't have "
                    "today's current exchange rate in my search results.\n\n"
                    "For the most accurate, real-time exchange rate, please check:\n"
                    "- **Google Finance** (search 'EUR to INR')\n"
                    "- **XE.com** - currency converter\n"
                    "- **Your bank's** current rates\n\n"
                    "Once you have the current rate, I can help you understand the platform fees "
                    "and total costs for your transfer.\n\n[Low]"
                )
                confidence = "low"
                fallback_used = True
        
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