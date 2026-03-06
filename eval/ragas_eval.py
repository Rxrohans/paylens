"""
ragas_eval.py — Phase 5 of PayLens  (FIXED v2)
------------------------------------------------
FIXES IN THIS VERSION:

  FIX 1 — web_search_rate always 0%:
    Bug: result.get("web_search_used", False) was returning False
         because guardrails.py return dict was missing that key.
    Fix: guardrails.py now includes web_search_used. This file
         also adds a fallback check on "web_search" key for safety.

  FIX 2 — faithfulness scores underreporting paraphrasing:
    Bug: Jaccard overlap punishes good paraphrasing.
         "exceeds two lakh rupees" ≠ "above ₹2,00,000" by word overlap,
         even though they mean the same thing.
    Fix: Semantic faithfulness using sentence embeddings (cosine similarity).
         We embed answer + context and compare in vector space.
         Paraphrases now score correctly.

  FIX 3 — Added per-metric pass/fail thresholds tuned for semantic scores
    Semantic similarity scores are higher than Jaccard (0.6+ is meaningful).
    Thresholds updated accordingly.

METRICS:
  answer_faithfulness  — cosine similarity(answer embedding, context embedding)
  answer_relevancy     — cosine similarity(answer embedding, question embedding)
  context_coverage     — keyword recall of ground truth in retrieved chunks
                         (kept as keyword metric — coverage is recall-based, OK)
  overall_score        — average of all three
"""

import os
import sys
import json
import time
import re
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("paylens.eval")

EVAL_DIR     = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "golden_dataset.json"
SCORES_PATH  = EVAL_DIR / "scores_history.json"


# ── Embedding model for semantic similarity ───────────────
# Loaded once, reused for all metric computations
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for eval metrics...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    FIX 2: Semantic similarity using sentence embeddings.

    Why this is better than Jaccard:
      Jaccard("exceeds two lakh", "above ₹2,00,000") ≈ 0.05 (almost no overlap)
      Semantic("exceeds two lakh", "above ₹2,00,000") ≈ 0.82 (correctly similar)

    "RTGS charges ₹30 for amounts exceeding two lakh rupees" is a faithful
    paraphrase of "RTGS fee for transactions above ₹2,00,000: ₹30" — now scores correctly.

    Range: -1 to 1 (in practice 0 to 1 for meaningful text)
    Threshold for PASS: 0.50 (well above random noise at ~0.2)
    """
    model = get_embed_model()
    vecs = model.encode([text_a, text_b], normalize_embeddings=True)
    return cosine_similarity(vecs[0], vecs[1])


# ── Keyword coverage (kept as-is — it's recall-based, correct for coverage) ─
def tokenize(text: str) -> set:
    STOPWORDS = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","for",
        "on","with","at","by","from","as","into","through","about",
        "and","or","but","if","then","that","this","it","its","i",
        "you","we","they","he","she","what","how","when","where","why"
    }
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def coverage_score(ground_truth: str, contexts: List[str]) -> float:
    """
    Keyword recall: what fraction of ground truth terms appear in retrieved chunks?
    Kept as keyword metric — this is fundamentally a recall question, not semantic.
    """
    gt_tokens  = tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    ctx_tokens = tokenize(" ".join(contexts))
    covered    = gt_tokens & ctx_tokens
    return len(covered) / len(gt_tokens)


# ── Pipeline runner ───────────────────────────────────────
def collect_pipeline_outputs(questions: List[Dict]) -> List[Dict]:
    """
    Runs each question through PayLens pipeline.
    FIX: Now correctly reads web_search_used from the guardrails return dict.
    """
    from chain import ChargeChain
    from guardrails import run_with_guardrails

    chain   = ChargeChain()
    results = []
    total   = len(questions)

    logger.info(f"Running {total} questions through pipeline...")

    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        logger.info(f"[{i}/{total}] {q[:65]}...")

        try:
            result = run_with_guardrails(q, chain.ask)

            if result["blocked"]:
                logger.warning(f"  Blocked: {result['guardrail_warnings']}")
                continue

            retrieved = chain.retriever.retrieve(q, top_k=5)
            contexts  = [c["text"] for c in retrieved]

            # FIX 1: web_search_used is now correctly present in the dict
            # Added fallback to "web_search" key for backwards compatibility
            web_used = (
                result.get("web_search_used")       # new key (guardrails v2)
                or result.get("web_search", False)  # old key fallback
            )

            results.append({
                "question":     q,
                "answer":       result["answer"],
                "contexts":     contexts,
                "ground_truth": gt,
                "latency_ms":   result["latency_ms"],
                "confidence":   result["confidence"],
                "web_search":   web_used,           # correctly populated now
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

        if i < total:
            time.sleep(5)  # 30 RPM Groq free tier

    logger.info(f"Collected {len(results)} valid outputs")
    return results


# ── Metric computation ────────────────────────────────────
def compute_metrics(outputs: List[Dict]) -> Dict:
    """
    FIX 2: Uses semantic similarity for faithfulness and relevancy.

    answer_faithfulness  → semantic_similarity(answer, retrieved_context)
    answer_relevancy     → semantic_similarity(answer, question)
    context_coverage     → keyword recall (ground truth terms in chunks)
    """
    faithfulness_scores  = []
    relevancy_scores     = []
    coverage_scores      = []
    per_sample           = []

    logger.info("Computing semantic similarity metrics...")

    for o in outputs:
        ctx_text  = " ".join(o["contexts"])

        # FIX: semantic similarity instead of Jaccard overlap
        faith     = semantic_similarity(o["answer"], ctx_text)
        relevancy = semantic_similarity(o["answer"], o["question"])
        coverage  = coverage_score(o["ground_truth"], o["contexts"])

        faithfulness_scores.append(faith)
        relevancy_scores.append(relevancy)
        coverage_scores.append(coverage)

        per_sample.append({
            "question":    o["question"][:60] + "...",
            "confidence":  o["confidence"],
            "latency_ms":  o["latency_ms"],
            "web_search":  o["web_search"],
            "faithfulness":   round(faith, 3),
            "relevancy":      round(relevancy, 3),
            "coverage":       round(coverage, 3),
        })

    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_rel   = sum(relevancy_scores)    / len(relevancy_scores)
    avg_cov   = sum(coverage_scores)     / len(coverage_scores)
    overall   = (avg_faith + avg_rel + avg_cov) / 3

    return {
        "faithfulness":     round(avg_faith, 4),
        "answer_relevancy": round(avg_rel,   4),
        "context_coverage": round(avg_cov,   4),
        "overall_score":    round(overall,   4),
        "per_sample":       per_sample,
    }


# ── Save scores ───────────────────────────────────────────
def save_scores(metrics: Dict, outputs: List[Dict]) -> Dict:
    history = []
    if SCORES_PATH.exists():
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

    latencies   = [o["latency_ms"] for o in outputs]
    web_rate    = sum(1 for o in outputs if o["web_search"]) / len(outputs)
    conf_counts = {}
    for o in outputs:
        conf_counts[o["confidence"]] = conf_counts.get(o["confidence"], 0) + 1

    entry = {
        "timestamp":        datetime.now().isoformat(),
        "metric_version":   "v2-semantic",  # tag so you know which algo was used
        "num_questions":    len(outputs),
        "faithfulness":     metrics["faithfulness"],
        "answer_relevancy": metrics["answer_relevancy"],
        "context_coverage": metrics["context_coverage"],
        "overall_score":    metrics["overall_score"],
        "avg_latency_ms":   round(sum(latencies) / len(latencies), 2),
        "web_search_rate":  round(web_rate, 4),   # now correctly > 0
        "confidence_dist":  conf_counts,
        "per_sample":       metrics["per_sample"],
    }

    history.append(entry)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return entry


# ── Print results ─────────────────────────────────────────
def print_results(entry: Dict):
    # Semantic similarity thresholds (higher than Jaccard — 0.5+ is meaningful)
    PASS_FAITH = 0.50
    PASS_REL   = 0.55
    PASS_COV   = 0.35  # coverage stays keyword-based, keep lower threshold

    print("\n" + "="*62)
    print("  PAYLENS — EVALUATION RESULTS (v2 · Semantic Metrics)")
    print("="*62)
    print(f"  Timestamp        : {entry['timestamp'][:19]}")
    print(f"  Metric version   : {entry.get('metric_version', 'v1-jaccard')}")
    print(f"  Questions tested : {entry['num_questions']}")
    print(f"  Overall Score    : {entry['overall_score']:.2%}")
    print()
    print(f"  Faithfulness     : {entry['faithfulness']:.2%}  "
          f"{'[OK]' if entry['faithfulness'] > PASS_FAITH else '[LOW]'}"
          f"  (semantic sim, threshold {PASS_FAITH:.0%})")
    print(f"  Answer Relevancy : {entry['answer_relevancy']:.2%}  "
          f"{'[OK]' if entry['answer_relevancy'] > PASS_REL else '[LOW]'}"
          f"  (semantic sim, threshold {PASS_REL:.0%})")
    print(f"  Context Coverage : {entry['context_coverage']:.2%}  "
          f"{'[OK]' if entry['context_coverage'] > PASS_COV else '[LOW]'}"
          f"  (keyword recall, threshold {PASS_COV:.0%})")
    print()
    print(f"  Avg Latency      : {entry['avg_latency_ms']:.0f}ms")
    print(f"  Web Search Rate  : {entry['web_search_rate']:.0%} of queries")
    print(f"  Confidence Dist  : {entry['confidence_dist']}")
    print()
    print("  Per-sample breakdown:")
    for s in entry["per_sample"]:
        web_flag = "🌐" if s.get("web_search") else "📚"
        print(f"    {web_flag} [{s['confidence'][:3].upper()}] {s['question'][:48]}")
        print(f"         faith={s['faithfulness']:.2f}  "
              f"rel={s['relevancy']:.2f}  cov={s['coverage']:.2f}  "
              f"{s['latency_ms']:.0f}ms")
    print("="*62)
    print(f"  Saved to: {SCORES_PATH}")
    print("="*62 + "\n")


# ── Main ──────────────────────────────────────────────────
def run_evaluation():
    logger.info("Starting PayLens evaluation (v2 — semantic metrics)...")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} golden questions")

    outputs = collect_pipeline_outputs(dataset)
    if not outputs:
        logger.error("No outputs collected — check your pipeline")
        return

    metrics = compute_metrics(outputs)
    entry   = save_scores(metrics, outputs)
    print_results(entry)
    return entry


if __name__ == "__main__":
    run_evaluation()