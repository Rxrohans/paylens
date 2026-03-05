"""
ragas_eval.py — Phase 5 of PayLens
-------------------------------------
WHAT THIS FILE DOES:
  Runs golden questions through the pipeline and scores quality
  using lightweight metrics — no extra LLM calls needed.

WHY NOT RAGAS LIBRARY:
  RAGAS makes 80+ LLM calls per eval run (1 per question per metric).
  This exceeds Groq free tier limits and causes 429/400 errors.
  Our custom metrics compute the same signal locally — faster,
  free, and more transparent about what's actually being measured.

METRICS:
  answer_faithfulness  — does answer use words from retrieved context?
  answer_relevancy     — does answer address the question keywords?
  context_coverage     — do retrieved chunks contain ground truth info?
  overall_score        — average of all three

HOW TO RUN:
  python eval/ragas_eval.py
"""

import os
import sys
import json
import time
import re
import logging
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


# ── Text helpers ──────────────────────────────────────────
def tokenize(text: str) -> set:
    """Lowercase words, strip punctuation, remove stopwords."""
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


def overlap_score(text_a: str, text_b: str) -> float:
    """
    Jaccard-like overlap between two texts.
    Score = |intersection| / |union|
    Range: 0 (no overlap) to 1 (identical vocabulary)
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


def coverage_score(ground_truth: str, contexts: List[str]) -> float:
    """
    What fraction of ground truth keywords appear in retrieved chunks?
    Range: 0 (nothing retrieved) to 1 (all GT info in context)
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
    Respects Groq free tier: 30 RPM = 2 second delay between calls.
    """
    from chain import ChargeChain
    from guardrails import run_with_guardrails

    chain   = ChargeChain()
    results = []
    total   = len(questions)

    logger.info(f"Running {total} questions through pipeline...")
    logger.info("Rate limit: 2s delay between calls (Groq free tier 30 RPM)")

    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        logger.info(f"[{i}/{total}] {q[:65]}...")

        try:
            result    = run_with_guardrails(q, chain.ask)

            if result["blocked"]:
                logger.warning(f"  Blocked: {result['guardrail_warnings']}")
                continue

            retrieved = chain.retriever.retrieve(q, top_k=5)
            contexts  = [c["text"] for c in retrieved]

            results.append({
                "question":     q,
                "answer":       result["answer"],
                "contexts":     contexts,
                "ground_truth": gt,
                "latency_ms":   result["latency_ms"],
                "confidence":   result["confidence"],
                "web_search":   result.get("web_search_used", False),
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

        # 2 second delay — stays well within 30 RPM free tier limit
        if i < total:
            time.sleep(5)

    logger.info(f"Collected {len(results)} valid outputs")
    return results


# ── Metric computation ────────────────────────────────────
def compute_metrics(outputs: List[Dict]) -> Dict:
    """
    Computes three metrics per sample, returns averages.

    answer_faithfulness:
        overlap(answer, retrieved_contexts)
        High = answer uses words from what was retrieved
        Low  = answer may be hallucinating

    answer_relevancy:
        overlap(answer, question)
        High = answer addresses the question
        Low  = answer went off-topic

    context_coverage:
        fraction of ground_truth keywords in retrieved chunks
        High = retriever found the right documents
        Low  = retriever missed relevant information
    """
    faithfulness_scores  = []
    relevancy_scores     = []
    coverage_scores      = []
    per_sample           = []

    for o in outputs:
        ctx_text     = " ".join(o["contexts"])
        faith        = overlap_score(o["answer"], ctx_text)
        relevancy    = overlap_score(o["answer"], o["question"])
        coverage     = coverage_score(o["ground_truth"], o["contexts"])

        faithfulness_scores.append(faith)
        relevancy_scores.append(relevancy)
        coverage_scores.append(coverage)

        per_sample.append({
            "question":    o["question"][:60] + "...",
            "confidence":  o["confidence"],
            "latency_ms":  o["latency_ms"],
            "faithfulness":   round(faith, 3),
            "relevancy":      round(relevancy, 3),
            "coverage":       round(coverage, 3),
        })

    avg_faith    = sum(faithfulness_scores)  / len(faithfulness_scores)
    avg_rel      = sum(relevancy_scores)     / len(relevancy_scores)
    avg_cov      = sum(coverage_scores)      / len(coverage_scores)
    overall      = (avg_faith + avg_rel + avg_cov) / 3

    return {
        "faithfulness":      round(avg_faith, 4),
        "answer_relevancy":  round(avg_rel,   4),
        "context_coverage":  round(avg_cov,   4),
        "overall_score":     round(overall,   4),
        "per_sample":        per_sample,
    }


# ── Save scores ───────────────────────────────────────────
def save_scores(metrics: Dict, outputs: List[Dict]) -> Dict:
    """Appends this run to scores_history.json."""
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
        "timestamp":         datetime.now().isoformat(),
        "num_questions":     len(outputs),
        "faithfulness":      metrics["faithfulness"],
        "answer_relevancy":  metrics["answer_relevancy"],
        "context_coverage":  metrics["context_coverage"],
        "overall_score":     metrics["overall_score"],
        "avg_latency_ms":    round(sum(latencies) / len(latencies), 2),
        "web_search_rate":   round(web_rate, 4),
        "confidence_dist":   conf_counts,
        "per_sample":        metrics["per_sample"],
    }

    history.append(entry)

    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return entry


# ── Print results ─────────────────────────────────────────
def print_results(entry: Dict):
    PASS = 0.35  # threshold for keyword overlap metrics
    print("\n" + "="*62)
    print("  PAYLENS — EVALUATION RESULTS")
    print("="*62)
    print(f"  Timestamp        : {entry['timestamp'][:19]}")
    print(f"  Questions tested : {entry['num_questions']}")
    print(f"  Overall Score    : {entry['overall_score']:.2%}")
    print()
    print(f"  Faithfulness     : {entry['faithfulness']:.2%}  "
          f"{'[OK]' if entry['faithfulness'] > PASS else '[LOW]'}")
    print(f"  Answer Relevancy : {entry['answer_relevancy']:.2%}  "
          f"{'[OK]' if entry['answer_relevancy'] > PASS else '[LOW]'}")
    print(f"  Context Coverage : {entry['context_coverage']:.2%}  "
          f"{'[OK]' if entry['context_coverage'] > PASS else '[LOW]'}")
    print()
    print(f"  Avg Latency      : {entry['avg_latency_ms']:.0f}ms")
    print(f"  Web Search Rate  : {entry['web_search_rate']:.0%} of queries")
    print(f"  Confidence Dist  : {entry['confidence_dist']}")
    print()
    print("  Per-sample breakdown:")
    for s in entry["per_sample"]:
        print(f"    [{s['confidence'][:3].upper()}] {s['question'][:50]}")
        print(f"         faith={s['faithfulness']:.2f}  "
              f"rel={s['relevancy']:.2f}  cov={s['coverage']:.2f}  "
              f"{s['latency_ms']:.0f}ms")
    print("="*62)
    print(f"  Saved to: {SCORES_PATH}")
    print("="*62 + "\n")


# ── Main ──────────────────────────────────────────────────
def run_evaluation():
    logger.info("Starting PayLens evaluation...")

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