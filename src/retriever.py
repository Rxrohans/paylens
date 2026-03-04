"""
retriever.py — Phase 2 of ChargeClarity
----------------------------------------
WHAT THIS FILE DOES:
  Given a user's question, finds the TOP K most relevant chunks
  from the FAISS index. These chunks become the "context" passed
  to the LLM in Phase 3.

CONCEPT — The Retrieval Step in RAG:
  R-A-G = Retrieve → Augment → Generate
  This file handles the R.

  User asks: "Why does PayPal charge extra for USD payments?"
      ↓
  We embed that question (same model, same 384 dimensions)
      ↓
  FAISS finds the 5 most similar chunk vectors
      ↓
  We return those 5 chunks as context
      ↓
  LLM uses those chunks to answer (Phase 3)

WHY THIS MATTERS FOR PINE LABS JD:
  "Prepare & curate data for AI agents (RAG, embeddings, context)"
  This is the retrieval + context assembly pipeline.
"""

import time
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from embedder import load_index, load_embedding_model

# How many chunks to retrieve per query
TOP_K = 5


class ChargeRetriever:
    """
    Handles all retrieval operations for ChargeClarity.
    Loads the index once, reuses it for every query (efficient).
    """

    def __init__(self):
        print("🔍 Initializing ChargeRetriever...")
        self.index, self.chunks = load_index()
        self.model = load_embedding_model()
        print("   ✅ Retriever ready\n")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Main retrieval function.

        Args:
            query:  User's question as plain text
            top_k:  Number of chunks to return

        Returns:
            List of top_k chunk dicts, each with:
            {text, source, chunk_id, score}
            
        CONCEPT — Score:
            Score is cosine similarity (0 to 1).
            1.0 = identical meaning, 0.0 = completely unrelated.
            We typically see 0.4–0.8 for good matches.
        """
        start = time.time()

        # Step 1: Embed the query
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # Step 2: Search FAISS
        scores, indices = self.index.search(query_vec, top_k)

        # Step 3: Build results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 when not enough vectors
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        elapsed = time.time() - start

        # Log the retrieval (this is the "logs and traces" requirement in the JD)
        self._log_retrieval(query, results, elapsed)

        return results

    def retrieve_with_context(self, query: str, top_k: int = TOP_K) -> str:
        """
        Returns retrieved chunks formatted as a single context string.
        This is what gets passed directly to the LLM prompt.

        Format:
            [Source: paypal_fees_india | ID: paypal_fees_india_chunk_0012]
            <chunk text>

            [Source: razorpay_pricing | ID: ...]
            <chunk text>
            ...
        """
        chunks = self.retrieve(query, top_k)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source: {chunk['source']} | Score: {chunk['score']:.3f}]\n"
                f"{chunk['text']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _log_retrieval(self, query: str, results: List[Dict], elapsed: float):
        """
        Logs retrieval details to console.
        In production this would write to a log file / observability platform.

        WHY THIS MATTERS FOR PINE LABS JD:
            "Maintain logs, traces and explainability artefacts"
            Every query is traceable — what was asked, what was retrieved,
            how confident the system was, how fast it responded.
        """
        print(f"🔍 Query     : {query[:80]}...")
        print(f"   Latency   : {elapsed*1000:.1f}ms")
        print(f"   Top result: [{results[0]['source']}] score={results[0]['score']:.3f}" if results else "   No results")
        print()


# ──────────────────────────────────────────────
# Quick test — run this directly to verify retrieval works
# ──────────────────────────────────────────────
if __name__ == "__main__":
    retriever = ChargeRetriever()

    test_queries = [
        "Why is PayPal charging me extra for international payments?",
        "What is the transaction fee on Razorpay?",
        "How much does Stripe charge per payment in India?",
        "What is a currency conversion fee?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print(f"{'='*60}")
        chunks = retriever.retrieve(q, top_k=3)
        for i, c in enumerate(chunks, 1):
            print(f"\n  Result {i} (score={c['score']:.3f}) [{c['source']}]")
            print(f"  {c['text'][:200]}...")