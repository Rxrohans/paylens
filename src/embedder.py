"""
embedder.py — Phase 2 of ChargeClarity
----------------------------------------
WHAT THIS FILE DOES:
  Takes every chunk from chunker.py and converts it into a vector (embedding).
  Then stores all vectors in a FAISS index so we can search them later.

CONCEPT — The Embedding Model:
  We use "all-MiniLM-L6-v2" from HuggingFace.
  It's FREE, runs on your laptop CPU, and is surprisingly powerful.
  It converts any text → 384 numbers that capture its meaning.

CONCEPT — FAISS:
  FAISS (Facebook AI Similarity Search) is a vector database.
  It stores all our embeddings and can find the most similar ones
  to a query in milliseconds — even with millions of vectors.
  Think of it as a super-powered search engine that understands meaning.

WHY THIS MATTERS FOR PINE LABS JD:
  "Experience with embeddings, RAG or vector databases" — this is exactly that.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # progress bar

PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH = PROC_DIR / "faiss_index.bin"      # the FAISS index file
META_PATH  = PROC_DIR / "chunk_metadata.pkl"   # chunk texts + sources

# ──────────────────────────────────────────────
# EMBEDDING MODEL — free, local, no API needed
# ──────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM  = 384   # this model always outputs 384-dimensional vectors


def load_embedding_model() -> SentenceTransformer:
    """
    Loads the embedding model.
    First run: downloads ~80MB from HuggingFace (one time only).
    After that: loads from cache instantly.
    """
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✅ Model loaded — output dimension: {EMBED_DIM}")
    return model


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Converts a list of chunk dicts into a numpy matrix of embeddings.

    Args:
        chunks: List of {text, source, chunk_id} dicts from chunker.py
        model:  Loaded SentenceTransformer model

    Returns:
        numpy array of shape (num_chunks, 384)
        Each row = one chunk's embedding vector

    CONCEPT — Batching:
        We embed in batches of 64 for speed.
        Single embedding call per chunk would be very slow.
    """
    texts = [c["text"] for c in chunks]
    
    print(f"\n🔢 Embedding {len(texts)} chunks in batches...")
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True   # normalize so cosine similarity = dot product
    )
    
    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.1f}s  —  shape: {embeddings.shape}")
    return embeddings.astype("float32")  # FAISS needs float32


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS index from embedding vectors.

    CONCEPT — IndexFlatIP:
        IP = Inner Product (dot product).
        Since we normalized embeddings above, dot product = cosine similarity.
        Cosine similarity is the standard for semantic search.
        
        Other index types exist (IVF, HNSW) for millions of vectors,
        but for our use case FlatIP is perfect — simple and exact.

    Args:
        embeddings: numpy array (num_chunks, 384)

    Returns:
        FAISS index ready for searching
    """
    print(f"\n🗂️  Building FAISS index...")
    
    index = faiss.IndexFlatIP(EMBED_DIM)  # flat = brute force exact search
    index.add(embeddings)
    
    print(f"   ✅ Index built — {index.ntotal} vectors stored")
    return index


def save_index(index: faiss.IndexFlatIP, chunks: List[Dict]):
    """
    Saves the FAISS index and chunk metadata to disk.
    
    WHY save separately?
        FAISS stores the vectors.
        We store the original texts + sources in pickle separately.
        When we search, FAISS gives us indices (0, 3, 7...)
        We use those indices to look up the actual text.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"\n💾 Saved:")
    print(f"   FAISS index    → {INDEX_PATH}  ({INDEX_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"   Chunk metadata → {META_PATH}   ({META_PATH.stat().st_size / 1024:.1f} KB)")


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Loads FAISS index + metadata from disk.
    Called by retriever.py at query time.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError("No FAISS index found. Run build_vector_store() first.")
    
    index  = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"✅ Loaded FAISS index — {index.ntotal} vectors")
    return index, chunks


def build_vector_store():
    """
    Master function: chunks → embeddings → FAISS index → saved to disk.
    Run this once after ingestor + chunker have been run.
    """
    # Load chunks
    chunks_path = PROC_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError("No chunks.json found. Run chunker.py first!")
    
    with open(chunks_path, "r" , encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"📦 Loaded {len(chunks)} chunks from {chunks_path}")
    
    # Build the vector store
    model      = load_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks)
    
    print("\n🎉 Vector store ready! You can now run retriever.py")
    return index, chunks


if __name__ == "__main__":
    build_vector_store()