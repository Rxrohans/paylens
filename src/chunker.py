"""
chunker.py — Phase 1 of ChargeClarity
----------------------------------------
WHAT THIS FILE DOES:
  Takes the raw text from ingestor.py and splits it into small chunks.
  These chunks are what we'll embed into vectors and store in FAISS.

CONCEPT — Why do we chunk?
  LLMs can only read so much text at once (context window limit).
  Also, searching through one giant document is slow and imprecise.
  By splitting into small overlapping chunks, we can find EXACTLY
  the relevant paragraph about "PayPal cross-border fees" vs the whole PDF.

  Think of it like an index in a book — instead of reading cover to cover,
  you jump straight to the right page.

CONCEPT — What is chunk_overlap?
  If chunk A ends with "The fee is 2.9%" and chunk B starts fresh,
  the model might miss context. Overlap means chunk B starts a bit
  before where chunk A ended — so no sentence is ever cut in half.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

# LangChain's text splitter — does the chunking for us
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# CHUNKING SETTINGS — tune these for better results
# ──────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters per chunk (not tokens)
CHUNK_OVERLAP = 64    # characters shared between consecutive chunks


def chunk_text(text: str, source_name: str) -> List[Dict]:
    """
    Splits text into overlapping chunks with metadata.
    
    Args:
        text: Raw document text
        source_name: Name of the source file (for traceability)
    
    Returns:
        List of dicts: [{text, source, chunk_id}, ...]
    
    WHY metadata matters for Pine Labs JD:
        Every chunk knows where it came from.
        This enables "explainability artefacts" — you can always
        tell the user "this answer came from PayPal's fee page, section 3"
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # It tries to split on paragraphs first, then sentences, then words
        # This keeps meaning intact rather than cutting mid-sentence
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    raw_chunks = splitter.split_text(text)
    
    # Add metadata to each chunk
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text.strip(),
            "source": source_name,
            "chunk_id": f"{source_name}_chunk_{i:04d}",
            "char_count": len(chunk_text)
        })
    
    return chunks


def process_all_raw_files() -> List[Dict]:
    """
    Reads all .txt files in data/raw/, chunks them,
    and saves the result to data/processed/chunks.json
    """
    print("\n✂️  Starting chunking pipeline...\n")
    all_chunks = []
    
    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print("⚠️  No .txt files found in data/raw/. Run ingestor.py first!")
        return []
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        if len(text.strip()) < 100:
            print(f"  ⚠️  Skipping — too short ({len(text)} chars)")
            continue
        
        chunks = chunk_text(text, source_name=txt_file.stem)
        all_chunks.extend(chunks)
        print(f"  ✅ {len(chunks)} chunks created")
    
    # Save all chunks to JSON (our processed dataset)
    output_path = PROC_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n📊 Chunking Summary:")
    print(f"   Total chunks   : {len(all_chunks)}")
    print(f"   Avg chunk size : {sum(c['char_count'] for c in all_chunks) // max(len(all_chunks),1)} chars")
    print(f"   Saved to       : {output_path}\n")
    
    return all_chunks


def load_chunks() -> List[Dict]:
    """Load already-processed chunks from disk."""
    path = PROC_DIR / "chunks.json"
    if not path.exists():
        raise FileNotFoundError("No chunks found. Run process_all_raw_files() first.")
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    process_all_raw_files()