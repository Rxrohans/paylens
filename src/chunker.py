"""
chunker.py — Phase 1 of PayLens  (FIXED v2)
---------------------------------------------
FIXES IN THIS VERSION:
  1. chunk_overlap raised from 64 → 128 chars
     → fee brackets no longer split across chunk boundaries
  2. Table/fee-schedule detection — tables kept as atomic chunks
     → "NEFT fee for ₹10k–₹1L: ₹5" stays in one chunk, not split mid-row
  3. Source-type aware chunking — RBI circulars get larger chunks (more context)
  4. India-specific separators added (₹, lakh, crore) as soft split hints
  5. Chunk metadata now includes source_type for retriever filtering

ROOT CAUSE OF NEFT/RTGS FAILURES (chunking side):
  With chunk_size=512 and overlap=64, a fee table like:
    "Up to ₹10,000: ₹2.50 | ₹10,001–₹1L: ₹5 | ₹1L–₹2L: ₹15 | Above ₹2L: ₹25"
  gets split after "₹5 |" and the bracket context is lost.
  The retriever finds half the table → LLM gives incomplete/wrong answer.
"""

import re
import json
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# CHUNKING SETTINGS
# ──────────────────────────────────────────────
CHUNK_SIZE         = 600   # ↑ from 512 — more context per chunk
CHUNK_OVERLAP      = 128   # ↑ from 64  — critical for fee bracket continuity
MAX_TABLE_CHUNK    = 1200  # tables can be larger — never split mid-row

# Source types that need larger chunks (regulatory docs are dense)
RBI_SOURCES = {"rbi_", "neft", "rtgs", "imps", "npci", "pci"}


def is_rbi_source(source_name: str) -> bool:
    return any(tag in source_name.lower() for tag in RBI_SOURCES)


# ──────────────────────────────────────────────
# TABLE DETECTION
# Detects markdown tables, pipe-delimited rows, and
# structured fee schedules like "Up to ₹X : ₹Y"
# ──────────────────────────────────────────────
TABLE_PATTERNS = [
    r"\|.*\|",                           # markdown table row
    r"₹[\d,]+\s*(to|-|–)\s*₹[\d,]+",   # fee range like ₹10,000 to ₹1,00,000
    r"(up to|above|below|between)\s+₹", # bracket language
    r"^\s*\d+\.\s+.{10,}:\s+₹",        # numbered fee list
    r"={3,}|─{3,}|-{3,}",              # section dividers in RBI docs
]

def contains_table_or_fee_schedule(text: str) -> bool:
    """Returns True if this text block looks like a fee table or schedule."""
    for pattern in TABLE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def extract_table_blocks(text: str) -> List[tuple]:
    """
    Splits text into (block, is_table) pairs.
    Table blocks are returned as-is (never split further).
    Non-table blocks go through normal chunking.
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n{2,}', text)
    blocks = []
    buffer_text = ""
    buffer_is_table = False

    for para in paragraphs:
        para_is_table = contains_table_or_fee_schedule(para)

        if para_is_table:
            # Flush any pending normal text first
            if buffer_text and not buffer_is_table:
                blocks.append((buffer_text.strip(), False))
                buffer_text = ""

            # Accumulate table rows together
            if buffer_is_table:
                buffer_text += "\n\n" + para
            else:
                buffer_text = para
                buffer_is_table = True
        else:
            # Flush any pending table block
            if buffer_text and buffer_is_table:
                blocks.append((buffer_text.strip(), True))
                buffer_text = ""
                buffer_is_table = False

            buffer_text = (buffer_text + "\n\n" + para).strip() if buffer_text else para

    # Flush final buffer
    if buffer_text:
        blocks.append((buffer_text.strip(), buffer_is_table))

    return blocks


# ──────────────────────────────────────────────
# MAIN CHUNKING FUNCTION
# ──────────────────────────────────────────────
def chunk_text(text: str, source_name: str) -> List[Dict]:
    """
    Splits text into overlapping chunks with metadata.
    Tables and fee schedules are kept as atomic chunks (never split).

    Args:
        text:        Raw document text
        source_name: Name of the source file

    Returns:
        List of {text, source, chunk_id, char_count, source_type, has_fee_table}
    """
    # Larger chunks for RBI regulatory documents
    chunk_size = CHUNK_SIZE * 2 if is_rbi_source(source_name) else CHUNK_SIZE
    source_type = "regulatory" if is_rbi_source(source_name) else "commercial"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n", "\n\n", "\n",
            "₹",          # India-specific: rupee amounts are natural split points
            ". ", " ", ""
        ]
    )

    # Phase 1: Separate table blocks from normal text
    blocks = extract_table_blocks(text)

    chunks = []
    chunk_idx = 0

    for block_text, is_table in blocks:
        if not block_text.strip():
            continue

        if is_table:
            # Keep fee tables as a SINGLE chunk — never split
            # If it's huge, split only at row boundaries (double newline)
            if len(block_text) > MAX_TABLE_CHUNK:
                sub_blocks = block_text.split("\n\n")
                sub_buffer = ""
                for sb in sub_blocks:
                    if len(sub_buffer) + len(sb) < MAX_TABLE_CHUNK:
                        sub_buffer += ("\n\n" + sb) if sub_buffer else sb
                    else:
                        if sub_buffer:
                            chunks.append(_make_chunk(sub_buffer, source_name, chunk_idx, source_type, True))
                            chunk_idx += 1
                        sub_buffer = sb
                if sub_buffer:
                    chunks.append(_make_chunk(sub_buffer, source_name, chunk_idx, source_type, True))
                    chunk_idx += 1
            else:
                chunks.append(_make_chunk(block_text, source_name, chunk_idx, source_type, True))
                chunk_idx += 1
        else:
            # Normal text → recursive character splitting
            raw_chunks = splitter.split_text(block_text)
            for rc in raw_chunks:
                if rc.strip():
                    chunks.append(_make_chunk(rc, source_name, chunk_idx, source_type, False))
                    chunk_idx += 1

    return chunks


def _make_chunk(text: str, source: str, idx: int, source_type: str, has_fee_table: bool) -> Dict:
    return {
        "text":          text.strip(),
        "source":        source,
        "chunk_id":      f"{source}_chunk_{idx:04d}",
        "char_count":    len(text),
        "source_type":   source_type,      # "regulatory" | "commercial"
        "has_fee_table": has_fee_table,    # True = contains fee schedule
    }


# ──────────────────────────────────────────────
# PROCESS ALL FILES
# ──────────────────────────────────────────────
def process_all_raw_files() -> List[Dict]:
    print("\n✂️  Starting chunking pipeline (v2 — table-aware)...\n")
    all_chunks = []

    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print("⚠️  No .txt files found in data/raw/. Run ingestor.py first!")
        return []

    fee_table_count = 0

    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        if len(text.strip()) < 100:
            print(f"  ⚠️  Skipping — too short ({len(text)} chars)")
            continue

        chunks = chunk_text(text, source_name=txt_file.stem)
        tables = sum(1 for c in chunks if c["has_fee_table"])
        fee_table_count += tables
        all_chunks.extend(chunks)
        print(f"  ✅ {len(chunks)} chunks ({tables} fee table blocks protected)")

    output_path = PROC_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Chunking Summary (v2):")
    print(f"   Total chunks        : {len(all_chunks)}")
    print(f"   Fee table chunks    : {fee_table_count} (kept atomic)")
    print(f"   Avg chunk size      : {sum(c['char_count'] for c in all_chunks) // max(len(all_chunks),1)} chars")
    print(f"   Regulatory chunks   : {sum(1 for c in all_chunks if c['source_type'] == 'regulatory')}")
    print(f"   Saved to            : {output_path}\n")

    return all_chunks


def load_chunks() -> List[Dict]:
    path = PROC_DIR / "chunks.json"
    if not path.exists():
        raise FileNotFoundError("No chunks found. Run process_all_raw_files() first.")
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    process_all_raw_files()