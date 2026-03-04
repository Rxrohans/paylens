"""
ingestor.py — Phase 1 of ChargeClarity
----------------------------------------
WHAT THIS FILE DOES:
  Pulls documents (PDFs + web pages) about payment fees into our system.
  Think of this as the "librarian" — it fetches and stores raw text.

CONCEPT — Why do we need this?
  The LLM doesn't know PayPal's 2024 fee structure. We need to give it
  that knowledge. This file fetches those documents and saves them locally
  so our chunker (next file) can process them.
"""

import os
import requests
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

# Where we save raw documents
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# SOURCE 1: Load a PDF from a URL
# ──────────────────────────────────────────────
def ingest_pdf_from_url(url: str, save_name: str) -> str:
    """
    Downloads a PDF and extracts its text.
    
    Args:
        url: Direct link to the PDF
        save_name: What to name the saved text file (e.g. 'paypal_fees.txt')
    
    Returns:
        The extracted text as a string
    """
    print(f"📥 Downloading PDF: {save_name}")
    
    headers = {"User-Agent": "Mozilla/5.0"}  # some sites block bots without this
    response = requests.get(url, headers=headers, timeout=30)
    
    # Save the raw PDF first
    pdf_path = RAW_DIR / save_name.replace(".txt", ".pdf")
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    
    # Extract text from PDF
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # Save the extracted text
    txt_path = RAW_DIR / save_name
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"✅ Saved {len(full_text)} characters to {txt_path}")
    return full_text


# ──────────────────────────────────────────────
# SOURCE 2: Scrape a web page (like PayPal fees page)
# ──────────────────────────────────────────────
def ingest_webpage(url: str, save_name: str) -> str:
    """
    Scrapes a web page and extracts clean text.
    
    Args:
        url: Page URL (e.g. PayPal fees page)
        save_name: What to name the saved file
    
    Returns:
        Cleaned text from the page
    """
    print(f"🌐 Scraping webpage: {url}")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    
    # BeautifulSoup parses HTML — we only want the readable text, not HTML tags
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove navigation, scripts, ads — we only want body content
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    clean_text = soup.get_text(separator="\n", strip=True)
    
    # Save it
    txt_path = RAW_DIR / save_name
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    
    print(f"✅ Saved {len(clean_text)} characters to {txt_path}")
    return clean_text


# ──────────────────────────────────────────────
# SOURCE 3: Load a local text/PDF file you already have
# ──────────────────────────────────────────────
def ingest_local_file(file_path: str) -> str:
    """
    Loads a file you already have on your machine.
    Supports .txt and .pdf formats.
    """
    path = Path(file_path)
    print(f"📂 Loading local file: {path.name}")
    
    if path.suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "".join(page.extract_text() + "\n" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    
    # Copy to our raw directory with metadata tag
    save_path = RAW_DIR / path.name.replace(path.suffix, ".txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"✅ Loaded {len(text)} characters")
    return text


# ──────────────────────────────────────────────
# DOCUMENT REGISTRY — All sources for ChargeClarity
# Add more sources here as the project grows!
# ──────────────────────────────────────────────
SOURCES = [
    {
        "type": "webpage",
        "url": "https://www.paypal.com/in/webapps/mpp/paypal-fees",
        "save_name": "paypal_fees_india.txt",
        "description": "PayPal India fee structure"
    },
    {
        "type": "webpage",
        "url": "https://stripe.com/in/pricing",
        "save_name": "stripe_pricing_india.txt",
        "description": "Stripe India pricing"
    },
    {
        "type": "webpage",
        "url": "https://www.npci.org.in/what-we-do/upi/live-members",
        "save_name": "upi_charges.txt",
        "description": "NPCI UPI fee structure"
    },
]

def run_all_ingestions():
    """
    Runs all document ingestions in SOURCES.
    Call this once to populate your data/raw folder.
    """
    print("\n🚀 Starting ChargeClarity document ingestion...\n")
    results = []
    
    for source in SOURCES:
        try:
            if source["type"] == "webpage":
                text = ingest_webpage(source["url"], source["save_name"])
            elif source["type"] == "pdf":
                text = ingest_pdf_from_url(source["url"], source["save_name"])
            
            results.append({
                "source": source["description"],
                "file": source["save_name"],
                "chars": len(text),
                "status": "✅ Success"
            })
        except Exception as e:
            print(f"❌ Failed: {source['description']} — {e}")
            results.append({
                "source": source["description"],
                "file": source["save_name"],
                "chars": 0,
                "status": f"❌ {e}"
            })
    
    # Print summary
    print("\n📊 Ingestion Summary:")
    print("-" * 60)
    for r in results:
        print(f"{r['status']}  {r['source']:35s}  {r['chars']:,} chars")
    print("-" * 60)
    print(f"Raw files saved to: {RAW_DIR}\n")
    
    return results


# Run directly: python src/ingestor.py
if __name__ == "__main__":
    run_all_ingestions()