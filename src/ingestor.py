"""
ingestor.py — Phase 1 of PayLens  (FIXED v2)
----------------------------------------------
FIXES IN THIS VERSION:
  1. Added RBI static data: NEFT, RTGS, IMPS fee structures
  2. Added PCI-DSS compliance reference data
  3. Added Razorpay, Paytm sources
  4. ingest_static() for regulatory data that blocks scrapers
  5. Retry logic for flaky web pages

ROOT CAUSE OF NEFT/RTGS/IMPS FAILURES:
  v1 only had PayPal, Stripe, NPCI live members.
  NPCI live members page lists banks — not fee schedules.
  Zero RBI circular data was ever in the knowledge base.
  The LLM was answering from training data alone → hallucinations.
"""

import time
import requests
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ──────────────────────────────────────────────
# INGESTION HELPERS
# ──────────────────────────────────────────────

def ingest_webpage(url: str, save_name: str, retries: int = 2) -> str:
    print(f"🌐 Scraping: {url}")
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            clean_text = soup.get_text(separator="\n", strip=True)
            txt_path = RAW_DIR / save_name
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"  ✅ {len(clean_text):,} chars → {save_name}")
            return clean_text
        except Exception as e:
            if attempt < retries:
                print(f"  ⚠️  Attempt {attempt+1} failed ({e}), retrying...")
                time.sleep(2)
            else:
                print(f"  ❌ Failed after {retries+1} attempts: {e}")
                return ""


def ingest_pdf_from_url(url: str, save_name: str) -> str:
    print(f"📥 Downloading PDF: {save_name}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=45)
        response.raise_for_status()
        pdf_path = RAW_DIR / save_name.replace(".txt", ".pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        reader = PdfReader(str(pdf_path))
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"
        txt_path = RAW_DIR / save_name
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  ✅ {len(full_text):,} chars → {save_name}")
        return full_text
    except Exception as e:
        print(f"  ❌ PDF failed: {e}")
        return ""


def ingest_static(content: str, save_name: str) -> str:
    """
    Saves curated static text directly to raw/.
    Used for RBI/regulatory sources that block scrapers or require login.
    This is the MOST RELIABLE method for authoritative fee data.
    """
    txt_path = RAW_DIR / save_name
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅ Static content saved → {save_name} ({len(content):,} chars)")
    return content


# ──────────────────────────────────────────────
# STATIC RBI DATA
# Source: RBI circulars (public domain)
# RBI/2019-20/187 | RBI/2020-21/62 | DPSS guidelines
# ──────────────────────────────────────────────

RBI_PAYMENT_FEES_DATA = """
RBI PAYMENT SYSTEMS — FEE STRUCTURE FOR INDIA
Source: Reserve Bank of India (RBI) Official Circulars and NPCI Guidelines
Last updated: 2024

========================================
NEFT — NATIONAL ELECTRONIC FUNDS TRANSFER
========================================
Full form: National Electronic Funds Transfer
Operated by: Reserve Bank of India (RBI)
Settlement: Deferred Net Settlement — processes in half-hourly batches
Availability: 24x7x365 (since December 16, 2019)
Channels: Internet banking, mobile banking, bank branches

NEFT CHARGES — OUTWARD TRANSACTIONS (customer sending money):
  Transaction up to ₹10,000       : ₹2.50 + applicable GST
  Transaction ₹10,001 to ₹1,00,000 : ₹5.00 + applicable GST
  Transaction ₹1,00,001 to ₹2,00,000: ₹15.00 + applicable GST
  Transaction above ₹2,00,000     : ₹25.00 + applicable GST

NEFT INWARD CHARGES: FREE — no charge to the beneficiary (receiver)

RBI Waiver: In January 2020, RBI waived processing charges for member banks.
Banks were directed to pass these savings to customers.
Most major banks (SBI, HDFC, ICICI, Axis) now offer FREE online NEFT.
Branch NEFT may still carry the above charges.

NEFT TRANSACTION LIMITS:
  Minimum: ₹1 (no minimum limit)
  Maximum: No upper limit set by RBI
  Note: Individual banks may impose their own limits.
  Walk-in customers (non-account holders): ₹50,000 per transaction

NEFT INFORMATION REQUIRED:
  - Beneficiary name
  - Beneficiary account number
  - IFSC code of beneficiary bank
  - Bank name and branch

========================================
RTGS — REAL TIME GROSS SETTLEMENT
========================================
Full form: Real Time Gross Settlement
Operated by: Reserve Bank of India (RBI)
Settlement: Real-time (transaction by transaction, immediately)
Purpose: High-value transactions only
Availability: 24x7x365 (since December 14, 2020)

RTGS CHARGES — OUTWARD TRANSACTIONS:
  Transactions from ₹2,00,000 to ₹5,00,000: ₹24.50 + GST
  Transactions above ₹5,00,000             : ₹49.50 + GST

RTGS INWARD CHARGES: FREE — no charge to the beneficiary

RTGS TRANSACTION LIMITS:
  Minimum: ₹2,00,000 (₹2 lakh) — RTGS is only for high-value payments
  Maximum: No upper limit (as per RBI)
  Individual bank limits may apply.

KEY DIFFERENCE FROM NEFT:
  NEFT = batch processing every 30 minutes (good for regular payments)
  RTGS = instant settlement (good for large, urgent payments)
  Use RTGS when you need money to arrive immediately.
  Use NEFT when timing is not critical and amounts are smaller.

========================================
IMPS — IMMEDIATE PAYMENT SERVICE
========================================
Full form: Immediate Payment Service
Operated by: National Payments Corporation of India (NPCI)
Settlement: Real-time, 24x7x365
Purpose: Instant fund transfer at any amount

IMPS CHARGES (set by individual banks, not mandated by RBI):
  Up to ₹10,000       : ₹2.50 to ₹5.00 + GST (varies by bank)
  ₹10,001 to ₹1 lakh  : ₹5.00 to ₹15.00 + GST
  ₹1 lakh to ₹2 lakh  : ₹15.00 + GST (typically)
  Above ₹2 lakh        : ₹25.00 + GST (typically)
  Via mobile banking   : Often FREE (bank-specific)

NOTE: IMPS charges are bank-determined. Many banks offer free IMPS via mobile apps.
IMPS via USSD (*99#): ₹0.50 per transaction

IMPS TRANSACTION LIMITS:
  Minimum: ₹1
  Maximum: ₹5,00,000 (₹5 lakh) per transaction as per NPCI
  Some banks allow up to ₹2 lakh, others ₹5 lakh — check with your bank.

IMPS IDENTIFIERS:
  - MMID + Mobile number (mobile-to-mobile)
  - Account number + IFSC (account-to-account)
  - Aadhaar number (Aadhaar-linked)

IMPS vs NEFT vs RTGS COMPARISON:
  Feature         IMPS          NEFT          RTGS
  Settlement      Instant       30-min batch  Instant
  Min amount      ₹1            ₹1            ₹2,00,000
  Max amount      ₹5 lakh       No limit      No limit
  Availability    24x7          24x7          24x7
  Typical fee     ₹5-₹15+GST   ₹2.50-₹25+GST ₹24.50-₹49.50+GST
  Best for        Small-medium  Regular       High-value urgent

========================================
UPI — UNIFIED PAYMENTS INTERFACE
========================================
Full form: Unified Payments Interface
Operated by: NPCI
Settlement: Real-time

UPI CHARGES:
  Person-to-Person (P2P): FREE
  Person-to-Merchant (P2M): FREE (as per RBI and NPCI directives)
  Wallets to bank via UPI: FREE up to ₹2,000; ₹1 above ₹2,000 for PPIs

MDR (Merchant Discount Rate) on UPI:
  As of January 2020, RBI mandated ZERO MDR on UPI and RuPay transactions.
  Merchants cannot be charged for accepting UPI payments.

UPI TRANSACTION LIMITS:
  Standard: ₹1,00,000 per transaction
  UPI for capital markets: ₹2,00,000
  IPO applications: ₹5,00,000
  Medical/educational: ₹5,00,000

========================================
GST ON PAYMENT SERVICES
========================================
GST Rate on payment processing services: 18%
Applies to: NEFT fees, RTGS fees, IMPS fees, payment gateway fees
Does NOT apply to: The transaction amount itself (only on the service fee)

Example:
  NEFT transfer of ₹50,000 → Fee = ₹5 + 18% GST = ₹5.90 total charge

========================================
CROSS-BORDER / FOREX PAYMENTS
========================================
SWIFT charges (international wire transfers):
  Sending bank fee: ₹500 to ₹1,500 typically
  Correspondent bank charges: $15-$30 (deducted from amount)
  Currency conversion markup: 1% to 3.5% over mid-market rate

RBI guidelines on forex:
  Liberalized Remittance Scheme (LRS): Up to USD 2,50,000 per year per individual
  TCS (Tax Collected at Source): 20% on remittances above ₹7 lakh per year
  Form 15CA/15CB required for certain payments
"""


PCI_DSS_DATA = """
PCI-DSS — PAYMENT CARD INDUSTRY DATA SECURITY STANDARD
Source: PCI Security Standards Council (PCI SSC)
Current Version: PCI DSS v4.0 (released March 2022, mandatory since March 2024)

========================================
WHAT IS PCI-DSS?
========================================
PCI-DSS (Payment Card Industry Data Security Standard) is a set of security
standards designed to ensure that ALL companies that accept, process, store,
or transmit credit card information maintain a secure environment.

Created by: Visa, Mastercard, American Express, Discover, JCB (the major card brands)
Governed by: PCI Security Standards Council (PCI SSC)
Applies to: Any entity handling cardholder data — merchants, processors, gateways

========================================
PCI-DSS COMPLIANCE LEVELS
========================================
Level 1: Over 6 million card transactions per year
  - Annual on-site audit by Qualified Security Assessor (QSA)
  - Quarterly network scan by Approved Scanning Vendor (ASV)

Level 2: 1 to 6 million transactions per year
  - Annual Self-Assessment Questionnaire (SAQ)
  - Quarterly network scan

Level 3: 20,000 to 1 million e-commerce transactions per year
  - Annual SAQ
  - Quarterly network scan

Level 4: Fewer than 20,000 e-commerce OR up to 1 million other transactions
  - Annual SAQ
  - Quarterly network scan (recommended)

========================================
12 PCI-DSS REQUIREMENTS
========================================
1. Install and maintain a firewall to protect cardholder data
2. Do not use vendor-supplied defaults for passwords and security parameters
3. Protect stored cardholder data (encryption, masking)
4. Encrypt transmission of cardholder data across open, public networks
5. Use and regularly update anti-virus software
6. Develop and maintain secure systems and applications
7. Restrict access to cardholder data on a need-to-know basis
8. Assign a unique ID to each person with computer access
9. Restrict physical access to cardholder data
10. Track and monitor all access to network resources and cardholder data
11. Regularly test security systems and processes
12. Maintain an information security policy

========================================
PCI-DSS IN INDIA — CONTEXT
========================================
RBI mandates: All payment aggregators and gateways in India must be
PCI-DSS certified as per RBI Payment Aggregator guidelines (2020).

Razorpay: PCI-DSS Level 1 certified
PayU: PCI-DSS Level 1 certified
Paytm Payment Gateway: PCI-DSS certified
Stripe India: PCI-DSS Level 1 certified
PayPal India: PCI-DSS Level 1 certified

COST OF NON-COMPLIANCE:
  Card brand fines: $5,000 to $100,000 per month
  Increased transaction fees by acquiring banks
  Risk of losing ability to process card payments
  Data breach liability — average breach cost $4.45 million globally

========================================
TOKENIZATION AND PCI-DSS
========================================
Tokenization: Replacing card number with a non-sensitive equivalent (token)
Reduces PCI-DSS scope significantly — tokenized data is not cardholder data
All major Indian payment gateways use tokenization as per RBI mandate.
RBI Card-on-File tokenization mandate: Effective October 1, 2022.
"""


RAZORPAY_DATA = """
RAZORPAY — FEE STRUCTURE (INDIA)
Source: Razorpay official pricing page
Category: Payment Gateway — India's leading payment gateway

========================================
RAZORPAY STANDARD FEES
========================================
Domestic Cards (Debit/Credit):
  Standard: 2% per transaction
  International cards: 3% per transaction

UPI Transactions:
  UPI (P2M): 0% — FREE as per RBI mandate
  UPI Autopay: 0.10% for recurring mandates

Net Banking:
  Most banks: ₹15 flat per transaction (not percentage-based)

Wallets:
  Paytm, PhonePe, Amazon Pay: 2% per transaction

EMI Transactions:
  Bank EMI: 2% per transaction (EMI plans by bank)
  Cardless EMI: 3% per transaction

International Payments (via Razorpay):
  3% + currency conversion fee

Settlement Timeline:
  Standard: T+2 (2 business days after transaction)
  Early settlement: Available for fee (Razorpay RazorpayX)

GST: 18% applied on all Razorpay fees
Example: ₹10,000 transaction at 2% = ₹200 fee + ₹36 GST = ₹236 total deduction
You receive: ₹9,764

Razorpay minimum fee: No minimum
Razorpay monthly fee: ₹0 (no monthly subscription for standard plan)

========================================
RAZORPAY PAYMENT LINKS / PAGES
========================================
Standard transaction fee applies
No additional fee for payment links
No setup cost

========================================
RAZORPAY INTERNATIONAL ACCEPTANCE
========================================
Supports 100+ currencies
International cards: 3% + payment gateway fee
Cross-currency settlements: Available in USD, EUR, GBP, SGD, AED
"""


PAYPAL_INDIA_DATA = """
PAYPAL INDIA — FEE STRUCTURE
Source: PayPal India official fee page

========================================
PAYPAL FEES FOR INDIA (RECEIVING PAYMENTS)
========================================
Receiving from within India:
  Standard: 2.5% + ₹3 per transaction

Receiving international payments (export/freelance):
  Standard commercial rate: 4.4% + fixed fee
  Fixed fee by currency:
    USD: $0.30
    GBP: £0.20
    EUR: €0.35
    CAD: $0.30

TOTAL EFFECTIVE DEDUCTION on international payment:
  Currency conversion: ~3-4% spread over mid-market rate
  Transaction fee: 4.4% + fixed fee
  Total loss: ~7-8% on international payments is common

PayPal Currency Conversion:
  PayPal applies a currency conversion spread (markup) over the base exchange rate.
  This is typically 3% to 4% above the mid-market rate.
  This is separate from the transaction fee.

WhY PayPal Feels Expensive:
  1. Transaction fee (4.4%)
  2. Currency conversion markup (3-4%)
  3. Fixed per-transaction fee ($0.30 for USD)
  Combined, receiving USD 100 → you may get ₹7,800-8,100 instead of ~₹8,350

PayPal Withdrawal to Indian Bank Account:
  Withdrawal fee: FREE (no fee to withdraw to Indian bank)
  Time: 3-5 business days

PayPal GST in India:
  18% GST applies on PayPal's service fees (not on the transaction amount)

Alternatives for freelancers receiving international payments:
  Wise (TransferWise): ~0.5-1% fee, mid-market rate
  Payoneer: Free for same-currency transfers, 2% for USD withdrawal
  Razorpay: For domestic business payments
"""


# ──────────────────────────────────────────────
# SOURCES REGISTRY
# ──────────────────────────────────────────────
SOURCES = [
    # ── Static regulatory data (MOST IMPORTANT — always ingest these) ──
    {
        "type": "static",
        "content": RBI_PAYMENT_FEES_DATA,
        "save_name": "rbi_neft_rtgs_imps_upi_fees.txt",
        "description": "RBI NEFT, RTGS, IMPS, UPI fee structure (static)"
    },
    {
        "type": "static",
        "content": PCI_DSS_DATA,
        "save_name": "pci_dss_compliance_guide.txt",
        "description": "PCI-DSS compliance standard (static)"
    },
    {
        "type": "static",
        "content": RAZORPAY_DATA,
        "save_name": "razorpay_fees_india.txt",
        "description": "Razorpay fee structure (static)"
    },
    {
        "type": "static",
        "content": PAYPAL_INDIA_DATA,
        "save_name": "paypal_india_fees.txt",
        "description": "PayPal India fee structure (static)"
    },

    # ── Live scraped sources ──
    {
        "type": "webpage",
        "url": "https://www.paypal.com/in/webapps/mpp/paypal-fees",
        "save_name": "paypal_fees_live.txt",
        "description": "PayPal India fee page (live)"
    },
    {
        "type": "webpage",
        "url": "https://stripe.com/in/pricing",
        "save_name": "stripe_pricing_india.txt",
        "description": "Stripe India pricing (live)"
    },
    {
        "type": "webpage",
        "url": "https://razorpay.com/pricing/",
        "save_name": "razorpay_pricing_live.txt",
        "description": "Razorpay pricing page (live)"
    },
    {
        "type": "webpage",
        "url": "https://www.npci.org.in/what-we-do/upi/product-overview",
        "save_name": "npci_upi_overview.txt",
        "description": "NPCI UPI overview"
    },
]


def run_all_ingestions():
    print("\n🚀 Starting PayLens document ingestion (v2)...\n")
    results = []

    for source in SOURCES:
        try:
            if source["type"] == "static":
                text = ingest_static(source["content"], source["save_name"])
            elif source["type"] == "webpage":
                text = ingest_webpage(source["url"], source["save_name"])
            elif source["type"] == "pdf":
                text = ingest_pdf_from_url(source["url"], source["save_name"])
            else:
                text = ""

            results.append({
                "source": source["description"],
                "file": source["save_name"],
                "chars": len(text),
                "status": "✅ Success" if text else "⚠️  Empty"
            })
        except Exception as e:
            print(f"❌ Failed: {source['description']} — {e}")
            results.append({
                "source": source["description"],
                "file": source["save_name"],
                "chars": 0,
                "status": f"❌ {e}"
            })

    print("\n📊 Ingestion Summary:")
    print("-" * 70)
    for r in results:
        print(f"{r['status']}  {r['source']:45s}  {r['chars']:,} chars")
    print("-" * 70)
    print(f"Raw files saved to: {RAW_DIR}\n")
    return results


if __name__ == "__main__":
    run_all_ingestions()