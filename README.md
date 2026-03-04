---
title: PayLens
emoji: 💳
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.37.1
app_file: src/app.py
pinned: false
license: mit
short_description: AI-powered payment fee explainer for India
---

# PayLens 💳

**AI-powered payment fee explainer**

Confused by a PayPal deduction? Getting mysterious Stripe charges?
Ask PayLens in plain English and get a clear, sourced answer.

## What it covers
- PayPal India fees (transaction + currency conversion)
- Stripe India pricing
- Razorpay fees
- UPI / NEFT / IMPS / RTGS
- Currency conversion mechanics
- FEMA / LRS rules for freelancers
- GST and TDS on payment fees
- Chargebacks, disputes, PCI-DSS

## Tech Stack
- **RAG Pipeline:** FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Llama 3.1 via Groq API (free tier)
- **Hybrid Search:** RAG + DuckDuckGo live search
- **Guardrails:** PII detection, prompt injection prevention
- **Evaluation:** RAGAS framework
- **Framework:** LangChain + Streamlit

## Run Locally
```bash
git clone https://github.com/Rxrohans/paylens
cd paylens
pip install -r requirements.txt
cp env.example .env   # add your GROQ_API_KEY
python src/chunker.py
python src/embedder.py
streamlit run src/app.py
```