---
title: PayLens
emoji: 💳
colorFrom: purple
colorTo: indigo
sdk: docker
app_file: src/app.py
pinned: false
license: mit
short_description: AI-powered payment fee explainer with RAG + live search
---

<div align="center">

# 💳 PayLens

### AI-powered payment fee explainer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces/Rxrohans/paylens)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Llama%203.1-red?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

*"Why did PayPal deduct 7% from my payment?" — PayLens tells you exactly why, and what to do about it.*

**[→ Try the live demo](https://huggingface.co/spaces/Rxrohans/paylens)**

</div>

---

## 🎯 The Problem

 You see USD 500 in your PayPal. Quick mental math — that's around ₹41,500 at today's rate. You're happy.
Then the money actually arrives. ₹38,200. You got ₹3,300 less and you have no idea why. Transaction fee? Currency conversion markup? Some hidden platform charge? The app doesn't explain it clearly. The terms are a bit complex. You're just... sad.
PayLens tells you exactly what happened — how much each platform deducted, why, and what you can do differently next time.

PayLens solves this. Just Ask, get a clear answer

**Example questions it answers:**
- *"Why did PayPal charge me 7-8% when I received money from Outlier?"*
- *"What is the difference between NEFT and IMPS?"*
- *"Do I need to pay GST on my freelance income from abroad?"*
- *"How does currency conversion spread work?"*
- *"Is UPI free for merchants in India?"*

---

## 🏗️ System Architecture

```
User Question
      ↓
┌─────────────────────────────┐
│  INPUT GUARDRAILS           │
│  • PII detection            │
│  • Prompt injection check   │
│  • Topic relevance filter   │
│  • Length validation        │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  FAISS SEMANTIC RETRIEVAL   │
│  • all-MiniLM-L6-v2 embeds  │
│  • Top-7 chunk retrieval    │
│  • Cosine similarity search │
└─────────────────────────────┘
      ↓
  Confidence score >= 0.60?
    │                    │
   YES                   NO
    │                    ↓
    │         ┌─────────────────────┐
    │         │  DUCKDUCKGO SEARCH  │
    │         │  Live web results   │
    │         └─────────────────────┘
    │                    │
    └────────┬───────────┘
             ↓
┌─────────────────────────────┐
│  LLM SYNTHESIS              │
│  • Llama 3.1 via Groq       │
│  • Two prompt variants      │
│    (RAG-only / Hybrid)      │
│  • temperature=0            │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  OUTPUT GUARDRAILS          │
│  • Hallucination detection  │
│  • PII redaction from logs  │
│  • Confidence classification│
└─────────────────────────────┘
      ↓
  Structured Answer
  + Source badges
  + Confidence badge
  + Official links
  + JSONL audit log
```

---

## ✨ Features

### Core
| Feature | Details |
|---|---|
| **Hybrid RAG** | FAISS semantic search with DuckDuckGo live web fallback — triggers when KB confidence score < 0.60 |
| **Rich Knowledge Base** | 150+ chunks across 6 domain files covering payments, forex, taxes, regulations |
| **Structured Output** | Every answer includes confidence level, source attribution, latency, and official links |
| **Audit Logging** | Every query logged to JSONL with question, answer, sources, latency, confidence, PII flags |

### Safety & Compliance
| Feature | Details |
|---|---|
| **PII Detection** | Regex patterns for Aadhaar, PAN, card numbers, UPI IDs, email, phone — sanitized before logging |
| **Prompt Injection Blocking** | Pattern matching against known injection attempts |
| **Topic Guardrails** | Off-topic questions (cricket scores, geography, etc.) blocked with explanation |
| **Hallucination Signals** | Output scanner detects and flags potential hallucination patterns |

### Evaluation
| Feature | Details |
|---|---|
| **Golden Dataset** | 20 manually verified Q&A pairs covering all KB domains |
| **Custom Eval Pipeline** | Keyword overlap scoring — faithfulness, relevancy, context coverage |
| **Score History** | Every eval run saved to JSON — track quality changes over time |
| **Dashboard Tab** | Built-in Streamlit eval tab shows scores, trends, per-sample breakdown |

---

## 🧠 Knowledge Base

**6 curated domain files totalling 150+ indexed chunks:**

| File | What it covers |
|---|---|
| `payment_fees_manual.txt` | PayPal India, Stripe India, Razorpay fees — transaction fees, dispute fees, conversion spreads |
| `fintech_fundamentals.txt` | How payments work end-to-end, UPI/NEFT/IMPS/RTGS/SWIFT, card networks, MDR, PCI-DSS, chargebacks |
| `forex_economics.txt` | Exchange rates, bid/ask spread, FEMA rules, LRS scheme, why INR weakens, PPP |
| `india_tax_compliance.txt` | GST on payment fees, TDS, TCS on foreign remittances, ITR filing for freelancers, FIRC |
| `paypal_fees_india.txt` | Scraped PayPal India pricing page |
| `stripe_pricing_india.txt` | Scraped Stripe India pricing page |

All sources documented in [`data/SOURCES.md`](data/SOURCES.md).

**Live web search** (DuckDuckGo) handles questions outside the KB — no API key required.

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Llama 3.1 8B Instant via Groq | Free tier (14,400 req/day), fastest inference |
| Embeddings | `all-MiniLM-L6-v2` | Local, 384-dim, good quality/speed tradeoff |
| Vector Store | FAISS IndexFlatIP | Cosine similarity, in-memory, no server needed |
| RAG Framework | LangChain LCEL | Modular chain composition |
| Web Search | DuckDuckGo (`ddgs`) | Free, no API key, no restrictions |
| Guardrails | Custom (regex + pattern matching) | Full control, no external dependency |
| Evaluation | Custom keyword overlap | Zero LLM calls, rate-limit safe |
| UI | Streamlit | Rapid deployment, HuggingFace native |
| Deployment | HuggingFace Spaces (Docker) | Free, public URL, auto-deploy on push |
| Version Control | Git + GitHub | Branch-based workflow |

---

## 📊 Evaluation Results

*First eval run — 20 questions, March 2026*

| Metric | Score | What it measures |
|---|---|---|
| Context Coverage | 62.81% | Retriever finding right documents |
| Faithfulness | 14.63% | Answer grounded in retrieved context* |
| Answer Relevancy | 12.70% | Answer addresses the question* |
| Avg Latency | ~600ms (RAG) / ~2500ms (hybrid) | Response time |
| Web Search Rate | 45% of queries | Hybrid fallback trigger rate |
| Confidence Dist | 90% High, 10% Medium | LLM self-assessed confidence |

*Note: Faithfulness and Relevancy use keyword overlap scoring which underreports paraphrasing LLMs. A human evaluation of the same answers would score significantly higher. Context Coverage is the most reliable metric for this system.*

Run your own evaluation:
```bash
python eval/ragas_eval.py
```

---

## 📁 Project Structure

```
paylens/
│
├── src/                          ← Python source
│   ├── app.py                    ← Streamlit UI (tabs: Ask / Eval Dashboard)
│   ├── chain.py                  ← Hybrid RAG + web search logic
│   ├── retriever.py              ← FAISS semantic search
│   ├── embedder.py               ← Builds vector store from chunks
│   ├── chunker.py                ← Splits raw docs into chunks with metadata
│   ├── ingestor.py               ← Fetches/loads raw documents
│   └── guardrails.py             ← PII detection + injection prevention
│
├── eval/                         ← Evaluation pipeline
│   ├── ragas_eval.py             ← Runs eval, saves scores
│   ├── metrics_dashboard.py      ← Dashboard UI (embedded as app tab)
│   ├── golden_dataset.json       ← 20 verified Q&A pairs
│   └── scores_history.json       ← Score history (auto-generated)
│
├── data/
│   ├── raw/                      ← Knowledge base source files 
│   ├── processed/                ← Auto-generated, gitignored
│   │   ├── chunks.json           ← Chunked documents with metadata
│   │   ├── faiss_index.bin       ← FAISS vector index
│   │   └── chunk_metadata.pkl    ← Chunk source/score metadata
│   └── SOURCES.md                ← Data lineage registry
│
├── logs/                         ← Auto-generated, gitignored
│   ├── answers.jsonl             ← Full audit log of every query
│   └── chain.log                 ← System logs
│
├── .gitignore                    ← Excludes venv, .env, index, logs
├── .env                          ← API keys (gitignored)
├── env.example                   ← Safe template for .env
├── requirements.txt              ← Python dependencies
├── packages.txt                  ← HuggingFace Linux deps (libgomp1)
└── README.md                     ← This file (also HuggingFace Space config)
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/Rxrohans/paylens
cd paylens

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key
# Get one at: console.groq.com (free, no credit card)
cp env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# 5. Build knowledge base index
python src/chunker.py
python src/embedder.py

# 6. Launch
streamlit run src/app.py
# Opens at http://localhost:8501
```

---

## 🔄 Adding New Knowledge

No code changes needed:

```bash
# 1. Add your .txt file
nano data/raw/new_topic.txt

# 2. Document it
nano data/SOURCES.md

# 3. Rebuild index (2 commands)
python src/chunker.py
python src/embedder.py

# New knowledge is live immediately
```

---

## 🔧 Swapping the LLM

One line change in `src/chain.py`:

```python
# Current (fastest, 14,400 req/day free)
model="llama-3.1-8b-instant"

# Smarter answers (1,000 req/day free)
model="llama-3.3-70b-versatile"

# Latest Llama 4 (1,000 req/day free)
model="meta-llama/llama-4-scout-17b-16e-instruct"
```

---

## 🗺️ Roadmap

- [ ] Section-header based chunking (better than fixed-size for structured docs)
- [ ] Embedding similarity eval (more accurate than keyword overlap)
- [ ] Paytm, CCAvenue, PayU fee structures
- [ ] Southeast Asia + Middle East payment systems (Pine Labs markets)
- [ ] Conversation memory for multi-turn questions
- [ ] Query latency optimisation (cache frequent questions)
- [ ] Cost tracking per query


---

## 👤 Author

**Rohan Singh**

[![GitHub](https://img.shields.io/badge/GitHub-Rxrohans-black?style=flat-square)](https://github.com/Rxrohans)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Rxrohans-yellow?style=flat-square)](https://huggingface.co/Rxrohans)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.