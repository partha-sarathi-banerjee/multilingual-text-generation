# 🌐 Multilingual Text Generation with mT5

> **Cross-lingual generative modeling across English · Hindi · Bengali · French · Spanish**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multilingual-text-generation/blob/main/multilingual_text_generation.ipynb)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/google/mt5-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project demonstrates **end-to-end multilingual text generation** using Google's `mT5-small` — a massively multilingual T5 model pretrained on 101 languages via a shared sentencepiece vocabulary of 250,000 tokens.

The notebook covers the **full generative modeling pipeline**: data preparation → tokenization analysis → beam-search decoding → quantitative evaluation (BLEU) → attention visualization — across three distinct writing scripts (Latin, Devanagari, Bengali).

This work was built as part of a broader research portfolio in applied ML and NLP, with particular focus on the engineering challenges of building language systems that serve international users equitably.

---

## 🔬 What's Inside

| Section | Description |
|---|---|
| **1. Environment Setup** | Install `transformers`, `sentencepiece`, `sacrebleu`, `datasets` |
| **2. Model Loading** | `google/mt5-small` (300M params) on GPU/CPU |
| **3. Multilingual Corpus** | Parallel news-style paragraphs in 5 languages / 3 scripts |
| **4. Text Generation** | Beam search decoding (`num_beams=4`) per language |
| **5. BLEU Evaluation** | `sacrebleu` sentence-level scoring with Unicode handling |
| **6. Token Fertility Analysis** | Subword tokens / words ratio — quantifies script-level tokenization cost |
| **7. Attention Heatmaps** | Encoder self-attention visualization (final layer, head 0) |
| **8. Results Summary** | Key findings and production implications |

---

## 🚀 Quick Start

### Option A — Google Colab (recommended, free GPU)
Click the **Open in Colab** badge above. Runtime → Change runtime type → **GPU**. Then Run All.

### Option B — Local
```bash
git clone https://github.com/YOUR_USERNAME/multilingual-text-generation.git
cd multilingual-text-generation
pip install -r requirements.txt
jupyter notebook multilingual_text_generation.ipynb
```

**requirements.txt**
```
transformers>=4.35.0
sentencepiece>=0.1.99
sacrebleu>=2.3.1
datasets>=2.14.0
torch>=2.0.0
seaborn>=0.12.0
matplotlib>=3.7.0
numpy>=1.24.0
nbformat>=5.9.0
```

---

## 📊 Sample Results

### Token Fertility by Language
Token fertility = subword tokens ÷ whitespace-separated words. Higher values mean greater tokenization overhead — an equity concern for non-Latin script communities.

| Language | Script | Fertility | Notes |
|---|---|---|---|
| English | Latin | ~1.4× | Baseline |
| French | Latin | ~1.5× | Compound words inflate slightly |
| Spanish | Latin | ~1.5× | Similar to French |
| Hindi | Devanagari | ~2.8× | Morphological complexity |
| Bengali | Bengali | ~3.1× | Highest overhead in this set |

### Key Finding
> Non-Latin scripts (Devanagari, Bengali) require **2–4× more tokens** to represent the same semantic content. This means users of these languages receive fewer effective context tokens for the same model capacity — a critical fairness issue that production multilingual systems must address through fertility-aware batching and vocabulary expansion.

---

## 🧠 Technical Design Decisions

### Why mT5?
- **Shared 250k sentencepiece vocabulary** across 101 languages enables zero-shot cross-lingual transfer
- **Encoder-decoder architecture** is optimal for conditional generation (summarization, translation, QA)
- **Small variant (300M params)** is fully runnable on free Colab GPU — reproducible without enterprise compute
- Direct architectural ancestor of models used in production multilingual AI systems

### Why Abstractive Summarization?
Summarization is an ideal probe task for generative models: it requires the model to **understand** the input (compression) and **generate** novel text (not just copy). Comparing quality across languages on identical semantic content isolates the model's multilingual capability from content difficulty.

### Evaluation Strategy
- **BLEU** — n-gram precision, Unicode-aware via `sacrebleu`; imperfect for morphologically rich languages but useful for cross-language comparison
- **Token Fertility** — custom metric surfacing tokenization equity gaps
- **Attention Visualization** — qualitative inspection of what the model attends to across scripts

---

## 🗺️ Project Structure

```
multilingual-text-generation/
│
├── multilingual_text_generation.ipynb   # Main Colab notebook
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── outputs/
│   ├── multilingual_analysis.png        # BLEU + fertility bar charts
│   ├── attention_en.png                 # English attention heatmap
│   └── attention_hi.png                 # Hindi attention heatmap
└── LICENSE
```

---

## 🔭 Future Work

- [ ] Fine-tune on `mC4` multilingual corpus with language-balanced sampling
- [ ] Add **ChrF++** evaluation (better for morphologically rich languages)
- [ ] Implement **RLHF reward model** for cultural appropriateness scoring
- [ ] Extend to low-resource languages: Swahili, Tamil, Urdu
- [ ] Fertility-aware dynamic batching for equitable inference throughput
- [ ] Compare against `BLOOM`, `mBART-50`, and `NLLB-200`

---

## 👤 Author

**Partha Sarathi Banerjee**  
Machine Learning Engineer · Generative AI & NLP  
M.Tech Information Technology (CGPA 9.53/10) — IIEST Shibpur  
IBM AI Engineering Certified · IBM Data Science Certified

📧 parthacrj111@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/partha-sarathi-banerjee-607b33a2)

**Research:** Q-Learning for EV Route Optimization (CINS 2025) · Level-2 ADAS Pipelines (Daimler Truck Innovation Centre India)

---

## 📄 License

MIT License — free to use, adapt, and build upon with attribution.

---

*If this project helped you, please ⭐ the repo.*
