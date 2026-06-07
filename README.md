# 🌐 Multilingual Text Generation with mT5

> Cross-lingual generative modelling across English · Hindi · Bengali · French · Spanish — with a focus on tokenization equity for non-Latin scripts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parthacrj111/multilingual-text-generation/blob/main/multilingual_text_generation.ipynb)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/google/mt5-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

This project builds an end-to-end multilingual text generation pipeline using Google's `mT5-small` — a 300M parameter encoder-decoder model pretrained on 101 languages. The full pipeline covers data preparation, tokenization analysis, beam-search decoding, BLEU evaluation, and encoder attention visualization across three distinct writing systems: Latin, Devanagari, and Bengali.

The core research question is simple: **do multilingual models serve all languages equally?** The short answer is no, and this project quantifies exactly why.

**The main finding:** Hindi and Bengali users consume 2–4× more tokens to represent the same semantic content compared to English or French users. With a fixed context window, this translates directly to fewer effective input tokens — a measurable equity gap baked into the model's shared vocabulary design. This matters for anyone building production multilingual systems and isn't measured nearly enough.

Runs entirely on free Colab GPU. No paid compute required.

---

## What's inside

| Section | What it does |
|---|---|
| Environment setup | Installs `transformers`, `sentencepiece`, `sacrebleu`, `datasets` |
| Model loading | `google/mt5-small` (300M params) on GPU or CPU |
| Multilingual corpus | Parallel news-style paragraphs across 5 languages / 3 scripts |
| Text generation | Beam search decoding (`num_beams=4`) per language |
| BLEU evaluation | `sacrebleu` sentence-level scoring with Unicode handling |
| Token fertility analysis | Subword tokens ÷ words — quantifies tokenization overhead per script |
| Attention heatmaps | Encoder self-attention visualization (final layer, head 0) |
| Results summary | Key findings and what they mean for production systems |

---

## Results

### Token fertility by language

Token fertility = subword tokens ÷ whitespace-separated words. Higher = more tokenization overhead = less effective context for the same model capacity.

| Language | Script | Fertility | Notes |
|---|---|---|---|
| English | Latin | ~1.4× | Baseline |
| French | Latin | ~1.5× | Slight inflation from compound words |
| Spanish | Latin | ~1.5× | Similar to French |
| Hindi | Devanagari | ~2.8× | Morphological complexity |
| Bengali | Bengali | ~3.1× | Highest overhead in this set |

**Why this matters in production:** A 2,048-token context window effectively becomes ~660 tokens for Bengali input. Fertility-aware batching and vocabulary expansion are not optional for equitable multilingual deployment — they're load-bearing.

---

## Quick start

**Option A — Colab (recommended)**

Click the badge above. Runtime → Change runtime type → GPU → Run All.

**Option B — Local**

```bash
git clone https://github.com/parthacrj111/multilingual-text-generation.git
cd multilingual-text-generation
pip install -r requirements.txt
jupyter notebook multilingual_text_generation.ipynb
```

---

## Repository structure

```
multilingual-text-generation/
├── multilingual_text_generation.ipynb   # Main notebook
├── Copy_of_multilingual_text_generation.ipynb  # Original Colab draft (archived)
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Design decisions worth knowing

**Why mT5?** The shared 250k SentencePiece vocabulary across 101 languages enables zero-shot cross-lingual transfer without fine-tuning. The small variant (300M params) is fully reproducible on free compute — no enterprise GPU required — which matters for accessibility.

**Why summarization as the probe task?** It forces the model to both understand (compress) and generate (not just copy), which isolates multilingual capability from content difficulty. Good test bed for comparing cross-lingual performance on identical semantic content.

**Why token fertility as a metric?** BLEU measures output quality. Fertility measures input fairness. Most multilingual benchmarks report the former and ignore the latter. This project argues both matter.

---

## What's next

- [ ] Fine-tune on `mC4` with language-balanced sampling
- [ ] Add ChrF++ evaluation (better for morphologically rich languages)
- [ ] Implement RLHF reward model for cultural appropriateness scoring
- [ ] Extend to low-resource languages: Swahili, Tamil, Urdu
- [ ] Fertility-aware dynamic batching for equitable inference throughput
- [ ] Compare against BLOOM, mBART-50, and NLLB-200

---

## About the author

**Partha Sarathi Banerjee** — ML Engineer, Generative AI & NLP  
M.Tech IT, IIEST Shibpur (9.53/10 CGPA) · IBM AI Engineering Certified · Published @ CINS 2025

[LinkedIn](https://linkedin.com/in/partha-sarathi-banerjee-607b33a2) · [parthacrj111@gmail.com](mailto:parthacrj111@gmail.com)

---

*If this was useful, a ⭐ helps other people find it.*


*If this project helped you, please ⭐ the repo.*
