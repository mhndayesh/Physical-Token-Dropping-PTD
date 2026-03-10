# Smarter, Not Bigger
### *Physical Token Dropping (PTD) Proof of Concept*

Welcome to the **Smarter, Not Bigger** Proof of Concept (POC) repository for Physical Token Dropping (PTD).

This project challenges the assumption that AI always needs to be bigger. Instead of scaling up parameters, we built a modified Transformer that **physically drops most tokens** during computation. Think of it like speed-reading: instead of reading every word on a page, you only look at the important ones. The model learns *which* tokens carry the meaning — and only computes on those.

## � Repository Structure

```
ptd_poc/
├── README.md                (This file)
├── docs/                    (Core whitepapers and walkthroughs)
├── src/                     (Model architecture code)
├── benchmarks/              (Speed and VRAM validation scripts)
└── eval/                    (Accuracy and perplexity validation scripts)
```

## �📖 The Core Documentation (`/docs`)

This repository contains five main documents that explain, prove, and validate the concept. **Start with the Master POC.**

1. 🏆 **[MASTER_POC.md](docs/MASTER_POC.md)** — *The main results document.* Start here. It contains the irrefutable proof, speedups vs. traditional PyTorch, the OOM boundaries, and the honest accuracy tradeoffs.
2. 📐 **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — *How 'Smarter, Not Bigger' Works.* An explanation of the "Physical Token Dropping" architecture step-by-step with an ASCII pipeline diagram.
3. 🧮 **[MATHEMATICAL_PROOFS.md](docs/MATHEMATICAL_PROOFS.md)** — *The Math Behind 'Smarter, Not Bigger'.* The formal proofs showing *why* memory gets cheaper linearly, attention gets cheaper quadratically, and where theory diverges from reality.
4. 🚶 **[WALKTHROUGH.md](docs/WALKTHROUGH.md)** — *Smarter, Not Bigger — How We Got Here.* The project story: the idea, the technical review that found bugs, the fixes, and the final scientific validations.
5. 🧪 **[TRAINING_RECIPE.md](docs/TRAINING_RECIPE.md)** — *How to Scale PTD.* A dedicated 3-phase training recipe (Teacher Distillation + Curriculum Sparsity + Mixed Data) designed specifically to handle the architectural shock of physical token dropping.
6. 📈 **[SCALABILITY.md](docs/SCALABILITY.md)** — *Honest Estimations at Scale.* A realistic look at how speed, VRAM, and accuracy will trade off when scaling PTD up to 7B+ parameters on long contexts.

## 💻 Source Code (`/src`)

- `transformer_0_5b.py` — The core bug-fixed implementation of PTD (with stochastic gating, segment routing, and physical gather/scatter logic).
- `sparse_transformer.py` — The original implementation (kept for historical context).

## 🚀 Scientific Validation (`/benchmarks` & `/eval`)

All claims in the documentation are backed by these empirical validation scripts:

### Benchmarks (Speed & VRAM)
- `scientific_validation.py` — The initial benchmark comparing against a strict `nn.TransformerEncoder` baseline and the Dense vs. Sparse OOM test.
- `oom_boundary_test.py` — The isolated subprocess script proving Sparse 30% can handle 4x longer sequences than dense models before crashing.
- `true_baseline_full.py` — The full benchmark suite comparing 10%-50% sparsity levels against the true PyTorch dense baseline.
- `benchmark_sparse.py` — General speed and VRAM overhead benchmarking.

### Eval (Accuracy & Perplexity)
- `true_baseline_accuracy.py`, `_1k.py`, `_2k.py` — Training scripts used to measure perplexity convergence of Sparse vs. Dense models over 200, 1000, and 2000 steps on TinyStories.
- `verify_accuracy.py` & `verify_tinystories.py` — Base sanity checks and initial perplexity measurements.

---

## ⚠️ Notes

> **A Final Note:** This is the work of an individual. While mathematically and scientifically rigorous tests have been conducted, there may still be mistakes, conflicts, or unoptimized edge cases that I have not caught. I appreciate and welcome all critiques, corrections, and contributions to help scale this concept further.
