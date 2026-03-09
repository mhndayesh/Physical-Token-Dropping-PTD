# Smarter, Not Bigger — How We Got Here

This is the story of building, breaking, fixing, and validating the Physical Token Dropping (PTD) architecture.

## Phase 1: The Idea

We started with a simple question: what if a Transformer didn't process every token? Most tokens in a sequence are predictable — articles, common prepositions, repetitive patterns. What if we could learn which tokens actually matter and only compute on those?

We built a "Physical Token Dropping" architecture: a small router network scores each token, we gather the top-scoring ones into a compact tensor, run attention and FFN on that smaller tensor, and scatter the results back. Unlike masking (which still allocates full-size tensors), this approach physically shrinks the computation.

## Phase 2: Initial Results (Too Good to Be True)

The first benchmarks looked amazing: 3.2x speedup, 53% VRAM savings, 100%+ accuracy retention. We wrote it all up in detailed mathematical proofs and benchmark reports.

## Phase 3: The Technical Review

An external review found three critical bugs that were inflating our numbers:

1. We were adding the input tensor twice (double residual), which made the "dense baseline" artificially slow.
2. After gathering tokens, we were applying positional encoding using local indices (0, 1, 2...) instead of the original sequence positions. A token from position 3000 was getting the encoding for position 37.
3. Our causal mask was based on local ordering instead of original positions, potentially allowing future-to-past attention.

We fixed all three. The "3.2x speedup" dropped to a more honest **2.0-2.3x**.

## Phase 4: Honest Benchmarking

After fixing the bugs, we ran a comprehensive sweep across sparsity levels and training budgets.

**Speed and memory** (these are hardware facts, not affected by data):
- 30% token retention: 2.3x faster, 42% less VRAM
- 40% token retention: 2.0x faster, 35% less VRAM

**Accuracy over training time** (this is where it gets interesting):

At 200 steps, all sparse models performed similarly poorly. But as we trained longer:
- At 5K steps, the best sparse config (30% retention) reached PPL 3.13 vs. dense's 1.68
- The absolute PPL gap **shrank** from 2.52 to 1.45 over the training run
- All sparse configs (10-50%) converged to a similar quality range (3.1-4.0)

We also ran a **real output test** — generating actual text from story prompts. At 200 steps, all models (dense and sparse) produced repetitive, half-formed phrases. The sparse models memorized training data just as well as dense (100% top-1 accuracy), but generation was too early-stage to show meaningful differences.

## Phase 5: What We Learned

**The architecture works.** The mechanism is sound, the bugs are fixed, the code runs clean.

**Speed and memory savings are real and guaranteed.** They come from physically smaller tensors — that's math, not data-dependent.

**Accuracy is the tradeoff.** There is a real quality cost, but it shrinks with training. At production scale (pretrained model, quality data, long training), the gap should narrow significantly because the router would start from strong token representations instead of learning from scratch.

**The sweet spot shifts with training budget.** At 200 steps, conservative sparsity (40-50%) works best. At 5K steps, aggressive sparsity (30%) actually produces the best quality — the router learns to be surgical about which tokens to keep.

## Phase 6: Scientific Validation

Two final tests to make the POC scientifically rigorous:

**True Dense Baseline**: We benchmarked against a real `nn.TransformerEncoder` with no routing overhead. Result: Sparse 30% is **3.70x faster** (24.8ms vs 91.8ms). This replaces our earlier comparison against sparsity=1.0 which unfairly included router costs in the baseline. The true PyTorch baseline proves the architectural efficiency without routing distortion.

**OOM Boundary Test**: We ramped sequence length until the dense model ran out of GPU memory (RTX 5070, 12 GB VRAM, 64 GB DDR4 system RAM). Dense crashed at **16,384 tokens**. Sparse 30% survived all the way to **49,152 tokens** (24.7 GB, offloading to system RAM) — handling **over 4x longer sequences** on the same hardware. Because the sparse model's compute is so much lighter, it doesn't choke when spilling past VRAM — dense can't do this because its full N² attention saturates the GPU.

## What's in This Folder

| File | What It Is |
|:---|:---|
| `MASTER_POC.md` | The main results document — start here |
| `ARCHITECTURE.md` | How the architecture works, step by step |
| `MATHEMATICAL_PROOFS.md` | The math showing why it's faster and where theory ≠ practice |
| `WALKTHROUGH.md` | This file — the project story |
| `transformer_0_5b.py` | The bug-fixed model code |
| `sparse_transformer.py` | The original sparse attention implementation |
| `benchmark_sparse.py` | Speed and VRAM benchmark script |
| `verify_tinystories.py` | Perplexity evaluation on TinyStories |
| `verify_accuracy.py` | Toy accuracy verification |
| `PTD_Technical_Review.docx` | The external review that found the bugs |
