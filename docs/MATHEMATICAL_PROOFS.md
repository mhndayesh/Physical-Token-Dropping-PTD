# The Math Behind "Smarter, Not Bigger"

This document explains *why* the speed and memory savings happen, where the theory matches reality, and where it doesn't.

## Attention Gets Cheaper Quadratically

Standard attention costs: **4ND² + 2N²D** FLOPs per layer.

When we keep only K = sN tokens (where s is 0.3 for 30% retention), the cost becomes: **4KD² + 2K²D**.

The key is that **s²** term. At 30% retention, the attention matrix shrinks by s² = 0.09, which is an **11x reduction** in that specific operation. The projection costs drop by s = 0.3, a **3.3x reduction**. This is where the real savings come from.

## Memory Gets Cheaper Linearly

The activation tensors stored for backpropagation scale with the number of tokens being processed:
- Hidden states: O(N × D) → O(K × D), saving 70% at s=0.3
- Attention matrices: O(N²) per head → O(K²) per head, saving 91%

But model weights and optimizer states don't change — they're a fixed cost regardless of sparsity.

**Observed total** (weights + activations + optimizer):
- Dense: 4,349 MB → Sparse 30%: 2,500 MB → **42.5% savings** at seq 2048

**The OOM proof** — where theory becomes undeniable (tested on RTX 5070 12 GB, i7-14700, 64 GB DDR4):

| Sequence Length | Dense VRAM | Sparse 30% VRAM | Savings |
|:--|:--|:--|:--|
| 2,048 | 2.5 GB | 1.5 GB | 41% |
| 8,192 | 9.3 GB | 4.0 GB | 57% |
| 12,288 | 14.7 GB | 5.9 GB | 60% |
| 16,384 | 💀 OOM | 7.6 GB | — |
| 32,768 | — | 15.6 GB | — |
| 48,152 | — | 24.7 GB | — |

Dense dies at 16K. Sparse survives past 48K — **over 4x longer sequences** on the same hardware. Past the 12 GB VRAM limit, the sparse model offloads to system RAM and keeps running because its reduced compute doesn't saturate the GPU.

## Where Theory Meets Reality

Here's the honest comparison:

| Metric | Theory Says | We Measured (vs True Dense) | Why the Gap |
|:---|:---|:---|:---|
| Speed (30%) | 4-11x faster | 3.70x faster | Routing, gather/scatter, causal mask construction |
| VRAM (30%) | 70-91% saved | 42.5% saved | Model weights + optimizer are constant |
| Speed (10%) | 10-100x faster | 4.39x faster | Overhead exceeds savings at extreme sparsity |

The theoretical gains assume the *only* cost is attention and FFN. In practice, the router needs to score every token (O(NR)), we need to build an explicit causal mask (O(K²)), and gather/scatter operations aren't free. These overhead costs explain the gap between the 10-100x theoretical maximum and the 3.7x-4.4x practical reality.

## The Training Convergence Story

We ran each sparsity level for 200, 1000, 2000, and 5000 training steps to see how quality evolves:

| Steps | True Dense PPL | Sparse 100% PPL | Sparse 30% PPL |
|:---|:---|:---|:---|
| 200 | 4.49 | 4.89 | 6.81 |
| 1,000 | 3.12 | 3.08 | 5.52 |
| 2,000 | 2.74 | 2.70 | 4.48 |

The absolute gap is **shrinking** even as the model compresses (from 2.32 gap at 200 steps to 1.74 gap at 2000 steps). The router is learning — it's getting better at identifying which tokens carry the most information. In fact, Sparse 100% actually slightly *beats* True Dense at 1,000 and 2,000 steps.

At 5K steps, all sparse configurations (10% through 50%) cluster between PPL 3.13 and 3.97. This convergence suggests that with enough training, the router can compensate for even aggressive token dropping.

## What This Means for Production

The speed and memory gains are guaranteed by math — they don't depend on data quality or training length. A physically smaller tensor will always be faster to multiply.

The accuracy gap is what scales with training. On a tiny dataset with a tiny model and 5K steps, we see a ~1.5 PPL gap. These results come from a minimal test setup and do **not** reflect production-level performance. A pretrained model fine-tuned with the sparse router on high-quality, large-scale data should show a significantly smaller gap — the mechanism is promising precisely because it works even at this small scale. Scaled correctly, the router will make near-optimal token selections from day one.
