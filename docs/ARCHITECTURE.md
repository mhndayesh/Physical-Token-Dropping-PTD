# How "Smarter, Not Bigger" Works

## The Core Idea

Normal Transformers process every token in the sequence — even the ones that don't matter much. Physical Token Dropping (PTD) asks: "What if we only computed on the tokens that matter?"

Instead of masking unimportant tokens (which still allocates full-size tensors and does all the multiply-adds), we **physically remove** them from the computation. Gather the important ones into a smaller tensor, do all the attention and FFN math on that smaller tensor, then scatter the results back. The GPU literally does less work.

## Step by Step

Here's what happens inside each block:

```
Full sequence arrives: (batch, 2048 tokens, 1024 dimensions)
         │
    ┌────┴────┐
    │  ROUTER │  A small network scores each token's importance.
    │         │  Picks the Top-K most important ones.
    └────┬────┘
         │
    ┌────┴────┐
    │ GATHER  │  Physically extract K tokens → smaller tensor.
    │         │  Keep their original position IDs for RoPE.
    └────┬────┘
         │
    ┌────┴────────────────┐
    │ ATTENTION + FFN     │  Normal Transformer layers, but on
    │ (on K tokens only)  │  a much smaller tensor. This is where
    │                     │  the speed comes from.
    └────┬────────────────┘
         │
    ┌────┴────┐
    │ SCATTER │  Put the updated tokens back into their
    │         │  original positions in the full sequence.
    └────┬────┘
         │
    Output: (batch, 2048 tokens, 1024 dimensions)
```

## The Router

The router is the brain of the operation. It uses a set of learnable "queries" that compete for tokens — each query looks for a different type of important token. The router scores every token, adds a tiny bit of random noise during training (so that no token gets permanently ignored), and picks the Top-K.

We route on blocks of 16 tokens at a time rather than individual tokens. This aligns better with how GPU memory works (cache lines, warps) and reduces routing overhead.

## Three Bugs We Fixed

The external Technical Review caught three critical issues:

**1. Double residual.** Each layer inside the block already adds a residual connection (`x + attention(x)`). But the outer block was adding the original input *again*. This was silently corrupting the model. Fix: clone the input, replace the selected positions, return without adding.

**2. Broken positions.** After gathering tokens, a token from position 3000 in the original sequence was getting the positional encoding for position 37 (its local index in the gathered tensor). RoPE needs the *original* position to encode relative distances correctly. Fix: pass the original indices through to the RoPE module.

**3. Wrong causal mask.** Using `is_causal=True` on the gathered tensor assumes local order equals original order. While we sort the indices (which helps), the correct approach is to build an explicit boolean mask from the original positions. Fix: construct a `(K × K)` mask where `position[i] >= position[j]`.

## Why It's Faster

At the theoretical level, attention is O(N²) — cutting N in half makes it 4x cheaper. But in practice, the routing, gathering, scattering, and mask construction all cost something. The net result at 450M scale:

| Tokens Kept | Theoretical Speedup | What We Actually Measured |
|:---|:---|:---|
| 50% | 4x | 1.84x |
| 30% | 11x | 2.29x |
| 10% | 100x | 2.18x |

The gap between theory and practice is the overhead. Below 20% retention, speed actually stops improving because the overhead becomes the bottleneck. That's an engineering problem, not a fundamental limit.
