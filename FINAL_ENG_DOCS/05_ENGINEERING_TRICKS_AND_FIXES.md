# Engineering Tricks and Fixes

Fixes included in actual_ptd
- Router parameters are moved to the same device and dtype as the base model to avoid BF16/FP32 mismatches.
- Attention mask is respected in the PTD path (token_mask is used throughout).
- Causal mask is built from original position ids, not local indices.
- RoPE uses the original positions for gathered tokens.
- Cache path falls back to dense HF forward to preserve generation semantics.

Practical tricks used
- Segment pooling: average only valid tokens to avoid padding bias.
- Segment routing: route by segments instead of single tokens to reduce overhead.
- Multi-query router: several queries compete to highlight different token types.
- Jitter noise in router scores to prevent deterministic early collapse.
- Top-k selection uses scores.detach to avoid gradient leakage through selection.
- Selection mask is tracked for selected-token loss and evaluation.
- Gate-usage regularization in Phase 2 to match keep-rate.
- Router entropy is logged to monitor how deterministic routing becomes.

Stability choices
- Block-level routing (block_size) reduces router frequency.
- Gradient clipping (norm 1.0) to reduce training spikes.
- Selected-token loss in Phase 3 to avoid full-token penalty.

Known limitations that remain
- Sparse KV-cache for generation is not implemented.
- Overhead of gather/scatter limits speedups at very low keep-rates.
- The router is not yet optimized for long-context or extreme sparsity.
