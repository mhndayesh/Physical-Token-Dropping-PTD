# Architecture and Mechanism

High-level flow for each block group
1. Start with full hidden states of shape (batch, seq, hidden).
2. Pad to an integer number of segments.
3. Pool tokens inside each segment to get segment embeddings.
4. Router scores each segment and selects top-k segments based on keep-rate.
5. Gather tokens from the selected segments into a compact tensor.
6. Build an additive causal mask from original positions.
7. Run the transformer layers on the compact tensor.
8. Scatter the updated tokens back into the full sequence.
9. Repeat for the next router block.

Router mechanism
- Router is MultiQueryRouter.
- Router parameter: k_proj is a linear projection from hidden size to router rank.
- Router parameter: queries are learned query vectors.
- For each segment embedding, compute scores with multi-query dot products.
- Select top-k segments by score using torch.topk.
- Optional jitter noise during training to avoid deterministic collapse.

Segmentation and pooling
- segment_size controls how many tokens are grouped.
- Segment pooling is a masked mean across tokens in the segment.
- segment_size and block_size are fixed by configuration in PTDConfig.

Gather and scatter
- Gather uses selected segment indices and expands to token indices.
- Scatter writes the updated compact tokens back into the full tensor.
- The non-selected tokens keep their previous states for this block.

Causal masking and positional encoding
- Causal mask is built from original position ids, not local indices.
- This ensures that attention never looks into the future after gather.
- RoPE is applied using the original position ids.

Selection mask
- selection_mask marks which original tokens were processed by PTD in a block.
- It is used during training and evaluation to compute selected-token loss.

Block grouping
- block_size groups multiple transformer layers under one routing decision.
- This reduces routing overhead and improves stability.
- Constraint: num_hidden_layers must be divisible by block_size.

Cache and generation
- When use_cache or past_key_values are provided, the model falls back to dense HF forward.
- This preserves generation correctness but does not provide sparse-cache speedups.

Relevant code
- actual_ptd/model.py
- PTDQwen2ForCausalLM._forward_hidden_with_aux
- MultiQueryRouter
