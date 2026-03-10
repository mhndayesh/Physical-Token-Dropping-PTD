# System Overview

What this system is
PTD V2 is a sparse transformer variant that physically drops tokens inside the forward pass. It does not just mask tokens. It gathers the selected tokens into a smaller tensor, runs the block layers on that smaller tensor, then scatters back into the full sequence. This reduces FLOPs and activation memory.

What is implemented in this repo
- A PTD wrapper around Qwen2 (Qwen/Qwen2.5-0.5B) in actual_ptd/model.py.
- Phase 2: router warm-up with distillation and soft gating.
- Phase 3: curriculum sparsity with full-model fine-tuning.
- A perplexity evaluator that compares dense vs PTD using the same dataset.

Core components
- Router: scores segments and selects top-k segments to keep.
- Gather/Scatter: physically compresses tokens for compute, then expands back.
- Causal mask from original positions: preserves correct attention ordering.
- Training schedule: gradually reduces keep-rate to avoid model shock.

Definitions used in this repo
- keep-rate: fraction of segments kept per block (for example 0.7 means keep 70 percent).
- segment: fixed-size chunk of tokens used for routing (segment_size).
- block: group of transformer layers between router decisions (block_size).
- selected tokens: tokens inside kept segments.

Where to look in code
- actual_ptd/model.py: PTDConfig, router, gather/scatter, forward path.
- actual_ptd/train_phase2.py: router warm-up with distillation.
- actual_ptd/train_phase3.py: curriculum sparsity training.
- actual_ptd/eval_perplexity.py: dense vs PTD PPL evaluation.
