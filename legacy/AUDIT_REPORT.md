# PTD Project Audit Report

Date: 2026-03-09
Scope: `ptd_poc/docs/*`, `qwen_ptd.py`, `train_0_5b.py`, `train_phase3.py`, eval and helper scripts.

## Executive summary

The core PTD idea is implemented and can run on toy/single-GPU experiments, but the current codebase is not production-ready for actual LLM-scale training/inference. The main blockers are model API incompatibility (mask/cache), training objective mismatch with the blueprint, and missing systems-level training infrastructure.

## Critical findings

1. Attention mask is ignored in patched model forward.
- Evidence: [`qwen_ptd.py:134`](C:\qwen-adaptation\qwen_ptd.py:134) accepts `attention_mask` but never uses it.
- Runtime check (tiny Qwen2 config): dense model changed logits when mask changed, PTD model did not (`max diff 0.0`).
- Impact: padded batches, packed sequences, and many downstream HF workflows become semantically incorrect.

2. KV cache is not implemented; `use_cache=True` is ignored.
- Evidence: [`qwen_ptd.py:136`](C:\qwen-adaptation\qwen_ptd.py:136), [`qwen_ptd.py:138`](C:\qwen-adaptation\qwen_ptd.py:138), [`qwen_ptd.py:256`](C:\qwen-adaptation\qwen_ptd.py:256) always returns `past_key_values=None`.
- Runtime check: dense returns non-`None` cache; PTD returns `None`.
- Impact: generation latency scales poorly with output length, blocking practical serving.

3. Output contract is changed in a non-HF-compatible way.
- Evidence: [`qwen_ptd.py:258`](C:\qwen-adaptation\qwen_ptd.py:258) stores `(selection_mask, indices)` in `hidden_states`.
- Baseline behavior: `Qwen2Model.forward` returns `hidden_states=None` unless requested.
- Impact: tools expecting HF standard output may break or misread outputs.

4. Phase 2 training does hard sparse dropping by default, conflicting with blueprint warm-up intent.
- Blueprint: warm-up at 100% retention or soft routing first in [`TRAINING_RECIPE.md:19`](C:\qwen-adaptation\ptd_poc\docs\TRAINING_RECIPE.md:19).
- Code: default Phase 2 sparsity is 30% in [`train_0_5b.py:39`](C:\qwen-adaptation\train_0_5b.py:39).
- Impact: router learns under severe information loss early, increasing instability and quality loss.

## High findings

1. Distillation loss in Phase 3 only supervises tokens selected by the last block.
- Evidence: mask is built from last block selection in [`qwen_ptd.py:247`](C:\qwen-adaptation\qwen_ptd.py:247) to [`qwen_ptd.py:252`](C:\qwen-adaptation\qwen_ptd.py:252), then used in [`train_phase3.py:194`](C:\qwen-adaptation\train_phase3.py:194).
- Impact: unselected tokens receive no direct distillation signal, which can hurt language quality.

2. Resume semantics are partial, not full.
- Evidence: stage/global step restored in [`train_phase3.py:155`](C:\qwen-adaptation\train_phase3.py:155), but no in-stage step offset is restored; loop restarts stage steps at 1 in [`train_phase3.py:176`](C:\qwen-adaptation\train_phase3.py:176).
- Impact: resumed runs can retrain part of a stage unexpectedly; reproducibility and accounting degrade.

3. Infrastructure is single-GPU prototype only.
- Evidence: no grad accumulation, no distributed/FSDP/ZeRO, no scheduler/warmup, no eval split loop in [`train_phase3.py`](C:\qwen-adaptation\train_phase3.py:1) and [`train_0_5b.py`](C:\qwen-adaptation\train_0_5b.py:1).
- Impact: not viable for billion-token or multi-node LLM training.

4. Data scale is far below meaningful adaptation for LLM behavior.
- Evidence: prepared dataset is 5,000 x 256 tokens (`1.28M` tokens) in [`prepare_qwen_data.py:10`](C:\qwen-adaptation\prepare_qwen_data.py:10), [`prepare_qwen_data.py:42`](C:\qwen-adaptation\prepare_qwen_data.py:42).
- Impact: results are proof-of-concept only, not representative of real LLM adaptation quality.

## Medium findings

1. Docs and code are inconsistent on curriculum details.
- `train_phase3.py` schedule includes 0.99 stage in [`train_phase3.py:29`](C:\qwen-adaptation\train_phase3.py:29), while docstring header lists 0.9 first in [`train_phase3.py:6`](C:\qwen-adaptation\train_phase3.py:6).
- Impact: operational confusion and incorrect expectation-setting.

2. Several scripts assume internet access and can fail in restricted environments.
- Example: tokenizer/model loading without explicit local/offline mode in [`check_logits.py:8`](C:\qwen-adaptation\check_logits.py:8), [`prepare_qwen_data.py:15`](C:\qwen-adaptation\prepare_qwen_data.py:15).
- Impact: reproducibility issues in locked-down clusters.

3. Performance-critical gather/scatter and mask builds are pure Python/PyTorch indexing.
- Evidence: advanced indexing in [`qwen_ptd.py:202`](C:\qwen-adaptation\qwen_ptd.py:202), [`qwen_ptd.py:241`](C:\qwen-adaptation\qwen_ptd.py:241), mask construction in [`qwen_ptd.py:114`](C:\qwen-adaptation\qwen_ptd.py:114).
- Impact: overhead likely dominates at scale without custom kernels.

## Blueprint-to-code alignment

1. Aligned: PTD gather/process/scatter structure and segment routing.
- Blueprint concept in [`ARCHITECTURE.md`](C:\qwen-adaptation\ptd_poc\docs\ARCHITECTURE.md) matches core implementation in [`qwen_ptd.py`](C:\qwen-adaptation\qwen_ptd.py).

2. Partially aligned: 3-phase curriculum exists.
- Implemented by [`train_0_5b.py`](C:\qwen-adaptation\train_0_5b.py) + [`train_phase3.py`](C:\qwen-adaptation\train_phase3.py), but warm-up retention policy diverges from recipe.

3. Not aligned enough for production claims.
- Blueprint/scale docs discuss large-scale training strategy in [`SCALABILITY.md`](C:\qwen-adaptation\ptd_poc\docs\SCALABILITY.md) and [`TRAINING_RECIPE.md`](C:\qwen-adaptation\ptd_poc\docs\TRAINING_RECIPE.md), but current code lacks distributed training, robust eval, and inference cache compatibility.

## What will make it fail in actual LLM phase

1. Incorrect masking behavior with real packed/padded data.
2. No KV cache for serving or long generation throughput.
3. Prototype training loop cannot scale operationally (single GPU, no sharding/scheduler/eval discipline).
4. Router warm-up policy and supervision strategy likely underfit important tokens in dense tasks (code/math).
5. Static fixed top-k retention can corrupt dense-information inputs (no dynamic keep policy).

## Recommended next milestones

1. Restore HF compatibility in `qwen_ptd.py`:
- Honor `attention_mask`, `use_cache`, `past_key_values`, `cache_position`.
- Keep `hidden_states` contract standard; put PTD metadata in a separate field/wrapper.

2. Bring training loop to LLM standards:
- Add gradient accumulation, LR schedule with warmup, eval split, checkpoint metadata for exact resume.
- Add distributed strategy (FSDP/DeepSpeed/ZeRO).

3. Fix curriculum-policy mismatch:
- Phase 2 at 100% keep (or true soft routing) before hard dropping.
- Revisit Phase 3 loss masking strategy.

4. Build production evaluation harness:
- Per-domain eval (general, code, math, long-context retrieval), latency with/without cache, memory profiling.

5. Add scale-oriented kernels:
- Replace high-overhead gather/scatter and mask ops with fused Triton/CUDA kernels.

