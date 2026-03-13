# PTD Production Bridge Update

Date: 2026-03-14

This document records the practical production updates added on top of PTD V2 for real deployment workflows.

## What Was Added

Core runtime updates:
- `actual_ptd/model.py`
  - mandatory keep masks
  - recent-window protection
  - router confidence + protected-ratio fallback checks
  - `generate_prefill_dense()` for sparse prefill + dense decode serving

New data/training/serving scripts:
- `actual_ptd/prepare_business_dataset.py`
- `actual_ptd/train_phase2_business.py`
- `actual_ptd/train_phase3_business.py`
- `actual_ptd/train_full_production.py`
- `actual_ptd/serve_prefill_dense.py`
- `actual_ptd/eval_business_replay.py`
- `actual_ptd/prepare_general_hf_dataset.py`
- `actual_ptd/data_quality_report.py`
- `actual_ptd/compare_dense_vs_ptd.py`
- `actual_ptd/benchmark_long_context.py`

## Bug Fixes Included

- `serve_prefill_dense.py`: fixed tokenizer `offset_mapping` handling that caused:
  - `ValueError: too many values to unpack (expected 2)`
- `eval_business_replay.py`: switched JSONL load to `utf-8-sig` to avoid:
  - `JSONDecodeError: Unexpected UTF-8 BOM`
- `train_full_production.py`: improved orchestration behavior and path fallback handling when `_prod` data names are missing.

## Why This Path

The repository already documents that sparse-cache decode is approximate (not bit-exact).  
For safer operations, this bridge keeps PTD for prompt pruning and then decodes with dense Qwen (`prefill sparse, decode dense`).

## Benchmark Snapshot (Dense vs PTD-70 vs PTD-30)

Checkpoint used:
- `checkpoints/ptd_prod_phase3_stage4_keep70.pt`

### 4k Context Runtime (32 new tokens)

| Metric | Dense | PTD-70 | PTD-30 |
| --- | --- | --- | --- |
| latency | `1.2914s` | `1.0639s` | `0.7909s` |
| throughput | `24.7795 tok/s` | `30.0786 tok/s` | `40.4608 tok/s` |
| peak VRAM | `4150.63 MB` | `4234.92 MB` | `3345.85 MB` |
| fallback rate | `n/a` | `0.0` | `0.0` |

Key deltas:
- PTD-70 vs Dense: about `1.21x` faster, with `+84.29 MB` peak VRAM in this run.
- PTD-30 vs Dense: about `1.64x` faster, with `804.78 MB` lower peak VRAM.

### General Eval Quality (60 samples)

`force_ptd=true`, `recent_window=0`, `max_new_tokens=64`

| Metric | Dense | PTD-70 | PTD-30 |
| --- | --- | --- | --- |
| response token F1 | `0.2228` | `0.2078` | `0.1753` |
| latency mean | `1.3393s` | `1.3713s` | `1.3410s` |
| throughput | `47.0140 tok/s` | `46.2328 tok/s` | `46.9923 tok/s` |
| peak VRAM | `994.67 MB` | `1080.75 MB` | `1080.75 MB` |
| fallback rate | `n/a` | `0.0` | `0.0` |

Quality tradeoff:
- PTD-70 F1 delta vs Dense: `-0.0150`
- PTD-30 F1 delta vs Dense: `-0.0475`
- PTD-30 F1 delta vs PTD-70: `-0.0325` (about `15.64%` relative)

Reference logs used:
- `logs/long_context_4k_fallback.json`
- `logs/long_context_4k_keep30_fallback.json`
- `logs/accuracy_general_keep70_force_rw0.json`
- `logs/accuracy_general_keep30_force_rw0.json`

## Notes

- PTD-30 and PTD-70 inference settings above were tested on a keep70-trained checkpoint, so a keep30-trained checkpoint may change the quality/runtime tradeoff.
- 16k benchmark was started but interrupted; no final `16k` JSON report was produced.
- This bridge is an engineering deployment compromise, not a mathematically exact PTD-to-dense hidden-state handoff.
