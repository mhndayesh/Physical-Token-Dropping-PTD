# PTD V2 Corrections and Restart Guide

This file reflects the current `actual_ptd` pipeline.

## Main issue behind "opposite" Phase 3 results

If Phase 3 optimizes/logs **full-token KL**, loss will increase as keep-rate drops.
That behavior is expected, because dropped tokens are included in the denominator.

Observed pattern from current checkpoints:
- keep 99%: full KL ~6-10
- keep 90%: full KL ~19
- keep 70%: full KL ~47
- keep 50%: full KL ~76
- keep 30%: full KL ~114

This does **not** mean selected-token learning is failing.

## Correct objective for sparse adaptation

Use selected-token KL (`--mask-loss`, default in `actual_ptd/train_phase3.py`).

You should see logs like:
- `loss ... | full ... | sel ...`

Where:
- `loss` = optimized objective
- `full` = diagnostic full-token KL
- `sel` = selected-token KL (the key metric)

## Confirmed fixes in `actual_ptd`

1. BF16 dtype mismatch fixed (routers moved to model dtype/device).
2. Phase 2 soft-gating path enabled by default (`ste_gating=False`).
3. Phase 2 now includes gate-usage regularization toward `keep-rate` (`--sparsity-reg`) to reduce pass-all collapse.
4. Phase 3 supports selected-token objective (`--mask-loss`) and dual metrics (`full`, `sel`).
5. Attention mask is honored in PTD forward path.
6. HF generation/cache path falls back to dense forward to preserve correctness.

## Fresh restart commands (recommended)

1) Phase 2 (router warm-up):

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4
```

2) Phase 3 (selected-token objective, default):

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --schedule 0.99,0.9,0.7,0.5,0.3
```

3) Optional resume with lower LR:

```bash
python -m actual_ptd.train_phase3 --resume-ckpt checkpoints/ptd_v2_phase3_step004000.pt --batch 2 --lr 5e-6
```

## Important note for "actual LLM phase"

Current `actual_ptd/model.py` keeps generation correctness by delegating cache-based calls to dense forward.
So training/eval works, but sparse-cache serving speedups are not implemented yet.
