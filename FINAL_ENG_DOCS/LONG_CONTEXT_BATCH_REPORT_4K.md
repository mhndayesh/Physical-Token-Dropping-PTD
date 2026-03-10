# Long Context Batch Report (4K)

Date: 2026-03-10
Model: Qwen/Qwen2.5-0.5B
Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
Dataset: C:\new-arch-model\stress test\chats\100K
Question set: abstention
Samples: 20 chats
Sequence length: 4096 tokens (target, with prompt+answer)

## What This Test Measures
- Answer PPL (lower is better)
- Token accuracy on the answer (higher is better)
- Exact match on the answer (strict, usually low)
- PTD reports both selected-token and full-token metrics

## Commands Used

Batch test (dense + PTD):

```powershell
python -m actual_ptd.run_long_test_batch --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --data-root "C:\new-arch-model\stress test\chats\100K" --seq-len 4096 --question-set abstention --max-questions 20 --report-json long_test_batch_4k_gpu.json
```

## Summary Results (Averages)

Dense
- Answer PPL: 8.916
- Answer token accuracy: 0.597

PTD (keep 70%)
- Answer PPL (selected): 29.529
- Answer token accuracy (selected): 0.501
- Answer PPL (full): 1.34e23 (not meaningful for PTD)
- Answer token accuracy (full): 0.465

## Simple Interpretation
- Dense is more accurate on average for this test set.
- PTD keeps about 70% of tokens and loses accuracy on these long-context Q/A pairs.
- Full-token PPL for PTD is not a valid metric because dropped tokens are not computed.

## Files
- Input data: C:\new-arch-model\stress test\chats\100K
- Raw results: C:\qwen-adaptation\long_test_batch_4k_gpu.json
