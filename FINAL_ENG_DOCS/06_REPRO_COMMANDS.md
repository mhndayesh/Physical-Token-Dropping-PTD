# Repro Commands

Data check
Make sure the packed TinyStories file exists:

```bash
python - <<'PY'
import os
path = 'data/tinystories_packed_qwen.pt'
print(path, 'exists' if os.path.exists(path) else 'MISSING')
PY
```

Phase 2 (router warm-up)

```bash
python -m actual_ptd.train_phase2 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --steps 3000 \
  --batch 4 \
  --lr 1e-4
```

Phase 2 with diversity loss (example)

```bash
python -m actual_ptd.train_phase2 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --steps 3000 \
  --batch 4 \
  --lr 1e-4 \
  --diversity-reg 0.1
```

Phase 2 with block distillation (example)

```bash
python -m actual_ptd.train_phase2 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --steps 3000 \
  --batch 4 \
  --lr 1e-4 \
  --block-distill-weight 1.0
```

Phase 3 (curriculum sparsity)

```bash
python -m actual_ptd.train_phase3 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt \
  --batch 2 \
  --lr 1e-5 \
  --schedule 0.99,0.9,0.7,0.5,0.3
```

Optional coverage penalty (example)

```bash
python -m actual_ptd.train_phase3 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt \
  --batch 2 \
  --lr 1e-5 \
  --schedule 0.99,0.9,0.7,0.5,0.3 \
  --coverage-window 4 \
  --coverage-weight 0.1
```

Phase 3 with loss-plateau early stop (example)

```bash
python -m actual_ptd.train_phase3 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt \
  --batch 2 \
  --lr 1e-5 \
  --schedule 0.99,0.9,0.7,0.5,0.3 \
  --early-stop-window 200 \
  --early-stop-delta 0.0005
```

Phase 3 with transformer router (example)

```bash
python -m actual_ptd.train_phase3 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt \
  --batch 2 \
  --lr 1e-5 \
  --router-type transformer --router-dim 128 --router-heads 2 --router-layers 1
```

Phase 3 with per-block keep rates (example)

```bash
python -m actual_ptd.train_phase3 \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt \
  --batch 2 \
  --lr 1e-5 \
  --per-block-keep 1.0,0.9,0.8,0.7
```

HF dataset eval (WikiText-2)

```bash
python -m actual_ptd.eval_hf_dataset \
  --model Qwen/Qwen2.5-0.5B \
  --dataset wikitext \
  --subset wikitext-2-raw-v1 \
  --split test \
  --seq-len 1024 \
  --n-seq 100
```

Profile (dense forward)

```bash
python -m actual_ptd.profile_eval --mode dense --model Qwen/Qwen2.5-0.5B --seq-len 1024 --steps 5
```

Eval: keep 70 percent

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt \
  --keep-rate 0.7
```

Eval: keep 50 percent

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint checkpoints/ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```

Eval: keep 30 percent

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint checkpoints/ptd_v2_phase3_stage5_keep30.pt \
  --keep-rate 0.3
```

Note on full-token eval
To compute full-token PPL (not recommended for PTD training targets), add --full-loss.

Long-context single test (8K, dense + PTD)

```powershell
python -m actual_ptd.run_long_test --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --dense-use-cpu --report-json reports\long_test_8k.json
```

Long-context batch test (4K, 20 samples)

```powershell
python -m actual_ptd.run_long_test_batch --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --data-root "C:\new-arch-model\stress test\chats\100K" --seq-len 4096 --question-set abstention --max-questions 20 --report-json reports\long_test_batch_4k_gpu.json
```

KV-cache correctness check (PTD no-cache vs PTD cache)

```powershell
python -m actual_ptd.eval_kv_cache --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --report-json reports\kv_cache_report_8k.json
```

Dense-cache vs PTD-cache comparison (4K)

```powershell
python -m actual_ptd.eval_cache_compare --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 4096 --report-json reports\cache_compare_4k_keep70.json
```

Dense-cache vs PTD-cache comparison (8K)

```powershell
python -m actual_ptd.eval_cache_compare --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --report-json reports\cache_compare_8k_keep70.json
```

Export HF package (keep 70, full-state)

```powershell
python -m actual_ptd.export_hf_package --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --out-dir ptd_models/hf_keep70_full_state --base-model Qwen/Qwen2.5-0.5B --keep-rate 0.7 --package-type full_state --model-label "Qwen2.5-0.5B PTD Keep70"
```

Upload HF package

```powershell
huggingface-cli upload <user>/<repo> ptd_models/hf_keep70_full_state . --repo-type model
```
