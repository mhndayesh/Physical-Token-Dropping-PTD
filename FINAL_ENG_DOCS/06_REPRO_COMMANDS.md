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
python -m actual_ptd.run_long_test --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --dense-use-cpu --report-json long_test_8k.json
```

Long-context batch test (4K, 20 samples)

```powershell
python -m actual_ptd.run_long_test_batch --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --data-root "C:\new-arch-model\stress test\chats\100K" --seq-len 4096 --question-set abstention --max-questions 20 --report-json long_test_batch_4k_gpu.json
```
