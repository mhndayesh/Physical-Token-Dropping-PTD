# PTD V2 (Fresh Start)

This directory is a fresh implementation built from the repo concepts, with cleaner contracts for real model work:

- Keeps Hugging Face output contracts in `forward`.
- Uses explicit `forward_with_aux` for PTD-only metadata used by training.
- Honors `attention_mask` in PTD path.
- Supports optional PTD sparse KV-cache (`ptd_use_sparse_cache=True`) with dense-cache fallback.
- Uses a clean Phase 2/Phase 3 split aligned with the blueprint.

## Files

- `model.py`: `PTDQwen2ForCausalLM` + `PTDConfig`.
- `train_phase2.py`: Router warm-up (soft routing, no physical drop).
- `train_phase3.py`: Curriculum sparsity with full-model training.
- `eval_perplexity.py`: Dense vs PTD perplexity check.
- `eval_long_context.py`: Long-context eval (dense or PTD).
- `eval_hf_dataset.py`: Evaluate on Hugging Face datasets (e.g., WikiText-2).
- `profile_eval.py`: torch.profiler wrapper for dense/PTD forward.
- `eval_kv_cache.py`: Sparse no-cache vs sparse KV-cache correctness/perf check.
- `eval_cache_compare.py`: Dense-cache vs PTD-cache benchmark.
- `export_hf_package.py`: Build HF upload package from PTD checkpoint.
- `prepare_long_test.py`: Build fixed prompt pack for long-context tests.
- `run_long_test.py`: Single long-context test (dense + PTD).
- `run_long_test_batch.py`: Batch long-context test across many chats.

Latest benchmark (Qwen2.5-0.5B, keep=70%)
- 4K: same accuracy as dense on this sample, 44% lower total time, 64% lower peak VRAM.
- 8K: 4.76 points lower accuracy, 72% lower total time, 86% lower peak VRAM.

## Commands

Phase 2:

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4
```

Note: Phase 2 defaults to true soft gating (`ste_gating=False`) so router receives non-trivial distillation signal.
Phase 2 also applies gate-usage regularization toward `keep-rate` (`--sparsity-reg`, default `1.0`) to avoid pass-all collapse.
Optional: add diversity loss to encourage query specialization:

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4 --diversity-reg 0.1
```

Optional: block-level hidden-state distillation:

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4 --block-distill-weight 1.0
```

Phase 3:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5
```

Optional: add a coverage penalty to ensure each local window keeps at least one segment:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --coverage-window 4 --coverage-weight 0.1
```

Optional: loss-plateau early stop per stage:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --early-stop-window 200 --early-stop-delta 0.0005
```

Optional: transformer router (small attention router):

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --router-type transformer --router-dim 128 --router-heads 2 --router-layers 1
```

Optional: per-block keep rates (scaled by stage keep-rate):

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --per-block-keep 1.0,0.9,0.8,0.7
```

HF dataset eval (WikiText-2):

```bash
python -m actual_ptd.eval_hf_dataset --model Qwen/Qwen2.5-0.5B --dataset wikitext --subset wikitext-2-raw-v1 --split test --seq-len 1024 --n-seq 100
```

Profile (dense forward, 1024 tokens):

```bash
python -m actual_ptd.profile_eval --mode dense --model Qwen/Qwen2.5-0.5B --seq-len 1024 --steps 5
```

Eval:

```bash
python -m actual_ptd.eval_perplexity --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --checkpoint checkpoints/ptd_v2_phase3_stage5_keep30.pt --keep-rate 0.3
```

Long-context pack (8K):

```powershell
python -m actual_ptd.prepare_long_test --model Qwen/Qwen2.5-0.5B --data-root "C:\new-arch-model\stress test\chats\100K" --chat-id 1 --seq-len 8192 --question-set abstention --question-index 0 --out-dir long_context_test
```

Long-context single test (dense + PTD):

```powershell
python -m actual_ptd.run_long_test --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --dense-use-cpu --report-json reports\long_test_8k.json
```

Long-context batch test (4K, 20 samples):

```powershell
python -m actual_ptd.run_long_test_batch --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --data-root "C:\new-arch-model\stress test\chats\100K" --seq-len 4096 --question-set abstention --max-questions 20 --report-json reports\long_test_batch_4k_gpu.json
```

KV-cache validation (sparse no-cache vs sparse-cache):

```powershell
python -m actual_ptd.eval_kv_cache --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --report-json reports\kv_cache_report_8k.json
```

Note: PTD routing currently uses global top-k per forward pass, so sparse-cache decode is an approximation and will not be bit-exact to full-sequence PTD forward.

Dense-cache vs PTD-cache comparison:

```powershell
python -m actual_ptd.eval_cache_compare --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --report-json reports\cache_compare_8k_keep70.json
```

Build Hugging Face package (keep 70 full-state):

```powershell
python -m actual_ptd.export_hf_package --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --out-dir ptd_models/hf_keep70_full_state --base-model Qwen/Qwen2.5-0.5B --keep-rate 0.7 --package-type full_state --model-label "Qwen2.5-0.5B PTD Keep70"
```

Upload package:

```powershell
huggingface-cli upload <user>/<repo> ptd_models/hf_keep70_full_state . --repo-type model
```

Published PTD Qwen variant (keep70):
- `mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant`
- https://huggingface.co/mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant

Load published model with Auto classes:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant"
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
```
