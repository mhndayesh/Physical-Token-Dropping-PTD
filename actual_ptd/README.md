# PTD V2 (Fresh Start)

This directory is a fresh implementation built from the repo concepts, with cleaner contracts for real model work:

- Keeps Hugging Face output contracts in `forward`.
- Uses explicit `forward_with_aux` for PTD-only metadata used by training.
- Honors `attention_mask` in PTD path.
- Preserves generation semantics by delegating cache-based calls to dense forward.
- Uses a clean Phase 2/Phase 3 split aligned with the blueprint.

## Files

- `model.py`: `PTDQwen2ForCausalLM` + `PTDConfig`.
- `train_phase2.py`: Router warm-up (soft routing, no physical drop).
- `train_phase3.py`: Curriculum sparsity with full-model training.
- `eval_perplexity.py`: Dense vs PTD perplexity check.
- `eval_long_context.py`: Long-context eval (dense or PTD).
- `prepare_long_test.py`: Build fixed prompt pack for long-context tests.
- `run_long_test.py`: Single long-context test (dense + PTD).
- `run_long_test_batch.py`: Batch long-context test across many chats.

## Commands

Phase 2:

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4
```

Note: Phase 2 defaults to true soft gating (`ste_gating=False`) so router receives non-trivial distillation signal.
Phase 2 also applies gate-usage regularization toward `keep-rate` (`--sparsity-reg`, default `1.0`) to avoid pass-all collapse.

Phase 3:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5
```

Optional: add a coverage penalty to ensure each local window keeps at least one segment:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5 --coverage-window 4 --coverage-weight 0.1
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
python -m actual_ptd.run_long_test --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test\prompt.txt --ideal-answer-file long_context_test\ideal_answer.txt --seq-len 8192 --dense-use-cpu --report-json long_test_8k.json
```

Long-context batch test (4K, 20 samples):

```powershell
python -m actual_ptd.run_long_test_batch --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --data-root "C:\new-arch-model\stress test\chats\100K" --seq-len 4096 --question-set abstention --max-questions 20 --report-json long_test_batch_4k_gpu.json
```
