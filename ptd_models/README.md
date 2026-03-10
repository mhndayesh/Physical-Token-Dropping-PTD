# PTD Qwen2.5-0.5B Models

This folder documents PTD checkpoints and packaged PTD Qwen variants hosted on Hugging Face.

## Hugging Face Repo

- Legacy checkpoints repo: https://huggingface.co/mhndayesh/PDT
- PTD Qwen keep70 variant repo: https://huggingface.co/mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant

## Available Checkpoints

- Keep 50%: `ptd_v2_phase3_stage4_keep50.pt`
  - Direct download: https://huggingface.co/mhndayesh/PDT/resolve/main/ptd_v2_phase3_stage4_keep50.pt

- Keep 70% package (full-state):
  - Local export folder: `ptd_models/hf_keep70_full_state/`
  - Main weight file: `ptd_model_state.pt`
  - Recommended keep-rate: `0.7`
  - Published HF repo: `mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant`

## How To Use

Legacy raw checkpoints require the PTD runtime code in `actual_ptd/`.

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```

## Notes

- The exported keep70 package includes HF custom code so it can be loaded via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
- It also includes a direct helper loader (`hf_ptd_loader.py`) and PTD runtime model file.
- Upload command:

```powershell
huggingface-cli upload <user>/<repo> ptd_models/hf_keep70_full_state . --repo-type model
```
