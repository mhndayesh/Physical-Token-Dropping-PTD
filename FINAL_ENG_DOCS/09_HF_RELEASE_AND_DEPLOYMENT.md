# HF Release And Deployment

## Published Model

- Repo: `mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant`
- URL: https://huggingface.co/mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant
- Base model: `Qwen/Qwen2.5-0.5B`
- Variant: `PTD keep70 full-state`

## What Was Packaged

- `ptd_model_state.pt` (full PTD model state)
- `config.json` with HF auto-map entries
- `configuration_ptd_qwen2.py` and `modeling_ptd_qwen2.py` (HF custom classes)
- `model.py` (PTD runtime)
- `ptd_package_config.json` metadata
- Model card `README.md` with usage and links

## Why This Format

- Standard HF loading path is supported:
  - `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- PTD routing logic remains intact through custom code.
- Keep-rate can be adjusted at runtime (default/recommended is 0.7).

## User Loading Example

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

## Maintainer Release Commands

Export package:

```powershell
python -m actual_ptd.export_hf_package --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --out-dir ptd_models/hf_keep70_full_state --base-model Qwen/Qwen2.5-0.5B --keep-rate 0.7 --package-type full_state --model-label "PTD Qwen2.5-0.5B Keep70 Variant"
```

Upload package:

```powershell
huggingface-cli upload <user>/<repo> ptd_models/hf_keep70_full_state . --repo-type model
```

## Integration Notes

- This is a PTD custom model, not a vanilla dense Qwen checkpoint.
- Use `trust_remote_code=True` in HF auto-loading.
- For benchmark context and limits, see:
  - `FINAL_ENG_DOCS/04_EVALUATION_AND_RESULTS.md`
  - `FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md`
