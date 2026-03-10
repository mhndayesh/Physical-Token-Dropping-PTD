# Smarter, Not Bigger: Physical Token Dropping (PTD)

PTD is a sparse transformer approach that keeps only top-scored token segments during block execution.
This repository contains a working PTD V2 implementation on **Qwen2.5-0.5B (0.5B model)** with training and evaluation code.

## End Results (Qwen2.5-0.5B, Keep=70%, KV-Cache Inference)

Dense vs PTD cache-mode comparison on the same long-context test:

| Context | Quality Tradeoff vs Dense | Total Latency | Peak VRAM | KV Cache Size |
| --- | --- | --- | --- | --- |
| 4K | PPL `+1.72%`, accuracy `0.00` points | `44.38%` lower with PTD | `64.09%` lower with PTD | `28.73%` lower with PTD |
| 8K | PPL `+2.16%`, accuracy `-4.76` points | `72.11%` lower with PTD | `85.56%` lower with PTD | `28.79%` lower with PTD |

Simple summary:
- PTD gives major long-context speed and memory gains.
- Accuracy cost is small to moderate at keep=70 for this 0.5B model.

Detailed benchmark report:
- [FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md](FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md)

## Quick Navigation

- PTD V2 code and commands: [actual_ptd](actual_ptd)
- PTD V2 usage guide: [actual_ptd/README.md](actual_ptd/README.md)
- Engineering docs index: [FINAL_ENG_DOCS/README.md](FINAL_ENG_DOCS/README.md)
- Evaluation summary: [FINAL_ENG_DOCS/04_EVALUATION_AND_RESULTS.md](FINAL_ENG_DOCS/04_EVALUATION_AND_RESULTS.md)
- Sparse training scoreboard: [PTD_SCOREBOARD.md](PTD_SCOREBOARD.md)
- Cache benchmark report: [FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md](FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md)
- Original POC docs: [ptd_poc/docs](ptd_poc/docs)

## Repository Layout

```text
.
|-- actual_ptd/                 # PTD V2 runtime + training + eval
|-- FINAL_ENG_DOCS/             # Engineering documentation bundle
|-- PTD_SCOREBOARD.md           # Sparse vs dense PPL results
|-- reports/                    # Reports folder (JSON files are gitignored)
|-- tools/                      # Utility scripts
|-- legacy/                     # Legacy notes and scripts
|-- ptd_poc/                    # Original POC code + docs
`-- README.md
```

## PTD V2 Scope

- Base model: `Qwen/Qwen2.5-0.5B`
- Training: Phase 2 router warm-up + Phase 3 sparsity curriculum
- Inference: Dense cache, PTD sparse cache, and long-context tests

## Hugging Face Package (Keep 70)

Published model repo:
- https://huggingface.co/mhndayesh/PTD-Qwen2.5-0.5B-Keep70-Variant

Export upload-ready package from checkpoint:

```powershell
python -m actual_ptd.export_hf_package --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --out-dir ptd_models/hf_keep70_full_state --base-model Qwen/Qwen2.5-0.5B --keep-rate 0.7 --package-type full_state --model-label "Qwen2.5-0.5B PTD Keep70"
```

Upload folder to HF:

```powershell
huggingface-cli upload <your-username>/<your-repo> ptd_models/hf_keep70_full_state . --repo-type model
```

Load from published HF repo (standard AutoModel + remote code):

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

Load packaged model:

```python
from pathlib import Path
import sys

pkg = Path("ptd_models/hf_keep70_full_state").resolve()
sys.path.insert(0, str(pkg))
from hf_ptd_loader import load_ptd_model

model, meta = load_ptd_model(str(pkg), device="cuda", dtype="bfloat16", keep_rate=0.7)
```

## Legacy POC References

Core concept documents:
- [ptd_poc/docs/MASTER_POC.md](ptd_poc/docs/MASTER_POC.md)
- [ptd_poc/docs/ARCHITECTURE.md](ptd_poc/docs/ARCHITECTURE.md)
- [ptd_poc/docs/MATHEMATICAL_PROOFS.md](ptd_poc/docs/MATHEMATICAL_PROOFS.md)
- [ptd_poc/docs/WALKTHROUGH.md](ptd_poc/docs/WALKTHROUGH.md)
- [ptd_poc/docs/TRAINING_RECIPE.md](ptd_poc/docs/TRAINING_RECIPE.md)
- [ptd_poc/docs/SCALABILITY.md](ptd_poc/docs/SCALABILITY.md)
