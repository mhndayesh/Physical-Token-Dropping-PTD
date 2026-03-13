# Smarter, Not Bigger: Physical Token Dropping (PTD)

PTD is a sparse transformer approach that keeps only top-scored token segments during block execution.
This repository contains a working PTD V2 implementation on **Qwen2.5-0.5B (0.5B model)** with training and evaluation code.

## Latest Production Update (Dense vs PTD-70 vs PTD-30, 2026-03-14)

Current production path uses **sparse prefill + dense decode** (`serve_prefill_dense.py`) with fallback available.

### 3-way snapshot

| Angle | Dense | PTD-70 | PTD-30 |
| --- | --- | --- | --- |
| General eval quality (response token F1, 60 samples) | `0.2228` | `0.2078` | `0.1753` |
| 4K latency (32 new tokens) | `1.2914s` | `1.0639s` | `0.7909s` |
| 4K throughput | `24.7795 tok/s` | `30.0786 tok/s` | `40.4608 tok/s` |
| 4K peak VRAM | `4150.63 MB` | `4234.92 MB` | `3345.85 MB` |
| Fallback rate in these runs | `n/a` | `0.0` | `0.0` |

What this means:
- PTD-70 is the balanced setting: about `1.21x` faster than dense at 4K with a small quality drop (`-0.0150` F1).
- PTD-30 is the aggressive setting: about `1.64x` faster than dense and saves about `804.78 MB` VRAM at 4K, with larger quality drop (`-0.0475` F1 vs dense; `-0.0325` vs PTD-70).
- In these benchmark runs, fallback did not trigger. It remains enabled for safety in production.
- Keep30/keep70 inference comparison here was run on a **keep70-trained checkpoint** (`ptd_prod_phase3_stage4_keep70.pt`), so a true keep30-trained checkpoint may shift the tradeoff.

Production references:
- [docs/PRODUCTION_BRIDGE.md](docs/PRODUCTION_BRIDGE.md)
- [docs/OLD_VS_NEW_WORKFLOW.md](docs/OLD_VS_NEW_WORKFLOW.md)
- [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)

## Research Cache-Mode Results (Earlier Benchmark)

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
- Production bridge guide (prefill prune + dense decode): [docs/PRODUCTION_BRIDGE.md](docs/PRODUCTION_BRIDGE.md)
- Old vs new workflow report: [docs/OLD_VS_NEW_WORKFLOW.md](docs/OLD_VS_NEW_WORKFLOW.md)
- One-line production commands: [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)
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
- Production bridge: protected sparse prefill + dense decode path (`actual_ptd/serve_prefill_dense.py`)
- End-to-end practical loop: `actual_ptd/train_full_production.py`

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
