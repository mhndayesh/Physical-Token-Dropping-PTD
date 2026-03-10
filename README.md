# Smarter, Not Bigger: Physical Token Dropping (PTD)

Proof-of-concept repository for a transformer variant that physically keeps only top-scored tokens during block execution.

## Quick Navigation

- Engineering docs: [FINAL_ENG_DOCS](FINAL_ENG_DOCS)
- PTD V2 (Qwen) runtime: [actual_ptd](actual_ptd)
- Long-context batch report (4K): [LONG_CONTEXT_BATCH_REPORT_4K.md](FINAL_ENG_DOCS/LONG_CONTEXT_BATCH_REPORT_4K.md)
- Accuracy scoreboard: [PTD_SCOREBOARD.md](PTD_SCOREBOARD.md)
- POC concept docs: [ptd_poc/docs](ptd_poc/docs)
- Reports (JSON outputs): [reports](reports)
- Utilities and scripts: [tools](tools)
- Legacy notes: [legacy](legacy)

## Repository Layout (Current)

```text
.
??? actual_ptd/                 # PTD V2 runtime + training + eval
??? FINAL_ENG_DOCS/             # Engineering documentation bundle
??? PTD_SCOREBOARD.md           # Sparse vs dense PPL results
??? reports/                    # JSON outputs from test runs
??? tools/                      # Diagnostics and helper scripts
??? legacy/                     # Legacy notes and scripts
??? ptd_poc/                    # Original POC code + docs
??? README.md
```

## PTD V2 (Qwen2.5-0.5B)

Runtime and training code lives in `actual_ptd/`.

Key docs:
- [actual_ptd/README.md](actual_ptd/README.md) (commands and usage)
- [FINAL_ENG_DOCS](FINAL_ENG_DOCS) (full engineering documentation)

## Long-Context Testing

- Single test (8K): [actual_ptd/run_long_test.py](actual_ptd/run_long_test.py)
- Batch test (4K): [actual_ptd/run_long_test_batch.py](actual_ptd/run_long_test_batch.py)
- Latest 4K report: [FINAL_ENG_DOCS/LONG_CONTEXT_BATCH_REPORT_4K.md](FINAL_ENG_DOCS/LONG_CONTEXT_BATCH_REPORT_4K.md)

## Legacy POC

The original concept and proof-of-concept docs are under `ptd_poc/docs/`.

Core docs:
- [ptd_poc/docs/MASTER_POC.md](ptd_poc/docs/MASTER_POC.md)
- [ptd_poc/docs/ARCHITECTURE.md](ptd_poc/docs/ARCHITECTURE.md)
- [ptd_poc/docs/MATHEMATICAL_PROOFS.md](ptd_poc/docs/MATHEMATICAL_PROOFS.md)
- [ptd_poc/docs/WALKTHROUGH.md](ptd_poc/docs/WALKTHROUGH.md)
- [ptd_poc/docs/TRAINING_RECIPE.md](ptd_poc/docs/TRAINING_RECIPE.md)
- [ptd_poc/docs/SCALABILITY.md](ptd_poc/docs/SCALABILITY.md)
