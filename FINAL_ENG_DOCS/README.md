# FINAL_ENG_DOCS

Purpose
- Provide engineering-level documentation for the PTD V2 implementation in this repo.
- Explain mechanism, training methods, evaluation, results, fixes, and next steps.
- Make it easy for a CS professor to review and propose improvements.

Contents
- 01_SYSTEM_OVERVIEW.md
- 02_ARCHITECTURE_AND_MECHANISM.md
- 03_TRAINING_METHODS.md
- 04_EVALUATION_AND_RESULTS.md
- 05_ENGINEERING_TRICKS_AND_FIXES.md
- 06_REPRO_COMMANDS.md
- 07_LIMITATIONS_AND_NEXT_STEPS.md
- 08_FILE_MAP.md

Quick summary
PTD (Physical Token Dropping) selects a subset of tokens inside each block, runs attention and MLP only on that subset, and scatters the results back. This reduces compute and memory, trading off some accuracy. The current implementation is in actual_ptd/ and uses a 2-phase training pipeline with a curriculum on keep-rate.
