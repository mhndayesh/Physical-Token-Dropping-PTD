# File Map

Root
- actual_ptd/ : current PTD V2 implementation (model, training, eval)
- FINAL_ENG_DOCS/ : this documentation bundle
- PTD_SCOREBOARD.md : evaluation results summary
- LONG_CONTEXT_BATCH_REPORT_4K.md : long-context batch report (4K)
- CORRECTIONS.md : known issues and fixes
- TRAINING_GUIDE.md : legacy guide for older scripts
- ptd_poc/ : original POC and blueprint documents
- checkpoints/ : local training checkpoints (not tracked in git)
- data/ : packed dataset tensors

actual_ptd/
- model.py : PTDConfig, router, gather/scatter, PTDQwen2ForCausalLM
- train_phase2.py : router warm-up with distillation
- train_phase3.py : curriculum sparsity training
- eval_perplexity.py : dense vs PTD PPL
- eval_long_context.py : long-context single test (dense or PTD)
- prepare_long_test.py : builds fixed 4K/8K prompt packs
- run_long_test.py : single run dense + PTD with metrics
- run_long_test_batch.py : batch run across many chats/questions

ptd_poc/docs/
- MASTER_POC.md : original proof-of-concept overview
- ARCHITECTURE.md : conceptual mechanism description
- TRAINING_RECIPE.md : original training plan
- WALKTHROUGH.md : project narrative
- SCALABILITY.md : scaling discussion
- MATHEMATICAL_PROOFS.md : theoretical background
