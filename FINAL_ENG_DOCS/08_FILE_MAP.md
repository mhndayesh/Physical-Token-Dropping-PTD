# File Map

Root
- actual_ptd/ : current PTD V2 implementation (model, training, eval)
- FINAL_ENG_DOCS/ : this documentation bundle
- PTD_SCOREBOARD.md : evaluation results summary
- reports/cache_compare_4k_keep70.json : dense-cache vs PTD-cache (4K) raw output
- reports/cache_compare_8k_keep70.json : dense-cache vs PTD-cache (8K) raw output
- reports/kv_cache_report_4k.json : PTD no-cache vs PTD cache (4K) raw output
- reports/kv_cache_report_8k.json : PTD no-cache vs PTD cache (8K) raw output
- CORRECTIONS.md : known issues and fixes
- TRAINING_GUIDE.md : legacy guide for older scripts
- ptd_poc/ : original POC and blueprint documents
- checkpoints/ : local training checkpoints (not tracked in git)
- data/ : packed dataset tensors
- ptd_models/ : model packaging notes and export outputs
- ptd_models/hf_keep70_full_state/ : HF upload-ready keep70 package (generated)

actual_ptd/
- model.py : PTDConfig, router, gather/scatter, PTDQwen2ForCausalLM
- train_phase2.py : router warm-up with distillation
- train_phase3.py : curriculum sparsity training
- eval_perplexity.py : dense vs PTD PPL
- eval_long_context.py : long-context single test (dense or PTD)
- eval_hf_dataset.py : evaluate on HF datasets (WikiText-2, etc.)
- profile_eval.py : torch.profiler wrapper for dense/PTD
- eval_kv_cache.py : checks PTD cache correctness/perf
- eval_cache_compare.py : dense-cache vs PTD-cache benchmark
- export_hf_package.py : exports checkpoint into HF upload package folder
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

FINAL_ENG_DOCS/
- ROADMAP.md : suggested next steps and research ideas
- CACHE_COMPARE_REPORT_2026-03-10.md : dense-cache vs PTD-cache final benchmark report
