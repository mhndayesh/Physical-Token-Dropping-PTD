# Long Context Test Report (8K)

Date: 2026-03-10
Model: Qwen/Qwen2.5-0.5B
Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
Context source: long_context_test\prompt.txt
Ideal answer: long_context_test\ideal_answer.txt
Sequence length: 8191 tokens
Answer length: 21 tokens

## Results

Dense (CPU)
- Answer PPL: 13.206
- Answer token accuracy: 0.429
- Exact match: False
- Forward time: 15.976 s
- Throughput: 512.70 tokens/s

PTD (keep 70%, GPU)
- Answer PPL (selected): 13.172
- Answer PPL (full): 13.172
- Answer token accuracy (selected): 0.429
- Answer token accuracy (full): 0.429
- Exact match (full): False
- Forward time: 0.543 s
- Throughput: 15072.77 tokens/s
- Peak CUDA allocated: 9410.6 MB
- Peak CUDA reserved: 9556.0 MB

## Notes
- Dense was forced to CPU to avoid GPU OOM at 8K. PTD ran on GPU.
- The dense and PTD accuracy metrics are nearly identical on this single 8K example.
- Speed is not directly comparable because dense ran on CPU and PTD ran on GPU.

## Is this a "real" accuracy test?
This is a valid sanity check for long-context correctness, but it is not a full scientific evaluation.
It uses one question and one ideal answer from the probing set, so it is low sample size and highly noisy.

For a real accuracy evaluation, you should run many questions across multiple chats and report averages.
Recommended next steps:
- Sample at least 100 questions across multiple chat IDs.
- Report mean and std for PPL and token accuracy.
- Keep hardware consistent for speed comparisons (GPU vs GPU, or CPU vs CPU).
- Use both selected-token and full-token metrics for PTD.
