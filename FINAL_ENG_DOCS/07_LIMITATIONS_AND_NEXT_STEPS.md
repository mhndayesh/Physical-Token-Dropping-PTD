# Limitations and Next Steps

Current limitations
- Sparse KV-cache is not implemented, so generation uses dense forward.
- Speedups are limited by gather/scatter overhead at very low keep-rates.
- Keep-rate is static per stage and not learned.
- Router uses a simple MLP-style projection and may miss longer-range importance cues.
- Evaluation uses TinyStories packed data, which is small and not representative of full-scale pretraining.

Suggested research directions
- Implement sparse KV-cache and sparse attention for generation.
- Explore learned keep-rate or adaptive keep-rate per layer.
- Try alternative router architectures (lightweight attention or small transformer).
- Add auxiliary loss for router diversity or coverage.
- Sweep segment_size and block_size to trade off quality vs speed.
- Train with mixed data (general + high-density data like code/math).
- Measure speed and memory with profiling (torch.profiler, nvprof).

Production readiness gaps
- Integration with HF generate for sparse paths.
- Stable export format for PTD checkpoints with config metadata.
- Robust evaluation on standard LM benchmarks.
