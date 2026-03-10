# PTD Accuracy Scoreboard (Sparse vs Dense)

Date: 2026-03-10
Model: Qwen/Qwen2.5-0.5B
Dataset: data/tinystories_packed_qwen.pt
Eval script: actual_ptd/eval_perplexity.py (selected-token loss)

## Results

- Dense PPL (baseline): 7.813

## Baseline (no coverage penalty)

### Keep 70% (stage3_keep70)
- Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
- Sparse PPL: 9.358
- Delta vs dense: +19.77%
- Ratio (Sparse / Dense): 1.20x

### Keep 50% (stage4_keep50)
- Checkpoint: checkpoints/ptd_v2_phase3_stage4_keep50.pt
- Sparse PPL: 10.646
- Delta vs dense: +36.25%
- Ratio (Sparse / Dense): 1.36x

### Keep 30% (stage5_keep30)
- Checkpoint: checkpoints/ptd_v2_phase3_stage5_keep30.pt
- Sparse PPL: 12.698
- Delta vs dense: +62.52%
- Ratio (Sparse / Dense): 1.63x

## Coverage penalty (coverage-window=4, coverage-weight=0.1)

### Keep 70% (stage3_keep70)
- Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
- Sparse PPL: 9.450
- Delta vs dense: +20.95%
- Ratio (Sparse / Dense): 1.21x

### Keep 50% (stage4_keep50)
- Checkpoint: checkpoints/ptd_v2_phase3_stage4_keep50.pt
- Sparse PPL: 10.631
- Delta vs dense: +36.06%
- Ratio (Sparse / Dense): 1.36x

### Keep 30% (stage5_keep30)
- Checkpoint: checkpoints/ptd_v2_phase3_stage5_keep30.pt
- Sparse PPL: 12.466
- Delta vs dense: +59.55%
- Ratio (Sparse / Dense): 1.60x

## Simple Explanation

Lower PPL is better. Dense is still best because it uses all tokens.
As we drop more tokens, accuracy gets worse:
- 70% keep: small drop (about 20-21% worse)
- 50% keep: medium drop (about 36% worse)
- 30% keep: larger drop (about 60-63% worse)

So your model is behaving correctly: more speed (fewer tokens) means lower accuracy.

## Notes

- These numbers are computed on selected tokens only (same objective as training).
- Full-token PPL is not the training target and will look extremely large.
