# PTD Training Guide for Qwen2.5-0.5B

## Architecture Quick Reference

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` (494M params) |
| PTD blocks | 4 blocks × 6 layers |
| Router params | **57,856** (what we're training) |
| Default sparsity | 0.3 (keep 30% of segments) |
| Segment size | 16 tokens per routing unit |

---

## Phase 2: Router Warm-up (Distillation)

**Goal:** Teach the router which tokens matter, WITHOUT modifying the backbone.

### Command
```
python train_0_5b.py --steps 10000 --batch 4 --lr 1e-4 --sparsity 0.3
```

### Settings

| Setting | Value | Why |
|---|---|---|
| `--lr` | `1e-4` | Good for 57K params. High enough to learn fast, low enough to not diverge |
| `--batch` | `4` | Balance between speed and gradient stability |
| `--temperature` | `2.0` | Softens teacher logits so router learns smooth importance scores |
| `--sparsity` | `0.3` | Keep 30% — the sweet spot from POC benchmarks |
| `--steps` | `10000` | ~8 epochs over 5K samples. Router has few params, needs many passes |

### What to Watch

| Loss Range | Meaning | Action |
|---|---|---|
| **> 100** | Router is random, just starting | Keep training |
| **50–100** | Router is learning segment importance | Keep training |
| **10–50** | Router is getting selective | Good progress, keep going |
| **5–10** | Router is well-calibrated | ✅ **Can stop here for Phase 2** |
| **< 5** | Excellent convergence | ✅ Stop, move to Phase 3 |
| **Loss stuck / oscillating** | LR too high or too low | See troubleshooting below |

### When to Stop Phase 2

**Stop when loss is consistently below 10.** Then verify:
```
python verify_accuracy.py --sparsity 0.3 --checkpoint checkpoints/ptd_student_step010000.pt
```
- **PPL increase < 30% vs dense** → router is working well, move to Phase 3
- **PPL increase > 50%** → train more or lower sparsity to 0.5

---

## Phase 3: Curriculum Sparsity (The Squeeze)

**Goal:** Unfreeze the backbone and let it adapt to the router's decisions.

### Command
```bash
python train_phase3.py --router-ckpt checkpoints/ptd_student_step003000.pt --batch 2
```

### Memory Optimization (BF16)
Phase 3 uses **BFloat16 Mixed Precision** to fit the Teacher, Student, and Optimizer states on a single consumer GPU (~8GB VRAM).

### Settings

| Setting | Phase 3a (Gentle) | Phase 3b (Target) |
|---|---|---|
| `--lr` | `1e-5` | `5e-6` |
| `--sparsity` | `0.9` → `0.7` | `0.5` → `0.3` |
| Backbone | **Unfrozen** | **Unfrozen** |

### Sparsity Schedule (SPARSITY_SCHEDULE)
The `train_phase3.py` script automatically steps through:
1. **Stage 1 (90%)**: ~2000 steps (Current Loss: ~1.5)
2. **Stage 2 (70%)**: ~2000 steps
3. **Stage 3 (50%)**: ~2000 steps
4. **Stage 4 (30%)**: ~2000 steps (Target)

### Ideal Loss Targets (Phase 3)

In Phase 3, we are physically dropping tokens, so the loss will be higher than the near-zero values seen in Phase 2. As the backbone adapts, you should see the following:

| Retention | Stage | Ideal Loss | Status |
|---|---|---|---|
| **100%** | Phase 2 | < 1.0 (Target: 0.1) | ✅ Accomplished |
| **90%** | Stage 1 | 1.0 - 2.5 | ✅ Currently ~1.5 |
| **70%** | Stage 2 | 1.5 - 3.5 | ✅ Successfully passed |
| **50%** | Stage 3 | 1.0 - 4.5 | ⚡ Currently ~0.9 (Excellent!) |
| **30%** | Stage 4 | **< 6.0** | 🎯 Final Goal |

**Note**: If your loss at 30% retention is below **6.0**, your model is likely performing at near-dense accuracy with full PTD speedup.

---

## LR Cheat Sheet

| Phase | LR | Why |
|---|---|---|
| Phase 2 start | `1e-4` | Fast router learning, backbone frozen |
| Phase 2 (if loss stalls) | `3e-5` | Reduce if loss oscillates after step 5000 |
| Phase 3a | `1e-5` | Backbone unfrozen — must be gentle |
| Phase 3b | `5e-6` | Final polish, minimal backbone disruption |

**Rule of thumb:** If loss hasn't improved in 1000 steps, halve the LR.

---

## Resuming from Checkpoints

The `train_phase3.py` script supports full session resumption (model + optimizer + stage).

To resume your curriculum from where it left off:
```bash
python train_phase3.py --resume-ckpt checkpoints/ptd_phase3_step008000.pt
```

To **Start Final Refinement** (training more on top of the finished 30% model):
```bash
python train_phase3.py --resume-ckpt checkpoints/ptd_phase3_stage5_keep30.pt --lr 5e-6 --steps-per-stage 5000
```
This will "park" the model at 30% sparsity and continue updating backbone weights for another 5000 steps.

---

## Evaluation Commands

### Check perplexity (dense vs sparse)
```
python verify_accuracy.py --sparsity 0.3 --checkpoint checkpoints/ptd_student_step010000.pt
```

### Generate text to inspect quality
```
python verify_fine_tuned.py --sparsity 0.3 --checkpoint checkpoints/ptd_student_step010000.pt --compare-dense
```

### Logits sanity check (should always pass)
```
python check_logits.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Loss explodes (NaN) | LR too high | Halve LR, restart from last checkpoint |
| Loss stuck > 100 | Router not learning | Check `--temperature 2.0`, try `--lr 3e-4` |
| Loss oscillates wildly | Batch too small or LR too high | Increase `--batch` to 8 or lower LR |
| PPL barely improves after Phase 3 | Backbone needs more adaptation time | Double the steps for Phase 3 |
| OOM on GPU | Batch too large for soft routing | Reduce `--batch` to 2 |

---

## Full Pipeline Summary

```
# 1. Prepare data
python prepare_qwen_data.py --samples 5000

# 2. Phase 2: Router warm-up (target: loss < 10)
python train_0_5b.py --steps 10000 --batch 4 --lr 1e-4

# 3. Evaluate router quality
python verify_accuracy.py --checkpoint checkpoints/ptd_student_step010000.pt

# 4. Phase 3: Curriculum sparsity (unfrozen backbone)
python train_phase3.py --router-ckpt checkpoints/ptd_student_step010000.pt --batch 2

# 5. Final Refinement (Polish)
python train_phase3.py --resume-ckpt checkpoints/ptd_phase3_stage5_keep30.pt --lr 5e-6 --steps-per-stage 5000

# 6. Final evaluation
python verify_accuracy.py --sparsity 0.3 --checkpoint checkpoints/ptd_phase3_stage5_keep30.pt
python verify_fine_tuned.py --sparsity 0.3 --checkpoint checkpoints/ptd_phase3_stage5_keep30.pt --compare-dense
```
