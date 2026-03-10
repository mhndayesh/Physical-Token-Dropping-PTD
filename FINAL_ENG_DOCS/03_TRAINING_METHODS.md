# Training Methods

Training is split into two phases to avoid router shock.

Phase 2: Router warm-up (distillation)
Goal
Teach the router to select useful tokens without physically dropping any tokens.

Method
1. Load the dense teacher (Qwen2.5-0.5B).
2. Create PTD student with drop_tokens=False.
3. Freeze backbone weights and train only router parameters.
4. Compute KL divergence between student logits and teacher logits.
5. Apply gate-usage regularization to keep average gate near keep-rate.
6. Save router checkpoints every N steps.

Loss used
- loss_kl = KL( softmax(student/T) || softmax(teacher/T) ) * T^2
- loss_reg = mean( (gate_mean - target_gate)^2 )
- loss = loss_kl + sparsity_reg * loss_reg

Notes
- Soft gating is default (ste_gating=False). This allows dense gradient signal.
- gate-usage regularization helps prevent pass-all collapse.

Phase 3: Curriculum sparsity (full model)
Goal
Fine-tune the full model while gradually decreasing keep-rate.

Method
1. Load router checkpoint from Phase 2.
2. Enable drop_tokens=True and unfreeze all weights.
3. Use a keep-rate schedule such as 0.99,0.9,0.7,0.5,0.3.
4. At each stage, train for steps_per_stage steps.
5. Optimize selected-token KL by default (mask_loss=True).
6. Log both selected-token and full-token losses for diagnostics.
7. Optional: add a coverage penalty to ensure each local window keeps at least one segment.

Why selected-token loss
Full-token KL includes dropped tokens and grows as keep-rate drops.
Selected-token KL tracks the objective the sparse model is actually optimized for.

Optional coverage penalty
- coverage_penalty is computed from segment selection per block.
- It penalizes windows with zero selected segments.
- Controlled by --coverage-window and --coverage-weight in train_phase3.py.

Data
- Current POC uses data/tinystories_packed_qwen.pt (TinyStories packed).
- This is a small dataset used to validate mechanism and stability.

Relevant code
- actual_ptd/train_phase2.py
- actual_ptd/train_phase3.py
