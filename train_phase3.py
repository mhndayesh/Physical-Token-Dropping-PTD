"""
train_phase3.py  –  Phase 3: Curriculum Sparsity with Unfrozen Backbone
========================================================================
The backbone adapts to physically missing tokens through a gradual schedule:

  Stage 1:  sparsity=0.9 (keep 90%)  →  gentle warm-up
  Stage 2:  sparsity=0.7 (keep 70%)  →  moderate dropping
  Stage 3:  sparsity=0.5 (keep 50%)  →  aggressive
  Stage 4:  sparsity=0.3 (keep 30%)  →  target operating point

ALL parameters are trainable (backbone + routers).
Uses KL-divergence against the frozen dense teacher.

Requires: Phase 2 router checkpoint (from train_0_5b.py soft routing run).

Usage:
  python train_phase3.py
  python train_phase3.py --router-ckpt checkpoints/ptd_student_step003000.pt
  python train_phase3.py --steps-per-stage 2000
"""

import argparse, copy, math, os, time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

SPARSITY_SCHEDULE = [0.99, 0.9, 0.7, 0.5, 0.3]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data",            default="data/tinystories_packed_qwen.pt")
    p.add_argument("--router-ckpt",     default=None,
                   help="Phase 2 router checkpoint (router weights only)")
    p.add_argument("--resume-ckpt",     default=None,
                   help="Phase 3 session checkpoint (full model + optimizer)")
    p.add_argument("--steps-per-stage", type=int,   default=2000)
    p.add_argument("--batch",           type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--block-size",      type=int,   default=6)
    p.add_argument("--segment-size",    type=int,   default=16)
    p.add_argument("--save-every",      type=int,   default=500)
    p.add_argument("--log-every",       type=int,   default=50)
    p.add_argument("--temperature",     type=float, default=2.0)
    return p.parse_args()


def kl_distill_loss(student_logits, teacher_logits, T=2.0, mask=None):
    s_log_prob = F.log_softmax(student_logits.float() / T, dim=-1)
    t_prob     = F.softmax(teacher_logits.float()    / T, dim=-1)
    
    # kl shape: (batch, seq, vocab)
    kl = F.kl_div(s_log_prob, t_prob, reduction="none")
    kl = kl.sum(dim=-1) * (T ** 2)  # (batch, seq)
    
    if mask is not None:
        # Only average over the selected tokens
        kl = (kl * mask.float()).sum() / (mask.float().sum() + 1e-6)
    else:
        kl = kl.mean()
        
    return kl


def get_batch(data, batch_size, device):
    n = data.shape[0]
    idx = torch.randint(0, n, (batch_size,))
    x = data[idx].to(device)
    return x[:, :-1], x[:, 1:]


def set_sparsity(model, sparsity):
    """Update all router sparsity values in-place."""
    for router in model.model.ptd_routers:
        router.sparsity = sparsity


def main():
    args   = parse_args()
    device_id = torch.cuda.current_device()
    target_device = f"cuda:{device_id}"
    print(f"Device: {target_device}")

    # ── Load teacher (frozen, BF16 to save space) ─────────────────────────────
    print(f"Loading teacher (BF16): {args.model} ...")
    teacher = Qwen2ForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=target_device
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Load student (Trainable, BF16/FP32 balanced) ──────────────────────────
    print("Building PTD student (ALL params unfrozen) ...")
    student = Qwen2ForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=target_device
    )
    student = apply_ptd_to_qwen2(
        student,
        block_size   = args.block_size,
        sparsity     = SPARSITY_SCHEDULE[0],
        segment_size = args.segment_size,
    )

    # Load Phase 2 router weights if provided
    if args.router_ckpt:
        print(f"  Loading Phase 2 router checkpoint: {args.router_ckpt}")
        ckpt = torch.load(args.router_ckpt, map_location="cpu", weights_only=True)
        student.model.ptd_routers.load_state_dict(ckpt["router_state"])

    # ALL parameters trainable
    for p in student.parameters():
        p.requires_grad_(True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    student = student.to(target_device, dtype=dtype)
    
    print(f"Teacher on: {teacher.device} ({teacher.dtype})")
    print(f"Student on: {student.device} ({student.dtype})")
    print(f"Router 0 on: {next(student.model.ptd_routers[0].parameters()).device} "
          f"({next(student.model.ptd_routers[0].parameters()).dtype})")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    student.train()

    # ── Data ──────────────────────────────────────────────────────────────────
    if os.path.exists(args.data):
        data = torch.load(args.data, weights_only=True)
        print(f"Data: {data.shape} sequences loaded from {args.data}")
    else:
        print(f"[WARNING] data not found: {args.data}")
        data = torch.randint(0, 151936, (200, 257))

    optimizer = AdamW(student.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)

    start_stage = 0
    global_step = 0
    
    if args.resume_ckpt:
        print(f"  Resuming from session checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        student.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # Keep CLI LR authoritative when resuming.
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        start_stage = ckpt.get("stage", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"  Resuming at Stage {start_stage+1}, Global Step {global_step}, LR {args.lr:g}")

    # ── Curriculum loop ───────────────────────────────────────────────────────
    ckpt_path = "checkpoints/ptd_phase3_final.pt"  # Default final checkpoint path
    for stage_idx, sparsity in enumerate(SPARSITY_SCHEDULE):
        if stage_idx < start_stage:
            continue
            
        set_sparsity(student, sparsity)
        keep_pct = int(sparsity * 100)
        stage_name = f"Stage {stage_idx+1}/{len(SPARSITY_SCHEDULE)}"
        print(f"\n{'='*65}")
        print(f"{stage_name}: sparsity={sparsity} (keep {keep_pct}%)")
        print(f"{'='*65}")

        losses = []
        avg = 0.0  # Initialize avg to avoid unbound variable
        t0 = time.time()

        for step in range(1, args.steps_per_stage + 1):
            global_step += 1
            x, _ = get_batch(data, args.batch, target_device)

            with torch.no_grad():
                t_out = teacher(x)
            t_logits = t_out.logits.detach()

            # Call model.model directly to get selection mask
            s_model_out = student.model(x)
            s_hidden = s_model_out.last_hidden_state
            
            # selection_mask and indices from PTD forward
            mask, sel_indices = s_model_out.hidden_states
            
            # Compute logits from hidden states
            s_logits = student.lm_head(s_hidden)
            
            loss = kl_distill_loss(s_logits, t_logits, T=args.temperature, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg     = sum(losses[-args.log_every:]) / args.log_every
                print(f"  step {step:>5d}/{args.steps_per_stage}  "
                      f"|  loss {avg:.4f}  |  {elapsed:.1f}s  |  keep {keep_pct}%")

            if global_step % args.save_every == 0:
                ckpt_path = f"checkpoints/ptd_phase3_step{global_step:06d}.pt"
                torch.save({
                    "global_step":    global_step,
                    "stage":          stage_idx,
                    "sparsity":       sparsity,
                    "model_state":    student.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss":           avg if losses else 0,
                }, ckpt_path)
                print(f"  ✓ Checkpoint: {ckpt_path}")

        # Save at end of each stage
        ckpt_path = f"checkpoints/ptd_phase3_stage{stage_idx+1}_keep{keep_pct}.pt"
        avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
        torch.save({
            "global_step":    global_step,
            "stage":          stage_idx,
            "sparsity":       sparsity,
            "model_state":    student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss":           avg,
        }, ckpt_path)
        print(f"\n  ✓ Stage complete → {ckpt_path}  (loss={avg:.4f})")

    print(f"\n{'='*65}")
    print(f"Phase 3 complete. Total steps: {global_step}")
    print(f"Final checkpoint: {ckpt_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
