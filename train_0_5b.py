"""
train_0_5b.py  –  Phase 2: Router Warm-up via Teacher Distillation
===================================================================
Following the PTD TRAINING_RECIPE.md (Phase 2):

  1. Load a frozen dense Teacher  (Qwen/Qwen2.5-0.5B)
  2. Load a Student copy wrapped with PTD (trainable routers, frozen backbone)
  3. Feed the same tokens through both.
  4. Minimise KL-divergence between Student logits and Teacher logits.
  5. Only the router weights are trained at this stage.
     (Unfreezing the backbone for Phase 3 is straightforward – see comment below.)

The student runs at `sparsity` token-retention (default 0.3 = keep 30 %).
At sparsity=1.0 the wrapper is lossless so you can sanity-check loss=0.

Usage:
  python train_0_5b.py                     # default settings
  python train_0_5b.py --steps 5 --dry-run # quick smoke-test

Checkpoint saved to  checkpoints/ptd_student_stepXXXXX.pt
"""

import argparse, copy, math, os, time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data",         default="data/tinystories_packed_qwen.pt")
    p.add_argument("--steps",        type=int,   default=3000)
    p.add_argument("--batch",        type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--block-size",   type=int,   default=6)    # layers/PTD block
    p.add_argument("--sparsity",     type=float, default=0.3)  # keep 30 % of segs
    p.add_argument("--segment-size", type=int,   default=16)
    p.add_argument("--save-every",   type=int,   default=500)
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--temperature",  type=float, default=2.0)  # KL softening T
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def kl_distill_loss(student_logits, teacher_logits, T=2.0):
    """
    Soft KL-divergence loss:
      L = T² * KL( softmax(student/T) || softmax(teacher/T) )
    Scaled by T² so gradient magnitudes are independent of temperature.
    """
    s_log_prob = F.log_softmax(student_logits.float() / T, dim=-1)
    t_prob     = F.softmax(teacher_logits.float()    / T, dim=-1)
    kl         = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
    return kl * (T ** 2)


# ──────────────────────────────────────────────────────────────────────────────
def get_batch(data, batch_size, device):
    n = data.shape[0]
    idx = torch.randint(0, n, (batch_size,))
    x = data[idx].to(device)
    return x[:, :-1], x[:, 1:]   # input, labels


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.dry_run:
        args.steps    = 5
        args.log_every = 1
        print("[DRY RUN] Running 5 steps only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load teacher (frozen) ──────────────────────────────────────────────────
    print(f"Loading teacher: {args.model} ...")
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Load student (PTD-wrapped, routers trainable) ──────────────────────────
    print("Wrapping student with PTD ...")
    student = copy.deepcopy(teacher)
    student = apply_ptd_to_qwen2(
        student,
        block_size   = args.block_size,
        sparsity     = args.sparsity,
        segment_size = args.segment_size,
    )
    # Freeze backbone; only train the new router params
    for name, param in student.named_parameters():
        if "ptd_routers" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    print(f"Trainable params: {trainable:,}  /  Total: {total:,}")

    student = student.to(device)
    student.train()

    # ── Data ──────────────────────────────────────────────────────────────────
    if os.path.exists(args.data):
        data = torch.load(args.data, weights_only=True)
        print(f"Data: {data.shape} sequences loaded from {args.data}")
    else:
        print(f"[WARNING] Data file not found: {args.data}")
        print("Generating random token data for smoke-test (run prepare_qwen_data.py first).")
        data = torch.randint(0, 151936, (200, 257))   # 200 × 257 random tokens

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
    )
    os.makedirs("checkpoints", exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nStarting distillation: {args.steps} steps, sparsity={args.sparsity:.0%}, T={args.temperature}")
    print("-" * 65)

    losses = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, _ = get_batch(data, args.batch, device)

        with torch.no_grad():
            t_out = teacher(x)
        t_logits = t_out.logits.detach()

        s_out    = student(x)
        s_logits = s_out.logits

        loss = kl_distill_loss(s_logits, t_logits, T=args.temperature)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg     = sum(losses[-args.log_every:]) / args.log_every
            print(f"  step {step:>6d}/{args.steps}  |  loss {avg:.4f}  |  {elapsed:.1f}s elapsed")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = f"checkpoints/ptd_student_step{step:06d}.pt"
            torch.save({
                "step":           step,
                "router_state":   student.model.ptd_routers.state_dict(),
                "sparsity":       args.sparsity,
                "block_size":     args.block_size,
                "segment_size":   args.segment_size,
                "loss":           avg if losses else 0,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
