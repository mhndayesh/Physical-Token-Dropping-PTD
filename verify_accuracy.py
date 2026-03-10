"""
verify_accuracy.py  –  Measure perplexity drop: Dense vs PTD-Sparse
====================================================================
Computes perplexity on a slice of TinyStories using:
  (a) The original dense Qwen2.5-0.5B
  (b) The PTD-wrapped version at the specified sparsity level

Optionally loads trained router weights from a checkpoint.

Usage:
  python verify_accuracy.py
  python verify_accuracy.py --sparsity 0.3 --checkpoint checkpoints/ptd_student_step003000.pt
"""

import argparse, math, copy
import torch
import torch.nn.functional  as F
from transformers import AutoTokenizer, Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data",        default="data/tinystories_packed_qwen.pt")
    p.add_argument("--sparsity",    type=float, default=0.3)
    p.add_argument("--block-size",  type=int,   default=6)
    p.add_argument("--segment-size",type=int,   default=16)
    p.add_argument("--n-sequences", type=int,   default=100)
    p.add_argument("--checkpoint",  default=None)
    return p.parse_args()


def compute_perplexity(model, data, n_seq, device):
    model.eval()
    total_loss = 0.0
    total_toks = 0
    with torch.no_grad():
        for i in range(min(n_seq, data.shape[0])):
            x = data[i:i+1].to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            
            # Call model.model directly to get selection mask
            model_out = model.model(inp)
            hidden = model_out.last_hidden_state
            logits = model.lm_head(hidden)  # (1, n, vocab)
            
            # PTD models return (selection_mask, indices) in hidden_states.
            # Dense models return hidden_states=None unless explicitly requested.
            if model_out.hidden_states is not None:
                mask, indices = model_out.hidden_states
            
            # Use full sequence for perplexity (scattered-back logits)
            # This gives comparable perplexity to dense baseline
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum",
            )
            total_toks += tgt.numel()
            total_loss += loss.item()
            
    avg_loss = total_loss / (total_toks + 1e-6)
    ppl = math.exp(avg_loss)
    return ppl


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data = torch.load(args.data, weights_only=True)
        print(f"Loaded {data.shape[0]} sequences from {args.data}")
    except FileNotFoundError:
        print(f"Data not found at {args.data} – generating random tokens.")
        data = torch.randint(0, 151936, (100, 257))

    # ── Dense baseline ────────────────────────────────────────────────────────
    print(f"\nLoading dense model: {args.model} ...")
    dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    ppl_dense = compute_perplexity(dense, data, args.n_sequences, device)
    print(f"  Dense PPL:  {ppl_dense:.3f}")

    # ── PTD student ───────────────────────────────────────────────────────────
    print(f"\nWrapping with PTD (sparsity={args.sparsity:.0%}) ...")
    sparse = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    sparse = apply_ptd_to_qwen2(
        sparse,
        block_size   = args.block_size,
        sparsity     = args.sparsity,
        segment_size = args.segment_size,
    )

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            # Phase 3 full-model checkpoint
            sparse.load_state_dict(ckpt["model_state"])
            print(f"  (Phase 3 full-model, stage sparsity={ckpt.get('sparsity', '?')})")
        elif "router_state" in ckpt:
            # Phase 2 router-only checkpoint
            sparse.model.ptd_routers.load_state_dict(ckpt["router_state"])
            print(f"  (Phase 2 router-only)")

    # Ensure everything is on device and in correct dtype
    sparse = sparse.to(device, dtype=dtype)
    ppl_sparse = compute_perplexity(sparse, data, args.n_sequences, device)
    print(f"  Sparse PPL: {ppl_sparse:.3f}")

    delta_pct = (ppl_sparse - ppl_dense) / ppl_dense * 100
    print(f"\n  PPL increase vs dense: {delta_pct:+.1f}%")
    print(f"  (lower is better; ~10-20% expected at 30% retention before fine-tuning)")


if __name__ == "__main__":
    main()
