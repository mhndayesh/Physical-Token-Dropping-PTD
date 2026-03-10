"""
verify_fine_tuned.py  –  Text generation with the PTD-wrapped model
====================================================================
Generate text from prompts to visually inspect the quality of the
sparse model before and after router warm-up.

Usage:
  python verify_fine_tuned.py
  python verify_fine_tuned.py --checkpoint checkpoints/ptd_student_step003000.pt
  python verify_fine_tuned.py --sparsity 0.5 --max-new 200
"""

import argparse, copy
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

PROMPTS = [
    "Once upon a time, there was a little girl named Emma who loved to",
    "The clever fox thought for a moment and then decided to",
    "In a village near a great forest, the children discovered a",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--sparsity",     type=float, default=0.3)
    p.add_argument("--block-size",   type=int,   default=6)
    p.add_argument("--segment-size", type=int,   default=16)
    p.add_argument("--max-new",      type=int,   default=150)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--compare-dense", action="store_true",
                   help="Also generate from the original dense model for comparison")
    return p.parse_args()


def generate(model, tokenizer, prompt, max_new, device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens    = max_new,
            do_sample         = True,
            temperature       = 0.8,
            top_p             = 0.9,
            repetition_penalty= 1.1,
            pad_token_id      = tokenizer.eos_token_id,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model} ...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    base = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    # ── Optional: dense reference ─────────────────────────────────────────────
    if args.compare_dense:
        print("\n" + "="*65)
        print("DENSE REFERENCE (no token dropping)")
        print("="*65)
        for prompt in PROMPTS:
            print(f"\nPROMPT: {prompt}")
            print(generate(base, tokenizer, prompt, args.max_new, device))

    # ── PTD sparse ────────────────────────────────────────────────────────────
    print(f"\nWrapping with PTD (sparsity={args.sparsity:.0%}) ...")
    sparse = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    sparse = apply_ptd_to_qwen2(
        sparse,
        block_size   = args.block_size,
        sparsity     = args.sparsity,
        segment_size = args.segment_size,
    )

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            sparse.load_state_dict(ckpt["model_state"])
            print(f"  (Phase 3 full-model checkpoint)")
        elif "router_state" in ckpt:
            sparse.model.ptd_routers.load_state_dict(ckpt["router_state"])
            print(f"  (Phase 2 router-only checkpoint)")

    sparse = sparse.to(device, dtype=dtype)
    print(f"\n" + "="*65)
    print(f"PTD SPARSE MODEL (keep {args.sparsity:.0%} of segments)")
    print("="*65)

    for prompt in PROMPTS:
        print(f"\nPROMPT: {prompt}")
        print(generate(sparse, tokenizer, prompt, args.max_new, device))


if __name__ == "__main__":
    main()
