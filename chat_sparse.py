"""
chat_sparse.py  –  Interactive Chat with PTD Sparse Qwen
=======================================================
Run interactive chat with the 30% retention sparse model.
"""

import argparse
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--sparsity",     type=float, default=0.3)
    p.add_argument("--checkpoint",   default="checkpoints/ptd_phase3_stage5_keep30.pt")
    p.add_argument("--max-new",      type=int,   default=200)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print(f"Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Loading PTD Sparse Model (keep {args.sparsity:.0%})...")
    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = apply_ptd_to_qwen2(model, sparsity=args.sparsity)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "router_state" in ckpt:
            model.model.ptd_routers.load_state_dict(ckpt["router_state"])
            
    model = model.to(device, dtype=dtype)
    model.eval()

    print("\n" + "="*60)
    print("PTD SPARSE CHAT (30% retention)")
    print("Type 'quit' to exit.")
    print("="*60)

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["quit", "exit"]:
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation with PTD
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nSparse Qwen: {response}")

if __name__ == "__main__":
    main()
