import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from qwen_ptd import apply_ptd_to_qwen2
import copy

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu" # Force CPU for strict parity check
    
    print("Loading two clean model copies in FP32 on CPU...")
    # Loading separately to avoid deepcopy issues with patched methods
    model_dense = Qwen2ForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    
    model_sparse = Qwen2ForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    
    # Apply wrapper with 100% retention (sparsity=1.0)
    print("Applying PTD wrapper (100% retention)...")
    model_sparse = apply_ptd_to_qwen2(model_sparse, block_size=6, sparsity=1.0, segment_size=16)
    model_sparse.to(dtype=torch.float32) # ensure routers are FP32
    
    # Use precisely 16 tokens to avoid padding interference
    text = "The quick brown fox jumps over the lazy dog and then it went to the park today."
    # We'll just truncate or pad manually to exactly 16
    inputs = tokenizer(text, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'][:, :16].to(device)
    inputs['attention_mask'] = inputs['attention_mask'][:, :16].to(device)
    
    print(f"Testing with sequence length: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        out_dense = model_dense(**inputs)
        out_sparse_eval = model_sparse(**inputs)
        
    diff_eval = (out_dense.logits - out_sparse_eval.logits).abs().max().item()
    print(f"Max logit difference (EVAL): {diff_eval}")
    
    if diff_eval > 1e-4:
        print("Logits (Dense) [0, :5]:", out_dense.logits[0, 0, :5])
        print("Logits (Sparse) [0, :5]:", out_sparse_eval.logits[0, 0, :5])

    # ── Test TRAIN mode (STE should be 1.0) ──────────────────────────────────
    model_sparse.train()
    with torch.no_grad():
        out_sparse_train = model_sparse(**inputs)
    
    diff_train = (out_dense.logits - out_sparse_train.logits).abs().max().item()
    print(f"Max logit difference (TRAIN): {diff_train}")

    assert diff_eval < 1e-4, "EVAL logits do not match!"
    assert diff_train < 1e-4, "TRAIN logits do not match!"
    print("SUCCESS: Sparse model (100% retention) matches Dense in both modes.")
    
if __name__ == "__main__":
    main()
