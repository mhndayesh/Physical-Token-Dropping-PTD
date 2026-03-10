import torch
from transformers import Qwen2ForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    rotary = model.model.rotary_emb
    
    # Test: Full sequence
    pos_full = torch.arange(20).unsqueeze(0) # (1, 20)
    dummy_x = torch.randn(1, 20, 64)
    # rotary_emb(x, position_ids)
    cos_full, sin_full = rotary(dummy_x, pos_full)
    
    # Test: Sliced sequence (e.g. positions 10..15)
    pos_sp = torch.arange(10, 16).unsqueeze(0) # (1, 6)
    dummy_sp = torch.randn(1, 6, 64)
    cos_sp, sin_sp = rotary(dummy_sp, pos_sp)
    
    # Check if cos_sp matches cos_full[0, 10:16]
    target_cos = cos_full[:, 10:16, :]
    diff = (cos_sp - target_cos).abs().max().item()
    
    print(f"RoPE Full Shape: {cos_full.shape}")
    print(f"RoPE Sparse Shape: {cos_sp.shape}")
    print(f"Max diff (sparse vs sliced full): {diff}")
    
    if diff < 1e-6:
        print("SUCCESS: rotary_emb handles non-contiguous position_ids correctly.")
    else:
        print("FAILURE: rotary_emb does NOT handle position_ids correctly. It likely just uses length.")

if __name__ == "__main__":
    main()
