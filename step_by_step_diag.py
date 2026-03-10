import torch
import types
from transformers import Qwen2ForCausalLM, AutoTokenizer
from qwen_ptd import apply_ptd_to_qwen2

def main():
    device = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    ids = inputs['input_ids']
    
    # 1. Baseline: Layer 0 output
    with torch.no_grad():
        emb = model.model.embed_tokens(ids)
        # Rotary
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(emb, pos)
        # Mask
        mask = model.model._update_causal_mask(None, emb, None, 0, False)
        
        baseline_l0 = model.model.layers[0](
            emb,
            attention_mask=mask,
            position_ids=pos,
            position_embeddings=(cos, sin)
        )[0]

    # 2. PTD Wrapper logic for Layer 0
    # sparsity=1.0, no padding (ids=4 tokens)
    # We simulate exactly what _ptd_model_forward does
    with torch.no_grad():
        x_sp = emb
        sp_pos = pos
        cos_sp, sin_sp = model.model.rotary_emb(x_sp, sp_pos)
        
        # My custom mask
        from qwen_ptd import _causal_mask_sparse
        hf_mask = _causal_mask_sparse(sp_pos, torch.float32, device)
        
        ptd_l0 = model.model.layers[0](
            hidden_states       = x_sp,
            attention_mask      = hf_mask,
            position_ids        = sp_pos,
            position_embeddings = (cos_sp, sin_sp),
        )[0]
        
    diff = (baseline_l0 - ptd_l0).abs().max().item()
    print(f"Layer 0 output diff: {diff}")
    
    if diff > 1e-5:
        print("Mask Baseline Shape:", mask.shape)
        print("Mask PTD Shape:", hf_mask.shape)
        print("Mask Baseline (0,0,:,:):", mask[0, 0])
        print("Mask PTD (0,0,:,:):", hf_mask[0, 0])

if __name__ == "__main__":
    main()
