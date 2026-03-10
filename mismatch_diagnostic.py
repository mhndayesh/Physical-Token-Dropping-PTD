import torch
import copy
from transformers import Qwen2ForCausalLM, AutoTokenizer
from qwen_ptd import apply_ptd_to_qwen2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dense model
    print("Loading dense model...")
    model_dense = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model_dense.eval()
    
    # Text input
    text = "The quick brown fox"
    inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
    n = inputs['input_ids'].shape[1]
    
    # Get dense baseline hidden states after block 1 (layer 6)
    # We use a hook or just temporary return? Let's just run it and see.
    with torch.no_grad():
        out_dense = model_dense.model(inputs['input_ids'], output_hidden_states=True)
        # hidden_states is a tuple: (embed, layer1, layer2, ... layer24)
        # Block 1 in PTD is layers 0-5. So after layer 5 (index 6 in tuple)
        dense_h6 = out_dense.hidden_states[6] # (1, n, d)

    # Load sparse model (100% retention)
    print("Applying PTD wrapper (100% retention)...")
    model_sparse = copy.deepcopy(model_dense)
    model_sparse = apply_ptd_to_qwen2(model_sparse, block_size=6, sparsity=1.0)
    model_sparse.eval()
    
    with torch.no_grad():
        out_sparse = model_sparse.model(inputs['input_ids'])
        # In my PTD version, I return selection_mask in hidden_states[0]
        # and current_indices in hidden_states[1] (wait, I updated it to 0 and 1)
        
        # Let's check the FINAL hidden state first
        sparse_h_final = out_sparse.last_hidden_state
        dense_h_final = out_dense.last_hidden_state
        
        diff_final = (dense_h_final - sparse_h_final).abs().max().item()
        print(f"Final Hidden State Max Diff: {diff_final}")
        
    # Check if the layers are physically identical
    print(f"Layer 0 weight match: {torch.equal(model_dense.model.layers[0].self_attn.q_proj.weight, model_sparse.model.layers[0].self_attn.q_proj.weight)}")

if __name__ == "__main__":
    main()
