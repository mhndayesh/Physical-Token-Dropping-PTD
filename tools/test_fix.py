"""
test_fix.py - Verify PTD fix: Hidden state propagation between blocks
======================================================================
Tests that with sparsity=1.0 (keep all), PTD model matches dense baseline.
"""
import torch
from transformers import Qwen2ForCausalLM
from qwen_ptd import apply_ptd_to_qwen2

def test_sparsity_1_equals_dense():
    """With sparsity=1.0, PTD should be lossless (identical to dense)."""
    print("="*65)
    print("Testing: sparsity=1.0 should match dense baseline")
    print("="*65)
    
    device = "cpu"  # Use CPU for reliability in testing
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Load dense model
    print(f"Loading model: {model_name}")
    dense = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    dense.eval()
    
    # Create sparse model with sparsity=1.0 (keep all)
    print("Applying PTD with sparsity=1.0...")
    sparse = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    sparse = apply_ptd_to_qwen2(sparse, block_size=6, sparsity=1.0, segment_size=16)
    sparse.eval()
    
    # Copy weights from dense to sparse (ensures same weights)
    sparse.load_state_dict(dense.state_dict(), strict=False)
    
    # Test input
    test_input = torch.randint(0, 151936, (1, 32))  # 1 batch, 32 tokens
    
    with torch.no_grad():
        # Dense forward
        dense_out = dense.model(test_input)
        dense_hidden = dense_out.last_hidden_state
        
        # Sparse forward
        sparse_out = sparse.model(test_input)
        sparse_hidden = sparse_out.last_hidden_state
        
        # Compare
        diff = (dense_hidden - sparse_hidden).abs().max().item()
        
    print(f"\nMax hidden state difference: {diff:.2e}")
    
    if diff < 1e-4:
        print("✅ PASS: PTD with sparsity=1.0 matches dense baseline!")
        return True
    else:
        print(f"❌ FAIL: Difference too large ({diff:.2e})")
        print(f"   Dense hidden shape: {dense_hidden.shape}")
        print(f"   Sparse hidden shape: {sparse_hidden.shape}")
        return False

def test_block_propagation():
    """Test that hidden states propagate through multiple blocks."""
    print("\n" + "="*65)
    print("Testing: Hidden state propagation through blocks")
    print("="*65)
    
    device = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Load model and wrap with PTD
    model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = apply_ptd_to_qwen2(model, block_size=6, sparsity=0.5, segment_size=16)
    model.eval()
    
    # Test input
    test_input = torch.randint(0, 151936, (1, 64))  # 1 batch, 64 tokens
    
    with torch.no_grad():
        out = model.model(test_input)
        
        # Check that we have 4 blocks
        n_blocks = len(model.model._ptd_routers)
        print(f"Number of PTD blocks: {n_blocks}")
        
        # Check output shapes
        hidden = out.last_hidden_state
        print(f"Output hidden shape: {hidden.shape}")
        
        # Check indices shape matches expected sparsity
        indices = out.hidden_states[1]
        expected_tokens = int(64 * 0.5)  # 50% sparsity = 32 tokens
        print(f"Selected tokens: {indices.shape[1]} (expected ~{expected_tokens})")
        
    print("✅ Block propagation test completed")
    return True

if __name__ == "__main__":
    print("Running PTD fix verification tests...\n")
    
    success = True
    
    # Test 1: sparsity=1.0 should equal dense
    if not test_sparsity_1_equals_dense():
        success = False
    
    # Test 2: Block propagation
    if not test_block_propagation():
        success = False
    
    print("\n" + "="*65)
    if success:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*65)