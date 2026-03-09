import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LowRankRouter(nn.Module):
    """
    Predicts token importance using low-rank projections (d -> r).
    Selects Top-K tokens to form a routing mask.
    """
    def __init__(self, d_model, rank=16, sparsity=0.1, jitter=0.01):
        super().__init__()
        self.rank = rank
        self.sparsity = sparsity
        self.jitter = jitter
        self.q_proj = nn.Linear(d_model, rank, bias=False)
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, n, d = x.shape
        k = max(1, int(n * self.sparsity))
        
        # O(N) projection
        q_low = self.q_proj(x) # (b, n, rank)
        k_low = self.k_proj(x) # (b, n, rank)
        
        # O(N) scoring via global query pooling
        q_global = q_low.mean(dim=1, keepdim=True) # (b, 1, rank)
        
        # Token importance = dot product with global query
        # scores: (b, n)
        scores = torch.matmul(q_global, k_low.transpose(-2, -1)).squeeze(1)
        
        # Hardening: Stochastic Jitter to prevent token starvation
        if self.training and self.jitter > 0:
            noise = torch.randn_like(scores) * self.jitter
            scores = scores + noise
            
        _, indices = torch.topk(scores, k, dim=-1)
        # indices: (b, k)
        
        return indices

class SparseBlockAttention(nn.Module):
    """
    Executes attention only on a subset of tokens provided by indices.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, indices=None):
        b, n, d = x.shape
        
        if indices is not None:
            # Physical token dropping: Gather only the active tokens
            # indices: (b, k)
            k_len = indices.shape[1]
            
            # Batch-wise gathering
            # Flatten indices for easier gathering if needed, or use batch_index_select
            batch_indices = torch.arange(b, device=x.device).view(-1, 1).expand(-1, k_len)
            x_sparse = x[batch_indices, indices] # (b, k, d)
            
            # Run Attention on the subset
            q = self.q_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Standard Scaled Dot-Product Attention on (b, heads, k, head_dim)
            # Complexity: O(k^2) instead of O(n^2)
            attn_output = F.scaled_dot_product_attention(q, k, v)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(b, k_len, d)
            out = self.out_proj(attn_output)
            
            # Scatter back with Residual Preservation
            # Instead of returning 0 for dropped tokens, we return the original hidden state x
            res = x.clone()
            res[batch_indices, indices] = out
            return res
        else:
            # Dense fallback
            q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, d)
            return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    """
    A block of B layers. Layer 0 computes the mask; others reuse it.
    """
    def __init__(self, d_model, n_heads, block_size=4, sparsity=0.1):
        super().__init__()
        self.block_size = block_size
        self.router = LowRankRouter(d_model, sparsity=sparsity)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': SparseBlockAttention(d_model, n_heads),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(block_size)
        ])
        
    def forward(self, x):
        # Step 2: Prediction Phase (only in Layer 1 of Block)
        indices = self.router(x)
        
        for i, layer in enumerate(self.layers):
            # Step 3 & 4: Sparse Execution & Free Ride
            # Every layer in the block uses the same 'indices'
            
            # Attention
            norm_x = layer['norm1'](x)
            x = x + layer['attn'](norm_x, indices=indices)
            
            # FFN (could also be sparse, but let's stick to Attention for now)
            x = x + layer['ffn'](layer['norm2'](x))
            
        return x

class DynamicSparseTransformer(nn.Module):
    """
    Full Transformer model composed of multiple Block-Shared Dynamic Sparse blocks.
    """
    def __init__(self, d_model, n_heads, n_blocks, block_size=4, sparsity=0.1, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, sparsity)
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

if __name__ == "__main__":
    # Test Run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 12-layer model (3 blocks of 4 layers each)
    model = DynamicSparseTransformer(
        d_model=256, 
        n_heads=8, 
        n_blocks=3, 
        block_size=4,
        sparsity=0.1
    ).to(device)
    
    dummy_input = torch.randint(0, 50257, (8, 512)).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Successfully executed Dynamic Sparse Transformer!")
