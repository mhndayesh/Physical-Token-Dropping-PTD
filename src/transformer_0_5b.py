import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None, position_ids=None):
        # Bug #2 Fix: Support explicit position IDs for sparse tokens
        if position_ids is not None:
            # position_ids: (b, k) — original sequence positions
            # Build per-token cos/sin from the precomputed cache
            # cos_cached: (1, 1, max_seq, dim)
            cos = self.cos_cached.squeeze(0).squeeze(0)  # (max_seq, dim)
            sin = self.sin_cached.squeeze(0).squeeze(0)  # (max_seq, dim)
            # position_ids: (b, k) -> gather from cache
            cos_pos = cos[position_ids]  # (b, k, dim)
            sin_pos = sin[position_ids]  # (b, k, dim)
            # x is (b, n_heads, k, head_dim), need (b, 1, k, dim)
            cos_pos = cos_pos.unsqueeze(1)  # (b, 1, k, dim)
            sin_pos = sin_pos.unsqueeze(1)  # (b, 1, k, dim)
            return x * cos_pos + self._rotate_half(x) * sin_pos
        return x * self.cos_cached[:, :, :seq_len, :] + self._rotate_half(x) * self.sin_cached[:, :, :seq_len, :]

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

class MultiQueryRouter(nn.Module):
    def __init__(self, d_model, num_queries=8, rank=16, sparsity=0.1, jitter=0.01):
        super().__init__()
        self.num_queries = num_queries
        self.rank = rank
        self.sparsity = sparsity
        self.jitter = jitter
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(num_queries, rank))

    def forward(self, x):
        b, n, d = x.shape
        k = max(1, int(n * self.sparsity))
        keys = self.k_proj(x) # (b, n, rank)
        
        # Multi-Query overlap for richer scoring
        # scores: (b, q, n)
        scores = torch.matmul(self.queries.unsqueeze(0), keys.transpose(1, 2)) 
        token_scores = scores.max(dim=1).values # (b, n)
        
        # Hardening: Stochastic Jitter to prevent token starvation
        if self.training and self.jitter > 0:
            noise = torch.randn_like(token_scores) * self.jitter
            token_scores = token_scores + noise
            
        _, indices = torch.topk(token_scores, k, dim=-1)
        indices, _ = torch.sort(indices, dim=-1)
        return indices

class SwiGLU(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class SparseBlockAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_sparse, position_ids=None):
        b, k_len, d = x_sparse.shape
        q = self.q_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_sparse).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Bug #2 Fix: Use original position IDs for RoPE, not local 0..k-1
        q = self.rope(q, seq_len=k_len, position_ids=position_ids)
        k = self.rope(k, seq_len=k_len, position_ids=position_ids)
        # Bug #3 Fix: Build explicit causal mask from original positions
        # Since indices are sorted, is_causal=True on sorted order IS correct.
        # But for maximum rigor, we build a mask from original positions.
        if position_ids is not None:
            # position_ids: (b, k) — sorted original positions
            # causal: pos_i <= pos_j means token i can attend to token j
            pos_i = position_ids.unsqueeze(-1)  # (b, k, 1)
            pos_j = position_ids.unsqueeze(-2)  # (b, 1, k)
            causal_mask = (pos_i >= pos_j).unsqueeze(1)  # (b, 1, k, k)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask
            )
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, k_len, d)
        return self.out_proj(attn_output)

class SparseLayer(nn.Module):
    def __init__(self, d_model, n_heads, rope):
        super().__init__()
        self.attn = SparseBlockAttention(d_model, n_heads, rope)
        self.ffn = SwiGLU(d_model, 4 * d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x_sparse, position_ids=None):
        x_sparse = x_sparse + self.attn(self.norm1(x_sparse), position_ids=position_ids)
        x_sparse = x_sparse + self.ffn(self.norm2(x_sparse))
        return x_sparse

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, sparsity, rope, segment_size=16):
        super().__init__()
        self.segment_size = segment_size
        # Router now operates on segments instead of tokens
        self.router = MultiQueryRouter(d_model, sparsity=sparsity)
        self.layers = nn.ModuleList([
            SparseLayer(d_model, n_heads, rope) for _ in range(block_size)
        ])

    def forward(self, x):
        b, n, d = x.shape
        
        # Pad sequence length to be divisible by segment_size
        pad_len = (self.segment_size - (n % self.segment_size)) % self.segment_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
            
        n_padded = x_padded.shape[1]
        n_segments = n_padded // self.segment_size
        x_segments = x_padded.view(b, n_segments, self.segment_size, d)
        
        # Route on segment-wise average for efficiency
        x_pooled = x_segments.mean(dim=2) # (b, n_segments, d)
        seg_indices = self.router(x_pooled) # (b, k_segments)
        k_seg = seg_indices.shape[1]
        
        # Expand segment indices to token-level indices
        # indices: (b, k_seg * segment_size) — these ARE the original positions
        indices = (seg_indices.unsqueeze(-1) * self.segment_size + 
                  torch.arange(self.segment_size, device=x.device)).view(b, -1)
        
        k_len = indices.shape[1]
        batch_indices = torch.arange(b, device=x.device).view(-1, 1).expand(-1, k_len)
        
        # GATHER tokens at block start (Physically contiguous blocks)
        x_sparse = x_padded[batch_indices, indices]
        
        # Bug #2 Fix: Pass original position IDs for correct RoPE encoding
        # indices already contain the original sequence positions
        position_ids = indices  # (b, k_len)
        
        for layer in self.layers:
            x_sparse = layer(x_sparse, position_ids=position_ids)
            
        # Bug #1 Fix: No double residual. Clone x and replace selected tokens.
        # SparseLayer already applies internal residual connections,
        # so we must NOT add x again.
        res = x_padded.clone()
        res[batch_indices, indices] = x_sparse
        return res[:, :n, :]

class SparseTransformer05B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RoPE(config.d_model // config.n_heads, max_position_embeddings=config.max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.block_size, config.sparsity, self.rope)
            for _ in range(config.n_layers // config.block_size)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embedding.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

class Config:
    def __init__(self):
        self.d_model = 1024
        self.n_heads = 16
        self.n_layers = 24
        self.block_size = 4
        self.sparsity = 0.1
        self.vocab_size = 50257
        self.max_seq_len = 2048

if __name__ == "__main__":
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseTransformer05B(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.1f}M")
    dummy_input = torch.randint(0, config.vocab_size, (1, 512)).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
