"""
Scientific Validation #1: True Dense Baseline
Compares Physical Token Dropping (PTD) speed against a REAL nn.TransformerEncoder — no routing overhead.

Scientific Validation #2: OOM Boundary Test
Ramps up sequence length until dense crashes, proves sparse survives.
"""
import torch
import torch.nn as nn
import time
import gc

# === PART 1: TRUE DENSE BASELINE ===

class TrueDenseTransformer(nn.Module):
    """A pure PyTorch dense transformer — no router, no gather/scatter."""
    def __init__(self, d_model=1024, n_heads=16, n_layers=24, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.head.weight

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.embedding(x)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h)

def benchmark_latency(model, input_ids, warmup=3, runs=5):
    """Measures average inference latency."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(input_ids)
        torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000  # ms

def part1_true_baseline():
    print("="*60)
    print("PART 1: TRUE DENSE BASELINE COMPARISON")
    print("="*60)
    
    from transformer_0_5b import SparseTransformer05B, Config
    device = "cuda"
    seq_len = 2048
    
    # True Dense (nn.TransformerEncoder)
    print("\nBuilding True Dense (nn.TransformerEncoder, 24L, d=1024)...")
    true_dense = TrueDenseTransformer(d_model=1024, n_heads=16, n_layers=24).to(device).half()
    true_dense_params = sum(p.numel() for p in true_dense.parameters()) / 1e6
    
    dummy = torch.randint(0, 50257, (1, seq_len), device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = true_dense(dummy)
    true_dense_vram = torch.cuda.max_memory_allocated() / (1024**2)
    true_dense_lat = benchmark_latency(true_dense, dummy)
    
    print(f"  Params: {true_dense_params:.1f}M")
    print(f"  Latency: {true_dense_lat:.1f} ms")
    print(f"  VRAM: {true_dense_vram:.0f} MB")
    
    del true_dense
    torch.cuda.empty_cache()
    gc.collect()
    
    # Physical Token Dropping (PTD) 30%
    print("\nBuilding Physical Token Dropping (PTD) (30%, 24L, d=1024)...")
    cfg = Config()
    cfg.n_layers = 24
    cfg.sparsity = 0.3
    sparse = SparseTransformer05B(cfg).to(device).half()
    sparse_params = sum(p.numel() for p in sparse.parameters()) / 1e6
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = sparse(dummy)
    sparse_vram = torch.cuda.max_memory_allocated() / (1024**2)
    sparse_lat = benchmark_latency(sparse, dummy)
    
    print(f"  Params: {sparse_params:.1f}M")
    print(f"  Latency: {sparse_lat:.1f} ms")
    print(f"  VRAM: {sparse_vram:.0f} MB")
    
    del sparse
    torch.cuda.empty_cache()
    gc.collect()
    
    # Comparison
    speedup = true_dense_lat / sparse_lat
    vram_saved = (1 - sparse_vram / true_dense_vram) * 100
    
    print(f"\n{'─'*60}")
    print(f"RESULT (vs True Dense Baseline):")
    print(f"  True Dense:  {true_dense_lat:.1f} ms | {true_dense_vram:.0f} MB")
    print(f"  Sparse 30%:  {sparse_lat:.1f} ms | {sparse_vram:.0f} MB")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  VRAM Saved:  {vram_saved:.1f}%")
    print(f"{'─'*60}")
    
    return {
        "true_dense_lat": true_dense_lat,
        "true_dense_vram": true_dense_vram,
        "sparse_lat": sparse_lat,
        "sparse_vram": sparse_vram,
        "speedup": speedup,
        "vram_saved": vram_saved,
    }

# === PART 2: OOM BOUNDARY TEST ===

def part2_oom_boundary():
    print(f"\n{'='*60}")
    print("PART 2: OOM BOUNDARY TEST")
    print("Find where Dense dies and Sparse survives.")
    print(f"{'='*60}")
    
    from transformer_0_5b import SparseTransformer05B, Config
    device = "cuda"
    
    # Use smaller model to push sequence length higher
    seq_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576]
    
    dense_results = {}
    sparse_results = {}
    
    # Test Dense
    print("\n--- Dense Model (sparsity=1.0) ---")
    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            cfg = Config()
            cfg.sparsity = 1.0
            cfg.max_seq_len = seq_len + 64
            cfg.n_layers = 12
            cfg.block_size = 4
            model = SparseTransformer05B(cfg).to(device).half()
            dummy = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
            
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(dummy)
            out.sum().backward()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            
            dense_results[seq_len] = peak
            print(f"  SeqLen {seq_len:>6}: {peak:.0f} MB ✓")
            del model, dummy, out
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            dense_results[seq_len] = "OOM"
            print(f"  SeqLen {seq_len:>6}: *** OOM *** 💀")
            del model
            break
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test Sparse (30%)
    print("\n--- Sparse Model (30% retention) ---")
    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            cfg = Config()
            cfg.sparsity = 0.3
            cfg.max_seq_len = seq_len + 64
            cfg.n_layers = 12
            cfg.block_size = 4
            model = SparseTransformer05B(cfg).to(device).half()
            dummy = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
            
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(dummy)
            out.sum().backward()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            
            sparse_results[seq_len] = peak
            print(f"  SeqLen {seq_len:>6}: {peak:.0f} MB ✓")
            del model, dummy, out
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            sparse_results[seq_len] = "OOM"
            print(f"  SeqLen {seq_len:>6}: *** OOM *** 💀")
            del model
            break
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print(f"\n{'─'*60}")
    print("OOM BOUNDARY COMPARISON")
    print(f"{'─'*60}")
    print(f"{'SeqLen':>8}  {'Dense':>12}  {'Sparse 30%':>12}")
    print(f"{'─'*8}  {'─'*12}  {'─'*12}")
    
    all_seqs = sorted(set(list(dense_results.keys()) + list(sparse_results.keys())))
    dense_max = 0
    sparse_max = 0
    for s in all_seqs:
        d = dense_results.get(s, "—")
        sp = sparse_results.get(s, "—")
        d_str = f"{d:.0f} MB" if isinstance(d, float) else d
        sp_str = f"{sp:.0f} MB" if isinstance(sp, float) else sp
        print(f"{s:>8}  {d_str:>12}  {sp_str:>12}")
        if isinstance(d, float): dense_max = s
        if isinstance(sp, float): sparse_max = s
    
    print(f"\n  Dense max sequence:  {dense_max}")
    print(f"  Sparse max sequence: {sparse_max}")
    if sparse_max > dense_max:
        ratio = sparse_max / dense_max
        print(f"  Sparse handles {ratio:.1f}x longer sequences! 🚀")
    
    return dense_results, sparse_results

if __name__ == "__main__":
    baseline = part1_true_baseline()
    dense_oom, sparse_oom = part2_oom_boundary()
