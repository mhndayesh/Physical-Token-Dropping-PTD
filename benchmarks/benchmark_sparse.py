import torch
import torch.nn as nn
import time
import pandas as pd
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformer_0_5b import SparseTransformer05B, Config

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA is required for this benchmark script.")
    print(f"Comparing Scales on {device}...")
    
    class ScaleConfig(Config):
        def __init__(self, layers, sparsity):
            super().__init__()
            self.n_layers = layers
            self.sparsity = sparsity
            self.d_model = 1024
            self.n_heads = 16
            self.vocab_size = 50257

    test_configs = [
        {"name": "200M Dense (12L)", "layers": 12, "sparsity": 1.0},
        {"name": "200M Sparse (10%)", "layers": 12, "sparsity": 0.1},
        {"name": "500M Dense (24L)", "layers": 24, "sparsity": 1.0},
        {"name": "500M Sparse (10%)", "layers": 24, "sparsity": 0.1},
    ]

    seq_len = 2048 
    batch_size = 1
    results = []

    for item in test_configs:
        print(f"Benchmarking {item['name']}...")
        cfg = ScaleConfig(item['layers'], item['sparsity'])
        try:
            model = SparseTransformer05B(cfg).to(device).half()
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
            
            # Memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(dummy_input)
            out.sum().backward()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            
            # Latency (Inference)
            model.eval()
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            latency = (time.time() - start) / 5 * 1000
            
            results.append({
                "Model": item['name'],
                "Params (M)": f"{params:.1f}M",
                "Peak VRAM (MB)": peak_mem,
                "Latency (ms)": latency
            })
            del model, dummy_input, out
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append({
                "Model": item['name'],
                "Params (M)": "OOM",
                "Peak VRAM (MB)": "OOM",
                "Latency (ms)": "OOM"
            })
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n--- FINAL SCALE COMPARISON ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    benchmark()
